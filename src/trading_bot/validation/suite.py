"""Full validation suite for model edge assessment."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from trading_bot.backtest.simulator import BacktestSimulator, build_simulation_frame
from trading_bot.backtest.strategy import Strategy, get_default_strategy
from trading_bot.logging_config import get_logger
from trading_bot.models.model_registry import ModelRegistry
from trading_bot.schemas.datasets import infer_horizon_bars_from_target_column
from trading_bot.schemas.backtest import ExecutionConfig, StrategyConfig
from trading_bot.schemas.validation import ValidationReport
from trading_bot.settings import PROJECT_ROOT, AppSettings, load_settings
from trading_bot.storage.backtest_store import BacktestStore
from trading_bot.storage.parquet_store import ParquetStore
from trading_bot.storage.validation_store import ValidationStore
from trading_bot.validation.baselines import (
    generate_always_flat_signals,
    generate_buy_and_hold_signals,
    generate_random_matched_frequency_signals,
    generate_sma_crossover_signals,
)

logger = get_logger(__name__)


class ValidationSuiteError(Exception):
    """Raised when the validation suite cannot complete."""


class ValidationSuite:
    """Run model-vs-baseline validation, diagnostics, and audit reporting."""

    def __init__(
        self,
        *,
        model_registry: ModelRegistry,
        simulator: BacktestSimulator,
        market_store: ParquetStore,
        backtest_store: BacktestStore,
        validation_store: ValidationStore,
        settings: AppSettings,
    ) -> None:
        self.model_registry = model_registry
        self.simulator = simulator
        self.market_store = market_store
        self.backtest_store = backtest_store
        self.validation_store = validation_store
        self.settings = settings

    @classmethod
    def from_settings(cls, settings: AppSettings | None = None) -> ValidationSuite:
        """Create the validation suite from project settings."""
        resolved_settings = settings or load_settings()
        return cls(
            model_registry=ModelRegistry.from_settings(resolved_settings),
            simulator=BacktestSimulator.from_settings(resolved_settings),
            market_store=ParquetStore(PROJECT_ROOT / resolved_settings.market_data.processed_data_dir),
            backtest_store=BacktestStore(PROJECT_ROOT / resolved_settings.simulations.backtests_dir),
            validation_store=ValidationStore(PROJECT_ROOT / resolved_settings.validation_suite.output_dir),
            settings=resolved_settings,
        )

    def run_validation_suite(
        self,
        *,
        model_run_id: str,
        validation_id: str | None = None,
    ) -> dict[str, Any]:
        """Run standardized model and baseline backtests plus diagnostics."""
        resolved_run = self.model_registry.resolve_run(model_run_id)
        predictions_df = self.model_registry.read_predictions(model_run_id)
        feature_importance_df = self.model_registry.read_feature_importance(model_run_id, aggregate=True)
        market_df = self.market_store.read(resolved_run.symbol, resolved_run.timeframe)
        joined_df = build_simulation_frame(predictions_df, market_df)
        if joined_df.empty:
            raise ValidationSuiteError(
                f"No overlapping timestamps found for run_id={model_run_id} and market candles"
            )

        resolved_validation_id = validation_id or _default_validation_id()
        execution = _execution_config_from_settings(self.settings)
        model_strategy = Strategy(config=get_default_strategy(self.settings))

        model_result = self.simulator.run_simulation_from_inputs(
            simulation_type="backtest",
            model_run_id=model_run_id,
            symbol=resolved_run.symbol,
            timeframe=resolved_run.timeframe,
            joined_df=joined_df,
            signals_df=_model_signals(joined_df, model_strategy.config),
            strategy=model_strategy,
            execution_config=execution,
            simulation_id=f"{resolved_validation_id}__model",
            provenance={
                "validation_id": resolved_validation_id,
                "strategy_name": "model",
                "linked_model_run_id": model_run_id,
                "feature_version": resolved_run.metadata.get("feature_version"),
                "dataset_version": resolved_run.metadata.get("dataset_version"),
            },
        )
        model_trades_df = self.backtest_store.read_trades(model_result.simulation_id)
        model_equity_df = self.backtest_store.read_equity_curve(model_result.simulation_id)
        model_signals_df = self.backtest_store.read_signals(model_result.simulation_id)

        baseline_results = self._run_baselines(
            validation_id=resolved_validation_id,
            model_run_id=model_run_id,
            symbol=resolved_run.symbol,
            timeframe=resolved_run.timeframe,
            joined_df=joined_df,
            execution=execution,
            reference_trade_count=int(model_result.metrics.number_of_trades),
            reference_average_holding_bars=float(model_result.metrics.average_holding_bars),
        )

        comparison_df = self._build_strategy_comparison(model_result, baseline_results)
        diagnostics_df = build_prediction_diagnostics(
            predictions_df=joined_df,
            bucket_edges=self.settings.validation_suite.bucket_edges,
        )
        failure_analysis = build_failure_analysis(
            trades_df=model_trades_df,
            equity_curve_df=model_equity_df,
            signals_df=model_signals_df,
            top_failure_cases=self.settings.validation_suite.top_failure_cases,
            drawdown_case_count=self.settings.validation_suite.drawdown_case_count,
        )
        feature_importance_summary = build_feature_importance_audit(feature_importance_df)
        simulation_audit = build_simulation_audit(
            metrics=model_result.metrics.to_dict(),
            trades_df=model_trades_df,
            equity_curve_df=model_equity_df,
            overtrading_threshold=self.settings.validation_suite.overtrading_trades_per_day_threshold,
            tiny_edge_after_fees_threshold=self.settings.validation_suite.tiny_edge_after_fees_threshold,
        )
        conclusion = build_validation_conclusion(
            comparison_df=comparison_df,
            prediction_diagnostics_df=diagnostics_df,
            simulation_audit=simulation_audit,
        )

        artifact_paths = {
            "report_json_path": str(
                self.validation_store.report_dir(resolved_validation_id) / "validation_report.json"
            ),
            "report_markdown_path": str(
                self.validation_store.report_dir(resolved_validation_id) / "validation_report.md"
            ),
        }
        report = ValidationReport(
            validation_id=resolved_validation_id,
            model_run_id=model_run_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            symbol=resolved_run.symbol,
            timeframe=resolved_run.timeframe,
            target_column=str(resolved_run.metadata.get("target_column", "")),
            dataset_version=str(resolved_run.metadata.get("dataset_version", "")),
            feature_version=str(resolved_run.metadata.get("feature_version", "")),
            label_version=str(resolved_run.metadata.get("label_version", "")),
            horizon_bars=infer_horizon_bars_from_target_column(
                str(resolved_run.metadata.get("target_column", ""))
            ),
            strategy_comparison=comparison_df.to_dict(orient="records"),
            prediction_diagnostics={
                "rows": diagnostics_df.to_dict(orient="records"),
                "bucket_edges": self.settings.validation_suite.bucket_edges,
            },
            failure_analysis=failure_analysis,
            feature_importance_summary=feature_importance_summary,
            simulation_audit=simulation_audit,
            conclusion=conclusion,
            artifact_paths=artifact_paths,
        ).to_dict()

        self.validation_store.write_table(resolved_validation_id, "strategy_comparison", comparison_df)
        self.validation_store.write_table(resolved_validation_id, "prediction_diagnostics", diagnostics_df)
        self.validation_store.write_table(
            resolved_validation_id,
            "top_losing_trades",
            pd.DataFrame(failure_analysis.get("top_losing_trades", [])),
        )
        self.validation_store.write_report_json(resolved_validation_id, report)
        self.validation_store.write_report_markdown(
            resolved_validation_id,
            format_validation_report_markdown(report),
        )
        logger.info(
            "validation_suite_complete",
            validation_id=resolved_validation_id,
            model_run_id=model_run_id,
            has_edge=conclusion.get("has_edge"),
        )
        return report

    def _run_baselines(
        self,
        *,
        validation_id: str,
        model_run_id: str,
        symbol: str,
        timeframe: str,
        joined_df: pd.DataFrame,
        execution: ExecutionConfig,
        reference_trade_count: int,
        reference_average_holding_bars: float,
    ) -> list[dict[str, Any]]:
        """Run the configured baseline strategies on the same joined frame."""
        baseline_specs = [
            (
                "buy_and_hold",
                Strategy(
                    config=StrategyConfig(
                        strategy_version="buy_and_hold",
                        type="long_only_threshold",
                        entry_threshold=1.0,
                        exit_threshold=0.0,
                        minimum_holding_bars=1,
                        cooldown_bars=0,
                    )
                ),
                generate_buy_and_hold_signals(joined_df),
            ),
            (
                "always_flat",
                Strategy(
                    config=StrategyConfig(
                        strategy_version="always_flat",
                        type="long_only_threshold",
                        entry_threshold=1.0,
                        exit_threshold=0.0,
                        minimum_holding_bars=1,
                        cooldown_bars=0,
                    )
                ),
                generate_always_flat_signals(joined_df),
            ),
            (
                "random_matched_frequency",
                Strategy(
                    config=StrategyConfig(
                        strategy_version="random_matched_frequency",
                        type="long_only_threshold",
                        entry_threshold=1.0,
                        exit_threshold=0.0,
                        minimum_holding_bars=1,
                        cooldown_bars=0,
                    )
                ),
                generate_random_matched_frequency_signals(
                    joined_df,
                    reference_trade_count=reference_trade_count,
                    reference_average_holding_bars=reference_average_holding_bars,
                    random_seed=self.settings.validation_suite.random_seed,
                ),
            ),
            (
                "sma_crossover_20_50",
                Strategy(
                    config=StrategyConfig(
                        strategy_version="sma_crossover_20_50",
                        type="long_only_threshold",
                        entry_threshold=1.0,
                        exit_threshold=0.0,
                        minimum_holding_bars=1,
                        cooldown_bars=0,
                    )
                ),
                generate_sma_crossover_signals(joined_df, fast_window=20, slow_window=50),
            ),
        ]

        results: list[dict[str, Any]] = []
        for strategy_name, strategy, signals_df in baseline_specs:
            result = self.simulator.run_simulation_from_inputs(
                simulation_type="backtest",
                model_run_id=f"BASELINE:{strategy_name}",
                symbol=symbol,
                timeframe=timeframe,
                joined_df=joined_df,
                signals_df=signals_df,
                strategy=strategy,
                execution_config=execution,
                simulation_id=f"{validation_id}__{strategy_name}",
                provenance={
                    "validation_id": validation_id,
                    "strategy_name": strategy_name,
                    "linked_model_run_id": model_run_id,
                },
            )
            results.append(
                {
                    "strategy_name": strategy_name,
                    "result": result,
                }
            )
        return results

    @staticmethod
    def _build_strategy_comparison(
        model_result,
        baseline_results: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Return a ranked comparison table for model and baselines."""
        rows = [_comparison_row("model", model_result)]
        rows.extend(
            _comparison_row(str(entry["strategy_name"]), entry["result"])
            for entry in baseline_results
        )
        comparison_df = pd.DataFrame(rows)
        comparison_df = comparison_df.sort_values(
            by=["total_return", "sharpe_like"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)
        comparison_df["rank"] = comparison_df.index + 1
        return comparison_df.reindex(
            columns=[
                "rank",
                "strategy_name",
                "simulation_id",
                "model_run_id",
                "total_return",
                "max_drawdown",
                "sharpe_like",
                "win_rate",
                "number_of_trades",
                "average_trade_return",
                "exposure_ratio",
                "turnover",
                "fee_impact",
            ]
        )


def build_prediction_diagnostics(
    *,
    predictions_df: pd.DataFrame,
    bucket_edges: list[float],
) -> pd.DataFrame:
    """Bucket model probabilities and audit confidence quality."""
    if predictions_df.empty:
        return pd.DataFrame(
            columns=["bucket", "trade_count", "accuracy_per_bucket", "average_return_per_bucket"]
        )

    frame = predictions_df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="coerce")
    frame["y_pred"] = pd.to_numeric(frame["y_pred"], errors="coerce")
    frame["y_proba"] = pd.to_numeric(frame["y_proba"], errors="coerce")
    frame["next_close"] = pd.to_numeric(frame["close"], errors="coerce").shift(-1)
    frame["forward_return_1bar"] = (frame["next_close"] / pd.to_numeric(frame["close"], errors="coerce")) - 1.0
    frame = frame[frame["y_proba"] >= bucket_edges[0]].copy()

    if frame.empty:
        return pd.DataFrame(
            columns=["bucket", "trade_count", "accuracy_per_bucket", "average_return_per_bucket"]
        )

    bucket_labels = [
        f"{bucket_edges[index]:.1f}-{bucket_edges[index + 1]:.1f}"
        for index in range(len(bucket_edges) - 1)
    ]
    frame["bucket"] = pd.cut(
        frame["y_proba"],
        bins=bucket_edges,
        labels=bucket_labels,
        include_lowest=True,
        right=True,
    )

    frame["correct_prediction"] = (frame["y_true"] == frame["y_pred"]).astype(float)
    diagnostics_df = (
        frame.groupby("bucket", observed=False)
        .agg(
            trade_count=("timestamp", "count"),
            accuracy_per_bucket=("correct_prediction", "mean"),
            average_return_per_bucket=("forward_return_1bar", "mean"),
            mean_probability=("y_proba", "mean"),
        )
        .reset_index()
    )
    diagnostics_df["bucket"] = diagnostics_df["bucket"].astype(str)
    diagnostics_df["accuracy_per_bucket"] = diagnostics_df["accuracy_per_bucket"].fillna(0.0)
    diagnostics_df["average_return_per_bucket"] = diagnostics_df["average_return_per_bucket"].fillna(0.0)
    diagnostics_df["mean_probability"] = diagnostics_df["mean_probability"].fillna(0.0)
    return diagnostics_df


def build_failure_analysis(
    *,
    trades_df: pd.DataFrame,
    equity_curve_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    top_failure_cases: int,
    drawdown_case_count: int,
) -> dict[str, Any]:
    """Summarize the most costly trades and worst drawdown periods."""
    top_losing_trades = (
        trades_df.sort_values("pnl", ascending=True).head(top_failure_cases).to_dict(orient="records")
        if not trades_df.empty and "pnl" in trades_df.columns
        else []
    )

    drawdown_rows = (
        equity_curve_df.sort_values("drawdown", ascending=False)
        .head(drawdown_case_count)
        .merge(
            signals_df[["timestamp", "y_proba", "signal_action"]],
            on="timestamp",
            how="left",
        )
        .to_dict(orient="records")
        if not equity_curve_df.empty and "drawdown" in equity_curve_df.columns
        else []
    )

    overconfident_losses = [
        trade
        for trade in top_losing_trades
        if float(trade.get("return_pct", 0.0) or 0.0) < 0.0
        and float(trade.get("entry_price", 0.0) or 0.0) > 0.0
    ]

    return {
        "top_losing_trades": top_losing_trades,
        "largest_drawdown_periods": drawdown_rows,
        "overconfident_loss_count": len(overconfident_losses),
    }


def build_feature_importance_audit(feature_importance_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize top features and sentiment-feature presence."""
    if feature_importance_df.empty:
        return {"top_features": [], "sentiment_features_in_top_20": [], "has_sentiment_signal": False}

    top_features_df = feature_importance_df.head(20).copy()
    sentiment_rows = top_features_df[
        top_features_df["feature_name"].astype(str).str.contains(
            "sentiment|positive_news|negative_news",
            case=False,
            regex=True,
        )
    ]
    return {
        "top_features": top_features_df.to_dict(orient="records"),
        "sentiment_features_in_top_20": sentiment_rows.to_dict(orient="records"),
        "has_sentiment_signal": not sentiment_rows.empty,
    }


def build_simulation_audit(
    *,
    metrics: dict[str, Any],
    trades_df: pd.DataFrame,
    equity_curve_df: pd.DataFrame,
    overtrading_threshold: float,
    tiny_edge_after_fees_threshold: float,
) -> dict[str, Any]:
    """Audit trading behavior, cost drag, and suspicious simulation patterns."""
    if equity_curve_df.empty:
        return {"flags": ["empty_equity_curve"]}

    timestamp_series = pd.to_datetime(equity_curve_df["timestamp"], utc=True)
    span_days = max((timestamp_series.max() - timestamp_series.min()).total_seconds() / 86400.0, 1.0)
    trades_per_day = float(len(trades_df) / span_days)
    flags: list[str] = []

    fee_impact = float(metrics.get("fee_impact", 0.0) or 0.0)
    total_return = float(metrics.get("total_return", 0.0) or 0.0)
    realized_pnl = float(metrics.get("realized_pnl", 0.0) or 0.0)

    if trades_per_day > overtrading_threshold:
        flags.append("overtrading_detected")
    if total_return <= tiny_edge_after_fees_threshold and fee_impact > abs(realized_pnl):
        flags.append("tiny_edge_wiped_by_fees")
    if float(metrics.get("exposure_ratio", 0.0) or 0.0) > 0.95 and total_return <= 0.0:
        flags.append("high_exposure_without_positive_return")

    return {
        "trades_per_day": round(trades_per_day, 8),
        "fee_impact": fee_impact,
        "exposure_ratio": float(metrics.get("exposure_ratio", 0.0) or 0.0),
        "flags": flags,
    }


def build_validation_conclusion(
    *,
    comparison_df: pd.DataFrame,
    prediction_diagnostics_df: pd.DataFrame,
    simulation_audit: dict[str, Any],
) -> dict[str, Any]:
    """Decide whether the current system shows a meaningful edge."""
    if comparison_df.empty:
        return {
            "has_edge": False,
            "reason": "No strategy comparison results were available.",
        }

    model_row = comparison_df[comparison_df["strategy_name"] == "model"]
    if model_row.empty:
        return {"has_edge": False, "reason": "Model strategy result missing from comparison."}
    model_row = model_row.iloc[0]

    competitive = comparison_df[
        comparison_df["strategy_name"].isin(["buy_and_hold", "random_matched_frequency", "sma_crossover_20_50"])
    ]
    best_baseline_return = float(competitive["total_return"].max()) if not competitive.empty else 0.0
    best_baseline_name = (
        str(competitive.sort_values("total_return", ascending=False).iloc[0]["strategy_name"])
        if not competitive.empty
        else "none"
    )

    bucket_signal = False
    if not prediction_diagnostics_df.empty:
        sorted_diag = prediction_diagnostics_df.sort_values("mean_probability")
        bucket_signal = float(sorted_diag["average_return_per_bucket"].iloc[-1]) >= float(
            sorted_diag["average_return_per_bucket"].iloc[0]
        )

    flags = simulation_audit.get("flags", []) if isinstance(simulation_audit, dict) else []
    model_total_return = float(model_row["total_return"])
    has_edge = (
        model_total_return > 0.0
        and model_total_return > best_baseline_return
        and not flags
        and bucket_signal
    )

    return {
        "has_edge": has_edge,
        "best_baseline": best_baseline_name,
        "best_baseline_return": round(best_baseline_return, 8),
        "model_total_return": round(model_total_return, 8),
        "prediction_confidence_supports_edge": bucket_signal,
        "audit_flags": list(flags),
        "reason": (
            "Model beat all configured baselines after costs and confidence buckets improved with probability."
            if has_edge
            else "Model did not clearly beat simple baselines after costs, or the audit found weak confidence support."
        ),
    }


def format_validation_report_markdown(report: dict[str, Any]) -> str:
    """Return a compact Markdown rendering of a validation report."""
    conclusion = report.get("conclusion", {})
    lines = [
        f"# Validation Report: {report.get('validation_id', 'unknown')}",
        "",
        "## Overview",
        f"- Model run id: `{report.get('model_run_id')}`",
        f"- Symbol: `{report.get('symbol')}`",
        f"- Timeframe: `{report.get('timeframe')}`",
        f"- Generated at: `{report.get('generated_at')}`",
        "",
        "## Conclusion",
        f"- has_edge: `{conclusion.get('has_edge')}`",
        f"- reason: {conclusion.get('reason')}",
        "",
        "## Strategy Comparison",
    ]
    for row in report.get("strategy_comparison", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- {row.get('strategy_name')}: total_return={row.get('total_return')}, "
            f"max_drawdown={row.get('max_drawdown')}, trades={row.get('number_of_trades')}"
        )
    return "\n".join(lines) + "\n"


def _model_signals(joined_df: pd.DataFrame, strategy_config: StrategyConfig) -> pd.DataFrame:
    """Return prediction-threshold signals for the model strategy."""
    from trading_bot.backtest.signals import generate_threshold_signals

    return generate_threshold_signals(joined_df, strategy_config=strategy_config)


def _comparison_row(strategy_name: str, result) -> dict[str, Any]:
    """Return one normalized comparison row."""
    metrics = result.metrics.to_dict()
    row = {
        "strategy_name": strategy_name,
        "simulation_id": result.simulation_id,
        "model_run_id": result.model_run_id,
    }
    row.update(metrics)
    return row


def _execution_config_from_settings(settings: AppSettings) -> ExecutionConfig:
    """Return the configured execution assumptions for standardized comparisons."""
    configured = settings.backtest_execution
    return ExecutionConfig(
        fill_model=configured.fill_model,
        starting_cash=configured.starting_cash,
        fee_rate=configured.fee_rate,
        slippage_rate=configured.slippage_rate,
        max_position_fraction=configured.max_position_fraction,
        allow_fractional_position=configured.allow_fractional_position,
    )


def _default_validation_id() -> str:
    """Return a timestamp-based validation identifier."""
    return f"VAL{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
