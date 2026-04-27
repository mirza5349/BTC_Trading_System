"""Strategy threshold search and execution-optimization audit."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from trading_bot.backtest.execution import simulate_execution
from trading_bot.backtest.metrics import compute_simulation_metrics
from trading_bot.backtest.signals import generate_threshold_signals
from trading_bot.backtest.simulator import build_simulation_frame
from trading_bot.backtest.strategy import Strategy, get_default_strategy
from trading_bot.logging_config import get_logger
from trading_bot.models.model_registry import ModelRegistry
from trading_bot.optimization.reporting import format_optimization_report_markdown
from trading_bot.optimization.search_space import build_parameter_grid
from trading_bot.optimization.selection import select_best_candidates
from trading_bot.schemas.datasets import infer_horizon_bars_from_target_column
from trading_bot.schemas.backtest import ExecutionConfig, StrategyConfig
from trading_bot.schemas.optimization import OptimizationReport, StrategySearchParameters
from trading_bot.settings import PROJECT_ROOT, AppSettings, load_settings
from trading_bot.storage.optimization_store import OptimizationStore
from trading_bot.storage.parquet_store import ParquetStore
from trading_bot.validation.baselines import (
    generate_always_flat_signals,
    generate_buy_and_hold_signals,
    generate_random_matched_frequency_signals,
    generate_sma_crossover_signals,
)

logger = get_logger(__name__)


class StrategySearchRunnerError(Exception):
    """Raised when a strategy-search run cannot complete."""


class StrategySearchRunner:
    """Run a deterministic threshold search on stored model predictions."""

    def __init__(
        self,
        *,
        model_registry: ModelRegistry,
        market_store: ParquetStore,
        optimization_store: OptimizationStore,
        settings: AppSettings,
    ) -> None:
        self.model_registry = model_registry
        self.market_store = market_store
        self.optimization_store = optimization_store
        self.settings = settings

    @classmethod
    def from_settings(cls, settings: AppSettings | None = None) -> StrategySearchRunner:
        """Create the strategy-search runner from project settings."""
        resolved_settings = settings or load_settings()
        return cls(
            model_registry=ModelRegistry.from_settings(resolved_settings),
            market_store=ParquetStore(PROJECT_ROOT / resolved_settings.market_data.processed_data_dir),
            optimization_store=OptimizationStore(
                PROJECT_ROOT / resolved_settings.optimization_storage.output_dir
            ),
            settings=resolved_settings,
        )

    def run_search(
        self,
        *,
        model_run_id: str,
        symbol: str | None = None,
        timeframe: str | None = None,
        optimization_id: str | None = None,
    ) -> dict[str, Any]:
        """Run the configured threshold search and persist compact optimization artifacts."""
        resolved_run = self.model_registry.resolve_run(
            model_run_id,
            symbol=symbol,
            timeframe=timeframe,
        )
        predictions_df = self.model_registry.read_predictions(
            model_run_id,
            symbol=resolved_run.symbol,
            timeframe=resolved_run.timeframe,
            model_version=resolved_run.model_version,
            asset=resolved_run.asset,
        )
        market_df = self.market_store.read(resolved_run.symbol, resolved_run.timeframe)
        joined_df = build_simulation_frame(predictions_df, market_df)
        if joined_df.empty:
            raise StrategySearchRunnerError(
                f"No overlapping timestamps found for run_id={model_run_id} and market candles"
            )

        resolved_optimization_id = optimization_id or _default_optimization_id()
        execution = _execution_from_settings(self.settings)
        original_strategy = Strategy(config=get_default_strategy(self.settings))
        original_result = _simulate_strategy(
            candidate_id="original_model",
            model_run_id=model_run_id,
            joined_df=joined_df,
            strategy_config=original_strategy.config,
            execution_config=execution,
        )

        baseline_rows = _build_baseline_rows(
            joined_df=joined_df,
            model_run_id=model_run_id,
            execution_config=execution,
            reference_trade_count=int(original_result["number_of_trades"]),
            reference_average_holding_bars=float(original_result["average_holding_bars"]),
            random_seed=self.settings.validation_suite.random_seed,
        )
        buy_and_hold_return = _metric_from_rows(baseline_rows, "buy_and_hold", "total_return")
        random_baseline_return = _metric_from_rows(
            baseline_rows,
            "random_matched_frequency",
            "total_return",
        )

        search_space = {
            "entry_threshold": list(self.settings.optimization_search_space.entry_threshold),
            "exit_threshold": list(self.settings.optimization_search_space.exit_threshold),
            "minimum_holding_bars": list(self.settings.optimization_search_space.minimum_holding_bars),
            "cooldown_bars": list(self.settings.optimization_search_space.cooldown_bars),
            "max_position_fraction": list(self.settings.optimization_search_space.max_position_fraction),
        }
        candidates = build_parameter_grid(search_space)

        candidate_rows = [
            _simulate_candidate(
                candidate_id=f"C{index:04d}",
                model_run_id=model_run_id,
                joined_df=joined_df,
                candidate=candidate,
                execution_config=execution,
                buy_and_hold_return=buy_and_hold_return,
                random_baseline_return=random_baseline_return,
                original_trade_count=int(original_result["number_of_trades"]),
            )
            for index, candidate in enumerate(candidates, start=1)
        ]
        candidate_df = pd.DataFrame(candidate_rows)

        selection = select_best_candidates(
            candidate_df,
            max_allowed_drawdown=self.settings.optimization_selection.max_allowed_drawdown,
            max_allowed_trades=self.settings.optimization_selection.max_allowed_trades,
            max_allowed_fee_to_starting_cash=(
                self.settings.optimization_selection.max_allowed_fee_to_starting_cash
            ),
            minimum_total_return=self.settings.optimization_selection.minimum_total_return,
            top_k=self.settings.optimization_selection.top_k,
            ranking_metric=self.settings.optimization_selection.ranking_metric,
        )
        top_candidates_df = selection["top_candidates"]
        best_candidate = dict(selection["best_candidate"])

        comparison_df = pd.DataFrame([original_result, *baseline_rows])
        if best_candidate:
            comparison_df = pd.concat(
                [pd.DataFrame([_tag_best_candidate(best_candidate)]), comparison_df],
                ignore_index=True,
            )

        fee_audit = _build_fee_audit(
            best_candidate=best_candidate,
            original_result=original_result,
            starting_cash=execution.starting_cash,
        )
        drawdown_audit = _build_drawdown_audit(
            best_candidate=best_candidate,
            max_allowed_drawdown=self.settings.optimization_selection.max_allowed_drawdown,
        )
        recommendation = _build_recommendation(
            best_candidate=best_candidate,
            random_baseline_return=random_baseline_return,
            selection_limits=self.settings.optimization_selection,
        )

        artifact_dir = self.optimization_store.optimization_dir(
            model_run_id,
            resolved_optimization_id,
        )
        report = OptimizationReport(
            optimization_id=resolved_optimization_id,
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
            search_space=search_space,
            candidates_tested=int(len(candidate_df)),
            best_candidate=best_candidate,
            top_candidates=top_candidates_df.to_dict(orient="records"),
            rejection_counts=dict(selection["rejection_counts"]),
            fee_audit=fee_audit,
            drawdown_audit=drawdown_audit,
            baseline_comparison=comparison_df.to_dict(orient="records"),
            recommendation=recommendation,
            artifact_paths={
                "candidate_results_path": str(artifact_dir / "candidate_results.parquet"),
                "top_candidates_path": str(artifact_dir / "top_candidates.parquet"),
                "baseline_comparison_path": str(artifact_dir / "baseline_comparison.parquet"),
                "report_json_path": str(artifact_dir / "optimization_report.json"),
                "report_markdown_path": str(artifact_dir / "optimization_report.md"),
            },
        ).to_dict()

        self.optimization_store.write_candidate_results(model_run_id, resolved_optimization_id, candidate_df)
        self.optimization_store.write_top_candidates(model_run_id, resolved_optimization_id, top_candidates_df)
        self.optimization_store.write_baseline_comparison(
            model_run_id,
            resolved_optimization_id,
            comparison_df,
        )
        self.optimization_store.write_report_json(model_run_id, resolved_optimization_id, report)
        self.optimization_store.write_report_markdown(
            model_run_id,
            resolved_optimization_id,
            format_optimization_report_markdown(report),
        )

        logger.info(
            "strategy_search_complete",
            optimization_id=resolved_optimization_id,
            model_run_id=model_run_id,
            candidates_tested=len(candidate_df),
            optimized_has_edge=recommendation.get("optimized_has_edge"),
        )
        return report


def _simulate_candidate(
    *,
    candidate_id: str,
    model_run_id: str,
    joined_df: pd.DataFrame,
    candidate: StrategySearchParameters,
    execution_config: ExecutionConfig,
    buy_and_hold_return: float | None,
    random_baseline_return: float | None,
    original_trade_count: int,
) -> dict[str, Any]:
    """Simulate one threshold-search candidate and return a compact result row."""
    strategy_config = StrategyConfig(
        strategy_version="strategy_search",
        type="long_only_threshold",
        entry_threshold=candidate.entry_threshold,
        exit_threshold=candidate.exit_threshold,
        minimum_holding_bars=candidate.minimum_holding_bars,
        cooldown_bars=candidate.cooldown_bars,
    )
    execution = ExecutionConfig(
        fill_model=execution_config.fill_model,
        starting_cash=execution_config.starting_cash,
        fee_rate=execution_config.fee_rate,
        slippage_rate=execution_config.slippage_rate,
        max_position_fraction=candidate.max_position_fraction,
        allow_fractional_position=execution_config.allow_fractional_position,
    )
    return _simulate_strategy(
        candidate_id=candidate_id,
        model_run_id=model_run_id,
        joined_df=joined_df,
        strategy_config=strategy_config,
        execution_config=execution,
        buy_and_hold_return=buy_and_hold_return,
        random_baseline_return=random_baseline_return,
        original_trade_count=original_trade_count,
    )


def _simulate_strategy(
    *,
    candidate_id: str,
    model_run_id: str,
    joined_df: pd.DataFrame,
    strategy_config: StrategyConfig,
    execution_config: ExecutionConfig,
    buy_and_hold_return: float | None = None,
    random_baseline_return: float | None = None,
    original_trade_count: int | None = None,
) -> dict[str, Any]:
    """Simulate one strategy configuration and return a compact row."""
    signals_df = generate_threshold_signals(joined_df, strategy_config=strategy_config)
    frames = simulate_execution(
        joined_df,
        signals_df=signals_df,
        execution_config=execution_config,
    )
    metrics = compute_simulation_metrics(
        frames.equity_curve,
        frames.trades,
        starting_cash=execution_config.starting_cash,
    ).to_dict()
    number_of_trades = int(metrics.get("number_of_trades", 0) or 0)
    return {
        "candidate_id": candidate_id,
        "model_run_id": model_run_id,
        "entry_threshold": strategy_config.entry_threshold,
        "exit_threshold": strategy_config.exit_threshold,
        "minimum_holding_bars": strategy_config.minimum_holding_bars,
        "cooldown_bars": strategy_config.cooldown_bars,
        "max_position_fraction": execution_config.max_position_fraction,
        **metrics,
        "fee_to_starting_cash": (
            float(metrics.get("fee_impact", 0.0) or 0.0) / execution_config.starting_cash
            if execution_config.starting_cash
            else 0.0
        ),
        "return_minus_buy_and_hold": (
            float(metrics.get("total_return", 0.0) or 0.0) - buy_and_hold_return
            if buy_and_hold_return is not None
            else None
        ),
        "return_minus_random_baseline": (
            float(metrics.get("total_return", 0.0) or 0.0) - random_baseline_return
            if random_baseline_return is not None
            else None
        ),
        "trade_count_reduction_vs_original": (
            int(original_trade_count) - number_of_trades if original_trade_count is not None else None
        ),
    }


def _build_baseline_rows(
    *,
    joined_df: pd.DataFrame,
    model_run_id: str,
    execution_config: ExecutionConfig,
    reference_trade_count: int,
    reference_average_holding_bars: float,
    random_seed: int,
) -> list[dict[str, Any]]:
    """Return baseline comparison rows using the same execution assumptions."""
    baseline_specs = [
        (
            "buy_and_hold",
            generate_buy_and_hold_signals(joined_df),
        ),
        (
            "always_flat",
            generate_always_flat_signals(joined_df),
        ),
        (
            "random_matched_frequency",
            generate_random_matched_frequency_signals(
                joined_df,
                reference_trade_count=reference_trade_count,
                reference_average_holding_bars=reference_average_holding_bars,
                random_seed=random_seed,
            ),
        ),
        (
            "sma_crossover_20_50",
            generate_sma_crossover_signals(joined_df, fast_window=20, slow_window=50),
        ),
    ]

    rows: list[dict[str, Any]] = []
    for strategy_name, signals_df in baseline_specs:
        frames = simulate_execution(
            joined_df,
            signals_df=signals_df,
            execution_config=execution_config,
        )
        metrics = compute_simulation_metrics(
            frames.equity_curve,
            frames.trades,
            starting_cash=execution_config.starting_cash,
        ).to_dict()
        rows.append(
            {
                "candidate_id": strategy_name,
                "strategy_name": strategy_name,
                "model_run_id": f"BASELINE:{strategy_name}",
                **metrics,
            }
        )
    return rows


def _metric_from_rows(rows: list[dict[str, Any]], strategy_name: str, metric_name: str) -> float | None:
    """Return one metric from a strategy comparison row list."""
    for row in rows:
        if row.get("strategy_name") == strategy_name:
            value = row.get(metric_name)
            return float(value) if value is not None else None
    return None


def _tag_best_candidate(best_candidate: dict[str, Any]) -> dict[str, Any]:
    """Return a comparison row for the best optimized candidate."""
    tagged = dict(best_candidate)
    tagged["strategy_name"] = "best_optimized"
    return tagged


def _build_fee_audit(
    *,
    best_candidate: dict[str, Any],
    original_result: dict[str, Any],
    starting_cash: float,
) -> dict[str, Any]:
    """Summarize fee drag and trade-count reduction for the optimization search."""
    original_trade_count = int(original_result.get("number_of_trades", 0) or 0)
    best_trade_count = int(best_candidate.get("number_of_trades", 0) or 0) if best_candidate else 0
    fee_impact = float(best_candidate.get("fee_impact", 0.0) or 0.0) if best_candidate else 0.0
    return {
        "original_model_trade_count": original_trade_count,
        "best_candidate_trade_count": best_trade_count,
        "trade_count_reduction": original_trade_count - best_trade_count,
        "meaningfully_reduced_frequency": best_trade_count < original_trade_count,
        "best_candidate_fee_impact": fee_impact,
        "fee_to_starting_cash": fee_impact / starting_cash if starting_cash else 0.0,
    }


def _build_drawdown_audit(
    *,
    best_candidate: dict[str, Any],
    max_allowed_drawdown: float,
) -> dict[str, Any]:
    """Return a compact drawdown audit for the best candidate."""
    drawdown = float(best_candidate.get("max_drawdown", 0.0) or 0.0) if best_candidate else None
    return {
        "best_candidate_max_drawdown": drawdown,
        "max_allowed_drawdown": max_allowed_drawdown,
        "passes_drawdown_filter": drawdown is not None and drawdown <= max_allowed_drawdown,
    }


def _build_recommendation(
    *,
    best_candidate: dict[str, Any],
    random_baseline_return: float | None,
    selection_limits,
) -> dict[str, Any]:
    """Return the final recommendation from the optimization search."""
    if not best_candidate:
        return {
            "optimized_has_edge": False,
            "reason": "No candidate survived the configured drawdown, trade-count, fee, and return filters.",
        }

    total_return = float(best_candidate.get("total_return", 0.0) or 0.0)
    max_drawdown = float(best_candidate.get("max_drawdown", 1.0) or 1.0)
    number_of_trades = int(best_candidate.get("number_of_trades", 0) or 0)
    fee_to_starting_cash = float(best_candidate.get("fee_to_starting_cash", 0.0) or 0.0)
    return_minus_random = best_candidate.get("return_minus_random_baseline")
    beats_random = (
        return_minus_random is None or float(return_minus_random) > 0.0 or random_baseline_return is None
    )

    optimized_has_edge = (
        total_return > 0.0
        and beats_random
        and max_drawdown <= selection_limits.max_allowed_drawdown
        and number_of_trades <= selection_limits.max_allowed_trades
        and fee_to_starting_cash <= selection_limits.max_allowed_fee_to_starting_cash
    )

    if optimized_has_edge:
        reason = "Best candidate stayed profitable after costs and passed drawdown, fee, and trade-count filters."
    elif total_return <= 0.0:
        reason = "All viable threshold configurations still lose money after costs."
    elif number_of_trades > selection_limits.max_allowed_trades:
        reason = "The best candidate still overtrades under realistic cost assumptions."
    elif fee_to_starting_cash > selection_limits.max_allowed_fee_to_starting_cash:
        reason = "Fee drag remains too high relative to starting cash."
    elif max_drawdown > selection_limits.max_allowed_drawdown:
        reason = "The best candidate still exceeds the allowed drawdown threshold."
    else:
        reason = "The best candidate does not clearly beat a random-style baseline after costs."

    return {
        "optimized_has_edge": optimized_has_edge,
        "reason": reason,
        "best_candidate_total_return": total_return,
        "best_candidate_number_of_trades": number_of_trades,
        "best_candidate_fee_to_starting_cash": fee_to_starting_cash,
    }


def _execution_from_settings(settings: AppSettings) -> ExecutionConfig:
    """Return the standardized execution assumptions for optimization."""
    configured = settings.backtest_execution
    return ExecutionConfig(
        fill_model=configured.fill_model,
        starting_cash=configured.starting_cash,
        fee_rate=configured.fee_rate,
        slippage_rate=configured.slippage_rate,
        max_position_fraction=configured.max_position_fraction,
        allow_fractional_position=configured.allow_fractional_position,
    )


def _default_optimization_id() -> str:
    """Return a deterministic timestamp-based optimization identifier."""
    return f"OPT{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
