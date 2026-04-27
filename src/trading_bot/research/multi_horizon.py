"""Multi-horizon dataset, training, validation, optimization, and comparison workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from trading_bot.experiments.tracker import ExperimentTracker
from trading_bot.labeling.dataset_builder import SupervisedDatasetBuilder
from trading_bot.logging_config import get_logger
from trading_bot.models.model_registry import ModelRegistry, ResolvedRun
from trading_bot.models.train import WalkForwardTrainingPipeline
from trading_bot.optimization.strategy_search import StrategySearchRunner
from trading_bot.schemas.datasets import (
    binary_target_column_name,
)
from trading_bot.schemas.horizons import HorizonComparisonReport, HorizonRunSummary
from trading_bot.settings import PROJECT_ROOT, AppSettings, load_settings
from trading_bot.storage.horizon_store import HorizonStore
from trading_bot.storage.optimization_store import OptimizationStore
from trading_bot.storage.validation_store import ValidationStore
from trading_bot.validation.suite import ValidationSuite

logger = get_logger(__name__)


class MultiHorizonResearchError(Exception):
    """Raised when the multi-horizon workflow cannot complete safely."""


class MultiHorizonResearchRunner:
    """Coordinate end-to-end multi-horizon research on top of existing project workflows."""

    def __init__(
        self,
        *,
        dataset_builder: SupervisedDatasetBuilder,
        training_pipeline: WalkForwardTrainingPipeline,
        experiment_tracker: ExperimentTracker,
        model_registry: ModelRegistry,
        validation_suite: ValidationSuite,
        validation_store: ValidationStore,
        strategy_runner: StrategySearchRunner,
        optimization_store: OptimizationStore,
        horizon_store: HorizonStore,
        settings: AppSettings,
    ) -> None:
        self.dataset_builder = dataset_builder
        self.training_pipeline = training_pipeline
        self.experiment_tracker = experiment_tracker
        self.model_registry = model_registry
        self.validation_suite = validation_suite
        self.validation_store = validation_store
        self.strategy_runner = strategy_runner
        self.optimization_store = optimization_store
        self.horizon_store = horizon_store
        self.settings = settings

    @classmethod
    def from_settings(cls, settings: AppSettings | None = None) -> MultiHorizonResearchRunner:
        """Build the research runner from project settings."""
        resolved_settings = settings or load_settings()
        return cls(
            dataset_builder=SupervisedDatasetBuilder.from_settings(resolved_settings),
            training_pipeline=WalkForwardTrainingPipeline.from_settings(resolved_settings),
            experiment_tracker=ExperimentTracker.from_settings(resolved_settings),
            model_registry=ModelRegistry.from_settings(resolved_settings),
            validation_suite=ValidationSuite.from_settings(resolved_settings),
            validation_store=ValidationStore(
                PROJECT_ROOT / resolved_settings.validation_suite.output_dir
            ),
            strategy_runner=StrategySearchRunner.from_settings(resolved_settings),
            optimization_store=OptimizationStore(
                PROJECT_ROOT / resolved_settings.optimization_storage.output_dir
            ),
            horizon_store=HorizonStore(PROJECT_ROOT / resolved_settings.multi_horizon.output_dir),
            settings=resolved_settings,
        )

    def build_datasets(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        asset: str | None = None,
        feature_version: str | None = None,
        horizons: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Build one supervised dataset artifact per requested horizon."""
        resolved_feature_version = (
            feature_version or self.settings.multi_horizon.default_feature_version
        )
        rows: list[dict[str, Any]] = []
        for horizon_bars in self._resolve_horizons(horizons):
            label_version = self._label_version(horizon_bars)
            dataset_version = self._dataset_version(horizon_bars)
            try:
                dataset_df, artifact = self.dataset_builder.build_supervised_dataset(
                    symbol=symbol or self.settings.datasets.symbol,
                    timeframe=timeframe or self.settings.datasets.timeframe,
                    asset=asset or self.settings.datasets.asset,
                    feature_version=resolved_feature_version,
                    label_version=label_version,
                    dataset_version=dataset_version,
                    horizon_bars=horizon_bars,
                )
            except Exception as exc:
                raise MultiHorizonResearchError(
                    f"Failed to build dataset for horizon={horizon_bars}bars: {exc}"
                ) from exc
            rows.append(
                {
                    "horizon_bars": horizon_bars,
                    "horizon_minutes": self._horizon_minutes(horizon_bars),
                    "target_column": binary_target_column_name(horizon_bars),
                    "feature_version": resolved_feature_version,
                    "label_version": label_version,
                    "dataset_version": dataset_version,
                    "row_count": int(len(dataset_df)),
                    "path": artifact.path,
                }
            )
        return rows

    def train_models(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        asset: str | None = None,
        feature_version: str | None = None,
        horizons: list[int] | None = None,
        requested_device: str | None = None,
    ) -> list[dict[str, Any]]:
        """Train one walk-forward model per requested horizon and refresh the registry."""
        resolved_feature_version = (
            feature_version or self.settings.multi_horizon.default_feature_version
        )
        original_max_folds = self.training_pipeline.max_folds
        if self.settings.multi_horizon.max_folds_override is not None:
            self.training_pipeline.max_folds = self.settings.multi_horizon.max_folds_override

        try:
            rows: list[dict[str, Any]] = []
            for horizon_bars in self._resolve_horizons(horizons):
                label_version = self._label_version(horizon_bars)
                dataset_version = self._dataset_version(horizon_bars)
                model_version = self._model_version(horizon_bars)
                try:
                    result = self.training_pipeline.run_training(
                        symbol=symbol or self.settings.datasets.symbol,
                        timeframe=timeframe or self.settings.datasets.timeframe,
                        asset=asset or self.settings.datasets.asset,
                        feature_version=resolved_feature_version,
                        label_version=label_version,
                        dataset_version=dataset_version,
                        model_version=model_version,
                        target_column=binary_target_column_name(horizon_bars),
                        requested_device=requested_device or self.settings.training.device,
                    )
                except Exception as exc:
                    raise MultiHorizonResearchError(
                        f"Failed to train model for horizon={horizon_bars}bars: {exc}"
                    ) from exc
                rows.append(
                    {
                        "horizon_bars": horizon_bars,
                        "horizon_minutes": self._horizon_minutes(horizon_bars),
                        "target_column": result.target_column,
                        "label_version": label_version,
                        "dataset_version": dataset_version,
                        "model_version": model_version,
                        "feature_version": resolved_feature_version,
                        "model_run_id": result.run_id,
                        "roc_auc_mean": result.aggregate_metrics.get("roc_auc"),
                        "f1_mean": result.aggregate_metrics.get("f1"),
                        "log_loss_mean": result.aggregate_metrics.get("log_loss"),
                    }
                )
                self.experiment_tracker.register_run(result.run_id, metadata=result.metadata)
            return rows
        finally:
            self.training_pipeline.max_folds = original_max_folds

    def run_validation(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        horizons: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Run the validation suite for the latest model run at each requested horizon."""
        rows: list[dict[str, Any]] = []
        for horizon_bars in self._resolve_horizons(horizons):
            try:
                resolved_run = self._resolve_latest_run_for_horizon(
                    symbol=symbol,
                    timeframe=timeframe,
                    horizon_bars=horizon_bars,
                )
                report = self.validation_suite.run_validation_suite(model_run_id=resolved_run.run_id)
            except Exception as exc:
                raise MultiHorizonResearchError(
                    f"Failed to validate horizon={horizon_bars}bars: {exc}"
                ) from exc
            rows.append(
                {
                    "horizon_bars": horizon_bars,
                    "target_column": resolved_run.metadata.get("target_column"),
                    "model_run_id": resolved_run.run_id,
                    "validation_id": report.get("validation_id"),
                    "validation_has_edge": report.get("conclusion", {}).get("has_edge"),
                    "best_baseline": report.get("conclusion", {}).get("best_baseline"),
                    "model_total_return": report.get("conclusion", {}).get("model_total_return"),
                }
            )
        return rows

    def run_strategy_search(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        horizons: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Run strategy search for the latest model run at each requested horizon."""
        rows: list[dict[str, Any]] = []
        for horizon_bars in self._resolve_horizons(horizons):
            try:
                resolved_run = self._resolve_latest_run_for_horizon(
                    symbol=symbol,
                    timeframe=timeframe,
                    horizon_bars=horizon_bars,
                )
                report = self.strategy_runner.run_search(model_run_id=resolved_run.run_id)
            except Exception as exc:
                raise MultiHorizonResearchError(
                    f"Failed to optimize horizon={horizon_bars}bars: {exc}"
                ) from exc
            recommendation = report.get("recommendation", {})
            best_candidate = report.get("best_candidate", {})
            rows.append(
                {
                    "horizon_bars": horizon_bars,
                    "target_column": resolved_run.metadata.get("target_column"),
                    "model_run_id": resolved_run.run_id,
                    "optimization_id": report.get("optimization_id"),
                    "optimized_has_edge": recommendation.get("optimized_has_edge"),
                    "best_strategy_total_return": best_candidate.get("total_return"),
                    "best_strategy_number_of_trades": best_candidate.get("number_of_trades"),
                }
            )
        return rows

    def build_horizon_comparison_report(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        asset: str | None = None,
        feature_version: str | None = None,
        horizons: list[int] | None = None,
        comparison_id: str | None = None,
    ) -> dict[str, Any]:
        """Build and persist a ranked multi-horizon comparison report."""
        resolved_symbol = symbol or self.settings.datasets.symbol
        resolved_timeframe = timeframe or self.settings.datasets.timeframe
        resolved_asset = asset or self.settings.datasets.asset
        resolved_feature_version = (
            feature_version or self.settings.multi_horizon.default_feature_version
        )
        rows: list[dict[str, Any]] = []

        for horizon_bars in self._resolve_horizons(horizons):
            try:
                resolved_run = self._resolve_latest_run_for_horizon(
                    symbol=resolved_symbol,
                    timeframe=resolved_timeframe,
                    horizon_bars=horizon_bars,
                )
                aggregate_metrics = self.model_registry.read_aggregate_metrics(
                    resolved_run.run_id,
                    asset=resolved_run.asset,
                    symbol=resolved_run.symbol,
                    timeframe=resolved_run.timeframe,
                    model_version=resolved_run.model_version,
                )
                validation_report = self._latest_validation_report_for_run(resolved_run.run_id)
                optimization_report = self._latest_optimization_report_for_run(
                    resolved_run.run_id
                )
            except Exception as exc:
                raise MultiHorizonResearchError(
                    f"Failed to compare horizon={horizon_bars}bars: {exc}"
                ) from exc
            rows.append(
                HorizonRunSummary(
                    horizon_bars=horizon_bars,
                    horizon_minutes=self._horizon_minutes(horizon_bars),
                    target_column=str(resolved_run.metadata.get("target_column", "")),
                    label_version=str(resolved_run.metadata.get("label_version", "")),
                    dataset_version=str(resolved_run.metadata.get("dataset_version", "")),
                    model_version=resolved_run.model_version,
                    feature_version=str(resolved_run.metadata.get("feature_version", "")),
                    model_run_id=resolved_run.run_id,
                    validation_id=_as_optional_str(validation_report.get("validation_id")),
                    optimization_id=_as_optional_str(optimization_report.get("optimization_id")),
                    accuracy_mean=_float_or_none(aggregate_metrics.get("accuracy")),
                    f1_mean=_float_or_none(aggregate_metrics.get("f1")),
                    roc_auc_mean=_float_or_none(aggregate_metrics.get("roc_auc")),
                    log_loss_mean=_float_or_none(aggregate_metrics.get("log_loss")),
                    validation_has_edge=_as_optional_bool(
                        validation_report.get("conclusion", {}).get("has_edge")
                    ),
                    optimized_has_edge=_as_optional_bool(
                        optimization_report.get("recommendation", {}).get("optimized_has_edge")
                    ),
                    best_strategy_total_return=_float_or_none(
                        optimization_report.get("best_candidate", {}).get("total_return")
                    ),
                    best_strategy_max_drawdown=_float_or_none(
                        optimization_report.get("best_candidate", {}).get("max_drawdown")
                    ),
                    best_strategy_number_of_trades=_int_or_none(
                        optimization_report.get("best_candidate", {}).get("number_of_trades")
                    ),
                    best_strategy_fee_impact=_float_or_none(
                        optimization_report.get("best_candidate", {}).get("fee_impact")
                    ),
                    best_strategy_fee_to_starting_cash=_float_or_none(
                        optimization_report.get("best_candidate", {}).get("fee_to_starting_cash")
                    ),
                ).to_dict()
            )

        comparison_df = rank_horizon_results(
            pd.DataFrame(rows),
            max_allowed_drawdown=self.settings.optimization_selection.max_allowed_drawdown,
            max_allowed_trades=self.settings.optimization_selection.max_allowed_trades,
            max_allowed_fee_to_starting_cash=(
                self.settings.optimization_selection.max_allowed_fee_to_starting_cash
            ),
        )
        resolved_comparison_id = comparison_id or _default_comparison_id()
        recommendation = build_horizon_recommendation(comparison_df)

        artifact_paths = {
            "comparison_table_path": str(
                self.horizon_store.report_dir(resolved_comparison_id) / "horizon_comparison.parquet"
            ),
            "report_json_path": str(
                self.horizon_store.report_dir(resolved_comparison_id) / "horizon_report.json"
            ),
            "report_markdown_path": str(
                self.horizon_store.report_dir(resolved_comparison_id) / "horizon_report.md"
            ),
        }
        report = HorizonComparisonReport(
            comparison_id=resolved_comparison_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            asset=resolved_asset,
            symbol=resolved_symbol,
            timeframe=resolved_timeframe,
            feature_version=resolved_feature_version,
            horizons_tested=self._resolve_horizons(horizons),
            rows=comparison_df.to_dict(orient="records"),
            recommendation=recommendation,
            artifact_paths=artifact_paths,
        ).to_dict()

        self.horizon_store.write_comparison_table(resolved_comparison_id, comparison_df)
        self.horizon_store.write_report_json(resolved_comparison_id, report)
        if self.settings.reports.write_markdown:
            self.horizon_store.write_report_markdown(
                resolved_comparison_id,
                format_horizon_report_markdown(report),
            )

        logger.info(
            "multi_horizon_report_complete",
            comparison_id=resolved_comparison_id,
            symbol=resolved_symbol,
            timeframe=resolved_timeframe,
            best_horizon_bars=recommendation.get("best_horizon_bars"),
            no_viable_horizon=recommendation.get("no_viable_horizon"),
        )
        return report

    def _resolve_horizons(self, horizons: list[int] | None) -> list[int]:
        resolved = horizons or list(self.settings.multi_horizon.horizon_bars)
        cleaned = sorted({int(horizon) for horizon in resolved if int(horizon) > 0})
        if not cleaned:
            raise MultiHorizonResearchError("At least one positive horizon is required")
        return cleaned

    def _label_version(self, horizon_bars: int) -> str:
        return f"{self.settings.multi_horizon.label_version_prefix}_h{horizon_bars}"

    def _dataset_version(self, horizon_bars: int) -> str:
        return f"{self.settings.multi_horizon.dataset_version_prefix}_h{horizon_bars}"

    def _model_version(self, horizon_bars: int) -> str:
        return f"{self.settings.multi_horizon.model_version_prefix}_h{horizon_bars}"

    def _horizon_minutes(self, horizon_bars: int) -> int:
        timeframe = pd.Timedelta(self.settings.datasets.timeframe)
        return int(timeframe.total_seconds() * horizon_bars / 60)

    def _resolve_latest_run_for_horizon(
        self,
        *,
        symbol: str | None,
        timeframe: str | None,
        horizon_bars: int,
    ) -> ResolvedRun:
        target_column = binary_target_column_name(horizon_bars)
        model_version = self._model_version(horizon_bars)
        runs = [
            run
            for run in self.model_registry.list_runs(
                symbol=symbol or self.settings.datasets.symbol,
                timeframe=timeframe or self.settings.datasets.timeframe,
                model_version=model_version,
            )
            if str(run.get("target_column")) == target_column
        ]
        if not runs:
            raise MultiHorizonResearchError(
                "No stored model run found for "
                f"horizon={horizon_bars}bars target={target_column} model_version={model_version}"
            )
        runs.sort(key=lambda run: str(run.get("created_at", "")), reverse=True)
        return self.model_registry.resolve_run(
            str(runs[0]["run_id"]),
            symbol=symbol or self.settings.datasets.symbol,
            timeframe=timeframe or self.settings.datasets.timeframe,
            model_version=model_version,
        )

    def _latest_validation_report_for_run(self, model_run_id: str) -> dict[str, Any]:
        reports = [
            report
            for report in self.validation_store.list_reports()
            if str(report.get("model_run_id")) == model_run_id
        ]
        if not reports:
            return {}
        reports.sort(key=lambda report: str(report.get("generated_at", "")), reverse=True)
        return reports[0]

    def _latest_optimization_report_for_run(self, model_run_id: str) -> dict[str, Any]:
        reports = [
            report
            for report in self.optimization_store.list_reports()
            if str(report.get("model_run_id")) == model_run_id
        ]
        if not reports:
            return {}
        reports.sort(key=lambda report: str(report.get("generated_at", "")), reverse=True)
        return reports[0]


def rank_horizon_results(
    comparison_df: pd.DataFrame,
    *,
    max_allowed_drawdown: float,
    max_allowed_trades: int,
    max_allowed_fee_to_starting_cash: float,
) -> pd.DataFrame:
    """Rank horizon rows using the configured edge and risk filters."""
    if comparison_df.empty:
        return comparison_df

    ranked = comparison_df.copy()
    ranked["passes_filters"] = (
        ranked["optimized_has_edge"].fillna(False).astype(bool)
        & (pd.to_numeric(ranked["best_strategy_total_return"], errors="coerce").fillna(-1.0) > 0.0)
        & (
            pd.to_numeric(ranked["best_strategy_max_drawdown"], errors="coerce").fillna(1.0)
            <= max_allowed_drawdown
        )
        & (
            pd.to_numeric(ranked["best_strategy_number_of_trades"], errors="coerce")
            .fillna(max_allowed_trades + 1)
            .astype(int)
            <= max_allowed_trades
        )
        & (
            pd.to_numeric(ranked["best_strategy_fee_to_starting_cash"], errors="coerce")
            .fillna(max_allowed_fee_to_starting_cash + 1.0)
            <= max_allowed_fee_to_starting_cash
        )
    )
    ranked["ranking_score"] = (
        ranked["passes_filters"].astype(int) * 1000.0
        + ranked["optimized_has_edge"].fillna(False).astype(int) * 100.0
        + pd.to_numeric(ranked["best_strategy_total_return"], errors="coerce").fillna(-1.0) * 100.0
        - pd.to_numeric(ranked["best_strategy_max_drawdown"], errors="coerce").fillna(1.0) * 25.0
        - pd.to_numeric(ranked["best_strategy_fee_impact"], errors="coerce").fillna(999.0) * 0.1
        - pd.to_numeric(ranked["best_strategy_number_of_trades"], errors="coerce").fillna(9999.0)
        / 1000.0
        + pd.to_numeric(ranked["roc_auc_mean"], errors="coerce").fillna(0.0)
    )
    ranked = ranked.sort_values(
        by=[
            "passes_filters",
            "optimized_has_edge",
            "best_strategy_total_return",
            "best_strategy_max_drawdown",
            "best_strategy_fee_impact",
            "best_strategy_number_of_trades",
            "roc_auc_mean",
        ],
        ascending=[False, False, False, True, True, True, False],
        na_position="last",
    ).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    return ranked


def build_horizon_recommendation(comparison_df: pd.DataFrame) -> dict[str, Any]:
    """Build the final best-horizon or no-viable-horizon recommendation."""
    if comparison_df.empty:
        return {
            "best_horizon_bars": None,
            "no_viable_horizon": True,
            "reason": "No horizon rows were available for comparison.",
        }

    best_row = comparison_df.iloc[0].to_dict()
    if bool(best_row.get("passes_filters", False)):
        return {
            "best_horizon_bars": int(best_row["horizon_bars"]),
            "no_viable_horizon": False,
            "reason": "This horizon passed optimization filters and ranked highest after costs.",
            "best_model_run_id": best_row.get("model_run_id"),
            "best_validation_id": best_row.get("validation_id"),
            "best_optimization_id": best_row.get("optimization_id"),
        }
    return {
        "best_horizon_bars": None,
        "no_viable_horizon": True,
        "reason": (
            "No horizon satisfied the optimization edge, return, drawdown, trade-count, "
            "and fee filters. Revisit labels, features, or strategy assumptions."
        ),
        "best_available_horizon_bars": int(best_row["horizon_bars"]),
    }


def format_horizon_report_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown summary for the multi-horizon comparison."""
    recommendation = report.get("recommendation", {})
    lines = [
        f"# Horizon Comparison Report: {report.get('comparison_id', 'unknown')}",
        "",
        "## Overview",
        f"- Asset: `{report.get('asset')}`",
        f"- Symbol: `{report.get('symbol')}`",
        f"- Timeframe: `{report.get('timeframe')}`",
        f"- Feature version: `{report.get('feature_version')}`",
        f"- Horizons tested: `{report.get('horizons_tested')}`",
        "",
        "## Recommendation",
        f"- best_horizon_bars: `{recommendation.get('best_horizon_bars')}`",
        f"- no_viable_horizon: `{recommendation.get('no_viable_horizon')}`",
        f"- reason: {recommendation.get('reason')}",
        "",
        "## Ranked Horizons",
    ]
    for row in report.get("rows", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- h={row.get('horizon_bars')} return={row.get('best_strategy_total_return')} "
            f"drawdown={row.get('best_strategy_max_drawdown')} "
            f"trades={row.get('best_strategy_number_of_trades')} "
            f"optimized_has_edge={row.get('optimized_has_edge')}"
        )
    return "\n".join(lines) + "\n"


def _default_comparison_id() -> str:
    """Return a UTC timestamp-based id for one horizon-comparison run."""
    return datetime.now(timezone.utc).strftime("HCOMP%Y%m%dT%H%M%SZ")


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _as_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _as_optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)
