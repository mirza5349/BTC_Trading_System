"""Fee-aware and minimum-return target research orchestration."""

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
from trading_bot.schemas.targets import TargetComparisonReport, TargetConfig, TargetRunSummary
from trading_bot.settings import PROJECT_ROOT, AppSettings, load_settings
from trading_bot.storage.optimization_store import OptimizationStore
from trading_bot.storage.target_store import TargetStore
from trading_bot.storage.validation_store import ValidationStore
from trading_bot.validation.suite import ValidationSuite

logger = get_logger(__name__)


class FeeAwareTargetResearchError(Exception):
    """Raised when the fee-aware target workflow cannot complete safely."""


class FeeAwareTargetResearchRunner:
    """Coordinate end-to-end research across fee-aware target definitions."""

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
        target_store: TargetStore,
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
        self.target_store = target_store
        self.settings = settings

    @classmethod
    def from_settings(cls, settings: AppSettings | None = None) -> FeeAwareTargetResearchRunner:
        """Build the fee-aware target research runner from project settings."""
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
            target_store=TargetStore(PROJECT_ROOT / resolved_settings.fee_aware_targets.output_dir),
            settings=resolved_settings,
        )

    def build_datasets(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        asset: str | None = None,
        feature_version: str | None = None,
        target_keys: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Build one supervised dataset per target configuration."""
        resolved_feature_version = (
            feature_version or self.settings.fee_aware_targets.default_feature_version
        )
        rows: list[dict[str, Any]] = []
        for target_config in build_target_configs(self.settings, target_keys=target_keys):
            try:
                dataset_df, artifact = self.dataset_builder.build_supervised_dataset(
                    symbol=symbol or self.settings.datasets.symbol,
                    timeframe=timeframe or self.settings.datasets.timeframe,
                    asset=asset or self.settings.datasets.asset,
                    feature_version=resolved_feature_version,
                    label_version=target_config.label_version,
                    dataset_version=target_config.dataset_version,
                    horizon_bars=target_config.horizon_bars,
                    target_config=target_config,
                )
            except Exception as exc:
                raise FeeAwareTargetResearchError(
                    f"Failed to build dataset for target={target_config.key}: {exc}"
                ) from exc
            positive_rate = _positive_rate_from_dataset(dataset_df, target_config.target_name)
            rows.append(
                {
                    "target_key": target_config.key,
                    "target_name": target_config.target_name,
                    "threshold_type": target_config.threshold_type,
                    "horizon_bars": target_config.horizon_bars,
                    "positive_class_rate": positive_rate,
                    "feature_version": resolved_feature_version,
                    "label_version": target_config.label_version,
                    "dataset_version": target_config.dataset_version,
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
        target_keys: list[str] | None = None,
        requested_device: str | None = None,
    ) -> list[dict[str, Any]]:
        """Train one walk-forward model per target configuration."""
        resolved_feature_version = (
            feature_version or self.settings.fee_aware_targets.default_feature_version
        )
        original_max_folds = self.training_pipeline.max_folds
        if self.settings.fee_aware_targets.max_folds_override is not None:
            self.training_pipeline.max_folds = self.settings.fee_aware_targets.max_folds_override
        try:
            rows: list[dict[str, Any]] = []
            for target_config in build_target_configs(self.settings, target_keys=target_keys):
                try:
                    result = self.training_pipeline.run_training(
                        symbol=symbol or self.settings.datasets.symbol,
                        timeframe=timeframe or self.settings.datasets.timeframe,
                        asset=asset or self.settings.datasets.asset,
                        feature_version=resolved_feature_version,
                        label_version=target_config.label_version,
                        dataset_version=target_config.dataset_version,
                        model_version=target_config.model_version,
                        target_column=target_config.target_name,
                        requested_device=requested_device or self.settings.training.device,
                    )
                except Exception as exc:
                    raise FeeAwareTargetResearchError(
                        f"Failed to train target={target_config.key}: {exc}"
                    ) from exc
                result.metadata["target_config"] = target_config.to_dict()
                self.experiment_tracker.register_run(result.run_id, metadata=result.metadata)
                rows.append(
                    {
                        "target_key": target_config.key,
                        "target_name": target_config.target_name,
                        "model_run_id": result.run_id,
                        "label_version": target_config.label_version,
                        "dataset_version": target_config.dataset_version,
                        "model_version": target_config.model_version,
                        "roc_auc_mean": result.aggregate_metrics.get("roc_auc"),
                        "f1_mean": result.aggregate_metrics.get("f1"),
                        "log_loss_mean": result.aggregate_metrics.get("log_loss"),
                    }
                )
            return rows
        finally:
            self.training_pipeline.max_folds = original_max_folds

    def run_validation(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        target_keys: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Run validation for the latest stored model run at each target configuration."""
        rows: list[dict[str, Any]] = []
        for target_config in build_target_configs(self.settings, target_keys=target_keys):
            try:
                resolved_run = self._resolve_latest_run_for_target(
                    symbol=symbol,
                    timeframe=timeframe,
                    target_config=target_config,
                )
                report = self.validation_suite.run_validation_suite(model_run_id=resolved_run.run_id)
            except Exception as exc:
                raise FeeAwareTargetResearchError(
                    f"Failed to validate target={target_config.key}: {exc}"
                ) from exc
            rows.append(
                {
                    "target_key": target_config.key,
                    "target_name": target_config.target_name,
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
        target_keys: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Run strategy search for the latest stored model run at each target configuration."""
        rows: list[dict[str, Any]] = []
        for target_config in build_target_configs(self.settings, target_keys=target_keys):
            try:
                resolved_run = self._resolve_latest_run_for_target(
                    symbol=symbol,
                    timeframe=timeframe,
                    target_config=target_config,
                )
                report = self.strategy_runner.run_search(model_run_id=resolved_run.run_id)
            except Exception as exc:
                raise FeeAwareTargetResearchError(
                    f"Failed to optimize target={target_config.key}: {exc}"
                ) from exc
            best_candidate = report.get("best_candidate", {})
            recommendation = report.get("recommendation", {})
            rows.append(
                {
                    "target_key": target_config.key,
                    "target_name": target_config.target_name,
                    "model_run_id": resolved_run.run_id,
                    "optimization_id": report.get("optimization_id"),
                    "optimized_has_edge": recommendation.get("optimized_has_edge"),
                    "best_strategy_total_return": best_candidate.get("total_return"),
                    "best_strategy_number_of_trades": best_candidate.get("number_of_trades"),
                }
            )
        return rows

    def build_target_comparison_report(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        asset: str | None = None,
        feature_version: str | None = None,
        target_keys: list[str] | None = None,
        comparison_id: str | None = None,
    ) -> dict[str, Any]:
        """Build and persist a ranked fee-aware target comparison report."""
        resolved_symbol = symbol or self.settings.datasets.symbol
        resolved_timeframe = timeframe or self.settings.datasets.timeframe
        resolved_asset = asset or self.settings.datasets.asset
        resolved_feature_version = (
            feature_version or self.settings.fee_aware_targets.default_feature_version
        )
        rows: list[dict[str, Any]] = []
        for target_config in build_target_configs(self.settings, target_keys=target_keys):
            try:
                resolved_run = self._resolve_latest_run_for_target(
                    symbol=resolved_symbol,
                    timeframe=resolved_timeframe,
                    target_config=target_config,
                )
                aggregate_metrics = self.model_registry.read_aggregate_metrics(
                    resolved_run.run_id,
                    asset=resolved_run.asset,
                    symbol=resolved_run.symbol,
                    timeframe=resolved_run.timeframe,
                    model_version=resolved_run.model_version,
                )
                validation_report = self._latest_validation_report_for_run(resolved_run.run_id)
                optimization_report = self._latest_optimization_report_for_run(resolved_run.run_id)
                positive_class_rate = _positive_rate_from_metadata(
                    resolved_run.metadata,
                    target_config.target_name,
                )
            except Exception as exc:
                raise FeeAwareTargetResearchError(
                    f"Failed to compare target={target_config.key}: {exc}"
                ) from exc
            rows.append(
                TargetRunSummary(
                    target_key=target_config.key,
                    target_name=target_config.target_name,
                    horizon_bars=target_config.horizon_bars,
                    horizon_minutes=_horizon_minutes(self.settings, target_config.horizon_bars),
                    threshold_type=target_config.threshold_type,
                    threshold_return=target_config.threshold_return,
                    positive_class_rate=positive_class_rate,
                    label_version=target_config.label_version,
                    dataset_version=target_config.dataset_version,
                    model_version=target_config.model_version,
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

        comparison_df = rank_target_results(
            pd.DataFrame(rows),
            max_allowed_drawdown=self.settings.optimization_selection.max_allowed_drawdown,
            max_allowed_trades=self.settings.optimization_selection.max_allowed_trades,
            max_allowed_fee_to_starting_cash=(
                self.settings.optimization_selection.max_allowed_fee_to_starting_cash
            ),
            positive_class_rate_min=self.settings.fee_aware_targets.positive_class_rate_min,
            positive_class_rate_max=self.settings.fee_aware_targets.positive_class_rate_max,
        )
        resolved_comparison_id = comparison_id or _default_comparison_id()
        recommendation = build_target_recommendation(comparison_df)
        feature_versions = sorted(
            {
                str(value)
                for value in comparison_df.get("feature_version", pd.Series(dtype=object))
                .dropna()
                .tolist()
                if str(value)
            }
        )
        report_feature_version = feature_versions[0] if len(feature_versions) == 1 else "mixed"
        artifact_paths = {
            "comparison_table_path": str(
                self.target_store.report_dir(resolved_comparison_id) / "target_comparison.parquet"
            ),
            "report_json_path": str(
                self.target_store.report_dir(resolved_comparison_id) / "target_report.json"
            ),
            "report_markdown_path": str(
                self.target_store.report_dir(resolved_comparison_id) / "target_report.md"
            ),
        }
        report = TargetComparisonReport(
            comparison_id=resolved_comparison_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            asset=resolved_asset,
            symbol=resolved_symbol,
            timeframe=resolved_timeframe,
            feature_version=report_feature_version,
            feature_versions=feature_versions,
            target_keys_tested=[config.key for config in build_target_configs(self.settings, target_keys=target_keys)],
            rows=comparison_df.to_dict(orient="records"),
            recommendation=recommendation,
            artifact_paths=artifact_paths,
        ).to_dict()
        self.target_store.write_comparison_table(resolved_comparison_id, comparison_df)
        self.target_store.write_report_json(resolved_comparison_id, report)
        if self.settings.reports.write_markdown:
            self.target_store.write_report_markdown(
                resolved_comparison_id,
                format_target_report_markdown(report),
            )
        logger.info(
            "target_comparison_report_complete",
            comparison_id=resolved_comparison_id,
            best_target_name=recommendation.get("best_target_name"),
            no_viable_target=recommendation.get("no_viable_target"),
        )
        return report

    def _resolve_latest_run_for_target(
        self,
        *,
        symbol: str | None,
        timeframe: str | None,
        target_config: TargetConfig,
    ) -> ResolvedRun:
        runs = [
            run
            for run in self.model_registry.list_runs(
                symbol=symbol or self.settings.datasets.symbol,
                timeframe=timeframe or self.settings.datasets.timeframe,
                model_version=target_config.model_version,
            )
            if str(run.get("target_column")) == target_config.target_name
        ]
        if not runs:
            raise FeeAwareTargetResearchError(
                f"No stored model run found for target={target_config.key} ({target_config.target_name})"
            )
        runs.sort(key=lambda run: str(run.get("created_at", "")), reverse=True)
        return self.model_registry.resolve_run(
            str(runs[0]["run_id"]),
            symbol=symbol or self.settings.datasets.symbol,
            timeframe=timeframe or self.settings.datasets.timeframe,
            model_version=target_config.model_version,
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


def build_target_configs(
    settings: AppSettings,
    *,
    target_keys: list[str] | None = None,
) -> list[TargetConfig]:
    """Resolve configured fee-aware target aliases into stable target configs."""
    fee_rate = settings.fee_aware_targets.fee_rate
    slippage_rate = settings.fee_aware_targets.slippage_rate
    default_round_trip_cost = settings.fee_aware_targets.default_round_trip_cost
    label_prefix = settings.fee_aware_targets.label_version_prefix
    dataset_prefix = settings.fee_aware_targets.dataset_version_prefix
    model_prefix = settings.fee_aware_targets.model_version_prefix

    config_map = {
        "fee_4": TargetConfig(
            key="fee_4",
            target_name="target_long_net_positive_4bars",
            horizon_bars=4,
            threshold_type="fee_aware",
            label_version=f"{label_prefix}_h4",
            dataset_version=f"{dataset_prefix}_h4",
            model_version=f"{model_prefix}_h4",
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            round_trip_cost=default_round_trip_cost,
        ),
        "fee_8": TargetConfig(
            key="fee_8",
            target_name="target_long_net_positive_8bars",
            horizon_bars=8,
            threshold_type="fee_aware",
            label_version=f"{label_prefix}_h8",
            dataset_version=f"{dataset_prefix}_h8",
            model_version=f"{model_prefix}_h8",
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            round_trip_cost=default_round_trip_cost,
        ),
        "fee_16": TargetConfig(
            key="fee_16",
            target_name="target_long_net_positive_16bars",
            horizon_bars=16,
            threshold_type="fee_aware",
            label_version=f"{label_prefix}_h16",
            dataset_version=f"{dataset_prefix}_h16",
            model_version=f"{model_prefix}_h16",
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            round_trip_cost=default_round_trip_cost,
        ),
        "ret025_8": TargetConfig(
            key="ret025_8",
            target_name="target_return_gt_025pct_8bars",
            horizon_bars=8,
            threshold_type="minimum_return",
            threshold_return=0.0025,
            label_version=f"{label_prefix}_ret025_h8",
            dataset_version=f"{dataset_prefix}_ret025_h8",
            model_version=f"{model_prefix}_ret025_h8",
        ),
        "ret050_16": TargetConfig(
            key="ret050_16",
            target_name="target_return_gt_050pct_16bars",
            horizon_bars=16,
            threshold_type="minimum_return",
            threshold_return=0.005,
            label_version=f"{label_prefix}_ret050_h16",
            dataset_version=f"{dataset_prefix}_ret050_h16",
            model_version=f"{model_prefix}_ret050_h16",
        ),
        "ret075_16": TargetConfig(
            key="ret075_16",
            target_name="target_return_gt_075pct_16bars",
            horizon_bars=16,
            threshold_type="minimum_return",
            threshold_return=0.0075,
            label_version=f"{label_prefix}_ret075_h16",
            dataset_version=f"{dataset_prefix}_ret075_h16",
            model_version=f"{model_prefix}_ret075_h16",
        ),
    }
    requested_keys = target_keys or list(settings.fee_aware_targets.target_keys)
    configs: list[TargetConfig] = []
    for key in requested_keys:
        if key not in config_map:
            raise FeeAwareTargetResearchError(f"Unsupported target key: {key}")
        configs.append(config_map[key])
    return configs


def rank_target_results(
    comparison_df: pd.DataFrame,
    *,
    max_allowed_drawdown: float,
    max_allowed_trades: int,
    max_allowed_fee_to_starting_cash: float,
    positive_class_rate_min: float,
    positive_class_rate_max: float,
) -> pd.DataFrame:
    """Rank fee-aware target rows using edge, tradeability, and class-balance filters."""
    if comparison_df.empty:
        return comparison_df
    ranked = comparison_df.copy()
    positive_rates = pd.to_numeric(ranked.get("positive_class_rate"), errors="coerce")
    optimized_has_edge = ranked.get("optimized_has_edge", pd.Series(False, index=ranked.index))
    validation_has_edge = ranked.get("validation_has_edge", pd.Series(pd.NA, index=ranked.index))
    ranked["optimized_has_edge_passed"] = optimized_has_edge.fillna(False).astype(bool)
    ranked["return_filter_passed"] = (
        pd.to_numeric(ranked["best_strategy_total_return"], errors="coerce").fillna(-1.0) > 0.0
    )
    ranked["drawdown_filter_passed"] = (
        pd.to_numeric(ranked["best_strategy_max_drawdown"], errors="coerce").fillna(1.0)
        <= max_allowed_drawdown
    )
    ranked["trade_count_filter_passed"] = (
        pd.to_numeric(ranked["best_strategy_number_of_trades"], errors="coerce")
        .fillna(max_allowed_trades + 1)
        .astype(int)
        <= max_allowed_trades
    )
    ranked["fee_filter_passed"] = (
        pd.to_numeric(ranked["best_strategy_fee_to_starting_cash"], errors="coerce")
        .fillna(max_allowed_fee_to_starting_cash + 1.0)
        <= max_allowed_fee_to_starting_cash
    )
    ranked["class_balance_filter_passed"] = positive_rates.fillna(-1.0).between(
        positive_class_rate_min,
        positive_class_rate_max,
    )
    ranked["validation_has_edge_passed"] = validation_has_edge.map(
        lambda value: None if pd.isna(value) else bool(value)
    )
    ranked["passes_filters"] = (
        ranked["optimized_has_edge_passed"].astype(bool)
        & ranked["return_filter_passed"].astype(bool)
        & ranked["drawdown_filter_passed"].astype(bool)
        & ranked["trade_count_filter_passed"].astype(bool)
        & ranked["fee_filter_passed"].astype(bool)
        & ranked["class_balance_filter_passed"].astype(bool)
    )
    ranked["rejection_reason"] = ranked.apply(
        lambda row: _target_rejection_reason(row),
        axis=1,
    )
    ranked["ranking_score"] = (
        ranked["passes_filters"].astype(int) * 1000.0
        + ranked["optimized_has_edge_passed"].astype(int) * 100.0
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


def build_target_recommendation(comparison_df: pd.DataFrame) -> dict[str, Any]:
    """Build the final best-target or no-viable-target recommendation."""
    if comparison_df.empty:
        return {
            "best_target_name": None,
            "no_viable_target": True,
            "reason": "No target rows were available for comparison.",
        }
    best_row = comparison_df.iloc[0].to_dict()
    if bool(best_row.get("passes_filters", False)):
        return {
            "best_target_name": str(best_row.get("target_name")),
            "best_target_key": str(best_row.get("target_key")),
            "no_viable_target": False,
            "reason": "This target passed optimization filters and ranked highest after costs.",
            "rejection_reason": "eligible",
            "validation_has_edge_required": False,
            "best_model_run_id": best_row.get("model_run_id"),
            "best_validation_id": best_row.get("validation_id"),
            "best_optimization_id": best_row.get("optimization_id"),
        }
    return {
        "best_target_name": None,
        "no_viable_target": True,
        "reason": (
            "No target satisfied the optimization edge, return, drawdown, trade-count, "
            "class-balance, and fee filters. Improve features or target design before paper trading."
        ),
        "best_available_target_name": best_row.get("target_name"),
        "best_available_rejection_reason": best_row.get("rejection_reason"),
        "validation_has_edge_required": False,
    }


def format_target_report_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown summary for fee-aware target comparison."""
    recommendation = report.get("recommendation", {})
    lines = [
        f"# Target Comparison Report: {report.get('comparison_id', 'unknown')}",
        "",
        "## Overview",
        f"- Asset: `{report.get('asset')}`",
        f"- Symbol: `{report.get('symbol')}`",
        f"- Timeframe: `{report.get('timeframe')}`",
        f"- Feature version: `{report.get('feature_version')}`",
        f"- Feature versions: `{report.get('feature_versions')}`",
        f"- Target keys tested: `{report.get('target_keys_tested')}`",
        "",
        "## Recommendation",
        f"- best_target_name: `{recommendation.get('best_target_name')}`",
        f"- no_viable_target: `{recommendation.get('no_viable_target')}`",
        f"- reason: {recommendation.get('reason')}",
        "",
        "## Ranked Targets",
    ]
    for row in report.get("rows", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- target={row.get('target_name')} return={row.get('best_strategy_total_return')} "
            f"drawdown={row.get('best_strategy_max_drawdown')} "
            f"trades={row.get('best_strategy_number_of_trades')} "
            f"optimized_has_edge={row.get('optimized_has_edge')} "
            f"rejection_reason={row.get('rejection_reason')}"
        )
    return "\n".join(lines) + "\n"


def _default_comparison_id() -> str:
    """Return a UTC timestamp-based id for one target-comparison run."""
    return datetime.now(timezone.utc).strftime("TCOMP%Y%m%dT%H%M%SZ")


def _horizon_minutes(settings: AppSettings, horizon_bars: int) -> int:
    timeframe = pd.Timedelta(settings.datasets.timeframe)
    return int(timeframe.total_seconds() * horizon_bars / 60)


def _positive_rate_from_dataset(dataset_df: pd.DataFrame, target_name: str) -> float | None:
    if target_name not in dataset_df.columns or dataset_df.empty:
        return None
    values = pd.to_numeric(dataset_df[target_name], errors="coerce").dropna()
    if values.empty:
        return None
    return round(float(values.astype(int).mean()), 6)


def _positive_rate_from_metadata(metadata: dict[str, Any], target_name: str) -> float | None:
    extra = metadata.get("extra", {})
    if not isinstance(extra, dict):
        return None
    target_summary = extra.get("target_summary", {})
    if not isinstance(target_summary, dict):
        return None
    if "positive_rate" in target_summary:
        value = target_summary.get("positive_rate")
        return float(value) if value is not None else None
    summary = target_summary.get(target_name, {})
    if not isinstance(summary, dict):
        return None
    value = summary.get("positive_rate")
    return float(value) if value is not None else None


def _target_rejection_reason(row: pd.Series) -> str:
    """Return an explicit eligibility or rejection reason for one target row."""
    if bool(row.get("passes_filters", False)):
        return "eligible"
    failed_reasons: list[str] = []
    for column, reason in (
        ("optimized_has_edge_passed", "optimized_has_edge_failed"),
        ("return_filter_passed", "non_positive_total_return"),
        ("drawdown_filter_passed", "drawdown_limit_failed"),
        ("trade_count_filter_passed", "trade_count_limit_failed"),
        ("fee_filter_passed", "fee_limit_failed"),
        ("class_balance_filter_passed", "class_balance_filter_failed"),
    ):
        value = row.get(column)
        if pd.isna(value):
            failed_reasons.append(f"{column}_unknown")
        elif not bool(value):
            failed_reasons.append(reason)
    if not failed_reasons:
        failed_reasons.append("unknown_rejection")
    if row.get("validation_has_edge_passed") is False:
        failed_reasons.append("validation_has_edge_false_not_required")
    return ", ".join(failed_reasons)


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
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
