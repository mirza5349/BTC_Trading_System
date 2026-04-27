"""Experiment tracking built on top of stored step-6 training artifacts."""

from __future__ import annotations

from typing import Any

import pandas as pd

from trading_bot.logging_config import get_logger
from trading_bot.schemas.experiments import (
    ExperimentRunRecord,
    FeatureImportanceSummary,
    PredictionProbabilitySummary,
)
from trading_bot.settings import PROJECT_ROOT, AppSettings, load_settings
from trading_bot.storage.evaluation_store import EvaluationStore
from trading_bot.storage.experiment_store import ExperimentStore
from trading_bot.storage.report_store import ReportStore
from trading_bot.validation.evaluation_report import (
    build_evaluation_report,
    format_evaluation_report_markdown,
)

logger = get_logger(__name__)


class ExperimentTracker:
    """Register, summarize, and report on locally stored training runs."""

    def __init__(
        self,
        *,
        evaluation_store: EvaluationStore,
        experiment_store: ExperimentStore,
        report_store: ReportStore,
        default_status: str,
        write_markdown: bool,
        top_k_feature_importance: int,
    ) -> None:
        self.evaluation_store = evaluation_store
        self.experiment_store = experiment_store
        self.report_store = report_store
        self.default_status = default_status
        self.write_markdown = write_markdown
        self.top_k_feature_importance = top_k_feature_importance

    @classmethod
    def from_settings(cls, settings: AppSettings | None = None) -> ExperimentTracker:
        """Create an experiment tracker from project settings."""
        resolved_settings = settings or load_settings()
        return cls(
            evaluation_store=EvaluationStore(
                PROJECT_ROOT / resolved_settings.artifacts.evaluation_dir
            ),
            experiment_store=ExperimentStore(
                base_dir=PROJECT_ROOT / resolved_settings.experiments.summary_dir,
                registry_path=PROJECT_ROOT / resolved_settings.experiments.registry_path,
            ),
            report_store=ReportStore(PROJECT_ROOT / resolved_settings.reports.output_dir),
            default_status=resolved_settings.experiments.default_status_for_new_runs,
            write_markdown=resolved_settings.reports.write_markdown,
            top_k_feature_importance=resolved_settings.reports.top_k_feature_importance,
        )

    def refresh_registry(self) -> pd.DataFrame:
        """Scan stored evaluation artifacts and refresh the local registry."""
        runs = self.evaluation_store.list_runs()
        rows: list[dict[str, Any]] = []

        for metadata in runs:
            run_id = str(metadata["run_id"])
            summary = self.register_run(run_id, metadata=metadata)
            rows.append(summary["registry_row"])

        registry_df = self.experiment_store.upsert_registry_rows(rows)
        logger.info("experiment_registry_refreshed", run_count=len(registry_df))
        return registry_df

    def register_run(
        self,
        run_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build and persist an experiment summary and richer report for one run."""
        resolved_metadata = metadata or self._resolve_metadata(run_id)
        coordinates = self._coordinates_from_metadata(resolved_metadata)

        aggregate_metrics = self.evaluation_store.read_aggregate_metrics(**coordinates)
        fold_metrics_df = self.evaluation_store.read_fold_metrics(**coordinates)
        predictions_df = self.evaluation_store.read_predictions(**coordinates)
        feature_importance_df = self.evaluation_store.read_feature_importance(
            **coordinates,
            aggregate=True,
        )

        previous_summary = self.experiment_store.read_run_summary(run_id)
        status = str(previous_summary.get("status", self.default_status))
        notes = str(previous_summary.get("notes", ""))

        probability_summary = _summarize_probabilities(predictions_df, threshold=0.5)
        feature_summary = _summarize_feature_importance(
            feature_importance_df,
            top_k=self.top_k_feature_importance,
        )

        artifact_paths = {
            "evaluation_dir": str(
                self.evaluation_store.run_dir(
                    asset=coordinates["asset"],
                    symbol=coordinates["symbol"],
                    timeframe=coordinates["timeframe"],
                    model_version=coordinates["model_version"],
                    run_id=coordinates["run_id"],
                )
            ),
            "run_summary_path": str(self.experiment_store.summary_path(run_id)),
            "report_json_path": str(self.report_store.report_json_path(run_id)),
            "report_markdown_path": str(self.report_store.report_markdown_path(run_id)),
            "top_features_path": str(self.experiment_store.top_features_path(run_id)),
        }

        report = build_evaluation_report(
            run_metadata=resolved_metadata,
            aggregate_metrics=aggregate_metrics,
            fold_metrics_df=fold_metrics_df,
            predictions_df=predictions_df,
            feature_importance_df=feature_importance_df,
            target_distribution_summary=dict(resolved_metadata.get("extra", {}).get("target_summary", {})),
            artifact_paths=artifact_paths,
            status=status,
            notes=notes,
            top_k_feature_importance=self.top_k_feature_importance,
        )
        report_payload = report.to_dict()

        registry_row = ExperimentRunRecord(
            run_id=run_id,
            created_at=str(resolved_metadata.get("created_at", "")),
            symbol=str(resolved_metadata.get("symbol", "")),
            timeframe=str(resolved_metadata.get("timeframe", "")),
            dataset_version=str(resolved_metadata.get("dataset_version", "")),
            feature_version=str(_feature_version_from_metadata(resolved_metadata)),
            label_version=str(resolved_metadata.get("label_version", "")),
            model_version=str(resolved_metadata.get("model_version", "")),
            target_column=str(resolved_metadata.get("target_column", "")),
            requested_device=str(resolved_metadata.get("requested_device", "")),
            effective_device=str(resolved_metadata.get("effective_device", "")),
            fold_count=int(report_payload["fold_count"]),
            feature_count=int(report_payload["feature_count"]),
            accuracy_mean=_float_or_none(aggregate_metrics.get("accuracy")),
            precision_mean=_float_or_none(aggregate_metrics.get("precision")),
            recall_mean=_float_or_none(aggregate_metrics.get("recall")),
            f1_mean=_float_or_none(aggregate_metrics.get("f1")),
            roc_auc_mean=_float_or_none(aggregate_metrics.get("roc_auc")),
            log_loss_mean=_float_or_none(aggregate_metrics.get("log_loss")),
            status=status,
            notes=notes,
        ).to_dict()

        summary_payload = {
            "run_id": run_id,
            "registry_row": registry_row,
            "metadata": resolved_metadata,
            "aggregate_metrics": aggregate_metrics,
            "per_fold_metrics": fold_metrics_df.to_dict(orient="records"),
            "prediction_probability_summary": probability_summary.to_dict(),
            "feature_importance_summary": feature_summary.to_dict(),
            "artifact_paths": artifact_paths,
            "status": status,
            "notes": notes,
        }

        self.experiment_store.write_run_summary(run_id, summary_payload)
        if not feature_importance_df.empty:
            self.experiment_store.write_top_features(
                run_id,
                feature_importance_df.head(self.top_k_feature_importance).reset_index(drop=True),
            )
        self.report_store.write_json_report(run_id, report_payload)
        if self.write_markdown:
            self.report_store.write_markdown_report(
                run_id,
                format_evaluation_report_markdown(report_payload),
            )

        logger.info("experiment_run_registered", run_id=run_id, status=status)
        return summary_payload

    def generate_run_report(self, run_id: str, *, write_report: bool = True) -> dict[str, Any]:
        """Return a rich run report, generating it from stored artifacts if needed."""
        summary = self.register_run(run_id) if write_report else self.experiment_store.read_run_summary(run_id)
        if write_report:
            report_payload = self.report_store.read_json_report(run_id)
            return report_payload
        return summary

    def _resolve_metadata(self, run_id: str) -> dict[str, Any]:
        matches = [run for run in self.evaluation_store.list_runs() if str(run.get("run_id")) == run_id]
        if not matches:
            raise ValueError(f"No stored evaluation metadata found for run_id={run_id}")
        if len(matches) > 1:
            raise ValueError(f"Multiple runs matched run_id={run_id}")
        return matches[0]

    @staticmethod
    def _coordinates_from_metadata(metadata: dict[str, Any]) -> dict[str, str]:
        return {
            "asset": str(metadata.get("asset", "")),
            "symbol": str(metadata.get("symbol", "")),
            "timeframe": str(metadata.get("timeframe", "")),
            "model_version": str(metadata.get("model_version", "")),
            "run_id": str(metadata.get("run_id", "")),
        }


def _summarize_probabilities(
    predictions_df: pd.DataFrame,
    *,
    threshold: float,
) -> PredictionProbabilitySummary:
    """Return a compact validation-probability summary for one run."""
    if predictions_df.empty:
        return PredictionProbabilitySummary(
            row_count=0,
            mean_probability=None,
            std_probability=None,
            min_probability=None,
            max_probability=None,
            predicted_positive_rate=None,
            observed_positive_rate=None,
            threshold=threshold,
            confusion_matrix={},
            confidence_buckets=[],
        )

    probabilities = pd.to_numeric(predictions_df["y_proba"], errors="coerce")
    y_true = pd.to_numeric(predictions_df["y_true"], errors="coerce").fillna(0).astype(int)
    y_pred = pd.to_numeric(predictions_df["y_pred"], errors="coerce").fillna(0).astype(int)

    confusion_matrix = {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }

    buckets = pd.cut(
        probabilities,
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        include_lowest=True,
    )
    bucket_counts = buckets.value_counts(sort=False, dropna=False)
    confidence_buckets = [
        {"bucket": str(bucket), "count": int(count)}
        for bucket, count in bucket_counts.items()
    ]

    return PredictionProbabilitySummary(
        row_count=int(len(predictions_df)),
        mean_probability=_float_or_none(probabilities.mean()),
        std_probability=_float_or_none(probabilities.std(ddof=1)),
        min_probability=_float_or_none(probabilities.min()),
        max_probability=_float_or_none(probabilities.max()),
        predicted_positive_rate=_float_or_none(y_pred.mean()),
        observed_positive_rate=_float_or_none(y_true.mean()),
        threshold=threshold,
        confusion_matrix=confusion_matrix,
        confidence_buckets=confidence_buckets,
    )


def _summarize_feature_importance(
    feature_importance_df: pd.DataFrame,
    *,
    top_k: int,
) -> FeatureImportanceSummary:
    """Return the top feature-importance rows for one stored run."""
    if not isinstance(feature_importance_df, pd.DataFrame) or feature_importance_df.empty:
        return FeatureImportanceSummary()

    working_df = feature_importance_df.copy()
    importance_column = _resolve_importance_column(working_df)
    if importance_column is None:
        logger.warning(
            "feature_importance_summary_missing_importance_column",
            available_columns=list(working_df.columns),
        )
        return FeatureImportanceSummary()

    importance_values = _coerce_importance_series(
        working_df,
        importance_column=importance_column,
    )
    if importance_values is None:
        logger.warning(
            "feature_importance_summary_invalid_importance_values",
            importance_column=importance_column,
        )
        return FeatureImportanceSummary()

    working_df["importance"] = importance_values
    working_df = working_df.dropna(subset=["importance"]).reset_index(drop=True)
    if working_df.empty:
        logger.warning("feature_importance_summary_empty_after_normalization")
        return FeatureImportanceSummary()

    if "rank" not in working_df.columns:
        working_df["rank"] = range(1, len(working_df) + 1)

    top_features = (
        working_df.head(top_k)
        .loc[:, [column for column in ("feature_name", "importance", "rank") if column in working_df.columns]]
        .to_dict(orient="records")
    )
    total_importance = float(working_df["importance"].fillna(0.0).sum())

    return FeatureImportanceSummary(
        top_features=top_features,
        feature_count=int(len(working_df)),
        total_importance=total_importance,
    )


def _resolve_importance_column(feature_importance_df: pd.DataFrame) -> str | None:
    """Return the most appropriate importance column when present."""
    for column_name in ("importance", "gain", "split", "importance_mean", "mean_importance"):
        if column_name in feature_importance_df.columns:
            return column_name
    return None


def _coerce_importance_series(
    feature_importance_df: pd.DataFrame,
    *,
    importance_column: str,
) -> pd.Series | None:
    """Return importance values as a numeric pandas Series or None when malformed."""
    raw_values = feature_importance_df.get(importance_column)
    if raw_values is None:
        return None
    if isinstance(raw_values, pd.DataFrame):
        if raw_values.shape[1] == 0:
            return None
        raw_values = raw_values.iloc[:, 0]
    if not isinstance(raw_values, pd.Series):
        return None
    numeric_values = pd.to_numeric(raw_values, errors="coerce")
    return numeric_values.fillna(0.0)


def _feature_version_from_metadata(metadata: dict[str, Any]) -> str:
    """Return the feature version from run metadata when present."""
    feature_version = metadata.get("feature_version")
    if feature_version:
        return str(feature_version)

    extra = metadata.get("extra", {})
    if isinstance(extra, dict) and extra.get("feature_version"):
        return str(extra["feature_version"])

    return "unknown"


def _float_or_none(value: Any) -> float | None:
    """Return floats when finite and not missing."""
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(converted):
        return None
    return converted
