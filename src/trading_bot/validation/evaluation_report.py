"""Structured reporting helpers for walk-forward training runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class EvaluationReport:
    """Machine-readable summary of one training run and its stored artifacts."""

    run_id: str
    symbol: str
    timeframe: str
    dataset_version: str
    label_version: str
    model_version: str
    target_column: str
    feature_count: int
    fold_count: int
    requested_device: str
    effective_device: str
    lightgbm_version: str
    validation_start: str | None
    validation_end: str | None
    metrics: dict[str, float | int | None]
    run_metadata: dict[str, Any] = field(default_factory=dict)
    aggregate_metrics: dict[str, Any] = field(default_factory=dict)
    per_fold_metrics_summary: dict[str, Any] = field(default_factory=dict)
    target_distribution_summary: dict[str, Any] = field(default_factory=dict)
    prediction_probability_summary: dict[str, Any] = field(default_factory=dict)
    feature_importance_summary: dict[str, Any] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    status: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable payload."""
        return asdict(self)


def build_evaluation_report(
    *,
    run_metadata: dict[str, Any],
    aggregate_metrics: dict[str, Any],
    fold_metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame | None = None,
    feature_importance_df: pd.DataFrame | None = None,
    target_distribution_summary: dict[str, Any] | None = None,
    artifact_paths: dict[str, str] | None = None,
    status: str | None = None,
    notes: str = "",
    top_k_feature_importance: int = 20,
) -> EvaluationReport:
    """Build a compact but richer evaluation report from stored run artifacts."""
    validation_start: str | None = None
    validation_end: str | None = None
    if not fold_metrics_df.empty:
        validation_start = str(fold_metrics_df["validation_start"].min())
        validation_end = str(fold_metrics_df["validation_end"].max())

    per_fold_summary = summarize_fold_metrics(fold_metrics_df)
    prediction_summary = summarize_predictions(
        predictions_df if predictions_df is not None else pd.DataFrame()
    )
    feature_summary = summarize_feature_importance(
        feature_importance_df if feature_importance_df is not None else pd.DataFrame(),
        top_k=top_k_feature_importance,
    )

    return EvaluationReport(
        run_id=str(run_metadata["run_id"]),
        symbol=str(run_metadata["symbol"]),
        timeframe=str(run_metadata["timeframe"]),
        dataset_version=str(run_metadata["dataset_version"]),
        label_version=str(run_metadata["label_version"]),
        model_version=str(run_metadata["model_version"]),
        target_column=str(run_metadata["target_column"]),
        feature_count=len(run_metadata.get("feature_columns", [])),
        fold_count=int(run_metadata.get("fold_count", aggregate_metrics.get("fold_count", 0))),
        requested_device=str(run_metadata["requested_device"]),
        effective_device=str(run_metadata["effective_device"]),
        lightgbm_version=str(run_metadata["lightgbm_version"]),
        validation_start=validation_start,
        validation_end=validation_end,
        metrics=dict(aggregate_metrics),
        run_metadata=dict(run_metadata),
        aggregate_metrics=dict(aggregate_metrics),
        per_fold_metrics_summary=per_fold_summary,
        target_distribution_summary=dict(target_distribution_summary or {}),
        prediction_probability_summary=prediction_summary,
        feature_importance_summary=feature_summary,
        artifact_paths=dict(artifact_paths or {}),
        status=status,
        notes=notes,
    )


def summarize_fold_metrics(fold_metrics_df: pd.DataFrame) -> dict[str, Any]:
    """Return best, worst, and simple averages from per-fold metrics."""
    if fold_metrics_df.empty:
        return {
            "fold_count": 0,
            "best_fold_by_f1": None,
            "worst_fold_by_log_loss": None,
            "fold_rows": [],
        }

    working_df = fold_metrics_df.copy()
    best_fold_row = (
        working_df.sort_values("f1", ascending=False, na_position="last").iloc[0].to_dict()
        if "f1" in working_df.columns
        else None
    )
    worst_fold_row = (
        working_df.sort_values("log_loss", ascending=False, na_position="last").iloc[0].to_dict()
        if "log_loss" in working_df.columns
        else None
    )

    return {
        "fold_count": int(len(working_df)),
        "best_fold_by_f1": _json_ready_row(best_fold_row),
        "worst_fold_by_log_loss": _json_ready_row(worst_fold_row),
        "fold_rows": [_json_ready_row(row) for row in working_df.to_dict(orient="records")],
    }


def summarize_predictions(predictions_df: pd.DataFrame, *, threshold: float = 0.5) -> dict[str, Any]:
    """Return aggregate probability-quality and confusion-matrix summaries."""
    if predictions_df.empty:
        return {
            "row_count": 0,
            "threshold": threshold,
            "mean_probability": None,
            "prediction_positive_rate": None,
            "observed_positive_rate": None,
            "confusion_matrix": {},
            "confidence_buckets": [],
        }

    y_true = pd.to_numeric(predictions_df["y_true"], errors="coerce").fillna(0).astype(int)
    y_pred = pd.to_numeric(predictions_df["y_pred"], errors="coerce").fillna(0).astype(int)
    y_proba = pd.to_numeric(predictions_df["y_proba"], errors="coerce")

    confusion_matrix = {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }

    bucket_counts = pd.cut(
        y_proba,
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        include_lowest=True,
    ).value_counts(sort=False, dropna=False)

    return {
        "row_count": int(len(predictions_df)),
        "threshold": threshold,
        "mean_probability": _float_or_none(y_proba.mean()),
        "std_probability": _float_or_none(y_proba.std(ddof=1)),
        "min_probability": _float_or_none(y_proba.min()),
        "max_probability": _float_or_none(y_proba.max()),
        "prediction_positive_rate": _float_or_none(y_pred.mean()),
        "observed_positive_rate": _float_or_none(y_true.mean()),
        "confusion_matrix": confusion_matrix,
        "confidence_buckets": [
            {"bucket": str(bucket), "count": int(count)}
            for bucket, count in bucket_counts.items()
        ],
    }


def summarize_feature_importance(
    feature_importance_df: pd.DataFrame,
    *,
    top_k: int = 20,
) -> dict[str, Any]:
    """Return a stable top-k feature-importance summary."""
    if feature_importance_df.empty:
        return {
            "feature_count": 0,
            "top_features": [],
        }

    working_df = feature_importance_df.copy()
    columns = [
        column
        for column in ("feature_name", "importance", "rank", "mean_importance")
        if column in working_df.columns
    ]
    top_features = working_df.head(top_k).loc[:, columns].to_dict(orient="records")

    return {
        "feature_count": int(len(working_df)),
        "top_features": [_json_ready_row(row) for row in top_features],
    }


def format_evaluation_report_markdown(report: dict[str, Any]) -> str:
    """Render a compact human-readable Markdown report."""
    lines = [
        f"# Evaluation Report: {report.get('run_id', 'unknown')}",
        "",
        "## Run Metadata",
        f"- Symbol: {report.get('symbol')}",
        f"- Timeframe: {report.get('timeframe')}",
        f"- Dataset Version: {report.get('dataset_version')}",
        f"- Label Version: {report.get('label_version')}",
        f"- Model Version: {report.get('model_version')}",
        f"- Target Column: {report.get('target_column')}",
        f"- Requested Device: {report.get('requested_device')}",
        f"- Effective Device: {report.get('effective_device')}",
        f"- Feature Count: {report.get('feature_count')}",
        f"- Fold Count: {report.get('fold_count')}",
        "",
        "## Aggregate Metrics",
    ]

    for metric_name, metric_value in dict(report.get("aggregate_metrics", {})).items():
        lines.append(f"- {metric_name}: {metric_value}")

    lines.extend(
        [
            "",
            "## Prediction Summary",
        ]
    )
    for key, value in dict(report.get("prediction_probability_summary", {})).items():
        if key == "confidence_buckets":
            continue
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Top Features",
        ]
    )
    top_features = report.get("feature_importance_summary", {}).get("top_features", [])
    if not top_features:
        lines.append("- No feature importance available")
    else:
        for row in top_features:
            lines.append(
                f"- {row.get('feature_name')}: {row.get('importance', row.get('mean_importance'))}"
            )

    artifact_paths = dict(report.get("artifact_paths", {}))
    if artifact_paths:
        lines.extend(
            [
                "",
                "## Artifact Paths",
            ]
        )
        for key, value in artifact_paths.items():
            lines.append(f"- {key}: {value}")

    return "\n".join(lines) + "\n"


def _json_ready_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert pandas scalar values to JSON-safe primitives."""
    if row is None:
        return None
    return {
        key: (value.isoformat() if hasattr(value, "isoformat") else _float_or_none(value) if isinstance(value, float) else value)
        for key, value in row.items()
    }


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
