"""Schema metadata and helpers for local experiment tracking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import pandas as pd

RUN_STATUS_VALUES: tuple[str, ...] = (
    "candidate",
    "rejected",
    "archived",
    "promoted_for_backtest",
)

RunStatus = Literal["candidate", "rejected", "archived", "promoted_for_backtest"]

SUMMARY_METRIC_COLUMNS: list[str] = [
    "accuracy_mean",
    "precision_mean",
    "recall_mean",
    "f1_mean",
    "roc_auc_mean",
    "log_loss_mean",
]

RUN_REGISTRY_COLUMNS: list[str] = [
    "run_id",
    "created_at",
    "symbol",
    "timeframe",
    "dataset_version",
    "feature_version",
    "label_version",
    "model_version",
    "target_column",
    "requested_device",
    "effective_device",
    "fold_count",
    "feature_count",
    *SUMMARY_METRIC_COLUMNS,
    "status",
    "notes",
]

RECOMMENDED_COMPARISON_COLUMNS: list[str] = [
    "run_id",
    "dataset_version",
    "feature_version",
    "label_version",
    "model_version",
    "target_column",
    "feature_count",
    "requested_device",
    "effective_device",
    "fold_count",
    "status",
]


@dataclass(frozen=True)
class ExperimentRunRecord:
    """Compact row persisted in the local experiment registry."""

    run_id: str
    created_at: str
    symbol: str
    timeframe: str
    dataset_version: str
    feature_version: str
    label_version: str
    model_version: str
    target_column: str
    requested_device: str
    effective_device: str
    fold_count: int
    feature_count: int
    accuracy_mean: float | None
    precision_mean: float | None
    recall_mean: float | None
    f1_mean: float | None
    roc_auc_mean: float | None
    log_loss_mean: float | None
    status: RunStatus
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record."""
        return asdict(self)


@dataclass(frozen=True)
class PredictionProbabilitySummary:
    """Aggregate probability-quality summary for one validation run."""

    row_count: int
    mean_probability: float | None
    std_probability: float | None
    min_probability: float | None
    max_probability: float | None
    predicted_positive_rate: float | None
    observed_positive_rate: float | None
    threshold: float
    confusion_matrix: dict[str, int] = field(default_factory=dict)
    confidence_buckets: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable payload."""
        return asdict(self)


@dataclass(frozen=True)
class FeatureImportanceSummary:
    """Compact feature-importance summary for one stored run."""

    top_features: list[dict[str, Any]] = field(default_factory=list)
    feature_count: int = 0
    total_importance: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable payload."""
        return asdict(self)


def validate_run_status(status: str) -> RunStatus:
    """Validate and normalize a promotion status string."""
    if status not in RUN_STATUS_VALUES:
        raise ValueError(
            f"Unsupported run status {status!r}. Expected one of {RUN_STATUS_VALUES}."
        )
    return status


def default_registry_row() -> dict[str, Any]:
    """Return empty default values for the registry schema."""
    defaults: dict[str, Any] = {column: None for column in RUN_REGISTRY_COLUMNS}
    defaults["fold_count"] = 0
    defaults["feature_count"] = 0
    defaults["status"] = "candidate"
    defaults["notes"] = ""
    return defaults


def ensure_registry_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy containing the full stable registry schema."""
    normalized = df.copy()
    defaults = default_registry_row()
    for column, value in defaults.items():
        if column not in normalized.columns:
            normalized[column] = value
    return normalized.reindex(columns=RUN_REGISTRY_COLUMNS)


def metric_sort_ascending(metric_name: str) -> bool:
    """Return whether lower values are better for the requested metric."""
    return metric_name == "log_loss_mean"
