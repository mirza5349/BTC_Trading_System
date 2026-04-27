"""Schema metadata and validation helpers for model training artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from trading_bot.schemas.features import INDEX_COLUMNS
from trading_bot.schemas.datasets import is_supported_target_column

PREDICTION_COLUMNS: list[str] = [
    "symbol",
    "timeframe",
    "timestamp",
    "fold_id",
    "y_true",
    "y_pred",
    "y_proba",
]

FOLD_METRIC_COLUMNS: list[str] = [
    "fold_id",
    "train_start",
    "train_end",
    "validation_start",
    "validation_end",
    "n_train",
    "n_validation",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "log_loss",
    "tp",
    "tn",
    "fp",
    "fn",
]

ModelArtifactName = Literal["fold_model"]


@dataclass(frozen=True)
class WalkForwardFoldSummary:
    """Metadata for one walk-forward fold."""

    fold_id: int
    mode: str
    train_start_idx: int
    train_end_idx: int
    validation_start_idx: int
    validation_end_idx: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    n_train: int
    n_validation: int


@dataclass(frozen=True)
class ModelRunMetadata:
    """Structured metadata for one training run."""

    run_id: str
    asset: str
    symbol: str
    timeframe: str
    dataset_version: str
    label_version: str
    model_version: str
    target_column: str
    feature_columns: list[str]
    requested_device: str
    effective_device: str
    lightgbm_version: str
    random_seed: int
    probability_threshold: float
    walk_forward_mode: str
    min_train_rows: int
    validation_rows: int
    step_rows: int
    fold_count: int
    dataset_row_count: int
    created_at: str
    lightgbm_device_type: str | None = None
    dataset_path: str | None = None
    python_version: str | None = None
    platform: str | None = None
    gpu_available_check_result: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class TrainingDatasetSummary:
    """Basic metadata about a supervised training dataset."""

    row_count: int
    feature_columns: list[str]
    target_columns: list[str]
    min_timestamp: str | None
    max_timestamp: str | None


def is_target_column(column_name: str) -> bool:
    """Return whether a column is a known target column."""
    return is_supported_target_column(column_name)


def list_target_columns(columns: Sequence[str]) -> list[str]:
    """Return target columns in stable source order."""
    return [column for column in columns if is_target_column(column)]


def select_numeric_feature_columns(
    df: pd.DataFrame,
    *,
    target_columns: Sequence[str] | None = None,
    include_patterns: Sequence[str] | None = None,
    exclude_patterns: Sequence[str] | None = None,
) -> list[str]:
    """Return numeric feature columns excluding identifiers and targets."""
    resolved_target_columns = set(target_columns or list_target_columns(df.columns))
    include_patterns = list(include_patterns or [])
    exclude_patterns = list(exclude_patterns or [])

    numeric_columns = {
        column
        for column in df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    }

    selected = [
        column
        for column in df.columns
        if column in numeric_columns
        and column not in INDEX_COLUMNS
        and column not in resolved_target_columns
    ]

    if include_patterns:
        selected = [
            column
            for column in selected
            if any(pattern in column for pattern in include_patterns)
        ]

    if exclude_patterns:
        selected = [
            column
            for column in selected
            if not any(pattern in column for pattern in exclude_patterns)
        ]

    return selected


def summarize_training_dataset(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
) -> TrainingDatasetSummary:
    """Build lightweight training dataset metadata."""
    min_timestamp: str | None = None
    max_timestamp: str | None = None
    if not df.empty and "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"], utc=True)
        min_timestamp = timestamps.min().isoformat()
        max_timestamp = timestamps.max().isoformat()

    return TrainingDatasetSummary(
        row_count=len(df),
        feature_columns=list(feature_columns),
        target_columns=list(target_columns),
        min_timestamp=min_timestamp,
        max_timestamp=max_timestamp,
    )


def validate_training_dataset(
    df: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: Sequence[str],
) -> None:
    """Validate a supervised dataset before split generation."""
    missing_identifier_columns = [column for column in INDEX_COLUMNS if column not in df.columns]
    if missing_identifier_columns:
        raise ValueError(
            f"Supervised dataset is missing identifier columns: {missing_identifier_columns}"
        )

    if target_column not in df.columns:
        raise ValueError(f"Supervised dataset is missing target column: {target_column}")

    if not feature_columns:
        raise ValueError("No numeric feature columns were selected for training")

    duplicated_rows = int(df.duplicated(subset=INDEX_COLUMNS).sum())
    if duplicated_rows > 0:
        raise ValueError(f"Supervised dataset contains {duplicated_rows} duplicate key rows")

    timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Supervised dataset timestamps must be sorted ascending")

    target_values = pd.to_numeric(df[target_column], errors="coerce")
    if target_values.isna().any():
        raise ValueError(f"Primary target column {target_column} contains missing values")

    invalid_values = sorted(
        {
            float(value)
            for value in target_values.astype(float).unique().tolist()
            if value not in (0.0, 1.0)
        }
    )
    if invalid_values:
        raise ValueError(
            "Primary target column "
            f"{target_column} contains invalid binary values: {invalid_values}"
        )

    allowed_numeric_columns = df.select_dtypes(include=[np.number, "bool"]).columns
    non_numeric_columns = [
        column for column in feature_columns if column not in allowed_numeric_columns
    ]
    if non_numeric_columns:
        raise ValueError(f"Selected feature columns are not numeric: {non_numeric_columns}")


def summarize_target_distribution(series: pd.Series) -> dict[str, float | int]:
    """Return a compact binary target summary for logging and reports."""
    values = pd.to_numeric(series, errors="coerce").dropna().astype(int)
    counts = values.value_counts().sort_index().to_dict()
    total_count = int(values.shape[0])
    positive_count = int(counts.get(1, 0))
    return {
        "count": total_count,
        "zero_count": int(counts.get(0, 0)),
        "one_count": positive_count,
        "positive_rate": round(positive_count / total_count, 6) if total_count else 0.0,
    }
