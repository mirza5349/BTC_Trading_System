"""Schema metadata and validation helpers for labels and supervised datasets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import re
from typing import Any, Literal

import numpy as np
import pandas as pd

from trading_bot.schemas.features import INDEX_COLUMNS

REQUIRED_LABEL_INPUT_COLUMNS: list[str] = [
    "symbol",
    "timeframe",
    "timestamp",
    "close",
]

REQUIRED_TARGET_COLUMNS: list[str] = [
    "target_up_2bars",
]

OPTIONAL_TARGET_COLUMNS: list[str] = [
    "future_return_2bars",
    "target_up_4bars",
    "future_log_return_2bars",
]

ALL_TARGET_COLUMNS: list[str] = REQUIRED_TARGET_COLUMNS + OPTIONAL_TARGET_COLUMNS

LABEL_ARTIFACT_COLUMNS: list[str] = INDEX_COLUMNS + ALL_TARGET_COLUMNS

DatasetArtifactName = Literal["labels", "supervised_dataset"]

CUSTOM_BINARY_TARGET_PREFIXES: tuple[str, ...] = (
    "target_up_",
    "target_long_net_positive_",
    "target_return_gt_",
)
CUSTOM_REGRESSION_TARGET_PREFIXES: tuple[str, ...] = (
    "future_return_",
    "future_log_return_",
)


@dataclass(frozen=True)
class SupervisedDatasetSchema:
    """Metadata describing a supervised dataset layout."""

    identifier_columns: list[str]
    feature_columns: list[str]
    target_columns: list[str]
    primary_target: str
    label_version: str
    dataset_version: str


@dataclass(frozen=True)
class DatasetValidationResult:
    """Result of validating a label table or supervised dataset."""

    is_valid: bool
    row_count: int
    column_count: int
    missing_identifier_columns: list[str]
    missing_target_columns: list[str]
    duplicate_key_count: int
    invalid_binary_target_columns: dict[str, list[float]] = field(default_factory=dict)
    missing_value_summary: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def binary_target_column_name(horizon_bars: int) -> str:
    """Return the canonical binary target column name for a horizon."""
    return f"target_up_{horizon_bars}bars"


def future_return_column_name(horizon_bars: int) -> str:
    """Return the canonical future return target column name for a horizon."""
    return f"future_return_{horizon_bars}bars"


def future_log_return_column_name(horizon_bars: int) -> str:
    """Return the canonical future log return column name for a horizon."""
    return f"future_log_return_{horizon_bars}bars"


def infer_horizon_bars_from_target_column(target_column: str) -> int | None:
    """Extract the horizon in bars from a canonical target column name."""
    match = re.fullmatch(
        r"(?:target_up|future_return|future_log_return|target_long_net_positive)_(\d+)bars",
        str(target_column),
    )
    if match is None:
        match = re.fullmatch(r"target_return_gt_[0-9]+pct_(\d+)bars", str(target_column))
    if match is None:
        return None
    return int(match.group(1))


def is_binary_target_column(column_name: str) -> bool:
    """Return whether a column is one of the supported binary target columns."""
    return str(column_name).startswith(CUSTOM_BINARY_TARGET_PREFIXES)


def is_regression_target_column(column_name: str) -> bool:
    """Return whether a column is one of the supported regression target columns."""
    return str(column_name).startswith(CUSTOM_REGRESSION_TARGET_PREFIXES)


def is_supported_target_column(column_name: str) -> bool:
    """Return whether a column belongs to the supported supervised target set."""
    return is_binary_target_column(column_name) or is_regression_target_column(column_name)


def get_expected_target_columns(
    *,
    primary_horizon_bars: int = 2,
    include_regression_target: bool = True,
    include_optional_targets: bool = True,
    optional_horizon_bars: Sequence[int] | None = None,
) -> list[str]:
    """Return target columns in stable output order."""
    target_columns = [binary_target_column_name(primary_horizon_bars)]

    if include_regression_target:
        target_columns.append(future_return_column_name(primary_horizon_bars))

    if include_optional_targets:
        optional_horizons = list(optional_horizon_bars or [4])
        for horizon_bars in optional_horizons:
            if horizon_bars == primary_horizon_bars:
                continue
            target_columns.append(binary_target_column_name(horizon_bars))

        if include_regression_target:
            target_columns.append(future_log_return_column_name(primary_horizon_bars))

    ordered_columns: list[str] = []
    for column in LABEL_ARTIFACT_COLUMNS:
        if column in target_columns:
            ordered_columns.append(column)

    for column in target_columns:
        if column not in ordered_columns:
            ordered_columns.append(column)

    return ordered_columns


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    *,
    dataset_name: str,
) -> None:
    """Raise a ValueError when required columns are missing."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{dataset_name} is missing required columns: {missing_columns}")


def validate_dataset_df(
    df: pd.DataFrame,
    *,
    target_columns: Sequence[str],
    expected_row_count: int | None = None,
) -> DatasetValidationResult:
    """Validate a label table or supervised dataset before persistence."""
    errors: list[str] = []

    missing_identifier_columns = [column for column in INDEX_COLUMNS if column not in df.columns]
    missing_target_columns = [column for column in target_columns if column not in df.columns]

    if missing_identifier_columns:
        errors.append(f"Missing identifier columns: {missing_identifier_columns}")
    if missing_target_columns:
        errors.append(f"Missing target columns: {missing_target_columns}")

    duplicate_key_count = 0
    if all(column in df.columns for column in INDEX_COLUMNS):
        duplicate_key_count = int(df.duplicated(subset=INDEX_COLUMNS).sum())
        if duplicate_key_count > 0:
            errors.append(f"Found {duplicate_key_count} duplicate rows for {INDEX_COLUMNS}")

        timestamp_series = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if timestamp_series.isna().any():
            errors.append("Timestamps contain invalid values")
        elif not timestamp_series.is_monotonic_increasing:
            errors.append("Timestamps are not sorted ascending")

    if expected_row_count is not None and len(df) != expected_row_count:
        errors.append(f"Expected {expected_row_count} rows but found {len(df)} rows")

    invalid_binary_target_columns: dict[str, list[float]] = {}
    invalid_regression_target_columns: dict[str, str] = {}
    for column in target_columns:
        if column not in df.columns:
            continue
        if is_binary_target_column(column):
            invalid_values = sorted(
                {
                    float(value)
                    for value in pd.Series(df[column]).dropna().astype(float).unique().tolist()
                    if value not in (0.0, 1.0)
                }
            )
            if invalid_values:
                invalid_binary_target_columns[column] = invalid_values
                errors.append(f"Binary target {column} contains invalid values: {invalid_values}")
        else:
            numeric_series = pd.to_numeric(df[column], errors="coerce")
            non_missing_values = pd.Series(numeric_series).dropna().to_numpy(dtype=float)
            if not np.isfinite(non_missing_values).all():
                invalid_regression_target_columns[column] = "non-finite values detected"
                errors.append(f"Regression target {column} contains non-finite values")

    missing_value_summary: dict[str, int] = {}
    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        if missing_count > 0:
            missing_value_summary[column] = missing_count

    for column in target_columns:
        if column in missing_value_summary:
            errors.append(f"Target column {column} contains missing values")

    return DatasetValidationResult(
        is_valid=not errors,
        row_count=len(df),
        column_count=len(df.columns),
        missing_identifier_columns=missing_identifier_columns,
        missing_target_columns=missing_target_columns,
        duplicate_key_count=duplicate_key_count,
        invalid_binary_target_columns=invalid_binary_target_columns,
        missing_value_summary=missing_value_summary,
        errors=errors,
    )


def summarize_target_columns(
    df: pd.DataFrame,
    target_columns: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Build simple target distribution stats for logging and inspection."""
    summary: dict[str, dict[str, Any]] = {}

    for column in target_columns:
        if column not in df.columns:
            continue

        series = df[column].dropna()
        if is_binary_target_column(column):
            counts = series.astype(int).value_counts().sort_index().to_dict()
            positive_count = int(counts.get(1, 0))
            total_count = int(series.shape[0])
            summary[column] = {
                "type": "binary",
                "count": total_count,
                "zero_count": int(counts.get(0, 0)),
                "one_count": positive_count,
                "positive_rate": round(positive_count / total_count, 6) if total_count else 0.0,
            }
        else:
            numeric_series = series.astype(float)
            summary[column] = {
                "type": "regression",
                "count": int(numeric_series.shape[0]),
                "mean": round(float(numeric_series.mean()), 8),
                "std": round(float(numeric_series.std(ddof=1)), 8)
                if numeric_series.shape[0] > 1
                else 0.0,
                "min": round(float(numeric_series.min()), 8),
                "max": round(float(numeric_series.max()), 8),
            }

    return summary


def summarize_missing_values(df: pd.DataFrame) -> dict[str, int]:
    """Return a compact missing-value summary for non-empty columns only."""
    return {
        column: int(df[column].isna().sum())
        for column in df.columns
        if int(df[column].isna().sum()) > 0
    }
