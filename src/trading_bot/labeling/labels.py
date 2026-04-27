"""Leak-safe future-return label generation from BTC candle history."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from trading_bot.logging_config import get_logger
from trading_bot.schemas.datasets import (
    REQUIRED_LABEL_INPUT_COLUMNS,
    binary_target_column_name,
    future_log_return_column_name,
    future_return_column_name,
    get_expected_target_columns,
    is_binary_target_column,
    validate_required_columns,
)
from trading_bot.schemas.features import INDEX_COLUMNS
from trading_bot.schemas.targets import TargetConfig

logger = get_logger(__name__)

class LabelType(Enum):
    """Supported step-5 label types."""

    BINARY_CLASSIFICATION = "binary_classification"
    REGRESSION = "regression"


@dataclass(frozen=True)
class LabelDefinition:
    """Metadata describing a generated target column."""

    name: str
    label_type: LabelType
    horizon_bars: int


@dataclass(frozen=True)
class CustomTargetGenerationResult:
    """Generated custom targets plus metadata summaries for research workflows."""

    labels_df: pd.DataFrame
    target_columns: list[str]
    target_metadata: list[dict[str, object]]


def generate_future_return_labels(
    df: pd.DataFrame,
    *,
    primary_horizon_bars: int = 2,
    horizon_minutes: int | None = None,
    binary_threshold: float = 0.0,
    include_regression_target: bool = True,
    include_optional_targets: bool = True,
    optional_horizon_bars: Sequence[int] | None = None,
    drop_unlabeled_rows: bool = True,
) -> pd.DataFrame:
    """Generate deterministic future-return labels from candle close prices."""
    validate_required_columns(
        df,
        REQUIRED_LABEL_INPUT_COLUMNS,
        dataset_name="market data for label generation",
    )

    if df.empty:
        return df[INDEX_COLUMNS].copy()

    market = df.copy()
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)
    market = market.sort_values(INDEX_COLUMNS).reset_index(drop=True)

    resolved_primary_horizon = _resolve_horizon_bars(
        df=market,
        horizon_bars=primary_horizon_bars,
        horizon_minutes=horizon_minutes,
    )
    resolved_optional_horizons = _resolve_optional_horizons(
        primary_horizon_bars=resolved_primary_horizon,
        optional_horizon_bars=optional_horizon_bars,
        include_optional_targets=include_optional_targets,
    )

    logger.info(
        "labels_start",
        rows=len(market),
        primary_horizon_bars=resolved_primary_horizon,
        optional_horizon_bars=resolved_optional_horizons,
        include_regression_target=include_regression_target,
        include_optional_targets=include_optional_targets,
    )

    result = market[INDEX_COLUMNS].copy()
    grouped_close = market.groupby(["symbol", "timeframe"], sort=False)["close"]
    current_close = market["close"].astype(float)

    primary_future_close = grouped_close.shift(-resolved_primary_horizon)
    primary_future_return = (primary_future_close / current_close) - 1.0
    primary_target_column = binary_target_column_name(resolved_primary_horizon)

    result[primary_target_column] = np.where(
        primary_future_return > binary_threshold,
        1.0,
        0.0,
    )
    result.loc[primary_future_return.isna(), primary_target_column] = np.nan

    if include_regression_target:
        result[future_return_column_name(resolved_primary_horizon)] = primary_future_return

    if include_optional_targets and include_regression_target:
        result[future_log_return_column_name(resolved_primary_horizon)] = np.log(
            primary_future_close / current_close
        )

    for horizon_bars in resolved_optional_horizons:
        future_close = grouped_close.shift(-horizon_bars)
        future_return = (future_close / current_close) - 1.0
        target_column = binary_target_column_name(horizon_bars)
        result[target_column] = np.where(
            future_return > binary_threshold,
            1.0,
            0.0,
        )
        result.loc[future_return.isna(), target_column] = np.nan

    target_columns = get_expected_target_columns(
        primary_horizon_bars=resolved_primary_horizon,
        include_regression_target=include_regression_target,
        include_optional_targets=include_optional_targets,
        optional_horizon_bars=resolved_optional_horizons,
    )
    result = result.reindex(columns=INDEX_COLUMNS + target_columns)

    if drop_unlabeled_rows:
        rows_before_drop = len(result)
        result = result.dropna(subset=target_columns).reset_index(drop=True)
        dropped_rows = rows_before_drop - len(result)
    else:
        dropped_rows = 0

    for column in target_columns:
        if column.startswith("target_up_") and column in result.columns:
            result[column] = result[column].astype(int)

    logger.info(
        "labels_complete",
        rows=len(result),
        target_columns=target_columns,
        dropped_rows=dropped_rows,
        start_timestamp=result["timestamp"].min().isoformat() if not result.empty else None,
        end_timestamp=result["timestamp"].max().isoformat() if not result.empty else None,
    )

    return result


def generate_custom_target_labels(
    df: pd.DataFrame,
    *,
    target_configs: Sequence[TargetConfig],
    drop_unlabeled_rows: bool = True,
) -> CustomTargetGenerationResult:
    """Generate fee-aware and minimum-return binary labels from future returns."""
    validate_required_columns(
        df,
        REQUIRED_LABEL_INPUT_COLUMNS,
        dataset_name="market data for custom label generation",
    )
    if df.empty:
        return CustomTargetGenerationResult(
            labels_df=df[INDEX_COLUMNS].copy(),
            target_columns=[],
            target_metadata=[],
        )

    market = df.copy()
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)
    market = market.sort_values(INDEX_COLUMNS).reset_index(drop=True)
    grouped_close = market.groupby(["symbol", "timeframe"], sort=False)["close"]
    current_close = market["close"].astype(float)

    unique_configs: list[TargetConfig] = []
    seen_target_names: set[str] = set()
    for config in target_configs:
        if config.target_name in seen_target_names:
            continue
        seen_target_names.add(config.target_name)
        unique_configs.append(config)

    future_return_cache: dict[int, pd.Series] = {}
    result = market[INDEX_COLUMNS].copy()
    target_columns: list[str] = []
    target_metadata: list[dict[str, object]] = []

    logger.info("custom_labels_start", rows=len(market), target_count=len(unique_configs))

    for config in unique_configs:
        future_return = future_return_cache.get(config.horizon_bars)
        if future_return is None:
            future_close = grouped_close.shift(-int(config.horizon_bars))
            future_return = (future_close / current_close) - 1.0
            future_return_cache[config.horizon_bars] = future_return
            result[future_return_column_name(config.horizon_bars)] = future_return

        threshold = _threshold_for_target_config(config)
        result[config.target_name] = np.where(future_return > threshold, 1.0, 0.0)
        result.loc[future_return.isna(), config.target_name] = np.nan
        target_columns.extend([config.target_name, future_return_column_name(config.horizon_bars)])

        non_missing = pd.to_numeric(result[config.target_name], errors="coerce").dropna().astype(int)
        positive_rate = float(non_missing.mean()) if not non_missing.empty else 0.0
        metadata = {
            "target_key": config.key,
            "target_name": config.target_name,
            "horizon_bars": int(config.horizon_bars),
            "threshold_type": config.threshold_type,
            "threshold_return": config.threshold_return,
            "fee_rate": config.fee_rate,
            "slippage_rate": config.slippage_rate,
            "round_trip_cost": config.round_trip_cost,
            "positive_class_rate": round(positive_rate, 6),
            "positive_count": int(non_missing.sum()) if not non_missing.empty else 0,
            "row_count": int(non_missing.shape[0]),
        }
        target_metadata.append(metadata)
        logger.info("custom_label_generated", **metadata)

    stable_target_columns: list[str] = []
    for column in target_columns:
        if column not in stable_target_columns:
            stable_target_columns.append(column)
    result = result.reindex(columns=INDEX_COLUMNS + stable_target_columns)

    if drop_unlabeled_rows and stable_target_columns:
        rows_before_drop = len(result)
        result = result.dropna(subset=stable_target_columns).reset_index(drop=True)
        dropped_rows = rows_before_drop - len(result)
    else:
        dropped_rows = 0

    for column in stable_target_columns:
        if is_binary_target_column(column) and column in result.columns:
            result[column] = result[column].astype(int)

    logger.info(
        "custom_labels_complete",
        rows=len(result),
        target_columns=stable_target_columns,
        dropped_rows=dropped_rows,
    )
    return CustomTargetGenerationResult(
        labels_df=result,
        target_columns=stable_target_columns,
        target_metadata=target_metadata,
    )


def get_label_definitions(
    *,
    primary_horizon_bars: int = 2,
    include_regression_target: bool = True,
    include_optional_targets: bool = True,
    optional_horizon_bars: Sequence[int] | None = None,
) -> list[LabelDefinition]:
    """Return label metadata in stable output order."""
    optional_horizons = _resolve_optional_horizons(
        primary_horizon_bars=primary_horizon_bars,
        optional_horizon_bars=optional_horizon_bars,
        include_optional_targets=include_optional_targets,
    )

    definitions = [
        LabelDefinition(
            name=binary_target_column_name(primary_horizon_bars),
            label_type=LabelType.BINARY_CLASSIFICATION,
            horizon_bars=primary_horizon_bars,
        )
    ]

    if include_regression_target:
        definitions.append(
            LabelDefinition(
                name=future_return_column_name(primary_horizon_bars),
                label_type=LabelType.REGRESSION,
                horizon_bars=primary_horizon_bars,
            )
        )

    for horizon_bars in optional_horizons:
        definitions.append(
            LabelDefinition(
                name=binary_target_column_name(horizon_bars),
                label_type=LabelType.BINARY_CLASSIFICATION,
                horizon_bars=horizon_bars,
            )
        )

    if include_optional_targets and include_regression_target:
        definitions.append(
            LabelDefinition(
                name=future_log_return_column_name(primary_horizon_bars),
                label_type=LabelType.REGRESSION,
                horizon_bars=primary_horizon_bars,
            )
        )

    return definitions


def _resolve_optional_horizons(
    *,
    primary_horizon_bars: int,
    optional_horizon_bars: Sequence[int] | None,
    include_optional_targets: bool,
) -> list[int]:
    """Return stable optional horizons excluding the primary horizon."""
    if not include_optional_targets:
        return []

    optional_horizons = []
    for horizon_bars in optional_horizon_bars or [4]:
        if horizon_bars <= 0 or horizon_bars == primary_horizon_bars:
            continue
        if horizon_bars not in optional_horizons:
            optional_horizons.append(horizon_bars)
    return optional_horizons


def _threshold_for_target_config(config: TargetConfig) -> float:
    """Resolve the binary threshold for one custom target config."""
    if config.threshold_type == "fee_aware":
        if config.round_trip_cost is not None:
            return float(config.round_trip_cost)
        return float(2.0 * ((config.fee_rate or 0.0) + (config.slippage_rate or 0.0)))
    if config.threshold_type == "minimum_return":
        return float(config.threshold_return or 0.0)
    raise ValueError(f"Unsupported custom target threshold_type={config.threshold_type!r}")


def _resolve_horizon_bars(
    *,
    df: pd.DataFrame,
    horizon_bars: int,
    horizon_minutes: int | None,
) -> int:
    """Resolve a label horizon in bars, optionally converting from minutes."""
    if horizon_minutes is None:
        if horizon_bars <= 0:
            raise ValueError("horizon_bars must be greater than zero")
        return horizon_bars

    timeframe = str(df["timeframe"].iloc[0])
    timeframe_delta = pd.Timedelta(timeframe)
    horizon_delta = pd.Timedelta(minutes=horizon_minutes)

    if horizon_delta <= pd.Timedelta(0):
        raise ValueError("horizon_minutes must be greater than zero")
    if horizon_delta % timeframe_delta != pd.Timedelta(0):
        raise ValueError(
            f"horizon_minutes={horizon_minutes} does not divide evenly into timeframe {timeframe}"
        )

    resolved_bars = int(horizon_delta / timeframe_delta)
    if resolved_bars <= 0:
        raise ValueError("Resolved horizon bars must be greater than zero")
    return resolved_bars
