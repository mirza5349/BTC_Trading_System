"""Feature schema definitions, metadata, and validation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

INDEX_COLUMNS: list[str] = ["symbol", "timeframe", "timestamp"]

REQUIRED_MARKET_INPUT_COLUMNS: list[str] = [
    "symbol",
    "timeframe",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

REQUIRED_NEWS_INPUT_COLUMNS: list[str] = [
    "asset",
    "provider",
    "published_at",
    "title",
    "url",
    "source_name",
]

REQUIRED_MARKET_FEATURE_COLUMNS: list[str] = [
    "ret_1",
    "ret_2",
    "ret_4",
    "ret_8",
    "log_ret_1",
    "range_pct",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "volume_change_1",
    "rolling_volatility_4",
    "rolling_volatility_8",
    "rolling_mean_return_4",
    "rolling_mean_return_8",
    "rolling_volume_mean_4",
    "rolling_volume_mean_8",
    "volume_zscore_8",
    "close_vs_sma_4",
    "close_vs_sma_8",
    "high_low_spread",
]

BREAKOUT_MARKET_FEATURE_COLUMNS: list[str] = [
    "rolling_max_close_8",
    "rolling_min_close_8",
    "breakout_distance_high_8",
    "breakout_distance_low_8",
]

CALENDAR_MARKET_FEATURE_COLUMNS: list[str] = [
    "hour_of_day",
    "day_of_week",
]

MARKET_FEATURE_COLUMNS: list[str] = (
    REQUIRED_MARKET_FEATURE_COLUMNS
    + BREAKOUT_MARKET_FEATURE_COLUMNS
    + CALENDAR_MARKET_FEATURE_COLUMNS
)

REQUIRED_NEWS_FEATURE_COLUMNS: list[str] = [
    "news_count_1h",
    "news_count_4h",
    "unique_news_sources_4h",
    "minutes_since_last_news",
    "news_title_length_mean_4h",
]

OPTIONAL_NEWS_FEATURE_COLUMNS: list[str] = [
    "news_count_12h",
    "unique_urls_4h",
    "is_news_burst_1h",
    "is_news_burst_4h",
]

SENTIMENT_NEWS_FEATURE_COLUMNS: list[str] = [
    "news_sentiment_mean_1h",
    "news_sentiment_mean_4h",
    "news_sentiment_sum_1h",
    "news_sentiment_sum_4h",
    "positive_news_count_1h",
    "negative_news_count_1h",
    "neutral_news_count_1h",
    "positive_news_count_4h",
    "negative_news_count_4h",
    "neutral_news_count_4h",
    "news_sentiment_abs_mean_1h",
    "news_sentiment_abs_mean_4h",
    "max_positive_prob_4h",
    "max_negative_prob_4h",
    "negative_news_burst_1h",
    "positive_news_burst_1h",
    "sentiment_confidence_mean_4h",
    "minutes_since_last_negative_news",
    "minutes_since_last_positive_news",
]

NEWS_FEATURE_COLUMNS: list[str] = (
    REQUIRED_NEWS_FEATURE_COLUMNS
    + OPTIONAL_NEWS_FEATURE_COLUMNS
    + SENTIMENT_NEWS_FEATURE_COLUMNS
)

MERGED_FEATURE_COLUMNS: list[str] = INDEX_COLUMNS + MARKET_FEATURE_COLUMNS + NEWS_FEATURE_COLUMNS

ZERO_FILL_NEWS_FEATURE_COLUMNS: list[str] = [
    "news_count_1h",
    "news_count_4h",
    "news_count_12h",
    "unique_news_sources_4h",
    "unique_urls_4h",
    "is_news_burst_1h",
    "is_news_burst_4h",
    "news_sentiment_mean_1h",
    "news_sentiment_mean_4h",
    "news_sentiment_sum_1h",
    "news_sentiment_sum_4h",
    "positive_news_count_1h",
    "negative_news_count_1h",
    "neutral_news_count_1h",
    "positive_news_count_4h",
    "negative_news_count_4h",
    "neutral_news_count_4h",
    "news_sentiment_abs_mean_1h",
    "news_sentiment_abs_mean_4h",
    "max_positive_prob_4h",
    "max_negative_prob_4h",
    "negative_news_burst_1h",
    "positive_news_burst_1h",
    "sentiment_confidence_mean_4h",
]

FeatureDatasetName = Literal["market_features", "news_features", "feature_table"]


@dataclass(frozen=True)
class FeatureSchema:
    """Metadata describing a feature dataset layout."""

    name: FeatureDatasetName
    required_input_columns: list[str]
    feature_columns: list[str]
    index_columns: list[str] = field(default_factory=lambda: INDEX_COLUMNS.copy())
    unique_key: list[str] = field(default_factory=lambda: INDEX_COLUMNS.copy())


MARKET_FEATURE_SCHEMA = FeatureSchema(
    name="market_features",
    required_input_columns=REQUIRED_MARKET_INPUT_COLUMNS,
    feature_columns=MARKET_FEATURE_COLUMNS,
)

NEWS_FEATURE_SCHEMA = FeatureSchema(
    name="news_features",
    required_input_columns=INDEX_COLUMNS + REQUIRED_NEWS_INPUT_COLUMNS,
    feature_columns=NEWS_FEATURE_COLUMNS,
)

MERGED_FEATURE_SCHEMA = FeatureSchema(
    name="feature_table",
    required_input_columns=INDEX_COLUMNS,
    feature_columns=MARKET_FEATURE_COLUMNS + NEWS_FEATURE_COLUMNS,
)


@dataclass(frozen=True)
class FeatureValidationResult:
    """Result of validating a feature DataFrame."""

    is_valid: bool
    row_count: int
    column_count: int
    missing_index_columns: list[str]
    missing_feature_columns: list[str]
    duplicate_key_count: int
    nan_summary: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def get_expected_market_feature_columns(
    *,
    include_breakout_features: bool = True,
    include_calendar_features: bool = True,
) -> list[str]:
    """Return market feature columns in stable output order."""
    columns = REQUIRED_MARKET_FEATURE_COLUMNS.copy()
    if include_breakout_features:
        columns.extend(BREAKOUT_MARKET_FEATURE_COLUMNS)
    if include_calendar_features:
        columns.extend(CALENDAR_MARKET_FEATURE_COLUMNS)
    return columns


def get_expected_news_feature_columns(
    *,
    lookback_windows: Sequence[str] | None = None,
    include_optional_features: bool = True,
    include_sentiment_features: bool = False,
) -> list[str]:
    """Return news feature columns in stable output order."""
    windows = set(lookback_windows or ["1h", "4h"])
    columns = REQUIRED_NEWS_FEATURE_COLUMNS.copy()

    if "12h" in windows:
        columns.append("news_count_12h")

    if include_optional_features:
        columns.extend(
            [
                "unique_urls_4h",
                "is_news_burst_1h",
                "is_news_burst_4h",
            ]
        )

    if include_sentiment_features:
        columns.extend(
            [
                "news_sentiment_mean_1h",
                "news_sentiment_mean_4h",
                "news_sentiment_sum_1h",
                "news_sentiment_sum_4h",
                "positive_news_count_1h",
                "negative_news_count_1h",
                "neutral_news_count_1h",
                "positive_news_count_4h",
                "negative_news_count_4h",
                "neutral_news_count_4h",
                "news_sentiment_abs_mean_1h",
                "news_sentiment_abs_mean_4h",
                "max_positive_prob_4h",
                "max_negative_prob_4h",
                "negative_news_burst_1h",
                "positive_news_burst_1h",
                "sentiment_confidence_mean_4h",
                "minutes_since_last_negative_news",
                "minutes_since_last_positive_news",
            ]
        )

    ordered_columns: list[str] = []
    for column in NEWS_FEATURE_COLUMNS:
        if column in columns:
            ordered_columns.append(column)
    return ordered_columns


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    *,
    dataset_name: str,
) -> None:
    """Raise a ValueError when required columns are missing."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def validate_feature_df(
    df: pd.DataFrame,
    expected_feature_columns: Sequence[str] | None = None,
    *,
    expected_row_count: int | None = None,
) -> FeatureValidationResult:
    """Validate a feature DataFrame for downstream model readiness."""
    errors: list[str] = []

    expected_columns = list(
        expected_feature_columns or (MARKET_FEATURE_COLUMNS + NEWS_FEATURE_COLUMNS)
    )
    missing_index_columns = [column for column in INDEX_COLUMNS if column not in df.columns]
    missing_feature_columns = [column for column in expected_columns if column not in df.columns]

    if missing_index_columns:
        errors.append(f"Missing index columns: {missing_index_columns}")
    if missing_feature_columns:
        errors.append(f"Missing feature columns: {missing_feature_columns}")

    duplicate_key_count = 0
    if all(column in df.columns for column in INDEX_COLUMNS):
        duplicate_key_count = int(df.duplicated(subset=INDEX_COLUMNS).sum())
        if duplicate_key_count > 0:
            errors.append(f"Found {duplicate_key_count} duplicate rows for {INDEX_COLUMNS}")

        if not df["timestamp"].is_monotonic_increasing:
            errors.append("Timestamps are not sorted ascending")

        timestamp_series = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if timestamp_series.isna().any():
            errors.append("Timestamps contain invalid values")

    if expected_row_count is not None and len(df) != expected_row_count:
        errors.append(
            f"Expected {expected_row_count} rows but found {len(df)} rows"
        )

    nan_summary: dict[str, int] = {}
    for column in df.columns:
        if column in INDEX_COLUMNS:
            continue
        null_count = int(df[column].isna().sum())
        if null_count > 0:
            nan_summary[column] = null_count

    return FeatureValidationResult(
        is_valid=not errors,
        row_count=len(df),
        column_count=len(df.columns),
        missing_index_columns=missing_index_columns,
        missing_feature_columns=missing_feature_columns,
        duplicate_key_count=duplicate_key_count,
        nan_summary=nan_summary,
        errors=errors,
    )
