"""Leak-safe news event aggregation aligned to market timestamps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from trading_bot.logging_config import get_logger
from trading_bot.schemas.features import (
    INDEX_COLUMNS,
    NEWS_FEATURE_COLUMNS,
    ZERO_FILL_NEWS_FEATURE_COLUMNS,
    get_expected_news_feature_columns,
    validate_required_columns,
)

logger = get_logger(__name__)

_DEFAULT_LOOKBACK_WINDOWS: tuple[str, ...] = ("1h", "4h")


def compute_news_features(
    market_df: pd.DataFrame,
    news_df: pd.DataFrame,
    *,
    lookback_windows: Sequence[str] | None = None,
    fill_missing_with_zero: bool = True,
    include_optional_features: bool = True,
    include_sentiment_features: bool = False,
    burst_thresholds: Mapping[str, int] | None = None,
    positive_burst_threshold_1h: int = 2,
    negative_burst_threshold_1h: int = 2,
) -> pd.DataFrame:
    """Aggregate historical news into one feature row per market timestamp."""
    validate_required_columns(
        market_df,
        INDEX_COLUMNS,
        dataset_name="market timeline",
    )

    if market_df.empty:
        return market_df[INDEX_COLUMNS].copy()

    windows = _resolve_lookback_windows(lookback_windows)
    thresholds = {
        "1h": 3,
        "4h": 8,
        **dict(burst_thresholds or {}),
    }

    market = market_df[INDEX_COLUMNS].copy()
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)
    market = market.sort_values(INDEX_COLUMNS).reset_index(drop=True)

    logger.info(
        "news_features_start",
        market_rows=len(market),
        news_rows=len(news_df),
        lookback_windows=windows,
        include_optional_features=include_optional_features,
        include_sentiment_features=include_sentiment_features,
    )

    if news_df.empty:
        return _empty_news_features(
            market,
            lookback_windows=windows,
            fill_missing_with_zero=fill_missing_with_zero,
            include_optional_features=include_optional_features,
            include_sentiment_features=include_sentiment_features,
        )

    validate_required_columns(
        news_df,
        ["asset", "provider", "published_at", "title", "url", "source_name"],
        dataset_name="news events",
    )

    news = news_df.copy()
    news["published_at"] = pd.to_datetime(news["published_at"], utc=True)
    news = news.sort_values("published_at").reset_index(drop=True)
    news["_title_length"] = news["title"].astype(str).str.len().astype(float)

    published = news["published_at"].to_numpy(dtype="datetime64[ns]")
    market_timestamps = market["timestamp"].to_numpy(dtype="datetime64[ns]")
    right_edges = np.searchsorted(published, market_timestamps, side="right")
    source_names = news["source_name"].fillna("").astype(str).to_numpy()
    urls = news["url"].fillna("").astype(str).to_numpy()
    title_lengths = news["_title_length"].to_numpy(dtype=float)

    result = market.copy()

    for window_name, delta in windows.items():
        left_edges = np.searchsorted(
            published,
            market_timestamps - delta.to_timedelta64(),
            side="left",
        )
        counts = right_edges - left_edges
        result[f"news_count_{window_name}"] = counts.astype(float)

        if window_name == "4h":
            unique_sources = np.zeros(len(result), dtype=float)
            unique_urls = np.zeros(len(result), dtype=float)
            title_means = np.full(len(result), np.nan)

            for index, (left_edge, right_edge) in enumerate(
                zip(left_edges, right_edges, strict=False)
            ):
                if right_edge <= left_edge:
                    continue
                unique_sources[index] = len(set(source_names[left_edge:right_edge]))
                unique_urls[index] = len(set(urls[left_edge:right_edge]))
                title_means[index] = float(np.mean(title_lengths[left_edge:right_edge]))

            result["unique_news_sources_4h"] = unique_sources
            result["news_title_length_mean_4h"] = title_means

            if include_optional_features:
                result["unique_urls_4h"] = unique_urls
                result["is_news_burst_4h"] = (counts >= thresholds["4h"]).astype(float)

        if include_optional_features and window_name == "1h":
            result["is_news_burst_1h"] = (counts >= thresholds["1h"]).astype(float)

        if include_sentiment_features:
            _append_sentiment_window_features(
                result=result,
                news=news,
                window_name=window_name,
                left_edges=left_edges,
                right_edges=right_edges,
                positive_burst_threshold_1h=positive_burst_threshold_1h,
                negative_burst_threshold_1h=negative_burst_threshold_1h,
            )

    last_news_index = right_edges - 1
    minutes_since_last_news = np.full(len(result), np.nan)
    valid_mask = last_news_index >= 0
    if valid_mask.any():
        last_news_timestamps = news.loc[last_news_index[valid_mask], "published_at"].to_numpy(
            dtype="datetime64[ns]"
        )
        delta_minutes = (
            market_timestamps[valid_mask] - last_news_timestamps
        ) / np.timedelta64(1, "m")
        minutes_since_last_news[valid_mask] = delta_minutes.astype(float)
    result["minutes_since_last_news"] = minutes_since_last_news

    if include_sentiment_features:
        result["minutes_since_last_positive_news"] = _minutes_since_last_label(
            news,
            market_timestamps=market_timestamps,
            label="positive",
        )
        result["minutes_since_last_negative_news"] = _minutes_since_last_label(
            news,
            market_timestamps=market_timestamps,
            label="negative",
        )

    expected_columns = INDEX_COLUMNS + get_expected_news_feature_columns(
        lookback_windows=windows.keys(),
        include_optional_features=include_optional_features,
        include_sentiment_features=include_sentiment_features,
    )
    result = result.reindex(columns=expected_columns)

    if fill_missing_with_zero:
        for column in ZERO_FILL_NEWS_FEATURE_COLUMNS:
            if column in result.columns:
                result[column] = result[column].fillna(0.0)

    logger.info(
        "news_features_complete",
        rows=len(result),
        feature_columns=len(result.columns) - len(INDEX_COLUMNS),
        start_timestamp=result["timestamp"].min().isoformat(),
        end_timestamp=result["timestamp"].max().isoformat(),
    )

    return result


def _append_sentiment_window_features(
    *,
    result: pd.DataFrame,
    news: pd.DataFrame,
    window_name: str,
    left_edges: np.ndarray,
    right_edges: np.ndarray,
    positive_burst_threshold_1h: int,
    negative_burst_threshold_1h: int,
) -> None:
    """Append one window's sentiment-aware aggregates to the result frame."""
    scores = pd.to_numeric(
        news.get("sentiment_score", pd.Series(np.nan, index=news.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    labels = news.get(
        "sentiment_label",
        pd.Series("", index=news.index, dtype=object),
    ).fillna("").astype(str).to_numpy()
    prob_positive = pd.to_numeric(
        news.get("prob_positive", pd.Series(np.nan, index=news.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    prob_negative = pd.to_numeric(
        news.get("prob_negative", pd.Series(np.nan, index=news.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    confidence = pd.to_numeric(
        news.get("prob_confidence", pd.Series(np.nan, index=news.index)),
        errors="coerce",
    ).to_numpy(dtype=float)

    sentiment_mean = np.full(len(result), np.nan)
    sentiment_sum = np.zeros(len(result), dtype=float)
    positive_counts = np.zeros(len(result), dtype=float)
    negative_counts = np.zeros(len(result), dtype=float)
    neutral_counts = np.zeros(len(result), dtype=float)
    sentiment_abs_mean = np.full(len(result), np.nan)

    max_positive_prob = np.full(len(result), np.nan)
    max_negative_prob = np.full(len(result), np.nan)
    confidence_mean = np.full(len(result), np.nan)

    for index, (left_edge, right_edge) in enumerate(zip(left_edges, right_edges, strict=False)):
        if right_edge <= left_edge:
            continue

        window_scores = scores[left_edge:right_edge]
        window_labels = labels[left_edge:right_edge]
        window_positive_prob = prob_positive[left_edge:right_edge]
        window_negative_prob = prob_negative[left_edge:right_edge]
        window_confidence = confidence[left_edge:right_edge]

        finite_scores = window_scores[np.isfinite(window_scores)]
        if finite_scores.size > 0:
            sentiment_mean[index] = float(np.mean(finite_scores))
            sentiment_sum[index] = float(np.sum(finite_scores))
            sentiment_abs_mean[index] = float(np.mean(np.abs(finite_scores)))

        positive_counts[index] = float(np.sum(window_labels == "positive"))
        negative_counts[index] = float(np.sum(window_labels == "negative"))
        neutral_counts[index] = float(np.sum(window_labels == "neutral"))

        finite_positive = window_positive_prob[np.isfinite(window_positive_prob)]
        finite_negative = window_negative_prob[np.isfinite(window_negative_prob)]
        finite_confidence = window_confidence[np.isfinite(window_confidence)]

        if window_name == "4h":
            if finite_positive.size > 0:
                max_positive_prob[index] = float(np.max(finite_positive))
            if finite_negative.size > 0:
                max_negative_prob[index] = float(np.max(finite_negative))
            if finite_confidence.size > 0:
                confidence_mean[index] = float(np.mean(finite_confidence))

    result[f"news_sentiment_mean_{window_name}"] = sentiment_mean
    result[f"news_sentiment_sum_{window_name}"] = sentiment_sum
    result[f"positive_news_count_{window_name}"] = positive_counts
    result[f"negative_news_count_{window_name}"] = negative_counts
    result[f"neutral_news_count_{window_name}"] = neutral_counts
    result[f"news_sentiment_abs_mean_{window_name}"] = sentiment_abs_mean

    if window_name == "4h":
        result["max_positive_prob_4h"] = max_positive_prob
        result["max_negative_prob_4h"] = max_negative_prob
        result["sentiment_confidence_mean_4h"] = confidence_mean
    if window_name == "1h":
        result["negative_news_burst_1h"] = (
            negative_counts >= float(negative_burst_threshold_1h)
        ).astype(float)
        result["positive_news_burst_1h"] = (
            positive_counts >= float(positive_burst_threshold_1h)
        ).astype(float)


def _minutes_since_last_label(
    news: pd.DataFrame,
    *,
    market_timestamps: np.ndarray,
    label: str,
) -> np.ndarray:
    """Return minutes since the last article with the requested sentiment label."""
    label_series = news.get(
        "sentiment_label",
        pd.Series("", index=news.index, dtype=object),
    ).astype(str)
    labeled_news = news[label_series == label].copy()
    if labeled_news.empty:
        return np.full(len(market_timestamps), np.nan)

    published = labeled_news["published_at"].to_numpy(dtype="datetime64[ns]")
    right_edges = np.searchsorted(published, market_timestamps, side="right")
    last_indices = right_edges - 1
    result = np.full(len(market_timestamps), np.nan)
    valid_mask = last_indices >= 0
    if valid_mask.any():
        last_timestamps = published[last_indices[valid_mask]]
        delta_minutes = (market_timestamps[valid_mask] - last_timestamps) / np.timedelta64(1, "m")
        result[valid_mask] = delta_minutes.astype(float)
    return result


def _empty_news_features(
    market: pd.DataFrame,
    *,
    lookback_windows: Mapping[str, pd.Timedelta],
    fill_missing_with_zero: bool,
    include_optional_features: bool,
    include_sentiment_features: bool,
) -> pd.DataFrame:
    """Return a news feature table aligned to market rows with no event input."""
    result = market.copy()
    for column in NEWS_FEATURE_COLUMNS:
        if column == "news_count_12h" and "12h" not in lookback_windows:
            continue
        result[column] = np.nan

    for column in ZERO_FILL_NEWS_FEATURE_COLUMNS:
        if column in result.columns and fill_missing_with_zero:
            result[column] = 0.0

    expected_columns = INDEX_COLUMNS + get_expected_news_feature_columns(
        lookback_windows=lookback_windows.keys(),
        include_optional_features=include_optional_features,
        include_sentiment_features=include_sentiment_features,
    )
    return result.reindex(columns=expected_columns)


def _resolve_lookback_windows(lookback_windows: Sequence[str] | None) -> dict[str, pd.Timedelta]:
    """Parse configured lookback strings into stable timedeltas."""
    requested = list(lookback_windows or _DEFAULT_LOOKBACK_WINDOWS)
    resolved: dict[str, pd.Timedelta] = {}

    for window_name in requested:
        if window_name in resolved:
            continue
        resolved[window_name] = pd.Timedelta(window_name)

    if "1h" not in resolved:
        resolved["1h"] = pd.Timedelta("1h")
    if "4h" not in resolved:
        resolved["4h"] = pd.Timedelta("4h")

    ordered: dict[str, pd.Timedelta] = {}
    for window_name in ("1h", "4h", "12h"):
        if window_name in resolved:
            ordered[window_name] = resolved[window_name]

    for window_name, delta in resolved.items():
        if window_name not in ordered:
            ordered[window_name] = delta

    return ordered
