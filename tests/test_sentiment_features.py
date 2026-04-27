"""Tests for sentiment-aware news feature aggregation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trading_bot.features.feature_pipeline import FeaturePipeline
from trading_bot.features.news_features import compute_news_features
from trading_bot.features.feature_store import ParquetFeatureStore
from trading_bot.storage.enriched_news_store import EnrichedNewsStore
from trading_bot.storage.news_store import NewsStore
from trading_bot.storage.parquet_store import ParquetStore


def _make_market_timeline() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "timeframe": "15m",
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 10:00:00+00:00",
                    "2024-01-01 10:15:00+00:00",
                    "2024-01-01 10:30:00+00:00",
                ],
                utc=True,
            ),
        }
    )


def _make_enriched_news() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asset": ["BTC", "BTC", "BTC"],
            "provider": ["cryptocompare", "cryptocompare", "cryptocompare"],
            "published_at": pd.to_datetime(
                [
                    "2024-01-01 09:40:00+00:00",
                    "2024-01-01 09:55:00+00:00",
                    "2024-01-01 10:20:00+00:00",
                ],
                utc=True,
            ),
            "title": ["a", "b", "c"],
            "url": ["https://a", "https://b", "https://c"],
            "source_name": ["s1", "s2", "s1"],
            "sentiment_label": ["positive", "negative", "positive"],
            "prob_positive": [0.9, 0.1, 0.8],
            "prob_neutral": [0.05, 0.2, 0.1],
            "prob_negative": [0.05, 0.7, 0.1],
            "sentiment_score": [0.85, -0.6, 0.7],
            "prob_confidence": [0.9, 0.7, 0.8],
            "enrichment_version": ["v1", "v1", "v1"],
            "model_name": ["ProsusAI/finbert", "ProsusAI/finbert", "ProsusAI/finbert"],
            "requested_device": ["cpu", "cpu", "cpu"],
            "effective_device": ["cpu", "cpu", "cpu"],
            "enrichment_key": ["1", "2", "3"],
        }
    )


def test_compute_news_features_with_sentiment_is_leak_safe() -> None:
    """Sentiment aggregates should use only events published at or before the market timestamp."""
    result = compute_news_features(
        _make_market_timeline(),
        _make_enriched_news(),
        lookback_windows=["1h", "4h"],
        include_optional_features=True,
        include_sentiment_features=True,
    )

    row_at_1000 = result.iloc[0]
    row_at_1030 = result.iloc[2]

    assert row_at_1000["positive_news_count_1h"] == pytest.approx(1.0)
    assert row_at_1000["negative_news_count_1h"] == pytest.approx(1.0)
    assert row_at_1000["news_sentiment_sum_1h"] == pytest.approx(0.25)
    assert row_at_1000["max_positive_prob_4h"] == pytest.approx(0.9)

    assert row_at_1030["positive_news_count_1h"] == pytest.approx(2.0)
    assert row_at_1030["negative_news_count_1h"] == pytest.approx(1.0)
    assert row_at_1030["news_sentiment_sum_1h"] == pytest.approx(0.95)
    assert row_at_1030["minutes_since_last_positive_news"] == pytest.approx(10.0)


def test_feature_pipeline_uses_enriched_news_for_sentiment_version(tmp_path: Path) -> None:
    """The feature pipeline should load enriched news when building the sentiment feature version."""
    market_store = ParquetStore(tmp_path / "market")
    news_store = NewsStore(tmp_path / "news")
    enriched_news_store = EnrichedNewsStore(tmp_path / "enriched")
    feature_store = ParquetFeatureStore(tmp_path / "features")

    market_df = pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "timeframe": "15m",
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC"),
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.0 + idx for idx in range(10)],
            "volume": [10.0 + idx for idx in range(10)],
        }
    )
    market_store.write(market_df, "BTCUSDT", "15m")
    news_store.write(_make_enriched_news().loc[:, ["asset", "provider", "published_at", "title", "url", "source_name"]], "BTC", "cryptocompare")
    enriched_news_store.write(
        _make_enriched_news(),
        asset="BTC",
        provider="cryptocompare",
        model_name="ProsusAI/finbert",
        enrichment_version="v1",
    )

    pipeline = FeaturePipeline(
        market_store=market_store,
        news_store=news_store,
        enriched_news_store=enriched_news_store,
        feature_store=feature_store,
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        provider="cryptocompare",
        version="v1",
        market_windows=[1, 2, 4, 8],
        news_lookback_windows=["1h", "4h"],
        fill_missing_news_with_zero=True,
        include_breakout_features=True,
        include_calendar_features=True,
        include_optional_news_features=True,
        news_burst_threshold_1h=3,
        news_burst_threshold_4h=8,
        sentiment_feature_version="v2_sentiment",
        sentiment_feature_lookback_windows=["1h", "4h"],
        use_enriched_news_store=True,
        finbert_model_name="ProsusAI/finbert",
        enrichment_version="v1",
        positive_burst_threshold_1h=2,
        negative_burst_threshold_1h=2,
    )

    feature_df, _ = pipeline.build_news_features(version="v2_sentiment")

    assert "news_sentiment_mean_1h" in feature_df.columns
    assert "positive_news_count_4h" in feature_df.columns
