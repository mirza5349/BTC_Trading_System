"""Integration tests for the feature engineering pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trading_bot.features.feature_pipeline import FeaturePipeline, FeaturePipelineError
from trading_bot.features.feature_store import ParquetFeatureStore
from trading_bot.storage.enriched_news_store import EnrichedNewsStore
from trading_bot.storage.news_store import NewsStore
from trading_bot.storage.parquet_store import ParquetStore


def _make_market_df(periods: int = 10) -> pd.DataFrame:
    """Create deterministic BTC market candles for pipeline tests."""
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="15min", tz="UTC")
    close = pd.Series([100 * (1.1 ** index) for index in range(periods)], dtype=float)
    return pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "timeframe": "15m",
            "timestamp": timestamps,
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": pd.Series(range(10, 10 * (periods + 1), 10), dtype=float),
        }
    )


def _make_news_df() -> pd.DataFrame:
    """Create a few BTC news events for feature alignment tests."""
    return pd.DataFrame(
        {
            "asset": "BTC",
            "provider": "cryptocompare",
            "published_at": pd.to_datetime(
                [
                    "2024-01-01 00:10:00+00:00",
                    "2024-01-01 00:50:00+00:00",
                    "2024-01-01 01:20:00+00:00",
                ],
                utc=True,
            ),
            "title": [
                "First BTC item",
                "Second BTC item",
                "Third BTC item",
            ],
            "url": [
                "https://example.com/first",
                "https://example.com/second",
                "https://example.com/third",
            ],
            "source_name": [
                "source_a",
                "source_b",
                "source_a",
            ],
        }
    )


def _make_pipeline(
    tmp_path: Path,
) -> tuple[FeaturePipeline, ParquetStore, NewsStore, EnrichedNewsStore, ParquetFeatureStore]:
    """Build a feature pipeline backed by temporary local stores."""
    market_store = ParquetStore(tmp_path / "market")
    news_store = NewsStore(tmp_path / "news")
    enriched_news_store = EnrichedNewsStore(tmp_path / "enriched_news")
    feature_store = ParquetFeatureStore(tmp_path / "features")
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
    return pipeline, market_store, news_store, enriched_news_store, feature_store


def test_build_feature_table_persists_merged_dataset(tmp_path: Path) -> None:
    """The merged feature table should align news to the market timeline and persist locally."""
    pipeline, market_store, news_store, _, feature_store = _make_pipeline(tmp_path)
    market_store.write(_make_market_df(), "BTCUSDT", "15m")
    news_store.write(_make_news_df(), "BTC", "cryptocompare")

    feature_df, feature_set = pipeline.build_feature_table()

    assert len(feature_df) == 10
    assert feature_df["timestamp"].is_monotonic_increasing
    assert "ret_1" in feature_df.columns
    assert "news_count_1h" in feature_df.columns

    expected_path = (
        tmp_path
        / "features"
        / "asset=BTC"
        / "symbol=BTCUSDT"
        / "timeframe=15m"
        / "version=v1"
        / "features.parquet"
    )
    assert Path(feature_set.path) == expected_path
    assert expected_path.exists()
    assert feature_store.exists(
        "feature_table",
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        version="v1",
    )

    first_row = feature_df.iloc[0]
    row_at_0100 = feature_df.loc[
        feature_df["timestamp"] == pd.Timestamp("2024-01-01 01:00:00+00:00")
    ].iloc[0]

    assert first_row["news_count_1h"] == pytest.approx(0.0)
    assert first_row["unique_news_sources_4h"] == pytest.approx(0.0)
    assert pd.isna(first_row["minutes_since_last_news"])

    assert row_at_0100["news_count_1h"] == pytest.approx(2.0)
    assert row_at_0100["news_count_4h"] == pytest.approx(2.0)
    assert row_at_0100["unique_news_sources_4h"] == pytest.approx(2.0)
    assert row_at_0100["minutes_since_last_news"] == pytest.approx(10.0)

    info = pipeline.inspect_dataset("feature_table")
    assert info["row_count"] == 10
    assert info["path"] == str(expected_path)


def test_build_news_features_handles_missing_news_input(tmp_path: Path) -> None:
    """A missing news store should still produce aligned zero-filled news features."""
    pipeline, market_store, _, _, feature_store = _make_pipeline(tmp_path)
    market_store.write(_make_market_df(), "BTCUSDT", "15m")

    news_features, feature_set = pipeline.build_news_features()

    assert len(news_features) == 10
    assert news_features["news_count_1h"].eq(0.0).all()
    assert news_features["news_count_4h"].eq(0.0).all()
    assert news_features["unique_news_sources_4h"].eq(0.0).all()
    assert news_features["minutes_since_last_news"].isna().all()
    assert feature_store.exists(
        "news_features",
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        version="v1",
    )
    assert feature_set.row_count == 10


def test_build_feature_table_requires_market_data(tmp_path: Path) -> None:
    """The pipeline should fail fast when the market master timeline is missing."""
    pipeline, _, _, _, _ = _make_pipeline(tmp_path)

    with pytest.raises(FeaturePipelineError, match="No stored market data found"):
        pipeline.build_feature_table()
