"""Tests for news feature engineering."""

from __future__ import annotations

import pandas as pd
import pytest

from trading_bot.features.news_features import compute_news_features


def _make_market_timeline() -> pd.DataFrame:
    """Create a small 15-minute BTC market timeline."""
    timestamps = pd.to_datetime(
        [
            "2024-01-01 10:00:00+00:00",
            "2024-01-01 10:15:00+00:00",
            "2024-01-01 10:30:00+00:00",
            "2024-01-01 10:45:00+00:00",
        ],
        utc=True,
    )
    return pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "timeframe": "15m",
            "timestamp": timestamps,
        }
    )


def _make_news_events() -> pd.DataFrame:
    """Create deterministic BTC news events around the market timeline."""
    return pd.DataFrame(
        {
            "asset": "BTC",
            "provider": "cryptocompare",
            "published_at": pd.to_datetime(
                [
                    "2024-01-01 09:35:00+00:00",
                    "2024-01-01 09:55:00+00:00",
                    "2024-01-01 10:20:00+00:00",
                    "2024-01-01 11:00:00+00:00",
                ],
                utc=True,
            ),
            "title": [
                "Alpha",
                "Long title",
                "Gamma news",
                "Future bulletin",
            ],
            "url": [
                "https://example.com/alpha",
                "https://example.com/long-title",
                "https://example.com/gamma-news",
                "https://example.com/future-bulletin",
            ],
            "source_name": ["source_a", "source_b", "source_a", "source_c"],
        }
    )


def test_compute_news_features_aligns_without_future_leakage() -> None:
    """Only articles published at or before the candle timestamp should count."""
    market_timeline = _make_market_timeline()
    news_events = _make_news_events()

    result = compute_news_features(
        market_timeline,
        news_events,
        lookback_windows=["1h", "4h", "12h"],
        include_optional_features=True,
    )

    row_at_1000 = result.iloc[0]
    row_at_1030 = result.iloc[2]

    assert result.shape[0] == market_timeline.shape[0]
    assert result["timestamp"].is_monotonic_increasing

    assert row_at_1000["news_count_1h"] == pytest.approx(2.0)
    assert row_at_1000["news_count_4h"] == pytest.approx(2.0)
    assert row_at_1000["unique_news_sources_4h"] == pytest.approx(2.0)
    assert row_at_1000["minutes_since_last_news"] == pytest.approx(5.0)
    assert row_at_1000["news_title_length_mean_4h"] == pytest.approx((5 + 10) / 2)

    assert row_at_1030["news_count_1h"] == pytest.approx(3.0)
    assert row_at_1030["news_count_4h"] == pytest.approx(3.0)
    assert row_at_1030["news_count_12h"] == pytest.approx(3.0)
    assert row_at_1030["unique_news_sources_4h"] == pytest.approx(2.0)
    assert row_at_1030["unique_urls_4h"] == pytest.approx(3.0)
    assert row_at_1030["minutes_since_last_news"] == pytest.approx(10.0)
    assert row_at_1030["news_title_length_mean_4h"] == pytest.approx((5 + 10 + 10) / 3)
    assert row_at_1030["is_news_burst_1h"] == pytest.approx(1.0)
    assert row_at_1030["is_news_burst_4h"] == pytest.approx(0.0)

    # The article at 11:00 must not affect the 10:45 feature row.
    assert result.iloc[3]["news_count_1h"] == pytest.approx(2.0)


def test_compute_news_features_empty_input_zero_fills_count_columns() -> None:
    """Missing news should yield zero-filled count features and NaN recency stats."""
    market_timeline = _make_market_timeline()
    empty_news = pd.DataFrame()

    result = compute_news_features(
        market_timeline,
        empty_news,
        lookback_windows=["1h", "4h"],
        include_optional_features=True,
    )

    assert result["news_count_1h"].eq(0.0).all()
    assert result["news_count_4h"].eq(0.0).all()
    assert result["unique_news_sources_4h"].eq(0.0).all()
    assert result["unique_urls_4h"].eq(0.0).all()
    assert result["is_news_burst_1h"].eq(0.0).all()
    assert result["is_news_burst_4h"].eq(0.0).all()
    assert result["minutes_since_last_news"].isna().all()
    assert result["news_title_length_mean_4h"].isna().all()
