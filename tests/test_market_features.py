"""Tests for market feature engineering."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from trading_bot.features.market_features import compute_market_features


def _make_market_df(periods: int = 10) -> pd.DataFrame:
    """Create a deterministic BTC candle frame with simple geometry."""
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


def test_compute_market_features_expected_values() -> None:
    """Required BTC market features should be deterministic and leak-safe."""
    market_df = _make_market_df()

    result = compute_market_features(market_df)

    row = result.iloc[8]
    source_row = market_df.iloc[8]

    assert result.shape[0] == market_df.shape[0]
    assert result["timestamp"].is_monotonic_increasing

    assert row["ret_1"] == pytest.approx(0.1)
    assert row["ret_2"] == pytest.approx((1.1**2) - 1.0)
    assert row["ret_4"] == pytest.approx((1.1**4) - 1.0)
    assert row["ret_8"] == pytest.approx((1.1**8) - 1.0)
    assert row["log_ret_1"] == pytest.approx(math.log(1.1))
    assert row["range_pct"] == pytest.approx(0.03)
    assert row["body_pct"] == pytest.approx(1.0 / 6.0)
    assert row["upper_wick_pct"] == pytest.approx(1.0 / 3.0)
    assert row["lower_wick_pct"] == pytest.approx(0.5)
    assert row["volume_change_1"] == pytest.approx(0.125)
    assert row["rolling_mean_return_4"] == pytest.approx(0.1)
    assert row["rolling_mean_return_8"] == pytest.approx(0.1)
    assert row["rolling_volatility_4"] == pytest.approx(0.0)
    assert row["rolling_volatility_8"] == pytest.approx(0.0)
    assert row["rolling_volume_mean_4"] == pytest.approx(75.0)
    assert row["rolling_volume_mean_8"] == pytest.approx(55.0)

    volume_window = market_df["volume"].iloc[1:9]
    expected_volume_zscore = (
        source_row["volume"] - volume_window.mean()
    ) / volume_window.std(ddof=1)
    assert row["volume_zscore_8"] == pytest.approx(expected_volume_zscore)

    close_window_4 = market_df["close"].iloc[5:9]
    close_window_8 = market_df["close"].iloc[1:9]
    assert row["close_vs_sma_4"] == pytest.approx(
        (source_row["close"] - close_window_4.mean()) / close_window_4.mean()
    )
    assert row["close_vs_sma_8"] == pytest.approx(
        (source_row["close"] - close_window_8.mean()) / close_window_8.mean()
    )
    assert row["high_low_spread"] == pytest.approx((1.01 / 0.98) - 1.0)
    assert row["rolling_max_close_8"] == pytest.approx(close_window_8.max())
    assert row["rolling_min_close_8"] == pytest.approx(close_window_8.min())
    assert row["breakout_distance_high_8"] == pytest.approx(
        (source_row["close"] - close_window_8.max()) / close_window_8.max()
    )
    assert row["breakout_distance_low_8"] == pytest.approx(
        (source_row["close"] - close_window_8.min()) / close_window_8.min()
    )
    assert row["hour_of_day"] == 2
    assert row["day_of_week"] == 0

    assert result.loc[:6, "rolling_volatility_8"].isna().all()
    assert pd.isna(result.loc[0, "ret_1"])


def test_compute_market_features_do_not_change_historical_rows_when_future_changes() -> None:
    """Historical rows should be identical even if future candles are changed."""
    baseline_df = _make_market_df()
    changed_df = baseline_df.copy()
    changed_df.loc[changed_df.index[-1], ["open", "high", "low", "close", "volume"]] = [
        999_000.0,
        1_000_000.0,
        995_000.0,
        999_500.0,
        999_999.0,
    ]

    baseline_features = compute_market_features(baseline_df)
    changed_features = compute_market_features(changed_df)

    pd.testing.assert_frame_equal(
        baseline_features.iloc[:-1].reset_index(drop=True),
        changed_features.iloc[:-1].reset_index(drop=True),
    )
