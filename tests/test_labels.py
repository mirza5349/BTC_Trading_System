"""Tests for future-return label generation."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from trading_bot.labeling.labels import generate_custom_target_labels, generate_future_return_labels
from trading_bot.schemas.targets import TargetConfig


def _make_market_df() -> pd.DataFrame:
    """Create a deterministic BTC candle frame for label tests."""
    timestamps = pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC")
    close = pd.Series([100.0, 101.0, 102.0, 99.0, 100.0, 104.0, 103.0, 107.0])
    return pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "timeframe": "15m",
            "timestamp": timestamps,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": pd.Series([10, 11, 12, 13, 14, 15, 16, 17], dtype=float),
        }
    )


def test_generate_future_return_labels_binary_and_regression_targets() -> None:
    """The primary 2-bar targets should be computed from future closes only."""
    market_df = _make_market_df()

    result = generate_future_return_labels(
        market_df,
        primary_horizon_bars=2,
        include_regression_target=True,
        include_optional_targets=False,
        drop_unlabeled_rows=True,
    )

    assert result.shape[0] == 6
    assert result["timestamp"].is_monotonic_increasing
    assert list(result.columns) == [
        "symbol",
        "timeframe",
        "timestamp",
        "target_up_2bars",
        "future_return_2bars",
    ]

    first_row = result.iloc[0]
    second_row = result.iloc[1]

    assert first_row["target_up_2bars"] == 1
    assert first_row["future_return_2bars"] == pytest.approx((102.0 / 100.0) - 1.0)
    assert second_row["target_up_2bars"] == 0
    assert second_row["future_return_2bars"] == pytest.approx((99.0 / 101.0) - 1.0)


def test_generate_future_return_labels_optional_targets_drop_end_of_sample_rows() -> None:
    """Including the optional 4-bar target should drop the final 4 rows cleanly."""
    market_df = _make_market_df()

    result = generate_future_return_labels(
        market_df,
        primary_horizon_bars=2,
        include_regression_target=True,
        include_optional_targets=True,
        optional_horizon_bars=[4],
        drop_unlabeled_rows=True,
    )

    assert result.shape[0] == 4
    assert "target_up_4bars" in result.columns
    assert "future_log_return_2bars" in result.columns

    first_row = result.iloc[0]
    assert first_row["target_up_4bars"] == 0
    assert first_row["future_log_return_2bars"] == pytest.approx(math.log(102.0 / 100.0))


def test_generate_future_return_labels_supports_horizon_minutes() -> None:
    """A 30-minute horizon on 15m candles should resolve to 2 bars."""
    market_df = _make_market_df()

    result = generate_future_return_labels(
        market_df,
        primary_horizon_bars=1,
        horizon_minutes=30,
        include_regression_target=False,
        include_optional_targets=False,
        drop_unlabeled_rows=True,
    )

    assert "target_up_2bars" in result.columns
    assert result.shape[0] == 6
    assert result.iloc[0]["target_up_2bars"] == 1


def test_generate_custom_target_labels_fee_aware_threshold() -> None:
    """Fee-aware targets should only fire when future return beats round-trip cost."""
    market_df = _make_market_df()

    generated = generate_custom_target_labels(
        market_df,
        target_configs=[
            TargetConfig(
                key="fee_4",
                target_name="target_long_net_positive_4bars",
                horizon_bars=4,
                threshold_type="fee_aware",
                label_version="fee_h4",
                dataset_version="fee_h4",
                model_version="fee_h4",
                fee_rate=0.001,
                slippage_rate=0.0005,
                round_trip_cost=0.003,
            )
        ],
    )

    result = generated.labels_df
    assert "target_long_net_positive_4bars" in result.columns
    assert "future_return_4bars" in result.columns
    assert result.iloc[0]["target_long_net_positive_4bars"] == 0
    assert result.iloc[1]["target_long_net_positive_4bars"] == 1


def test_generate_custom_target_labels_minimum_return_threshold() -> None:
    """Minimum-return targets should apply the configured threshold exactly."""
    market_df = _make_market_df()

    generated = generate_custom_target_labels(
        market_df,
        target_configs=[
            TargetConfig(
                key="ret050_4",
                target_name="target_return_gt_050pct_4bars",
                horizon_bars=4,
                threshold_type="minimum_return",
                threshold_return=0.005,
                label_version="fee_ret050_h4",
                dataset_version="fee_ret050_h4",
                model_version="fee_ret050_h4",
            )
        ],
    )

    result = generated.labels_df
    assert result.iloc[0]["target_return_gt_050pct_4bars"] == 0
    assert result.iloc[1]["target_return_gt_050pct_4bars"] == 1
