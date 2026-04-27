"""Tests for probability-to-signal conversion."""

from __future__ import annotations

import pandas as pd

from trading_bot.backtest.signals import generate_threshold_signals
from trading_bot.schemas.backtest import StrategyConfig


def test_generate_threshold_signals_respects_thresholds_and_cooldown() -> None:
    """Signal generation should honor entry, exit, holding, and cooldown rules."""
    predictions_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="15min", tz="UTC"),
            "y_proba": [0.56, 0.57, 0.40, 0.58, 0.60, 0.44],
        }
    )
    config = StrategyConfig(
        entry_threshold=0.55,
        exit_threshold=0.45,
        minimum_holding_bars=1,
        cooldown_bars=1,
    )

    signals_df = generate_threshold_signals(predictions_df, strategy_config=config)

    assert signals_df["signal_action"].tolist() == [
        "enter_long",
        "hold_long",
        "exit_long",
        "flat",
        "enter_long",
        "exit_long",
    ]
    assert signals_df["desired_position"].tolist() == [1, 1, 0, 0, 1, 0]


def test_generate_threshold_signals_returns_empty_frame_for_empty_input() -> None:
    """Signal generation should handle empty prediction tables cleanly."""
    signals_df = generate_threshold_signals(
        pd.DataFrame(columns=["timestamp", "y_proba"]),
        strategy_config=StrategyConfig(),
    )

    assert signals_df.empty
    assert "signal_action" in signals_df.columns
