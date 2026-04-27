"""Tests for validation baseline strategies."""

from __future__ import annotations

import pandas as pd

from trading_bot.validation.baselines import (
    generate_always_flat_signals,
    generate_buy_and_hold_signals,
)


def test_generate_buy_and_hold_signals_enters_and_exits() -> None:
    """Buy-and-hold should enter once and force an exit near the end."""
    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC")}
    )

    signals_df = generate_buy_and_hold_signals(frame)

    assert signals_df["signal_action"].tolist() == [
        "enter_long",
        "hold_long",
        "hold_long",
        "exit_long",
        "flat",
    ]


def test_generate_always_flat_signals_never_trades() -> None:
    """Always-flat should remain flat for every timestamp."""
    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC")}
    )

    signals_df = generate_always_flat_signals(frame)

    assert signals_df["signal_action"].tolist() == ["flat", "flat", "flat", "flat"]
    assert signals_df["desired_position"].sum() == 0
