"""Baseline strategy signal generation for validation audits."""

from __future__ import annotations

import math
import random

import pandas as pd


def generate_buy_and_hold_signals(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a full-invested buy-and-hold signal stream."""
    return _signals_from_position_windows(frame, [(0, max(0, len(frame) - 2))])


def generate_always_flat_signals(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a no-trade signal stream."""
    return _empty_signal_frame(frame)


def generate_random_matched_frequency_signals(
    frame: pd.DataFrame,
    *,
    reference_trade_count: int,
    reference_average_holding_bars: float,
    random_seed: int,
) -> pd.DataFrame:
    """Return a deterministic random signal stream matching reference trade frequency."""
    if frame.empty or reference_trade_count <= 0:
        return _empty_signal_frame(frame)

    rng = random.Random(random_seed)
    max_entry_index = max(0, len(frame) - 2)
    hold_bars = max(1, int(math.ceil(reference_average_holding_bars or 1.0)))
    candidate_indices = list(range(max_entry_index + 1))
    rng.shuffle(candidate_indices)

    windows: list[tuple[int, int]] = []
    for entry_index in candidate_indices:
        exit_signal_index = min(entry_index + hold_bars, len(frame) - 2)
        if exit_signal_index <= entry_index:
            continue
        overlaps = any(
            not (exit_signal_index < existing_start or entry_index > existing_end)
            for existing_start, existing_end in windows
        )
        if overlaps:
            continue
        windows.append((entry_index, exit_signal_index))
        if len(windows) >= reference_trade_count:
            break

    return _signals_from_position_windows(frame, sorted(windows))


def generate_sma_crossover_signals(
    frame: pd.DataFrame,
    *,
    fast_window: int = 20,
    slow_window: int = 50,
) -> pd.DataFrame:
    """Return a long-only SMA crossover signal stream."""
    if frame.empty:
        return _empty_signal_frame(frame)

    working = frame.copy()
    working["sma_fast"] = pd.to_numeric(working["close"], errors="coerce").rolling(fast_window).mean()
    working["sma_slow"] = pd.to_numeric(working["close"], errors="coerce").rolling(slow_window).mean()
    desired = (working["sma_fast"] > working["sma_slow"]).fillna(False).astype(int).tolist()
    if len(desired) >= 2:
        desired[-1] = 0
    return _signals_from_desired_positions(working["timestamp"], desired)


def _signals_from_position_windows(
    frame: pd.DataFrame,
    windows: list[tuple[int, int]],
) -> pd.DataFrame:
    """Return signal actions from explicit entry/exit signal windows."""
    desired = [0 for _ in range(len(frame))]
    for start, end in windows:
        for index in range(max(0, start), min(len(frame), end + 1)):
            desired[index] = 1
        if 0 <= end < len(desired):
            desired[end] = 0
    return _signals_from_desired_positions(frame["timestamp"], desired)


def _signals_from_desired_positions(
    timestamps: pd.Series,
    desired_positions: list[int],
) -> pd.DataFrame:
    """Return the canonical signal schema from a 0/1 desired-position stream."""
    rows: list[dict[str, object]] = []
    previous_position = 0
    bars_held = 0

    for timestamp, desired_position in zip(pd.to_datetime(timestamps, utc=True), desired_positions):
        desired_position = int(bool(desired_position))
        if desired_position == 1 and previous_position == 0:
            action = "enter_long"
            bars_held = 0
        elif desired_position == 1 and previous_position == 1:
            action = "hold_long"
            bars_held += 1
        elif desired_position == 0 and previous_position == 1:
            action = "exit_long"
            bars_held = 0
        else:
            action = "flat"
            bars_held = 0

        rows.append(
            {
                "timestamp": timestamp,
                "signal_action": action,
                "desired_position": desired_position,
                "bars_held_signal_state": bars_held if desired_position == 1 else 0,
                "cooldown_remaining": 0,
            }
        )
        previous_position = desired_position

    return pd.DataFrame(rows)


def _empty_signal_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a flat signal frame for the provided timestamps."""
    timestamps = pd.to_datetime(
        frame.get("timestamp", pd.Series(dtype="datetime64[ns]")),
        utc=True,
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "signal_action": ["flat"] * len(timestamps),
            "desired_position": [0] * len(timestamps),
            "bars_held_signal_state": [0] * len(timestamps),
            "cooldown_remaining": [0] * len(timestamps),
        }
    )
