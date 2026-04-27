"""Tests for walk-forward split generation."""

from __future__ import annotations

import pandas as pd
import pytest

from trading_bot.validation.walk_forward import WalkForwardError, WalkForwardSplitter


def _make_time_ordered_df(row_count: int = 10) -> pd.DataFrame:
    """Create a deterministic timestamp-ordered frame for splitter tests."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=row_count, freq="15min", tz="UTC"),
            "value": range(row_count),
        }
    )


def test_expanding_window_split_boundaries_are_time_ordered() -> None:
    """Expanding-window folds should grow history and keep validation strictly ahead."""
    splitter = WalkForwardSplitter(
        mode="expanding_window",
        min_train_rows=4,
        validation_rows=2,
        step_rows=2,
    )

    splits = splitter.split(_make_time_ordered_df())

    assert len(splits) == 3
    assert [(split.train_start_idx, split.train_end_idx) for split in splits] == [
        (0, 3),
        (0, 5),
        (0, 7),
    ]
    assert [(split.validation_start_idx, split.validation_end_idx) for split in splits] == [
        (4, 5),
        (6, 7),
        (8, 9),
    ]
    assert all(
        split.train_end < split.validation_start
        for split in splits
        if split.train_end is not None
    )


def test_rolling_window_split_boundaries_shift_forward() -> None:
    """Rolling-window folds should keep a fixed trailing train window when configured."""
    splitter = WalkForwardSplitter(
        mode="rolling_window",
        min_train_rows=4,
        validation_rows=2,
        step_rows=2,
        rolling_train_rows=4,
    )

    splits = splitter.split(_make_time_ordered_df())

    assert len(splits) == 3
    assert [(split.train_start_idx, split.train_end_idx) for split in splits] == [
        (0, 3),
        (2, 5),
        (4, 7),
    ]
    assert [(split.validation_start_idx, split.validation_end_idx) for split in splits] == [
        (4, 5),
        (6, 7),
        (8, 9),
    ]


def test_split_raises_on_unsorted_timestamps() -> None:
    """The splitter should fail loudly when timestamps are not sorted ascending."""
    df = _make_time_ordered_df().iloc[::-1].reset_index(drop=True)
    splitter = WalkForwardSplitter(
        mode="expanding_window",
        min_train_rows=4,
        validation_rows=2,
        step_rows=2,
    )

    with pytest.raises(WalkForwardError, match="timestamps sorted ascending"):
        splitter.split(df)
