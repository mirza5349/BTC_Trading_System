"""Walk-forward split generation for time-ordered supervised datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from trading_bot.logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_WALK_FORWARD_MODES: tuple[str, ...] = ("expanding_window", "rolling_window")


@dataclass(frozen=True)
class WalkForwardSplit:
    """A single train/validation fold with explicit index and timestamp bounds."""

    fold_id: int
    mode: str
    train_start_idx: int
    train_end_idx: int
    validation_start_idx: int
    validation_end_idx: int
    n_train: int
    n_validation: int
    train_start: pd.Timestamp | None = None
    train_end: pd.Timestamp | None = None
    validation_start: pd.Timestamp | None = None
    validation_end: pd.Timestamp | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the fold."""
        return {
            "fold_id": self.fold_id,
            "mode": self.mode,
            "train_start_idx": self.train_start_idx,
            "train_end_idx": self.train_end_idx,
            "validation_start_idx": self.validation_start_idx,
            "validation_end_idx": self.validation_end_idx,
            "n_train": self.n_train,
            "n_validation": self.n_validation,
            "train_start": self.train_start.isoformat() if self.train_start is not None else None,
            "train_end": self.train_end.isoformat() if self.train_end is not None else None,
            "validation_start": (
                self.validation_start.isoformat() if self.validation_start is not None else None
            ),
            "validation_end": (
                self.validation_end.isoformat() if self.validation_end is not None else None
            ),
        }


@dataclass(frozen=True)
class WalkForwardResult:
    """Collection of generated walk-forward folds."""

    splits: list[WalkForwardSplit] = field(default_factory=list)

    @property
    def fold_count(self) -> int:
        """Return the number of generated folds."""
        return len(self.splits)


class WalkForwardError(Exception):
    """Raised when walk-forward splits cannot be created safely."""


class WalkForwardSplitter:
    """Generate leak-safe walk-forward splits for time-ordered datasets."""

    def __init__(
        self,
        *,
        mode: str = "expanding_window",
        min_train_rows: int = 1000,
        validation_rows: int = 250,
        step_rows: int = 250,
        max_folds: int | None = None,
        rolling_train_rows: int | None = None,
    ) -> None:
        if mode not in SUPPORTED_WALK_FORWARD_MODES:
            raise ValueError(
                f"Unsupported walk-forward mode {mode!r}. "
                f"Expected one of {SUPPORTED_WALK_FORWARD_MODES}."
            )
        if min_train_rows <= 0:
            raise ValueError("min_train_rows must be positive")
        if validation_rows <= 0:
            raise ValueError("validation_rows must be positive")
        if step_rows <= 0:
            raise ValueError("step_rows must be positive")

        self.mode = mode
        self.min_train_rows = min_train_rows
        self.validation_rows = validation_rows
        self.step_rows = step_rows
        self.max_folds = max_folds
        self.rolling_train_rows = rolling_train_rows

    def split(self, data: pd.DataFrame | int) -> list[WalkForwardSplit]:
        """Return walk-forward folds for a dataset length or a DataFrame."""
        n_rows, timestamps = _resolve_dataset_inputs(data)

        if n_rows < self.min_train_rows + self.validation_rows:
            logger.warning(
                "walk_forward_not_enough_rows",
                mode=self.mode,
                n_rows=n_rows,
                min_train_rows=self.min_train_rows,
                validation_rows=self.validation_rows,
            )
            return []

        splits: list[WalkForwardSplit] = []
        validation_start_idx = self.min_train_rows
        fold_id = 0

        while validation_start_idx + self.validation_rows <= n_rows:
            if self.max_folds is not None and fold_id >= self.max_folds:
                break

            if self.mode == "expanding_window":
                train_start_idx = 0
            else:
                window_rows = self.rolling_train_rows or self.min_train_rows
                train_start_idx = max(0, validation_start_idx - window_rows)

            train_end_idx = validation_start_idx - 1
            validation_end_idx = validation_start_idx + self.validation_rows - 1

            split = WalkForwardSplit(
                fold_id=fold_id,
                mode=self.mode,
                train_start_idx=train_start_idx,
                train_end_idx=train_end_idx,
                validation_start_idx=validation_start_idx,
                validation_end_idx=validation_end_idx,
                n_train=train_end_idx - train_start_idx + 1,
                n_validation=validation_end_idx - validation_start_idx + 1,
                train_start=_timestamp_at(timestamps, train_start_idx),
                train_end=_timestamp_at(timestamps, train_end_idx),
                validation_start=_timestamp_at(timestamps, validation_start_idx),
                validation_end=_timestamp_at(timestamps, validation_end_idx),
            )

            _validate_split_order(split)
            splits.append(split)

            validation_start_idx += self.step_rows
            fold_id += 1

        logger.info(
            "walk_forward_splits_generated",
            mode=self.mode,
            n_rows=n_rows,
            fold_count=len(splits),
            min_train_rows=self.min_train_rows,
            validation_rows=self.validation_rows,
            step_rows=self.step_rows,
            max_folds=self.max_folds,
        )
        return splits

    def describe(self, data: pd.DataFrame | int) -> WalkForwardResult:
        """Return the generated splits wrapped in a result object."""
        return WalkForwardResult(splits=self.split(data))


class WalkForwardValidator(WalkForwardSplitter):
    """Backward-compatible alias for the original placeholder class."""


def _resolve_dataset_inputs(data: pd.DataFrame | int) -> tuple[int, pd.Series | None]:
    """Normalize splitter inputs to a row count plus optional timestamps."""
    if isinstance(data, int):
        return data, None

    if data.empty:
        return 0, pd.Series(dtype="datetime64[ns, UTC]")

    if "timestamp" not in data.columns:
        raise WalkForwardError("Walk-forward splitter requires a timestamp column")

    timestamps = pd.to_datetime(data["timestamp"], utc=True, errors="raise")
    if not timestamps.is_monotonic_increasing:
        raise WalkForwardError("Walk-forward splitter requires timestamps sorted ascending")

    return len(data), timestamps.reset_index(drop=True)


def _timestamp_at(
    timestamps: pd.Series | None,
    index: int,
) -> pd.Timestamp | None:
    """Return the timestamp at the requested row index when available."""
    if timestamps is None:
        return None
    return pd.Timestamp(timestamps.iloc[index])


def _validate_split_order(split: WalkForwardSplit) -> None:
    """Fail if a fold would leak validation rows into training."""
    if split.train_end_idx >= split.validation_start_idx:
        raise WalkForwardError(
            "Walk-forward split is invalid because training rows overlap validation rows"
        )
    if (
        split.train_start is not None
        and split.validation_start is not None
        and (split.train_end is None or split.train_end >= split.validation_start)
    ):
        raise WalkForwardError(
            "Walk-forward split is invalid because training timestamps overlap validation"
        )
