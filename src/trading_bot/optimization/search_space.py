"""Search-space helpers for strategy-threshold optimization."""

from __future__ import annotations

from itertools import product
from typing import Any

from trading_bot.schemas.optimization import StrategySearchParameters


class StrategySearchSpaceError(Exception):
    """Raised when the configured strategy search space is invalid."""


def build_parameter_grid(search_space: dict[str, list[Any]]) -> list[StrategySearchParameters]:
    """Return deterministic valid parameter combinations from a search-space mapping."""
    required_keys = {
        "entry_threshold",
        "exit_threshold",
        "minimum_holding_bars",
        "cooldown_bars",
        "max_position_fraction",
    }
    missing_keys = sorted(required_keys - set(search_space))
    if missing_keys:
        raise StrategySearchSpaceError(f"Missing search-space keys: {missing_keys}")

    combinations: list[StrategySearchParameters] = []
    for values in product(
        search_space["entry_threshold"],
        search_space["exit_threshold"],
        search_space["minimum_holding_bars"],
        search_space["cooldown_bars"],
        search_space["max_position_fraction"],
    ):
        candidate = StrategySearchParameters(
            entry_threshold=float(values[0]),
            exit_threshold=float(values[1]),
            minimum_holding_bars=int(values[2]),
            cooldown_bars=int(values[3]),
            max_position_fraction=float(values[4]),
        )
        if is_valid_parameter_combination(candidate):
            combinations.append(candidate)
    return combinations


def is_valid_parameter_combination(candidate: StrategySearchParameters) -> bool:
    """Return whether one candidate respects simple threshold-search invariants."""
    return (
        candidate.entry_threshold > candidate.exit_threshold
        and candidate.entry_threshold > 0.0
        and candidate.exit_threshold >= 0.0
        and candidate.minimum_holding_bars >= 1
        and candidate.cooldown_bars >= 0
        and 0.0 < candidate.max_position_fraction <= 1.0
    )
