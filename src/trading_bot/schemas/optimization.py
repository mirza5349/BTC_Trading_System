"""Schemas for strategy-threshold optimization and reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class StrategySearchParameters:
    """One candidate strategy configuration in the search grid."""

    entry_threshold: float
    exit_threshold: float
    minimum_holding_bars: int
    cooldown_bars: int
    max_position_fraction: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return asdict(self)


@dataclass(frozen=True)
class OptimizationReport:
    """Top-level optimization report payload."""

    optimization_id: str
    model_run_id: str
    generated_at: str
    symbol: str = ""
    timeframe: str = ""
    target_column: str = ""
    dataset_version: str = ""
    feature_version: str = ""
    label_version: str = ""
    horizon_bars: int | None = None
    search_space: dict[str, Any] = field(default_factory=dict)
    candidates_tested: int = 0
    best_candidate: dict[str, Any] = field(default_factory=dict)
    top_candidates: list[dict[str, Any]] = field(default_factory=list)
    rejection_counts: dict[str, int] = field(default_factory=dict)
    fee_audit: dict[str, Any] = field(default_factory=dict)
    drawdown_audit: dict[str, Any] = field(default_factory=dict)
    baseline_comparison: list[dict[str, Any]] = field(default_factory=list)
    recommendation: dict[str, Any] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return asdict(self)
