"""Schemas for multi-horizon research orchestration and reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class HorizonRunSummary:
    """One row in the multi-horizon comparison table."""

    horizon_bars: int
    horizon_minutes: int
    target_column: str
    label_version: str
    dataset_version: str
    model_version: str
    feature_version: str
    model_run_id: str | None = None
    validation_id: str | None = None
    optimization_id: str | None = None
    accuracy_mean: float | None = None
    f1_mean: float | None = None
    roc_auc_mean: float | None = None
    log_loss_mean: float | None = None
    validation_has_edge: bool | None = None
    optimized_has_edge: bool | None = None
    best_strategy_total_return: float | None = None
    best_strategy_max_drawdown: float | None = None
    best_strategy_number_of_trades: int | None = None
    best_strategy_fee_impact: float | None = None
    best_strategy_fee_to_starting_cash: float | None = None
    passes_filters: bool = False
    ranking_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return asdict(self)


@dataclass(frozen=True)
class HorizonComparisonReport:
    """Top-level report for multi-horizon research."""

    comparison_id: str
    generated_at: str
    asset: str
    symbol: str
    timeframe: str
    feature_version: str
    horizons_tested: list[int] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    recommendation: dict[str, Any] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return asdict(self)
