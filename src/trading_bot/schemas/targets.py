"""Schemas and helpers for fee-aware and minimum-return target research."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TargetConfig:
    """Serializable configuration for one custom supervised target."""

    key: str
    target_name: str
    horizon_bars: int
    threshold_type: str
    label_version: str
    dataset_version: str
    model_version: str
    threshold_return: float | None = None
    fee_rate: float | None = None
    slippage_rate: float | None = None
    round_trip_cost: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return asdict(self)


@dataclass(frozen=True)
class TargetRunSummary:
    """One row in the fee-aware target comparison table."""

    target_key: str
    target_name: str
    horizon_bars: int
    horizon_minutes: int
    threshold_type: str
    threshold_return: float | None
    positive_class_rate: float | None
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
    optimized_has_edge_passed: bool | None = None
    return_filter_passed: bool | None = None
    drawdown_filter_passed: bool | None = None
    trade_count_filter_passed: bool | None = None
    fee_filter_passed: bool | None = None
    class_balance_filter_passed: bool | None = None
    validation_has_edge_passed: bool | None = None
    rejection_reason: str = ""
    passes_filters: bool = False
    ranking_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return asdict(self)


@dataclass(frozen=True)
class TargetComparisonReport:
    """Top-level report for fee-aware target comparison research."""

    comparison_id: str
    generated_at: str
    asset: str
    symbol: str
    timeframe: str
    feature_version: str
    feature_versions: list[str] = field(default_factory=list)
    target_keys_tested: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    recommendation: dict[str, Any] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return asdict(self)
