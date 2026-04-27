"""Schemas and constants for model-validation audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

BASELINE_STRATEGIES: tuple[str, ...] = (
    "buy_and_hold",
    "always_flat",
    "random_matched_frequency",
    "sma_crossover_20_50",
)

ValidationConclusion = Literal["has_edge", "no_edge"]


@dataclass(frozen=True)
class ValidationReport:
    """Top-level report for a model-vs-baselines validation run."""

    validation_id: str
    model_run_id: str
    generated_at: str
    symbol: str
    timeframe: str
    target_column: str = ""
    dataset_version: str = ""
    feature_version: str = ""
    label_version: str = ""
    horizon_bars: int | None = None
    strategy_comparison: list[dict[str, Any]] = field(default_factory=list)
    prediction_diagnostics: dict[str, Any] = field(default_factory=dict)
    failure_analysis: dict[str, Any] = field(default_factory=dict)
    feature_importance_summary: dict[str, Any] = field(default_factory=dict)
    simulation_audit: dict[str, Any] = field(default_factory=dict)
    conclusion: dict[str, Any] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return asdict(self)
