"""Helpers for locating and inspecting stored model training runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from trading_bot.settings import PROJECT_ROOT, AppSettings, load_settings
from trading_bot.storage.evaluation_store import EvaluationStore


class ModelRegistryError(Exception):
    """Raised when a stored training run cannot be resolved cleanly."""


@dataclass(frozen=True)
class ResolvedRun:
    """Coordinates for a stored training run."""

    asset: str
    symbol: str
    timeframe: str
    model_version: str
    run_id: str
    metadata: dict[str, Any]


class ModelRegistry:
    """Resolve stored run metadata and evaluation artifacts."""

    def __init__(self, *, evaluation_store: EvaluationStore) -> None:
        self.evaluation_store = evaluation_store

    @classmethod
    def from_settings(cls, settings: AppSettings | None = None) -> ModelRegistry:
        """Create a registry from project settings."""
        resolved_settings = settings or load_settings()
        return cls(
            evaluation_store=EvaluationStore(
                PROJECT_ROOT / resolved_settings.artifacts.evaluation_dir
            )
        )

    def list_runs(
        self,
        *,
        asset: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        model_version: str | None = None,
    ) -> list[dict[str, Any]]:
        """List stored model runs ordered by newest metadata first."""
        return self.evaluation_store.list_runs(
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
        )

    def resolve_run(
        self,
        run_id: str,
        *,
        asset: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        model_version: str | None = None,
    ) -> ResolvedRun:
        """Resolve a run id to its stored coordinates and metadata."""
        matches = [
            run
            for run in self.list_runs(
                asset=asset,
                symbol=symbol,
                timeframe=timeframe,
                model_version=model_version,
            )
            if str(run.get("run_id")) == run_id
        ]
        if not matches:
            raise ModelRegistryError(f"No stored model run found for run_id={run_id}")
        if len(matches) > 1:
            raise ModelRegistryError(
                "Multiple stored runs matched "
                f"run_id={run_id}; please specify symbol/timeframe/model_version"
            )

        metadata = matches[0]
        return ResolvedRun(
            asset=str(metadata["asset"]),
            symbol=str(metadata["symbol"]),
            timeframe=str(metadata["timeframe"]),
            model_version=str(metadata["model_version"]),
            run_id=str(metadata["run_id"]),
            metadata=metadata,
        )

    def read_fold_metrics(self, run_id: str, **filters: str | None) -> pd.DataFrame:
        """Load per-fold metrics for one stored run."""
        resolved = self.resolve_run(run_id, **filters)
        return self.evaluation_store.read_fold_metrics(
            asset=resolved.asset,
            symbol=resolved.symbol,
            timeframe=resolved.timeframe,
            model_version=resolved.model_version,
            run_id=resolved.run_id,
        )

    def read_predictions(self, run_id: str, **filters: str | None) -> pd.DataFrame:
        """Load validation predictions for one stored run."""
        resolved = self.resolve_run(run_id, **filters)
        return self.evaluation_store.read_predictions(
            asset=resolved.asset,
            symbol=resolved.symbol,
            timeframe=resolved.timeframe,
            model_version=resolved.model_version,
            run_id=resolved.run_id,
        )

    def read_feature_importance(
        self,
        run_id: str,
        *,
        aggregate: bool = True,
        **filters: str | None,
    ) -> pd.DataFrame:
        """Load stored feature importance for one run."""
        resolved = self.resolve_run(run_id, **filters)
        return self.evaluation_store.read_feature_importance(
            asset=resolved.asset,
            symbol=resolved.symbol,
            timeframe=resolved.timeframe,
            model_version=resolved.model_version,
            run_id=resolved.run_id,
            aggregate=aggregate,
        )

    def read_aggregate_metrics(self, run_id: str, **filters: str | None) -> dict[str, Any]:
        """Load aggregate metrics for one run."""
        resolved = self.resolve_run(run_id, **filters)
        return self.evaluation_store.read_aggregate_metrics(
            asset=resolved.asset,
            symbol=resolved.symbol,
            timeframe=resolved.timeframe,
            model_version=resolved.model_version,
            run_id=resolved.run_id,
        )

    def read_report(self, run_id: str, **filters: str | None) -> dict[str, Any]:
        """Load the compact evaluation report for one run."""
        resolved = self.resolve_run(run_id, **filters)
        return self.evaluation_store.read_report(
            asset=resolved.asset,
            symbol=resolved.symbol,
            timeframe=resolved.timeframe,
            model_version=resolved.model_version,
            run_id=resolved.run_id,
        )
