"""Query and management helpers for the local experiment registry."""

from __future__ import annotations

from typing import Any

import pandas as pd

from trading_bot.experiments.promotion import tag_run_status
from trading_bot.logging_config import get_logger
from trading_bot.schemas.experiments import (
    RECOMMENDED_COMPARISON_COLUMNS,
    RUN_REGISTRY_COLUMNS,
    SUMMARY_METRIC_COLUMNS,
    metric_sort_ascending,
)
from trading_bot.settings import PROJECT_ROOT, AppSettings, load_settings
from trading_bot.storage.experiment_store import ExperimentStore

logger = get_logger(__name__)


class ExperimentRegistryError(Exception):
    """Raised when a local experiment record cannot be resolved."""


class ExperimentRegistry:
    """Query and update local run summaries and registry rows."""

    def __init__(self, *, experiment_store: ExperimentStore, default_sort_metric: str) -> None:
        self.experiment_store = experiment_store
        self.default_sort_metric = default_sort_metric

    @classmethod
    def from_settings(cls, settings: AppSettings | None = None) -> ExperimentRegistry:
        """Build a registry from project settings."""
        resolved_settings = settings or load_settings()
        return cls(
            experiment_store=ExperimentStore(
                base_dir=PROJECT_ROOT / resolved_settings.experiments.summary_dir,
                registry_path=PROJECT_ROOT / resolved_settings.experiments.registry_path,
            ),
            default_sort_metric=resolved_settings.experiments.default_sort_metric,
        )

    def read_registry(self) -> pd.DataFrame:
        """Return the persisted experiment registry."""
        return self.experiment_store.read_registry()

    def list_runs(
        self,
        *,
        status: str | None = None,
        model_version: str | None = None,
        dataset_version: str | None = None,
        target_column: str | None = None,
        device: str | None = None,
        sort_by: str | None = None,
    ) -> pd.DataFrame:
        """Filter and sort the local experiment registry."""
        registry_df = self.read_registry()
        if registry_df.empty:
            return registry_df

        filtered = registry_df.copy()
        if status is not None:
            filtered = filtered[filtered["status"] == status]
        if model_version is not None:
            filtered = filtered[filtered["model_version"] == model_version]
        if dataset_version is not None:
            filtered = filtered[filtered["dataset_version"] == dataset_version]
        if target_column is not None:
            filtered = filtered[filtered["target_column"] == target_column]
        if device is not None:
            filtered = filtered[
                (filtered["requested_device"] == device) | (filtered["effective_device"] == device)
            ]

        sort_column = sort_by or self.default_sort_metric
        if sort_column in filtered.columns:
            filtered = filtered.sort_values(
                by=[sort_column, "created_at", "run_id"],
                ascending=[metric_sort_ascending(sort_column), False, False],
                na_position="last",
            )
        else:
            filtered = filtered.sort_values(by=["created_at", "run_id"], ascending=[False, False])

        return filtered.reset_index(drop=True)

    def get_run_summary(self, run_id: str) -> dict[str, Any]:
        """Load one run summary from local experiment storage."""
        summary = self.experiment_store.read_run_summary(run_id)
        if not summary:
            raise ExperimentRegistryError(f"No experiment summary found for run_id={run_id}")
        return summary

    def set_run_status(self, *, run_id: str, status: str, note: str | None = None) -> dict[str, Any]:
        """Update one run summary and registry row with a promotion decision."""
        summary = self.get_run_summary(run_id)
        updated_summary = tag_run_status(summary=summary, status=status, note=note)
        self.experiment_store.write_run_summary(run_id, updated_summary)

        registry_df = self.read_registry()
        if registry_df.empty or run_id not in registry_df["run_id"].tolist():
            raise ExperimentRegistryError(f"No experiment registry row found for run_id={run_id}")

        registry_df = registry_df.copy()
        registry_df.loc[registry_df["run_id"] == run_id, "status"] = updated_summary["status"]
        if note is not None:
            registry_df.loc[registry_df["run_id"] == run_id, "notes"] = updated_summary["notes"]
        self.experiment_store.write_registry(registry_df)

        logger.info("experiment_status_updated", run_id=run_id, status=status)
        return updated_summary

    def comparison_columns(self) -> list[str]:
        """Return the default stable comparison columns."""
        return RECOMMENDED_COMPARISON_COLUMNS + SUMMARY_METRIC_COLUMNS

    def registry_columns(self) -> list[str]:
        """Return the stable registry schema."""
        return RUN_REGISTRY_COLUMNS.copy()
