"""Utilities for comparing experiment runs on stable metrics and metadata."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from trading_bot.experiments.registry import ExperimentRegistry
from trading_bot.logging_config import get_logger
from trading_bot.schemas.experiments import (
    RECOMMENDED_COMPARISON_COLUMNS,
    SUMMARY_METRIC_COLUMNS,
    metric_sort_ascending,
)
from trading_bot.settings import AppSettings, load_settings

logger = get_logger(__name__)


class RunComparator:
    """Build comparison tables for two or more tracked runs."""

    def __init__(self, *, registry: ExperimentRegistry) -> None:
        self.registry = registry

    @classmethod
    def from_settings(cls, settings: AppSettings | None = None) -> RunComparator:
        """Create a comparator from project settings."""
        resolved_settings = settings or load_settings()
        return cls(registry=ExperimentRegistry.from_settings(resolved_settings))

    def compare_runs(
        self,
        run_ids: list[str],
        *,
        sort_by: str | None = None,
    ) -> pd.DataFrame:
        """Return a machine-loadable comparison table for the requested runs."""
        registry_df = self.registry.read_registry()
        if registry_df.empty:
            return pd.DataFrame(columns=RECOMMENDED_COMPARISON_COLUMNS + SUMMARY_METRIC_COLUMNS)

        filtered = registry_df[registry_df["run_id"].isin(run_ids)].copy()
        if filtered.empty:
            return filtered

        missing_run_ids = [run_id for run_id in run_ids if run_id not in filtered["run_id"].tolist()]
        if missing_run_ids:
            logger.warning("compare_runs_missing_ids", missing_run_ids=missing_run_ids)

        sort_column = sort_by or self.registry.default_sort_metric

        if sort_column in filtered.columns:
            filtered = filtered.sort_values(
                by=[sort_column, "created_at" if "created_at" in filtered.columns else "run_id"],
                ascending=[metric_sort_ascending(sort_column), False],
                na_position="last",
            )

        ordered_columns = RECOMMENDED_COMPARISON_COLUMNS + SUMMARY_METRIC_COLUMNS
        comparison_df = filtered.reindex(columns=ordered_columns)
        return comparison_df.reset_index(drop=True)

    def build_sentiment_ablation_summary(
        self,
        *,
        baseline_run_id: str,
        new_run_id: str,
    ) -> dict[str, Any]:
        """Return a provenance-rich baseline-vs-sentiment comparison summary."""
        comparison_df = self.compare_runs([baseline_run_id, new_run_id])
        if comparison_df.empty:
            return {}

        index_by_run_id = {
            str(row["run_id"]): row
            for row in comparison_df.to_dict(orient="records")
        }
        if baseline_run_id not in index_by_run_id or new_run_id not in index_by_run_id:
            return {}

        baseline_row = dict(index_by_run_id[baseline_run_id])
        new_row = dict(index_by_run_id[new_run_id])

        baseline_summary = self._safe_run_summary(baseline_run_id)
        new_summary = self._safe_run_summary(new_run_id)

        metric_deltas: dict[str, float | None] = {}
        metric_improvements: dict[str, str] = {}
        for metric_name in SUMMARY_METRIC_COLUMNS:
            baseline_value = _float_or_none(baseline_row.get(metric_name))
            new_value = _float_or_none(new_row.get(metric_name))
            if baseline_value is None or new_value is None:
                metric_deltas[metric_name] = None
                metric_improvements[metric_name] = "unknown"
                continue

            delta = new_value - baseline_value
            if metric_name == "log_loss_mean":
                delta = baseline_value - new_value
            metric_deltas[metric_name] = round(delta, 8)
            metric_improvements[metric_name] = (
                "improved" if delta > 0 else "worse" if delta < 0 else "unchanged"
            )

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "baseline_run_id": baseline_run_id,
            "new_run_id": new_run_id,
            "baseline_run": baseline_row,
            "new_run": new_row,
            "metric_deltas": metric_deltas,
            "metric_improvements": metric_improvements,
            "provenance": {
                "baseline_summary": baseline_summary,
                "new_summary": new_summary,
            },
        }

    def compare_feature_importance(
        self,
        run_ids: list[str],
        *,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Return a simple cross-run top-feature table."""
        rows: list[dict[str, Any]] = []
        for run_id in run_ids:
            top_features = self.registry.experiment_store.read_top_features(run_id)
            if top_features.empty:
                continue
            subset = top_features.head(top_k).copy()
            subset["run_id"] = run_id
            rows.extend(subset.to_dict(orient="records"))

        if not rows:
            return pd.DataFrame(columns=["run_id", "feature_name", "importance", "rank"])

        comparison_df = pd.DataFrame(rows)
        ordered_columns = [
            column
            for column in ("run_id", "feature_name", "importance", "rank")
            if column in comparison_df.columns
        ]
        return comparison_df.reindex(columns=ordered_columns).reset_index(drop=True)

    def _safe_run_summary(self, run_id: str) -> dict[str, Any]:
        """Return the stored run summary when available, otherwise an empty payload."""
        try:
            return self.registry.get_run_summary(run_id)
        except Exception:
            return {}


def _float_or_none(value: Any) -> float | None:
    """Return a finite float when available."""
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(converted):
        return None
    return converted
