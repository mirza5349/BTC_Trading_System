"""Filesystem-backed local persistence for experiment registry artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from trading_bot.logging_config import get_logger
from trading_bot.schemas.experiments import RUN_REGISTRY_COLUMNS, ensure_registry_columns

logger = get_logger(__name__)

RUN_SUMMARY_FILENAME = "run_summary.json"
TOP_FEATURES_FILENAME = "top_features.parquet"


class ExperimentStoreError(Exception):
    """Raised when experiment artifacts cannot be persisted or loaded."""


class ExperimentStore:
    """Persist a local run registry and per-run experiment summaries."""

    def __init__(self, *, base_dir: str | Path, registry_path: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.registry_path = Path(registry_path)

    def run_dir(self, run_id: str) -> Path:
        """Return the local experiment directory for one run."""
        return self.base_dir / f"run_id={run_id}"

    def summary_path(self, run_id: str) -> Path:
        """Return the summary JSON path for one run."""
        return self.run_dir(run_id) / RUN_SUMMARY_FILENAME

    def top_features_path(self, run_id: str) -> Path:
        """Return the persisted top-feature parquet path for one run."""
        return self.run_dir(run_id) / TOP_FEATURES_FILENAME

    def read_registry(self) -> pd.DataFrame:
        """Load the local run registry."""
        if not self.registry_path.exists():
            return pd.DataFrame(columns=RUN_REGISTRY_COLUMNS)

        try:
            df = pd.read_parquet(self.registry_path)
        except Exception as exc:
            raise ExperimentStoreError(
                f"Failed to read experiment registry from {self.registry_path}: {exc}"
            ) from exc
        return ensure_registry_columns(df)

    def write_registry(self, df: pd.DataFrame) -> Path:
        """Persist the local run registry in a stable schema."""
        path = self.registry_path
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            normalized = ensure_registry_columns(df)
            normalized.to_parquet(path, index=False, engine="pyarrow")
        except Exception as exc:
            raise ExperimentStoreError(
                f"Failed to write experiment registry to {path}: {exc}"
            ) from exc

        logger.info("experiment_registry_write_complete", path=str(path), rows=len(df))
        return path

    def upsert_registry_rows(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        """Insert or replace registry rows by run id."""
        existing = self.read_registry()
        incoming = ensure_registry_columns(pd.DataFrame(rows))
        if incoming.empty:
            return existing
        if existing.empty:
            incoming = incoming.drop_duplicates(subset=["run_id"], keep="last")
            incoming = incoming.sort_values(["created_at", "run_id"], ascending=[False, False])
            incoming = incoming.reset_index(drop=True)
            self.write_registry(incoming)
            return incoming

        combined = pd.concat([existing, incoming], ignore_index=True)
        combined = combined.drop_duplicates(subset=["run_id"], keep="last")
        combined = combined.sort_values(["created_at", "run_id"], ascending=[False, False])
        combined = combined.reset_index(drop=True)
        self.write_registry(combined)
        return combined

    def write_run_summary(self, run_id: str, payload: dict[str, Any]) -> Path:
        """Persist one experiment run summary as JSON."""
        path = self.summary_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        except Exception as exc:
            raise ExperimentStoreError(f"Failed to write run summary to {path}: {exc}") from exc
        return path

    def read_run_summary(self, run_id: str) -> dict[str, Any]:
        """Load one stored experiment summary JSON."""
        path = self.summary_path(run_id)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            raise ExperimentStoreError(f"Failed to read run summary from {path}: {exc}") from exc

    def write_top_features(self, run_id: str, df: pd.DataFrame) -> Path:
        """Persist top feature-importance rows for one run."""
        path = self.top_features_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(path, index=False, engine="pyarrow")
        except Exception as exc:
            raise ExperimentStoreError(f"Failed to write top features to {path}: {exc}") from exc
        return path

    def read_top_features(self, run_id: str) -> pd.DataFrame:
        """Load stored top feature-importance rows for one run."""
        path = self.top_features_path(run_id)
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            raise ExperimentStoreError(f"Failed to read top features from {path}: {exc}") from exc
