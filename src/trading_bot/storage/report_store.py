"""Filesystem-backed persistence for generated evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from trading_bot.logging_config import get_logger

logger = get_logger(__name__)


class ReportStoreError(Exception):
    """Raised when generated reports cannot be persisted or loaded."""


class ReportStore:
    """Persist machine-readable and human-readable experiment reports."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def run_dir(self, run_id: str) -> Path:
        """Return the report directory for one run."""
        return self.base_dir / f"run_id={run_id}"

    def report_json_path(self, run_id: str) -> Path:
        """Return the JSON report path for one run."""
        return self.run_dir(run_id) / "evaluation_report.json"

    def report_markdown_path(self, run_id: str) -> Path:
        """Return the Markdown report path for one run."""
        return self.run_dir(run_id) / "evaluation_report.md"

    def write_json_report(self, run_id: str, payload: dict[str, Any]) -> Path:
        """Persist a JSON evaluation report."""
        path = self.report_json_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        except Exception as exc:
            raise ReportStoreError(f"Failed to write JSON report to {path}: {exc}") from exc
        return path

    def read_json_report(self, run_id: str) -> dict[str, Any]:
        """Load a stored JSON evaluation report."""
        path = self.report_json_path(run_id)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            raise ReportStoreError(f"Failed to read JSON report from {path}: {exc}") from exc

    def write_markdown_report(self, run_id: str, content: str) -> Path:
        """Persist a Markdown evaluation report."""
        path = self.report_markdown_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(content)
        except Exception as exc:
            raise ReportStoreError(
                f"Failed to write Markdown report to {path}: {exc}"
            ) from exc
        return path

    def read_markdown_report(self, run_id: str) -> str:
        """Load a stored Markdown evaluation report."""
        path = self.report_markdown_path(run_id)
        if not path.exists():
            return ""
        try:
            return path.read_text()
        except Exception as exc:
            raise ReportStoreError(
                f"Failed to read Markdown report from {path}: {exc}"
            ) from exc

    def write_comparison_table(self, name: str, df: pd.DataFrame) -> Path:
        """Persist a run-comparison table as Parquet."""
        path = self.base_dir / "comparisons" / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(path, index=False, engine="pyarrow")
        except Exception as exc:
            raise ReportStoreError(
                f"Failed to write comparison table to {path}: {exc}"
            ) from exc
        logger.info("comparison_table_write_complete", path=str(path), rows=len(df))
        return path

    def write_comparison_json(self, name: str, payload: dict[str, Any]) -> Path:
        """Persist a machine-loadable comparison or ablation JSON payload."""
        path = self.base_dir / "comparisons" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        except Exception as exc:
            raise ReportStoreError(
                f"Failed to write comparison JSON to {path}: {exc}"
            ) from exc
        logger.info("comparison_json_write_complete", path=str(path))
        return path
