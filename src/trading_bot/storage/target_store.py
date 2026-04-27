"""Filesystem-backed persistence for fee-aware target comparison artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class TargetStore:
    """Persist fee-aware target comparison tables and reports locally."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def report_dir(self, comparison_id: str) -> Path:
        """Return the output directory for one target-comparison run."""
        return self.base_dir / f"target_comparison_id={comparison_id}"

    def write_comparison_table(self, comparison_id: str, df: pd.DataFrame) -> Path:
        """Persist the machine-loadable target comparison table."""
        path = self.report_dir(comparison_id) / "target_comparison.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, engine="pyarrow")
        return path

    def read_comparison_table(self, comparison_id: str) -> pd.DataFrame:
        """Load a stored target comparison table."""
        path = self.report_dir(comparison_id) / "target_comparison.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def write_report_json(self, comparison_id: str, payload: dict[str, Any]) -> Path:
        """Persist the machine-loadable target comparison report."""
        path = self.report_dir(comparison_id) / "target_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return path

    def read_report_json(self, comparison_id: str) -> dict[str, Any]:
        """Load a stored target comparison report."""
        path = self.report_dir(comparison_id) / "target_report.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def write_report_markdown(self, comparison_id: str, content: str) -> Path:
        """Persist the human-readable target comparison report."""
        path = self.report_dir(comparison_id) / "target_report.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def list_reports(self) -> list[dict[str, Any]]:
        """List stored target comparison reports."""
        if not self.base_dir.exists():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(self.base_dir.glob("target_comparison_id=*/target_report.json")):
            payload = json.loads(path.read_text())
            payload["report_path"] = str(path)
            rows.append(payload)
        return rows
