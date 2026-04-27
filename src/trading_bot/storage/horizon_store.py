"""Filesystem-backed persistence for multi-horizon comparison artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class HorizonStore:
    """Persist multi-horizon comparison tables and reports locally."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def report_dir(self, comparison_id: str) -> Path:
        """Return the output directory for one horizon-comparison run."""
        return self.base_dir / f"horizon_comparison_id={comparison_id}"

    def write_comparison_table(self, comparison_id: str, df: pd.DataFrame) -> Path:
        """Persist the machine-loadable horizon comparison table."""
        path = self.report_dir(comparison_id) / "horizon_comparison.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, engine="pyarrow")
        return path

    def read_comparison_table(self, comparison_id: str) -> pd.DataFrame:
        """Load a stored horizon comparison table."""
        path = self.report_dir(comparison_id) / "horizon_comparison.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def write_report_json(self, comparison_id: str, payload: dict[str, Any]) -> Path:
        """Persist the machine-loadable horizon comparison report."""
        path = self.report_dir(comparison_id) / "horizon_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return path

    def read_report_json(self, comparison_id: str) -> dict[str, Any]:
        """Load a stored horizon comparison report."""
        path = self.report_dir(comparison_id) / "horizon_report.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def write_report_markdown(self, comparison_id: str, content: str) -> Path:
        """Persist the human-readable horizon comparison report."""
        path = self.report_dir(comparison_id) / "horizon_report.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def list_reports(self) -> list[dict[str, Any]]:
        """List stored horizon comparison reports."""
        if not self.base_dir.exists():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(self.base_dir.glob("horizon_comparison_id=*/horizon_report.json")):
            payload = json.loads(path.read_text())
            payload["report_path"] = str(path)
            rows.append(payload)
        return rows

    def latest_report(
        self,
        *,
        symbol: str,
        timeframe: str,
    ) -> dict[str, Any]:
        """Return the most recent stored report for one symbol/timeframe pair."""
        matching_reports = [
            report
            for report in self.list_reports()
            if str(report.get("symbol")) == symbol and str(report.get("timeframe")) == timeframe
        ]
        if not matching_reports:
            return {}
        matching_reports.sort(key=lambda report: str(report.get("generated_at", "")), reverse=True)
        return matching_reports[0]
