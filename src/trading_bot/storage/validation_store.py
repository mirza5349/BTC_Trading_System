"""Filesystem-backed persistence for validation-suite reports and tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class ValidationStoreError(Exception):
    """Raised when validation artifacts cannot be persisted or loaded."""


class ValidationStore:
    """Persist validation reports and tabular diagnostics to local storage."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def report_dir(self, validation_id: str) -> Path:
        """Return the report directory for one validation suite run."""
        return self.base_dir / f"validation_id={validation_id}"

    def write_report_json(self, validation_id: str, payload: dict[str, Any]) -> Path:
        """Persist a machine-loadable validation report."""
        path = self.report_dir(validation_id) / "validation_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return path

    def read_report_json(self, validation_id: str) -> dict[str, Any]:
        """Load a persisted validation report."""
        path = self.report_dir(validation_id) / "validation_report.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def write_report_markdown(self, validation_id: str, content: str) -> Path:
        """Persist a Markdown validation report."""
        path = self.report_dir(validation_id) / "validation_report.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def write_table(self, validation_id: str, name: str, df: pd.DataFrame) -> Path:
        """Persist a table under a validation report directory."""
        path = self.report_dir(validation_id) / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, engine="pyarrow")
        return path

    def read_table(self, validation_id: str, name: str) -> pd.DataFrame:
        """Load a persisted validation table."""
        path = self.report_dir(validation_id) / f"{name}.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def list_reports(self) -> list[dict[str, Any]]:
        """List stored validation reports."""
        if not self.base_dir.exists():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(self.base_dir.glob("validation_id=*/validation_report.json")):
            payload = json.loads(path.read_text())
            payload["report_path"] = str(path)
            rows.append(payload)
        return rows
