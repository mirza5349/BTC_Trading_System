"""Filesystem-backed persistence for strategy-search artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class OptimizationStore:
    """Persist compact strategy-search outputs under model-run and optimization ids."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def optimization_dir(self, model_run_id: str, optimization_id: str) -> Path:
        """Return the output directory for one strategy-search run."""
        return (
            self.base_dir
            / f"model_run_id={model_run_id}"
            / f"optimization_id={optimization_id}"
        )

    def write_candidate_results(
        self,
        model_run_id: str,
        optimization_id: str,
        df: pd.DataFrame,
    ) -> Path:
        """Persist all candidate result rows as Parquet."""
        return self._write_table(model_run_id, optimization_id, "candidate_results.parquet", df)

    def read_candidate_results(self, model_run_id: str, optimization_id: str) -> pd.DataFrame:
        """Load stored candidate results."""
        return self._read_table(model_run_id, optimization_id, "candidate_results.parquet")

    def write_top_candidates(
        self,
        model_run_id: str,
        optimization_id: str,
        df: pd.DataFrame,
    ) -> Path:
        """Persist the top-k candidate table as Parquet."""
        return self._write_table(model_run_id, optimization_id, "top_candidates.parquet", df)

    def read_top_candidates(self, model_run_id: str, optimization_id: str) -> pd.DataFrame:
        """Load the top-k candidate table."""
        return self._read_table(model_run_id, optimization_id, "top_candidates.parquet")

    def write_baseline_comparison(
        self,
        model_run_id: str,
        optimization_id: str,
        df: pd.DataFrame,
    ) -> Path:
        """Persist the optimized-vs-baselines comparison table."""
        return self._write_table(model_run_id, optimization_id, "baseline_comparison.parquet", df)

    def read_baseline_comparison(self, model_run_id: str, optimization_id: str) -> pd.DataFrame:
        """Load the optimized-vs-baselines comparison table."""
        return self._read_table(model_run_id, optimization_id, "baseline_comparison.parquet")

    def write_report_json(
        self,
        model_run_id: str,
        optimization_id: str,
        payload: dict[str, Any],
    ) -> Path:
        """Persist the machine-loadable optimization report."""
        path = self.optimization_dir(model_run_id, optimization_id) / "optimization_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return path

    def read_report_json(self, optimization_id: str) -> dict[str, Any]:
        """Load an optimization report by scanning for the optimization id."""
        path = self.find_report_path(optimization_id)
        if path is None or not path.exists():
            return {}
        return json.loads(path.read_text())

    def write_report_markdown(
        self,
        model_run_id: str,
        optimization_id: str,
        content: str,
    ) -> Path:
        """Persist a Markdown optimization report."""
        path = self.optimization_dir(model_run_id, optimization_id) / "optimization_report.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def list_reports(self) -> list[dict[str, Any]]:
        """List stored optimization reports."""
        if not self.base_dir.exists():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(self.base_dir.glob("model_run_id=*/optimization_id=*/optimization_report.json")):
            payload = json.loads(path.read_text())
            payload["report_path"] = str(path)
            rows.append(payload)
        return rows

    def find_model_run_id(self, optimization_id: str) -> str | None:
        """Return the stored model run id for one optimization id."""
        path = self.find_report_path(optimization_id)
        if path is None:
            return None
        return path.parents[1].name.split("=", 1)[1]

    def find_report_path(self, optimization_id: str) -> Path | None:
        """Return the JSON report path for one optimization id when present."""
        matches = list(
            self.base_dir.glob(
                f"model_run_id=*/optimization_id={optimization_id}/optimization_report.json"
            )
        )
        return matches[0] if matches else None

    def _write_table(
        self,
        model_run_id: str,
        optimization_id: str,
        filename: str,
        df: pd.DataFrame,
    ) -> Path:
        """Persist a table artifact as Parquet."""
        path = self.optimization_dir(model_run_id, optimization_id) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, engine="pyarrow")
        return path

    def _read_table(self, model_run_id: str, optimization_id: str, filename: str) -> pd.DataFrame:
        """Load a stored Parquet table when present."""
        path = self.optimization_dir(model_run_id, optimization_id) / filename
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)
