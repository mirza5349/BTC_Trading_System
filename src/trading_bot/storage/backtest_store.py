"""Filesystem-backed storage for backtest artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class SimulationStoreError(Exception):
    """Raised when simulation artifacts cannot be persisted or loaded."""


class _BaseSimulationStore:
    """Common persistence helpers for offline simulation artifacts."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def simulation_dir(self, simulation_id: str) -> Path:
        """Return the output directory for one simulation."""
        return self.base_dir / f"simulation_id={simulation_id}"

    def write_summary(self, simulation_id: str, payload: dict[str, Any]) -> Path:
        """Persist summary metadata as JSON."""
        path = self.simulation_dir(simulation_id) / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return path

    def read_summary(self, simulation_id: str) -> dict[str, Any]:
        """Load a stored summary JSON payload."""
        path = self.simulation_dir(simulation_id) / "summary.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def write_equity_curve(self, simulation_id: str, df: pd.DataFrame) -> Path:
        """Persist an equity curve as Parquet."""
        return self._write_parquet(simulation_id, "equity_curve.parquet", df)

    def read_equity_curve(self, simulation_id: str) -> pd.DataFrame:
        """Load a stored equity curve."""
        return self._read_parquet(simulation_id, "equity_curve.parquet")

    def write_trades(self, simulation_id: str, df: pd.DataFrame) -> Path:
        """Persist a trade log as Parquet."""
        return self._write_parquet(simulation_id, "trades.parquet", df)

    def read_trades(self, simulation_id: str) -> pd.DataFrame:
        """Load a stored trade log."""
        return self._read_parquet(simulation_id, "trades.parquet")

    def write_signals(self, simulation_id: str, df: pd.DataFrame) -> Path:
        """Persist generated signals as Parquet."""
        return self._write_parquet(simulation_id, "signals.parquet", df)

    def read_signals(self, simulation_id: str) -> pd.DataFrame:
        """Load stored signals."""
        return self._read_parquet(simulation_id, "signals.parquet")

    def write_report(self, simulation_id: str, content: str) -> Path:
        """Persist a Markdown report."""
        path = self.simulation_dir(simulation_id) / "report.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def list_summaries(self) -> list[dict[str, Any]]:
        """List stored simulation summaries."""
        if not self.base_dir.exists():
            return []

        summaries: list[dict[str, Any]] = []
        for path in sorted(self.base_dir.glob("simulation_id=*/summary.json")):
            payload = json.loads(path.read_text())
            payload["summary_path"] = str(path)
            summaries.append(payload)
        return summaries

    def _write_parquet(self, simulation_id: str, filename: str, df: pd.DataFrame) -> Path:
        """Persist a Parquet artifact."""
        path = self.simulation_dir(simulation_id) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, engine="pyarrow")
        return path

    def _read_parquet(self, simulation_id: str, filename: str) -> pd.DataFrame:
        """Load a Parquet artifact when present."""
        path = self.simulation_dir(simulation_id) / filename
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)


class BacktestStore(_BaseSimulationStore):
    """Persist backtest outputs under the backtests directory."""
