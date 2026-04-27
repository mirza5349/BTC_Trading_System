"""Filesystem-backed storage for scheduled offline paper-trading loops."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class PaperLoopStore:
    """Persist one ongoing paper-loop account and its rolling artifacts."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def loop_dir(self, loop_id: str) -> Path:
        """Return the root directory for one paper loop."""
        return self.base_dir / f"loop_id={loop_id}"

    def write_json(self, loop_id: str, filename: str, payload: dict[str, Any]) -> Path:
        """Persist a JSON payload under the loop directory."""
        path = self.loop_dir(loop_id) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return path

    def read_json(self, loop_id: str, filename: str) -> dict[str, Any]:
        """Load a JSON payload when present."""
        path = self.loop_dir(loop_id) / filename
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def write_table(self, loop_id: str, filename: str, df: pd.DataFrame) -> Path:
        """Persist a Parquet table under the loop directory."""
        path = self.loop_dir(loop_id) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, engine="pyarrow")
        return path

    def read_table(self, loop_id: str, filename: str) -> pd.DataFrame:
        """Load a Parquet table when present."""
        path = self.loop_dir(loop_id) / filename
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def write_config(self, loop_id: str, payload: dict[str, Any]) -> Path:
        """Persist the selected loop configuration."""
        return self.write_json(loop_id, "config.json", payload)

    def read_config(self, loop_id: str) -> dict[str, Any]:
        """Load stored loop configuration."""
        return self.read_json(loop_id, "config.json")

    def write_status(self, loop_id: str, payload: dict[str, Any]) -> Path:
        """Persist the latest runtime status."""
        return self.write_json(loop_id, "status.json", payload)

    def read_status(self, loop_id: str) -> dict[str, Any]:
        """Load the latest runtime status."""
        return self.read_json(loop_id, "status.json")

    def write_summary(self, loop_id: str, payload: dict[str, Any]) -> Path:
        """Persist the latest account summary."""
        return self.write_json(loop_id, "summary.json", payload)

    def read_summary(self, loop_id: str) -> dict[str, Any]:
        """Load the latest account summary."""
        return self.read_json(loop_id, "summary.json")

    def write_predictions(self, loop_id: str, df: pd.DataFrame) -> Path:
        """Persist the accumulated prediction log."""
        return self.write_table(loop_id, "predictions.parquet", df)

    def read_predictions(self, loop_id: str) -> pd.DataFrame:
        """Load the accumulated prediction log."""
        return self.read_table(loop_id, "predictions.parquet")

    def write_signals(self, loop_id: str, df: pd.DataFrame) -> Path:
        """Persist generated strategy signals."""
        return self.write_table(loop_id, "signals.parquet", df)

    def read_signals(self, loop_id: str) -> pd.DataFrame:
        """Load stored signals."""
        return self.read_table(loop_id, "signals.parquet")

    def write_trades(self, loop_id: str, df: pd.DataFrame) -> Path:
        """Persist the closed-trade log."""
        return self.write_table(loop_id, "trades.parquet", df)

    def read_trades(self, loop_id: str) -> pd.DataFrame:
        """Load the closed-trade log."""
        return self.read_table(loop_id, "trades.parquet")

    def write_orders(self, loop_id: str, df: pd.DataFrame) -> Path:
        """Persist the simulated order log."""
        return self.write_table(loop_id, "orders.parquet", df)

    def read_orders(self, loop_id: str) -> pd.DataFrame:
        """Load the simulated order log."""
        return self.read_table(loop_id, "orders.parquet")

    def write_positions(self, loop_id: str, df: pd.DataFrame) -> Path:
        """Persist position snapshots."""
        return self.write_table(loop_id, "positions.parquet", df)

    def read_positions(self, loop_id: str) -> pd.DataFrame:
        """Load position snapshots."""
        return self.read_table(loop_id, "positions.parquet")

    def write_equity_curve(self, loop_id: str, df: pd.DataFrame) -> Path:
        """Persist the full equity curve."""
        return self.write_table(loop_id, "equity_curve.parquet", df)

    def read_equity_curve(self, loop_id: str) -> pd.DataFrame:
        """Load the full equity curve."""
        return self.read_table(loop_id, "equity_curve.parquet")

    def list_statuses(self) -> list[dict[str, Any]]:
        """List stored loop status payloads."""
        if not self.base_dir.exists():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(self.base_dir.glob("loop_id=*/status.json")):
            payload = json.loads(path.read_text())
            payload["status_path"] = str(path)
            rows.append(payload)
        return rows
