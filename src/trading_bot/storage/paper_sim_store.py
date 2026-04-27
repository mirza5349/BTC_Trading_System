"""Filesystem-backed storage for offline paper-simulation artifacts."""

from __future__ import annotations

from trading_bot.storage.backtest_store import _BaseSimulationStore


class PaperSimulationStore(_BaseSimulationStore):
    """Persist offline paper-simulation outputs under a separate directory."""
