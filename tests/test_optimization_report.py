"""Tests for optimization reporting and persisted artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trading_bot.optimization.reporting import format_optimization_report_markdown
from trading_bot.storage.optimization_store import OptimizationStore


def test_format_optimization_report_markdown_renders_recommendation() -> None:
    """Markdown reporting should include the recommendation and best candidate."""
    report = {
        "optimization_id": "OPT001",
        "model_run_id": "RUN001",
        "candidates_tested": 12,
        "recommendation": {
            "optimized_has_edge": False,
            "reason": "Fee drag remains too high relative to starting cash.",
        },
        "best_candidate": {
            "candidate_id": "C0001",
            "total_return": -0.01,
        },
        "top_candidates": [{"candidate_id": "C0001", "total_return": -0.01, "number_of_trades": 120}],
    }

    markdown = format_optimization_report_markdown(report)

    assert "OPT001" in markdown
    assert "optimized_has_edge" in markdown
    assert "C0001" in markdown


def test_optimization_store_round_trips_report_and_tables(tmp_path: Path) -> None:
    """Optimization storage should persist reports and compact candidate tables."""
    store = OptimizationStore(tmp_path / "optimization")
    candidate_df = pd.DataFrame([{"candidate_id": "C0001", "total_return": 0.05}])
    report = {
        "optimization_id": "OPT001",
        "model_run_id": "RUN001",
        "generated_at": "2024-01-01T00:00:00+00:00",
        "candidates_tested": 1,
    }

    store.write_candidate_results("RUN001", "OPT001", candidate_df)
    store.write_report_json("RUN001", "OPT001", report)

    loaded_df = store.read_candidate_results("RUN001", "OPT001")
    loaded_report = store.read_report_json("OPT001")

    assert loaded_df.iloc[0]["candidate_id"] == "C0001"
    assert loaded_report["model_run_id"] == "RUN001"
    assert store.find_model_run_id("OPT001") == "RUN001"
