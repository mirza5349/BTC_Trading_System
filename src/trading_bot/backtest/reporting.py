"""Reporting helpers for backtests and offline paper simulations."""

from __future__ import annotations

from typing import Any

from trading_bot.schemas.backtest import BacktestResult


def build_simulation_summary_payload(result: BacktestResult) -> dict[str, Any]:
    """Return a machine-loadable summary payload for a completed simulation."""
    return result.to_dict()


def build_simulation_markdown_report(result: BacktestResult) -> str:
    """Return a compact Markdown report for a completed simulation."""
    metrics = result.metrics.to_dict()
    strategy = result.config.strategy.to_dict()
    execution = result.config.execution.to_dict()

    lines = [
        f"# Simulation Report: {result.simulation_id}",
        "",
        "## Overview",
        f"- Simulation type: `{result.simulation_type}`",
        f"- Model run id: `{result.model_run_id}`",
        f"- Symbol: `{result.symbol}`",
        f"- Timeframe: `{result.timeframe}`",
        f"- Date range: `{result.start_timestamp}` to `{result.end_timestamp}`",
        "",
        "## Strategy",
        *(f"- {key}: `{value}`" for key, value in strategy.items()),
        "",
        "## Execution",
        *(f"- {key}: `{value}`" for key, value in execution.items()),
        "",
        "## Metrics",
        *(f"- {key}: `{value}`" for key, value in metrics.items()),
        "",
        "## Artifacts",
        *(f"- {key}: `{value}`" for key, value in result.artifact_paths.items()),
    ]
    return "\n".join(lines) + "\n"
