"""Reporting helpers for strategy-threshold optimization."""

from __future__ import annotations

from typing import Any


def format_optimization_report_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown optimization report."""
    recommendation = report.get("recommendation", {})
    lines = [
        f"# Optimization Report: {report.get('optimization_id', 'unknown')}",
        "",
        "## Overview",
        f"- Model run id: `{report.get('model_run_id')}`",
        f"- Candidates tested: `{report.get('candidates_tested')}`",
        "",
        "## Recommendation",
        f"- optimized_has_edge: `{recommendation.get('optimized_has_edge')}`",
        f"- reason: {recommendation.get('reason')}",
        "",
        "## Best Candidate",
    ]
    best_candidate = report.get("best_candidate", {})
    if isinstance(best_candidate, dict) and best_candidate:
        for key, value in best_candidate.items():
            lines.append(f"- {key}: `{value}`")
    else:
        lines.append("- No eligible candidate found")
    lines.extend(["", "## Top Candidates"])
    for row in report.get("top_candidates", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- candidate_id={row.get('candidate_id')} total_return={row.get('total_return')} "
            f"trades={row.get('number_of_trades')} drawdown={row.get('max_drawdown')}"
        )
    return "\n".join(lines) + "\n"
