"""Helpers for lightweight run promotion and status tagging."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from trading_bot.schemas.experiments import validate_run_status


def tag_run_status(
    *,
    summary: dict[str, Any],
    status: str,
    note: str | None = None,
) -> dict[str, Any]:
    """Return a summary payload updated with a promotion decision."""
    resolved_status = validate_run_status(status)
    updated = dict(summary)
    updated["status"] = resolved_status
    if note is not None:
        updated["notes"] = note
    updated["status_updated_at"] = datetime.now(timezone.utc).isoformat()
    return updated
