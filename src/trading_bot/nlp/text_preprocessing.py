"""Deterministic text preprocessing for local news sentiment scoring."""

from __future__ import annotations

from typing import Any

import pandas as pd

from trading_bot.schemas.nlp import TextMode, validate_text_mode


def normalize_text(text: Any) -> str:
    """Return a whitespace-normalized text string."""
    if text is None:
        return ""
    normalized = " ".join(str(text).split())
    return normalized.strip()


def assemble_news_text(
    row: pd.Series | dict[str, Any],
    *,
    text_mode: str,
    max_length: int = 256,
) -> str:
    """Build a deterministic model input string from a news row."""
    resolved_mode: TextMode = validate_text_mode(text_mode)
    title = normalize_text(_value(row, "title"))
    summary = normalize_text(_value(row, "summary"))

    if resolved_mode == "title_only":
        combined = title
    else:
        combined = " ".join(part for part in (title, summary) if part)

    if not combined:
        return ""

    if max_length <= 0:
        return combined

    truncated = combined[:max_length].rstrip()
    return truncated


def _value(row: pd.Series | dict[str, Any], key: str) -> Any:
    """Read a key from either a series or a dict-like row."""
    if isinstance(row, pd.Series):
        return row.get(key)
    return row.get(key)
