"""Schema contracts and helpers for local news NLP enrichment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

import pandas as pd

TEXT_MODE_VALUES: tuple[str, ...] = ("title_only", "title_plus_summary")
TextMode = Literal["title_only", "title_plus_summary"]

SENTIMENT_LABEL_VALUES: tuple[str, ...] = ("positive", "neutral", "negative")

REQUIRED_ENRICHED_NEWS_COLUMNS: list[str] = [
    "asset",
    "provider",
    "published_at",
    "title",
    "url",
    "source_name",
    "sentiment_label",
    "prob_positive",
    "prob_neutral",
    "prob_negative",
    "sentiment_score",
    "enrichment_version",
    "model_name",
    "requested_device",
    "effective_device",
]

RECOMMENDED_ENRICHED_NEWS_COLUMNS: list[str] = [
    "provider_article_id",
    "summary",
    "text_mode",
    "scored_text",
    "scored_text_length",
    "ingested_at",
    "enriched_at",
    "enrichment_key",
    "prob_confidence",
]

ALL_ENRICHED_NEWS_COLUMNS: list[str] = (
    REQUIRED_ENRICHED_NEWS_COLUMNS + RECOMMENDED_ENRICHED_NEWS_COLUMNS
)


@dataclass(frozen=True)
class SentimentPrediction:
    """Normalized sentiment output for one text input."""

    sentiment_label: str
    prob_positive: float
    prob_neutral: float
    prob_negative: float
    sentiment_score: float

    @property
    def confidence(self) -> float:
        """Return the maximum class probability."""
        return max(self.prob_positive, self.prob_neutral, self.prob_negative)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "sentiment_label": self.sentiment_label,
            "prob_positive": self.prob_positive,
            "prob_neutral": self.prob_neutral,
            "prob_negative": self.prob_negative,
            "sentiment_score": self.sentiment_score,
            "prob_confidence": self.confidence,
        }


def validate_text_mode(text_mode: str) -> TextMode:
    """Validate the configured text assembly mode."""
    if text_mode not in TEXT_MODE_VALUES:
        raise ValueError(f"Unsupported text_mode {text_mode!r}. Expected one of {TEXT_MODE_VALUES}.")
    return text_mode


def normalize_sentiment_label(label: str) -> str:
    """Normalize common label spellings to stable output values."""
    normalized = str(label).strip().lower()
    alias_map = {
        "pos": "positive",
        "positive": "positive",
        "label_2": "positive",
        "neu": "neutral",
        "neutral": "neutral",
        "label_1": "neutral",
        "neg": "negative",
        "negative": "negative",
        "label_0": "negative",
    }
    if normalized in alias_map:
        return alias_map[normalized]
    raise ValueError(f"Unsupported sentiment label {label!r}")


def build_enrichment_key(
    *,
    provider: str,
    provider_article_id: str | None,
    url: str,
    published_at: Any,
    enrichment_version: str,
) -> str:
    """Return a stable enrichment key for idempotent local scoring."""
    provider_article_id = str(provider_article_id or "").strip()
    if provider_article_id:
        return f"{provider}::{provider_article_id}::{enrichment_version}"

    normalized_url = normalize_url(url)
    timestamp = pd.Timestamp(published_at)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return f"{provider}::{normalized_url}::{timestamp.isoformat()}::{enrichment_version}"


def normalize_url(url: str) -> str:
    """Return a deterministic lowercase URL form for fallback deduplication."""
    parsed = urlsplit(str(url).strip())
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    query = parsed.query
    normalized = f"{scheme}://{netloc}{path}"
    if query:
        normalized = f"{normalized}?{query}"
    return normalized


def validate_enriched_news_df(df: pd.DataFrame) -> None:
    """Raise when enriched news is missing required persisted columns."""
    missing_columns = [column for column in REQUIRED_ENRICHED_NEWS_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"enriched news data is missing required columns: {missing_columns}")
