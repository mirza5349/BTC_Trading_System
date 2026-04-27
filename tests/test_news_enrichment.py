"""Tests for idempotent local news enrichment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_bot.nlp.finbert_service import FinBERTService
from trading_bot.nlp.news_enrichment import NewsEnrichmentService
from trading_bot.storage.enriched_news_store import EnrichedNewsStore
from trading_bot.storage.news_store import NewsStore


class DummyFinBERTService(FinBERTService):
    """Deterministic fake FinBERT service for enrichment tests."""

    def __init__(self) -> None:
        super().__init__(
            model_name="ProsusAI/finbert",
            batch_size=16,
            requested_device="cpu",
            allow_cuda_fallback_to_cpu=True,
            max_length=256,
        )
        self._effective_device = "cpu"

    def score_texts(self, texts: list[str]):
        predictions = []
        for text in texts:
            positive = 0.9 if "bullish" in text.lower() else 0.2
            negative = 0.1 if "bullish" in text.lower() else 0.6
            neutral = 0.0 if "bullish" in text.lower() else 0.2
            predictions.append(
                type(
                    "Prediction",
                    (),
                    {
                        "to_dict": lambda self, positive=positive, neutral=neutral, negative=negative: {
                            "sentiment_label": "positive" if positive > negative else "negative",
                            "prob_positive": positive,
                            "prob_neutral": neutral,
                            "prob_negative": negative,
                            "sentiment_score": positive - negative,
                            "prob_confidence": max(positive, neutral, negative),
                        }
                    },
                )()
            )
        return predictions


def _make_news_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asset": ["BTC", "BTC"],
            "provider": ["cryptocompare", "cryptocompare"],
            "published_at": pd.to_datetime(
                ["2024-01-01 00:10:00+00:00", "2024-01-01 00:20:00+00:00"],
                utc=True,
            ),
            "title": ["Bullish ETF headline", "Risk-off miner concern"],
            "summary": ["bullish catalyst", ""],
            "url": ["https://example.com/a", "https://example.com/b"],
            "source_name": ["source_a", "source_b"],
            "provider_article_id": ["1", "2"],
            "ingested_at": pd.to_datetime(
                ["2024-01-01 00:11:00+00:00", "2024-01-01 00:21:00+00:00"],
                utc=True,
            ),
        }
    )


def test_news_enrichment_is_idempotent_and_persists_sentiment(tmp_path: Path) -> None:
    """Repeated enrichment runs should not duplicate stored enriched records."""
    news_store = NewsStore(tmp_path / "news")
    enriched_store = EnrichedNewsStore(tmp_path / "enriched_news")
    news_store.write(_make_news_df(), "BTC", "cryptocompare")

    service = NewsEnrichmentService(
        news_store=news_store,
        enriched_news_store=enriched_store,
        finbert_service=DummyFinBERTService(),
        default_asset="BTC",
        default_provider="cryptocompare",
        default_text_mode="title_plus_summary",
        default_enrichment_version="v1",
    )

    first_result = service.enrich_news()
    second_result = service.enrich_news()

    enriched_df = enriched_store.read(
        asset="BTC",
        provider="cryptocompare",
        model_name="ProsusAI/finbert",
        enrichment_version="v1",
    )

    assert first_result.scored_rows == 2
    assert second_result.scored_rows == 0
    assert second_result.skipped_existing_rows == 2
    assert len(enriched_df) == 2
    assert "sentiment_label" in enriched_df.columns
    assert enriched_df["enrichment_key"].nunique() == 2
