"""Local BTC news enrichment pipeline using FinBERT sentiment scoring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from trading_bot.logging_config import get_logger
from trading_bot.nlp.finbert_service import FinBERTService
from trading_bot.nlp.text_preprocessing import assemble_news_text
from trading_bot.schemas.nlp import build_enrichment_key
from trading_bot.settings import PROJECT_ROOT, AppSettings, load_settings
from trading_bot.storage.enriched_news_store import EnrichedNewsStore
from trading_bot.storage.news_store import NewsStore

logger = get_logger(__name__)


@dataclass(frozen=True)
class NewsEnrichmentResult:
    """Summary of one local news enrichment run."""

    asset: str
    provider: str
    model_name: str
    enrichment_version: str
    requested_device: str
    effective_device: str
    input_rows: int
    scored_rows: int
    skipped_existing_rows: int
    skipped_empty_text_rows: int
    failure_rows: int
    stored_rows: int


class NewsEnrichmentService:
    """Read stored news, enrich with FinBERT, and persist idempotent local outputs."""

    def __init__(
        self,
        *,
        news_store: NewsStore,
        enriched_news_store: EnrichedNewsStore,
        finbert_service: FinBERTService,
        default_asset: str,
        default_provider: str,
        default_text_mode: str,
        default_enrichment_version: str,
    ) -> None:
        self.news_store = news_store
        self.enriched_news_store = enriched_news_store
        self.finbert_service = finbert_service
        self.default_asset = default_asset
        self.default_provider = default_provider
        self.default_text_mode = default_text_mode
        self.default_enrichment_version = default_enrichment_version

    @classmethod
    def from_settings(cls, settings: AppSettings | None = None) -> NewsEnrichmentService:
        """Build the enrichment service from project settings."""
        resolved_settings = settings or load_settings()
        return cls(
            news_store=NewsStore(PROJECT_ROOT / resolved_settings.news_data.processed_data_dir),
            enriched_news_store=EnrichedNewsStore(PROJECT_ROOT / resolved_settings.storage.enriched_news_dir),
            finbert_service=FinBERTService(
                model_name=resolved_settings.nlp.model_name,
                batch_size=resolved_settings.nlp.batch_size,
                requested_device=resolved_settings.nlp.device,
                allow_cuda_fallback_to_cpu=resolved_settings.nlp.allow_cuda_fallback_to_cpu,
                max_length=resolved_settings.nlp.max_length,
            ),
            default_asset=resolved_settings.news_data.asset,
            default_provider=resolved_settings.news_data.provider,
            default_text_mode=resolved_settings.nlp.text_mode,
            default_enrichment_version=resolved_settings.nlp.enrichment_version,
        )

    def enrich_news(
        self,
        *,
        asset: str | None = None,
        provider: str | None = None,
        enrichment_version: str | None = None,
        text_mode: str | None = None,
    ) -> NewsEnrichmentResult:
        """Score stored news locally and persist enriched rows idempotently."""
        resolved_asset = asset or self.default_asset
        resolved_provider = provider or self.default_provider
        resolved_version = enrichment_version or self.default_enrichment_version
        resolved_text_mode = text_mode or self.default_text_mode

        news_df = self.news_store.read(resolved_asset, resolved_provider)
        existing_df = self.enriched_news_store.read(
            asset=resolved_asset,
            provider=resolved_provider,
            model_name=self.finbert_service.model_name,
            enrichment_version=resolved_version,
        )

        if news_df.empty:
            return NewsEnrichmentResult(
                asset=resolved_asset,
                provider=resolved_provider,
                model_name=self.finbert_service.model_name,
                enrichment_version=resolved_version,
                requested_device=self.finbert_service.requested_device,
                effective_device=self.finbert_service.effective_device,
                input_rows=0,
                scored_rows=0,
                skipped_existing_rows=0,
                skipped_empty_text_rows=0,
                failure_rows=0,
                stored_rows=0,
            )

        working_df = news_df.copy()
        working_df["published_at"] = pd.to_datetime(working_df["published_at"], utc=True)
        working_df["enrichment_key"] = working_df.apply(
            lambda row: build_enrichment_key(
                provider=str(row.get("provider", resolved_provider)),
                provider_article_id=str(row.get("provider_article_id", "")),
                url=str(row.get("url", "")),
                published_at=row.get("published_at"),
                enrichment_version=resolved_version,
            ),
            axis=1,
        )

        existing_keys = set(existing_df.get("enrichment_key", pd.Series(dtype=str)).astype(str).tolist())
        pending_df = working_df[~working_df["enrichment_key"].isin(existing_keys)].copy()
        skipped_existing_rows = int(len(working_df) - len(pending_df))

        pending_df["scored_text"] = pending_df.apply(
            lambda row: assemble_news_text(
                row,
                text_mode=resolved_text_mode,
                max_length=self.finbert_service.max_length * 4,
            ),
            axis=1,
        )
        pending_df["scored_text_length"] = pending_df["scored_text"].astype(str).str.len()

        empty_text_mask = pending_df["scored_text"].astype(str).str.strip() == ""
        skipped_empty_text_rows = int(empty_text_mask.sum())
        empty_text_df = pending_df[empty_text_mask].copy()
        scored_candidates_df = pending_df[~empty_text_mask].copy()

        scored_rows = 0
        failure_rows = 0
        enriched_rows: list[dict[str, object]] = []

        if not scored_candidates_df.empty:
            predictions = self.finbert_service.score_texts(
                scored_candidates_df["scored_text"].astype(str).tolist()
            )
            scored_rows = len(predictions)

            for (_, row), prediction in zip(
                scored_candidates_df.iterrows(),
                predictions,
                strict=False,
            ):
                enriched_rows.append(
                    _build_enriched_row(
                        row=row,
                        prediction=prediction.to_dict(),
                        enrichment_version=resolved_version,
                        model_name=self.finbert_service.model_name,
                        text_mode=resolved_text_mode,
                        requested_device=self.finbert_service.requested_device,
                        effective_device=self.finbert_service.effective_device,
                    )
                )

        for _, row in empty_text_df.iterrows():
            enriched_rows.append(
                _build_enriched_row(
                    row=row,
                    prediction={},
                    enrichment_version=resolved_version,
                    model_name=self.finbert_service.model_name,
                    text_mode=resolved_text_mode,
                    requested_device=self.finbert_service.requested_device,
                    effective_device=self.finbert_service.effective_device,
                )
            )

        enriched_df = pd.DataFrame(enriched_rows)
        stored_rows = self.enriched_news_store.write(
            enriched_df,
            asset=resolved_asset,
            provider=resolved_provider,
            model_name=self.finbert_service.model_name,
            enrichment_version=resolved_version,
        ) if not enriched_df.empty else len(existing_df)

        logger.info(
            "news_enrichment_complete",
            asset=resolved_asset,
            provider=resolved_provider,
            enrichment_version=resolved_version,
            input_rows=len(news_df),
            scored_rows=scored_rows,
            skipped_existing_rows=skipped_existing_rows,
            skipped_empty_text_rows=skipped_empty_text_rows,
            failure_rows=failure_rows,
            stored_rows=stored_rows,
        )

        return NewsEnrichmentResult(
            asset=resolved_asset,
            provider=resolved_provider,
            model_name=self.finbert_service.model_name,
            enrichment_version=resolved_version,
            requested_device=self.finbert_service.requested_device,
            effective_device=self.finbert_service.effective_device,
            input_rows=len(news_df),
            scored_rows=scored_rows,
            skipped_existing_rows=skipped_existing_rows,
            skipped_empty_text_rows=skipped_empty_text_rows,
            failure_rows=failure_rows,
            stored_rows=stored_rows,
        )


def _build_enriched_row(
    *,
    row: pd.Series,
    prediction: dict[str, object],
    enrichment_version: str,
    model_name: str,
    text_mode: str,
    requested_device: str,
    effective_device: str,
) -> dict[str, object]:
    """Return one enriched-news record preserving original news metadata."""
    enriched_at = datetime.now(timezone.utc).isoformat()
    return {
        "asset": row.get("asset"),
        "provider": row.get("provider"),
        "published_at": row.get("published_at"),
        "title": row.get("title"),
        "url": row.get("url"),
        "source_name": row.get("source_name"),
        "provider_article_id": row.get("provider_article_id", ""),
        "summary": row.get("summary", ""),
        "text_mode": text_mode,
        "scored_text": row.get("scored_text", ""),
        "scored_text_length": int(row.get("scored_text_length", 0)),
        "sentiment_label": prediction.get("sentiment_label"),
        "prob_positive": prediction.get("prob_positive"),
        "prob_neutral": prediction.get("prob_neutral"),
        "prob_negative": prediction.get("prob_negative"),
        "sentiment_score": prediction.get("sentiment_score"),
        "prob_confidence": prediction.get("prob_confidence"),
        "enrichment_version": enrichment_version,
        "model_name": model_name,
        "requested_device": requested_device,
        "effective_device": effective_device,
        "ingested_at": row.get("ingested_at"),
        "enriched_at": enriched_at,
        "enrichment_key": row.get("enrichment_key"),
    }
