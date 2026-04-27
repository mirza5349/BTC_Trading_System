"""News data ingestion service.

Coordinates downloading historical BTC news from exchange-neutral
news APIs, normalizing to the canonical schema, validating, filtering
for BTC relevance, and persisting to local Parquet storage.

The original abstract interfaces (NewsDataProvider, NewsArticle, etc.)
are preserved for backward compatibility and future use.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from trading_bot.ingestion.news_client import CryptoCompareNewsClient
from trading_bot.logging_config import get_logger
from trading_bot.schemas.news_data import (
    is_btc_relevant,
    normalize_news_df,
    validate_news,
)
from trading_bot.storage.news_store import NewsStore

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes (from step 1, preserved for backward compatibility)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NewsArticle:
    """A single news article or headline."""

    title: str
    source: str
    published_at: datetime
    url: str = ""
    body: str = ""
    symbol: str = "BTC"

    @property
    def has_body(self) -> bool:
        """Check if the full article body is available."""
        return bool(self.body.strip())


# ---------------------------------------------------------------------------
# Abstract provider interface (from step 1, preserved)
# ---------------------------------------------------------------------------
class NewsDataProvider(ABC):
    """Abstract base class for news data providers."""

    @abstractmethod
    async def fetch_latest_news(
        self,
        symbol: str,
        limit: int = 50,
    ) -> list[NewsArticle]:
        """Fetch the latest news articles for a symbol."""
        ...

    @abstractmethod
    async def fetch_news_range(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[NewsArticle]:
        """Fetch news articles within a date range."""
        ...


# ---------------------------------------------------------------------------
# Ingestion result
# ---------------------------------------------------------------------------
@dataclass
class NewsIngestionResult:
    """Result of a news data ingestion run."""

    asset: str
    provider: str
    fetched_rows: int = 0
    valid_rows: int = 0
    stored_rows: int = 0
    duplicates_removed: int = 0
    invalid_rows_removed: int = 0
    btc_filtered_out: int = 0
    start_timestamp: str = ""
    end_timestamp: str = ""
    is_success: bool = False
    error: str = ""


# ---------------------------------------------------------------------------
# News ingestion service
# ---------------------------------------------------------------------------
class NewsIngestionService:
    """Coordinates historical news download, validation, and storage.

    This is the main entry point for the news ingestion pipeline. It:
    1. Downloads raw news data from CryptoCompare (via CryptoCompareNewsClient)
    2. Normalizes column names and dtypes to the canonical schema
    3. Applies BTC relevance filtering
    4. Validates data quality (dedup, missing fields, timestamps)
    5. Persists to local Parquet storage with idempotent merge

    Usage::

        service = NewsIngestionService(
            store=NewsStore("data/processed/news"),
        )
        result = await service.ingest(
            asset="BTC",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 3, 1, tzinfo=timezone.utc),
        )
    """

    def __init__(
        self,
        store: NewsStore,
        client: CryptoCompareNewsClient | None = None,
        btc_keywords: list[str] | None = None,
    ) -> None:
        """Initialize the news ingestion service.

        Args:
            store: News store for persistence.
            client: CryptoCompare API client. Uses default if None.
            btc_keywords: Keywords for BTC relevance filtering. Uses defaults if None.
        """
        self.store = store
        self.client = client or CryptoCompareNewsClient()
        self.btc_keywords = btc_keywords

    async def ingest(
        self,
        asset: str = "BTC",
        provider: str = "cryptocompare",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> NewsIngestionResult:
        """Run the full news ingestion pipeline.

        Pipeline: fetch → normalize → filter → validate → store

        Args:
            asset: Asset tag (e.g., "BTC").
            provider: Data provider name.
            start: Start datetime (UTC). Defaults to 2024-01-01.
            end: End datetime (UTC). Defaults to now.

        Returns:
            NewsIngestionResult with pipeline stats.
        """
        if start is None:
            start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        if end is None:
            end = datetime.now(timezone.utc)

        logger.info(
            "news_ingestion_start",
            asset=asset,
            provider=provider,
            start=start.isoformat(),
            end=end.isoformat(),
        )

        result = NewsIngestionResult(asset=asset, provider=provider)

        try:
            # 1. Fetch raw data
            raw_df = await self.client.fetch_news(
                categories=asset,
                start_time=start,
                end_time=end,
            )

            if raw_df.empty:
                result.error = "No news data returned from provider"
                logger.warning("news_ingestion_no_data", asset=asset, provider=provider)
                return result

            result.fetched_rows = len(raw_df)

            # 2. Normalize to canonical schema
            normalized = normalize_news_df(raw_df, asset=asset, provider=provider)

            # 3. BTC relevance filter
            pre_filter_count = len(normalized)
            if asset.upper() == "BTC":
                relevance_mask = normalized.apply(
                    lambda row: is_btc_relevant(
                        f"{row.get('title', '')} {row.get('category', '')} {row.get('tags', '')}",
                        keywords=self.btc_keywords,
                    ),
                    axis=1,
                )
                normalized = normalized[relevance_mask].copy()
                result.btc_filtered_out = pre_filter_count - len(normalized)

                if result.btc_filtered_out > 0:
                    logger.info(
                        "news_btc_filter",
                        kept=len(normalized),
                        filtered_out=result.btc_filtered_out,
                    )

            if normalized.empty:
                result.error = "All articles filtered out as non-BTC"
                logger.warning("news_ingestion_all_filtered", asset=asset)
                return result

            # 4. Validate
            clean_df, validation = validate_news(normalized)

            result.valid_rows = validation.valid_rows
            result.duplicates_removed = validation.duplicate_count
            result.invalid_rows_removed = (
                validation.missing_required_count + validation.invalid_timestamp_count
            )

            if validation.errors:
                for err in validation.errors:
                    logger.warning("news_validation_issue", issue=err)

            if clean_df.empty:
                result.error = "All rows failed validation"
                logger.error("news_ingestion_all_invalid", asset=asset, provider=provider)
                return result

            # 5. Store
            total_rows = self.store.write(clean_df, asset, provider)
            result.stored_rows = total_rows
            result.start_timestamp = clean_df["published_at"].min().isoformat()
            result.end_timestamp = clean_df["published_at"].max().isoformat()
            result.is_success = True

            logger.info(
                "news_ingestion_complete",
                asset=asset,
                provider=provider,
                fetched=result.fetched_rows,
                valid=result.valid_rows,
                stored=total_rows,
                btc_filtered=result.btc_filtered_out,
            )

        except Exception as exc:
            result.error = str(exc)
            logger.error("news_ingestion_failed", error=str(exc), asset=asset)

        return result

    def ingest_sync(
        self,
        asset: str = "BTC",
        provider: str = "cryptocompare",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> NewsIngestionResult:
        """Synchronous wrapper around ingest() for CLI convenience.

        Args:
            asset: Asset tag.
            provider: Data provider name.
            start: Start datetime (UTC).
            end: End datetime (UTC).

        Returns:
            NewsIngestionResult with pipeline stats.
        """
        return asyncio.run(self.ingest(asset, provider, start, end))
