"""CryptoCompare news API client.

Uses the free CryptoCompare public news endpoint to fetch historical
crypto news articles. Supports BTC category filtering and timestamp-based
pagination for retrieving large backlogs.

API docs: https://min-api.cryptocompare.com/documentation?key=News
Endpoint: GET /data/v2/news/?categories=BTC&lTs={timestamp}
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx
import pandas as pd

from trading_bot.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CRYPTOCOMPARE_BASE_URL = "https://min-api.cryptocompare.com"
NEWS_ENDPOINT = "/data/v2/news/"

# Delay between paginated requests (seconds) to stay under rate limits
REQUEST_DELAY = 0.25

# Maximum pages to fetch in a single ingestion run (safety limit)
MAX_PAGES = 500


class NewsClientError(Exception):
    """Raised when the news API returns an error."""


class CryptoCompareNewsClient:
    """Client for CryptoCompare public news API.

    Downloads historical BTC news articles with automatic pagination
    using the ``lTs`` (last timestamp) parameter. Each page returns
    up to 50 articles by default, ordered newest-first.

    Usage::

        client = CryptoCompareNewsClient()
        df = await client.fetch_news(
            categories="BTC",
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2024, 3, 1, tzinfo=timezone.utc),
        )
    """

    def __init__(
        self,
        base_url: str = CRYPTOCOMPARE_BASE_URL,
        api_key: str = "",
        timeout: float = 30.0,
        max_retries: int = 3,
        max_pages: int = MAX_PAGES,
    ) -> None:
        """Initialize the CryptoCompare news client.

        Args:
            base_url: API base URL.
            api_key: Optional API key for higher rate limits.
            timeout: HTTP request timeout in seconds.
            max_retries: Number of retries for failed requests.
            max_pages: Safety limit on pages per ingestion run.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_pages = max_pages

    async def fetch_news(
        self,
        categories: str = "BTC",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch historical news with automatic backward pagination.

        CryptoCompare returns articles newest-first. We paginate backward
        using ``lTs`` until we reach ``start_time`` or run out of articles.

        Args:
            categories: Comma-separated category filter (e.g., "BTC").
            start_time: Earliest published_at to include (UTC).
            end_time: Latest published_at to include (UTC). Defaults to now.

        Returns:
            DataFrame with raw news article data.

        Raises:
            NewsClientError: If the API returns an error.
        """
        if end_time is None:
            end_time = datetime.now(timezone.utc)
        if start_time is None:
            start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

        start_ts = int(start_time.timestamp())
        end_ts = int(end_time.timestamp())

        logger.info(
            "news_fetch_start",
            categories=categories,
            start=start_time.isoformat(),
            end=end_time.isoformat(),
        )

        all_articles: list[dict[str, Any]] = []
        lts: int | None = end_ts  # Start from end, paginate backward
        page = 0
        seen_ids: set[str] = set()

        headers: dict[str, str] = {}
        if self.api_key:
            headers["authorization"] = f"Apikey {self.api_key}"

        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=headers,
        ) as client:
            while page < self.max_pages:
                page += 1

                params: dict[str, Any] = {
                    "categories": categories,
                    "extraParams": "btc_ml_trading_bot",
                }
                if lts is not None:
                    params["lTs"] = lts

                articles = await self._request_with_retry(client, params)

                if not articles:
                    logger.debug("news_fetch_empty_page", page=page)
                    break

                # Filter to time range and deduplicate
                new_articles: list[dict[str, Any]] = []
                oldest_ts = end_ts

                for article in articles:
                    pub_ts = article.get("published_on", 0)
                    article_id = str(article.get("id", ""))

                    # Skip duplicates
                    if article_id in seen_ids:
                        continue
                    seen_ids.add(article_id)

                    # Skip articles outside our range
                    if pub_ts < start_ts:
                        continue
                    if pub_ts > end_ts:
                        continue

                    new_articles.append(article)
                    oldest_ts = min(oldest_ts, pub_ts)

                all_articles.extend(new_articles)

                # Check if we've gone past start_time
                page_oldest = min(
                    (a.get("published_on", 0) for a in articles),
                    default=0,
                )
                if page_oldest <= start_ts:
                    logger.debug("news_fetch_reached_start", page=page)
                    break

                # Set lTs for next page (oldest timestamp from current page)
                lts = page_oldest

                logger.debug(
                    "news_fetch_page",
                    page=page,
                    new_articles=len(new_articles),
                    total=len(all_articles),
                )

                # Rate limiting
                await asyncio.sleep(REQUEST_DELAY)

        if not all_articles:
            logger.warning("news_fetch_no_data", categories=categories)
            return pd.DataFrame()

        # Convert to DataFrame
        df = self._parse_articles(all_articles)

        logger.info(
            "news_fetch_complete",
            categories=categories,
            total_articles=len(df),
            first=df["published_at"].min().isoformat() if len(df) > 0 else None,
            last=df["published_at"].max().isoformat() if len(df) > 0 else None,
        )

        return df

    async def _request_with_retry(
        self,
        client: httpx.AsyncClient,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Make a single news request with retry logic.

        Args:
            client: The httpx AsyncClient.
            params: Query parameters.

        Returns:
            List of article dicts.

        Raises:
            NewsClientError: After all retries are exhausted.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = await client.get(NEWS_ENDPOINT, params=params)

                if response.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning("news_rate_limited", wait_seconds=wait, attempt=attempt)
                    await asyncio.sleep(wait)
                    continue

                response.raise_for_status()
                data = response.json()

                # CryptoCompare wraps results in {"Data": [...]}
                if isinstance(data, dict):
                    articles = data.get("Data", [])
                    if isinstance(articles, list):
                        return articles
                    return []

                return []

            except httpx.HTTPStatusError as exc:
                last_error = exc
                logger.warning(
                    "news_http_error",
                    status=exc.response.status_code,
                    attempt=attempt,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
                last_error = exc
                logger.warning(
                    "news_connection_error",
                    error=str(exc),
                    attempt=attempt,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)

        raise NewsClientError(
            f"Failed after {self.max_retries} retries: {last_error}"
        )

    @staticmethod
    def _parse_articles(articles: list[dict[str, Any]]) -> pd.DataFrame:
        """Parse raw CryptoCompare article dicts into a normalized DataFrame.

        Args:
            articles: List of raw article dictionaries from the API.

        Returns:
            DataFrame with columns mapped to the internal schema.
        """
        records: list[dict[str, Any]] = []

        for article in articles:
            source_info = article.get("source_info", {}) or {}

            records.append({
                "provider_article_id": str(article.get("id", "")),
                "published_at": pd.to_datetime(
                    article.get("published_on", 0), unit="s", utc=True
                ),
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "source_name": article.get("source", ""),
                "summary": article.get("body", ""),  # CryptoCompare "body" is actually summary
                "body": "",  # Full body not available in free tier
                "author": "",
                "language": source_info.get("lang", "EN"),
                "category": article.get("categories", ""),
                "tags": article.get("tags", ""),
                "raw_symbol_refs": article.get("categories", ""),
            })

        return pd.DataFrame(records)
