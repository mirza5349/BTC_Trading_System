"""Binance public API client for historical OHLCV data.

Uses the Binance public REST API (no authentication required for historical
klines). Handles pagination, rate limiting, and retry logic.
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
BINANCE_BASE_URL = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"

# Binance kline response column order
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

# Timeframe to milliseconds mapping
TIMEFRAME_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
}

# Binance maximum klines per request
MAX_LIMIT = 1000

# Delay between paginated requests (seconds) to stay under rate limits
REQUEST_DELAY = 0.15


class BinanceClientError(Exception):
    """Raised when the Binance API returns an error."""


class BinanceClient:
    """Client for Binance public REST API.

    Downloads historical OHLCV kline data with automatic pagination
    for large date ranges.
    """

    def __init__(
        self,
        base_url: str = BINANCE_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
        request_limit: int = MAX_LIMIT,
    ) -> None:
        """Initialize the Binance client.

        Args:
            base_url: Binance API base URL.
            timeout: HTTP request timeout in seconds.
            max_retries: Number of retries for failed requests.
            request_limit: Maximum klines per request (max 1000).
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.request_limit = min(request_limit, MAX_LIMIT)

    async def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """Fetch historical klines with automatic pagination.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT").
            interval: Kline interval (e.g., "15m", "1h").
            start_time: Start datetime (UTC).
            end_time: End datetime (UTC).

        Returns:
            DataFrame with raw kline data from Binance.

        Raises:
            BinanceClientError: If the API returns an error.
        """
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        logger.info(
            "binance_fetch_start",
            symbol=symbol,
            interval=interval,
            start=start_time.isoformat(),
            end=end_time.isoformat(),
        )

        all_klines: list[list[Any]] = []
        current_start_ms = start_ms
        page = 0

        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
        ) as client:
            while current_start_ms < end_ms:
                page += 1
                params: dict[str, Any] = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": current_start_ms,
                    "endTime": end_ms,
                    "limit": self.request_limit,
                }

                klines = await self._request_with_retry(client, params)

                if not klines:
                    logger.debug("binance_fetch_empty_page", page=page)
                    break

                all_klines.extend(klines)

                # Move start to after the last candle's open time
                last_open_time = int(klines[-1][0])
                interval_ms = TIMEFRAME_MS.get(interval, 60_000)
                current_start_ms = last_open_time + interval_ms

                logger.debug(
                    "binance_fetch_page",
                    page=page,
                    rows=len(klines),
                    total=len(all_klines),
                )

                # Rate limiting
                if current_start_ms < end_ms:
                    await asyncio.sleep(REQUEST_DELAY)

        if not all_klines:
            logger.warning("binance_fetch_no_data", symbol=symbol, interval=interval)
            return pd.DataFrame(columns=KLINE_COLUMNS)

        df = pd.DataFrame(all_klines, columns=KLINE_COLUMNS)

        # Drop the 'ignore' column
        df = df.drop(columns=["ignore"], errors="ignore")

        # Convert timestamps from ms to datetime
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        # Coerce numeric columns
        numeric_cols = [
            "open", "high", "low", "close", "volume",
            "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["number_of_trades"] = pd.to_numeric(df["number_of_trades"], errors="coerce").astype(
            "Int64"
        )

        logger.info(
            "binance_fetch_complete",
            symbol=symbol,
            interval=interval,
            total_candles=len(df),
            first=df["open_time"].iloc[0].isoformat() if len(df) > 0 else None,
            last=df["open_time"].iloc[-1].isoformat() if len(df) > 0 else None,
        )

        return df

    async def _request_with_retry(
        self,
        client: httpx.AsyncClient,
        params: dict[str, Any],
    ) -> list[list[Any]]:
        """Make a single klines request with retry logic.

        Args:
            client: The httpx AsyncClient.
            params: Query parameters.

        Returns:
            List of kline arrays.

        Raises:
            BinanceClientError: After all retries are exhausted.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = await client.get(KLINES_ENDPOINT, params=params)

                if response.status_code == 429:
                    # Rate limited — back off
                    wait = 2 ** attempt
                    logger.warning("binance_rate_limited", wait_seconds=wait, attempt=attempt)
                    await asyncio.sleep(wait)
                    continue

                if response.status_code == 418:
                    # IP ban — long backoff
                    logger.error("binance_ip_banned", attempt=attempt)
                    await asyncio.sleep(60)
                    continue

                response.raise_for_status()
                data: list[list[Any]] = response.json()
                return data

            except httpx.HTTPStatusError as exc:
                last_error = exc
                logger.warning(
                    "binance_http_error",
                    status=exc.response.status_code,
                    attempt=attempt,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
                last_error = exc
                logger.warning(
                    "binance_connection_error",
                    error=str(exc),
                    attempt=attempt,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)

        raise BinanceClientError(
            f"Failed after {self.max_retries} retries: {last_error}"
        )
