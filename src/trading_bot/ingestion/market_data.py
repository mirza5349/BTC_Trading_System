"""Market data ingestion service.

Coordinates downloading historical OHLCV data from exchange APIs,
normalizing it to the canonical schema, validating, and persisting
to local Parquet storage.

The original abstract interfaces (MarketDataProvider, OHLCV, etc.)
are preserved for backward compatibility and future use.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from trading_bot.ingestion.binance_client import BinanceClient
from trading_bot.logging_config import get_logger
from trading_bot.schemas.market_data import (
    REQUIRED_COLUMNS,
    normalize_candle_df,
    validate_candles,
)
from trading_bot.storage.parquet_store import ParquetStore

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes (from step 1, preserved)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OHLCV:
    """Single OHLCV candle."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = "BTCUSDT"

    @property
    def typical_price(self) -> float:
        """Return (high + low + close) / 3."""
        return (self.high + self.low + self.close) / 3.0


@dataclass
class OrderBookSnapshot:
    """Order book snapshot at a point in time."""

    timestamp: datetime
    bids: list[tuple[float, float]] = field(default_factory=list)
    asks: list[tuple[float, float]] = field(default_factory=list)
    symbol: str = "BTCUSDT"

    @property
    def spread(self) -> float:
        """Return best ask - best bid, or 0 if book is empty."""
        if not self.bids or not self.asks:
            return 0.0
        return self.asks[0][0] - self.bids[0][0]


# ---------------------------------------------------------------------------
# Abstract provider interface (from step 1, preserved)
# ---------------------------------------------------------------------------
class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""

    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[OHLCV]:
        """Fetch historical OHLCV candles."""

    @abstractmethod
    async def fetch_latest_price(self, symbol: str) -> float:
        """Fetch the latest ticker price."""

    @abstractmethod
    async def fetch_order_book(
        self, symbol: str, depth: int = 20
    ) -> OrderBookSnapshot:
        """Fetch a snapshot of the order book."""


# ---------------------------------------------------------------------------
# Ingestion result
# ---------------------------------------------------------------------------
@dataclass
class IngestionResult:
    """Result of a market data ingestion run."""

    symbol: str
    timeframe: str
    fetched_rows: int = 0
    valid_rows: int = 0
    stored_rows: int = 0
    duplicates_removed: int = 0
    invalid_rows_removed: int = 0
    start_timestamp: str = ""
    end_timestamp: str = ""
    is_success: bool = False
    error: str = ""


# ---------------------------------------------------------------------------
# Market data ingestion service
# ---------------------------------------------------------------------------
class MarketDataService:
    """Coordinates historical market data download, validation, and storage.

    This is the main entry point for the ingestion pipeline. It:
    1. Downloads raw kline data from Binance (via BinanceClient)
    2. Normalizes column names and dtypes to the canonical schema
    3. Validates data quality (dedup, OHLC checks, negatives)
    4. Persists to local Parquet storage with idempotent merge

    Usage::

        service = MarketDataService(
            store=ParquetStore("data/processed/market"),
        )
        result = await service.ingest(
            symbol="BTCUSDT",
            timeframe="15m",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 3, 1, tzinfo=timezone.utc),
        )
    """

    def __init__(
        self,
        store: ParquetStore,
        client: BinanceClient | None = None,
    ) -> None:
        """Initialize the ingestion service.

        Args:
            store: Parquet store for persistence.
            client: Binance API client. Uses default if None.
        """
        self.store = store
        self.client = client or BinanceClient()

    async def ingest(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "15m",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> IngestionResult:
        """Run the full ingestion pipeline: fetch → normalize → validate → store.

        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe (e.g., "15m", "1h").
            start: Start datetime (UTC). Defaults to 2024-01-01.
            end: End datetime (UTC). Defaults to now.

        Returns:
            IngestionResult with pipeline stats.
        """
        if start is None:
            start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        if end is None:
            end = datetime.now(timezone.utc)

        logger.info(
            "ingestion_start",
            symbol=symbol,
            timeframe=timeframe,
            start=start.isoformat(),
            end=end.isoformat(),
        )

        result = IngestionResult(symbol=symbol, timeframe=timeframe)

        try:
            # 1. Fetch raw data
            raw_df = await self.client.fetch_klines(
                symbol=symbol,
                interval=timeframe,
                start_time=start,
                end_time=end,
            )

            if raw_df.empty:
                result.error = "No data returned from exchange"
                logger.warning("ingestion_no_data", symbol=symbol, timeframe=timeframe)
                return result

            result.fetched_rows = len(raw_df)

            # 2. Normalize to canonical schema
            normalized = self._normalize_binance_response(raw_df, symbol, timeframe)

            # 3. Validate
            clean_df, validation = validate_candles(normalized)

            result.valid_rows = validation.valid_rows
            result.duplicates_removed = validation.duplicate_count
            result.invalid_rows_removed = (
                validation.invalid_ohlc_count + validation.negative_price_count
            )

            if validation.errors:
                for err in validation.errors:
                    logger.warning("ingestion_validation_issue", issue=err)

            if clean_df.empty:
                result.error = "All rows failed validation"
                logger.error("ingestion_all_invalid", symbol=symbol, timeframe=timeframe)
                return result

            # 4. Store
            total_rows = self.store.write(clean_df, symbol, timeframe)
            result.stored_rows = total_rows
            result.start_timestamp = clean_df["timestamp"].min().isoformat()
            result.end_timestamp = clean_df["timestamp"].max().isoformat()
            result.is_success = True

            logger.info(
                "ingestion_complete",
                symbol=symbol,
                timeframe=timeframe,
                fetched=result.fetched_rows,
                valid=result.valid_rows,
                stored=total_rows,
            )

        except Exception as exc:
            result.error = str(exc)
            logger.error("ingestion_failed", error=str(exc), symbol=symbol)

        return result

    def ingest_sync(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "15m",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> IngestionResult:
        """Synchronous wrapper around ingest() for CLI convenience.

        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe.
            start: Start datetime (UTC).
            end: End datetime (UTC).

        Returns:
            IngestionResult with pipeline stats.
        """
        return asyncio.run(self.ingest(symbol, timeframe, start, end))

    @staticmethod
    def _normalize_binance_response(
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> pd.DataFrame:
        """Convert Binance kline response to canonical schema.

        Args:
            df: Raw DataFrame from BinanceClient.fetch_klines().
            symbol: Trading pair symbol.
            timeframe: Candle timeframe string.

        Returns:
            DataFrame conforming to the canonical candle schema.
        """
        # Map Binance column names to our schema
        renamed = df.rename(columns={"open_time": "timestamp"})
        return normalize_candle_df(renamed, symbol=symbol, timeframe=timeframe, source="binance")
