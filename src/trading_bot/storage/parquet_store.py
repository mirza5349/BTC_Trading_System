"""Parquet-based local storage for market data.

Stores normalized OHLCV candles as Parquet files organized by
symbol and timeframe. Supports idempotent writes with deduplication
on merge.

Directory layout:
    data/processed/market/BTCUSDT/15m/candles.parquet
    data/processed/market/BTCUSDT/1h/candles.parquet
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from trading_bot.logging_config import get_logger
from trading_bot.schemas.market_data import UNIQUE_KEY, validate_candles

logger = get_logger(__name__)


class ParquetStoreError(Exception):
    """Raised on storage read/write failures."""


class ParquetStore:
    """Local Parquet file store for candle data.

    Organizes data under a base directory:
        {base_dir}/{symbol}/{timeframe}/candles.parquet

    Provides idempotent append behavior — repeated writes with
    overlapping data are merged by deduplicating on the unique key
    (symbol + timeframe + timestamp), keeping the latest record.
    """

    FILENAME = "candles.parquet"

    def __init__(self, base_dir: str | Path) -> None:
        """Initialize the Parquet store.

        Args:
            base_dir: Root directory for data storage.
        """
        self.base_dir = Path(base_dir)

    def _path_for(self, symbol: str, timeframe: str) -> Path:
        """Return the Parquet file path for a symbol/timeframe pair."""
        return self.base_dir / symbol / timeframe / self.FILENAME

    def write(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> int:
        """Write or append candle data, deduplicating on the unique key.

        If a file already exists, the new data is merged with existing data.
        Duplicate timestamps (per symbol+timeframe) are resolved by keeping
        the latest record (from the new data).

        Args:
            df: DataFrame conforming to the candle schema.
            symbol: Trading pair symbol.
            timeframe: Candle timeframe string.

        Returns:
            Total row count after write.

        Raises:
            ParquetStoreError: On write failure.
        """
        if df.empty:
            logger.debug("parquet_write_skip_empty", symbol=symbol, timeframe=timeframe)
            return 0

        path = self._path_for(symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Load existing data if present
            if path.exists():
                existing = pd.read_parquet(path)
                logger.debug(
                    "parquet_merge_existing",
                    existing_rows=len(existing),
                    new_rows=len(df),
                )
                # Concatenate and deduplicate
                merged = pd.concat([existing, df], ignore_index=True)
                merged = merged.drop_duplicates(subset=UNIQUE_KEY, keep="last")
                merged = merged.sort_values("timestamp").reset_index(drop=True)
            else:
                merged = df.sort_values("timestamp").reset_index(drop=True)

            # Write
            merged.to_parquet(path, index=False, engine="pyarrow")

            logger.info(
                "parquet_write_complete",
                path=str(path),
                rows=len(merged),
                symbol=symbol,
                timeframe=timeframe,
            )
            return len(merged)

        except Exception as exc:
            raise ParquetStoreError(
                f"Failed to write {path}: {exc}"
            ) from exc

    def read(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Read candle data from the store, optionally filtered by time range.

        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe string.
            start: Optional start datetime filter (inclusive).
            end: Optional end datetime filter (inclusive).

        Returns:
            DataFrame of candle data, or empty DataFrame if not found.
        """
        path = self._path_for(symbol, timeframe)

        if not path.exists():
            logger.debug("parquet_read_not_found", path=str(path))
            return pd.DataFrame()

        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            logger.error("parquet_read_error", path=str(path), error=str(exc))
            return pd.DataFrame()

        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Apply time filters
        if start is not None:
            start_utc = pd.Timestamp(start).tz_localize("UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start).tz_convert("UTC")
            df = df[df["timestamp"] >= start_utc]
        if end is not None:
            end_utc = pd.Timestamp(end).tz_localize("UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).tz_convert("UTC")
            df = df[df["timestamp"] <= end_utc]

        return df.sort_values("timestamp").reset_index(drop=True)

    def get_info(self, symbol: str, timeframe: str) -> dict[str, Any]:
        """Get metadata about a stored dataset.

        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe string.

        Returns:
            Dict with row_count, min_timestamp, max_timestamp, file_size_mb.
            Empty dict if file doesn't exist.
        """
        path = self._path_for(symbol, timeframe)

        if not path.exists():
            return {}

        try:
            df = pd.read_parquet(path)
            if df.empty:
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "row_count": 0,
                    "path": str(path),
                }

            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "row_count": len(df),
                "min_timestamp": df["timestamp"].min().isoformat(),
                "max_timestamp": df["timestamp"].max().isoformat(),
                "file_size_mb": round(path.stat().st_size / (1024 * 1024), 3),
                "path": str(path),
            }
        except Exception as exc:
            logger.error("parquet_info_error", path=str(path), error=str(exc))
            return {"error": str(exc)}

    def exists(self, symbol: str, timeframe: str) -> bool:
        """Check if data exists for a symbol/timeframe pair."""
        return self._path_for(symbol, timeframe).exists()

    def get_latest_timestamp(
        self, symbol: str, timeframe: str
    ) -> datetime | None:
        """Get the latest timestamp in the stored data.

        Useful for incremental ingestion — start fetching from this point.

        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe string.

        Returns:
            Latest timestamp as a datetime, or None if no data exists.
        """
        path = self._path_for(symbol, timeframe)
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path, columns=["timestamp"])
            if df.empty:
                return None
            ts = pd.to_datetime(df["timestamp"], utc=True).max()
            return ts.to_pydatetime()
        except Exception:
            return None
