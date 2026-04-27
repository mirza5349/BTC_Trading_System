"""Tests for market data ingestion pipeline.

Tests normalization, validation, and the ingestion service with mocked
API responses. No live network calls.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from trading_bot.ingestion.binance_client import BinanceClient
from trading_bot.ingestion.market_data import IngestionResult, MarketDataService
from trading_bot.schemas.market_data import (
    REQUIRED_COLUMNS,
    normalize_candle_df,
    validate_candles,
)


# ============================================================================
# Fixtures
# ============================================================================
def _make_raw_candles(n: int = 10, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Create a realistic raw candle DataFrame mimicking Binance output."""
    base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = pd.date_range(base_ts, periods=n, freq="15min", tz="UTC")
    close_times = timestamps + pd.Timedelta(minutes=15) - pd.Timedelta(milliseconds=1)

    rng = np.random.default_rng(42)
    opens = 42000 + rng.standard_normal(n) * 100
    highs = opens + rng.uniform(50, 200, n)
    lows = opens - rng.uniform(50, 200, n)
    closes = opens + rng.standard_normal(n) * 80
    volumes = rng.uniform(10, 500, n)

    return pd.DataFrame({
        "open_time": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "close_time": close_times,
        "quote_asset_volume": volumes * closes,
        "number_of_trades": rng.integers(100, 5000, n),
        "taker_buy_base_volume": volumes * 0.6,
        "taker_buy_quote_volume": volumes * closes * 0.6,
    })


# ============================================================================
# Schema normalization tests
# ============================================================================
class TestNormalization:
    """Test candle DataFrame normalization."""

    def test_normalize_adds_metadata_columns(self) -> None:
        """Normalization should add symbol, timeframe, and source columns."""
        raw = _make_raw_candles(5)
        result = normalize_candle_df(raw, symbol="BTCUSDT", timeframe="15m")

        assert "symbol" in result.columns
        assert "timeframe" in result.columns
        assert "source" in result.columns
        assert (result["symbol"] == "BTCUSDT").all()
        assert (result["timeframe"] == "15m").all()
        assert (result["source"] == "binance").all()

    def test_normalize_renames_open_time_to_timestamp(self) -> None:
        """open_time should be renamed to timestamp."""
        raw = _make_raw_candles(3)
        renamed = raw.rename(columns={"open_time": "timestamp"})
        result = normalize_candle_df(renamed, symbol="BTCUSDT", timeframe="15m")

        assert "timestamp" in result.columns
        assert result["timestamp"].dt.tz is not None  # Should be UTC

    def test_normalize_ensures_utc_timestamps(self) -> None:
        """Timestamps should be timezone-aware UTC."""
        raw = _make_raw_candles(3)
        renamed = raw.rename(columns={"open_time": "timestamp"})
        result = normalize_candle_df(renamed)

        assert str(result["timestamp"].dt.tz) == "UTC"

    def test_normalize_coerces_numeric_columns(self) -> None:
        """Numeric columns should be float64."""
        raw = _make_raw_candles(3)
        # Simulate string prices (as they come from JSON)
        raw["open"] = raw["open"].astype(str)
        raw["close"] = raw["close"].astype(str)
        renamed = raw.rename(columns={"open_time": "timestamp"})
        result = normalize_candle_df(renamed)

        assert result["open"].dtype == np.float64
        assert result["close"].dtype == np.float64

    def test_normalize_adds_missing_optional_columns(self) -> None:
        """Missing optional columns should be added with defaults."""
        raw = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"),
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [95.0, 96.0, 97.0],
            "close": [103.0, 104.0, 105.0],
            "volume": [10.0, 20.0, 30.0],
        })
        result = normalize_candle_df(raw)

        assert "quote_asset_volume" in result.columns
        assert "number_of_trades" in result.columns
        assert "source" in result.columns

    def test_normalize_sorts_by_timestamp(self) -> None:
        """Output should be sorted by timestamp ascending."""
        raw = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01 02:00", "2024-01-01 01:00", "2024-01-01 03:00"], utc=True),
            "open": [100, 99, 101],
            "high": [105, 104, 106],
            "low": [95, 94, 96],
            "close": [103, 102, 104],
            "volume": [10, 20, 30],
        })
        result = normalize_candle_df(raw)
        ts = result["timestamp"].tolist()
        assert ts == sorted(ts)


# ============================================================================
# Validation tests
# ============================================================================
class TestValidation:
    """Test candle data validation."""

    def test_validate_passes_clean_data(self) -> None:
        """Clean data should pass validation."""
        df = pd.DataFrame({
            "symbol": ["BTCUSDT"] * 3,
            "timeframe": ["15m"] * 3,
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"),
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [95.0, 96.0, 97.0],
            "close": [103.0, 104.0, 105.0],
            "volume": [10.0, 20.0, 30.0],
        })
        clean, result = validate_candles(df)

        assert result.is_valid
        assert result.valid_rows == 3
        assert result.duplicate_count == 0
        assert result.invalid_ohlc_count == 0

    def test_validate_removes_duplicates(self) -> None:
        """Duplicate timestamps should be removed, keeping the last."""
        ts = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        df = pd.DataFrame({
            "symbol": ["BTCUSDT"] * 3,
            "timeframe": ["15m"] * 3,
            "timestamp": [ts, ts, ts + pd.Timedelta(minutes=15)],
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [95.0, 96.0, 97.0],
            "close": [103.0, 104.0, 105.0],
            "volume": [10.0, 20.0, 30.0],
        })
        clean, result = validate_candles(df)

        assert result.duplicate_count == 1
        assert result.valid_rows == 2

    def test_validate_removes_negative_prices(self) -> None:
        """Rows with negative prices should be removed."""
        df = pd.DataFrame({
            "symbol": ["BTCUSDT"] * 3,
            "timeframe": ["15m"] * 3,
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"),
            "open": [100.0, -1.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [95.0, 96.0, 97.0],
            "close": [103.0, 104.0, 105.0],
            "volume": [10.0, 20.0, 30.0],
        })
        clean, result = validate_candles(df)

        assert result.negative_price_count == 1
        assert result.valid_rows == 2

    def test_validate_removes_invalid_ohlc(self) -> None:
        """Rows where low > high should be removed."""
        df = pd.DataFrame({
            "symbol": ["BTCUSDT"] * 3,
            "timeframe": ["15m"] * 3,
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"),
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 90.0, 107.0],   # Row 2: high=90 < low=96
            "low": [95.0, 96.0, 97.0],
            "close": [103.0, 104.0, 105.0],
            "volume": [10.0, 20.0, 30.0],
        })
        clean, result = validate_candles(df)

        assert result.invalid_ohlc_count == 1
        assert result.valid_rows == 2

    def test_validate_detects_missing_columns(self) -> None:
        """Missing required columns should fail validation."""
        df = pd.DataFrame({
            "symbol": ["BTCUSDT"],
            "open": [100.0],
            # missing: timeframe, timestamp, high, low, close, volume
        })
        clean, result = validate_candles(df)

        assert not result.is_valid
        assert len(result.missing_column_names) > 0

    def test_validate_empty_df(self) -> None:
        """Empty DataFrame should pass validation with 0 rows."""
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        clean, result = validate_candles(df)

        assert result.is_valid
        assert result.valid_rows == 0

    def test_validate_sorts_output(self) -> None:
        """Validated output should be sorted by timestamp."""
        df = pd.DataFrame({
            "symbol": ["BTCUSDT"] * 3,
            "timeframe": ["15m"] * 3,
            "timestamp": pd.to_datetime(
                ["2024-01-01 02:00", "2024-01-01 01:00", "2024-01-01 03:00"], utc=True
            ),
            "open": [100.0, 99.0, 101.0],
            "high": [105.0, 104.0, 106.0],
            "low": [95.0, 94.0, 96.0],
            "close": [103.0, 102.0, 104.0],
            "volume": [10.0, 20.0, 30.0],
        })
        clean, _ = validate_candles(df)

        ts = clean["timestamp"].tolist()
        assert ts == sorted(ts)


# ============================================================================
# Ingestion service tests (mocked API)
# ============================================================================
class TestMarketDataService:
    """Test the ingestion service with mocked Binance client."""

    def _make_service(self, tmp_path, raw_df: pd.DataFrame | None = None):
        """Create a MarketDataService with a mocked client."""
        from trading_bot.storage.parquet_store import ParquetStore

        store = ParquetStore(base_dir=tmp_path / "processed")
        mock_client = AsyncMock(spec=BinanceClient)

        if raw_df is not None:
            mock_client.fetch_klines.return_value = raw_df
        else:
            mock_client.fetch_klines.return_value = _make_raw_candles(20)

        return MarketDataService(store=store, client=mock_client), store

    def test_ingest_success(self, tmp_path) -> None:
        """Ingestion with valid data should succeed."""
        service, store = self._make_service(tmp_path)
        result = service.ingest_sync(
            symbol="BTCUSDT",
            timeframe="15m",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )

        assert result.is_success
        assert result.fetched_rows == 20
        assert result.stored_rows > 0
        assert result.error == ""

    def test_ingest_stores_to_parquet(self, tmp_path) -> None:
        """Ingested data should be persisted as Parquet."""
        service, store = self._make_service(tmp_path)
        service.ingest_sync(
            symbol="BTCUSDT",
            timeframe="15m",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )

        assert store.exists("BTCUSDT", "15m")
        df = store.read("BTCUSDT", "15m")
        assert len(df) > 0
        assert "timestamp" in df.columns
        assert "close" in df.columns

    def test_ingest_handles_empty_response(self, tmp_path) -> None:
        """Ingestion should handle empty API response gracefully."""
        service, store = self._make_service(
            tmp_path,
            raw_df=pd.DataFrame(),
        )
        result = service.ingest_sync(
            symbol="BTCUSDT",
            timeframe="15m",
        )

        assert not result.is_success
        assert "No data" in result.error

    def test_ingest_idempotent(self, tmp_path) -> None:
        """Running ingestion twice with same data should not create duplicates."""
        service, store = self._make_service(tmp_path)

        result1 = service.ingest_sync(
            symbol="BTCUSDT",
            timeframe="15m",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        result2 = service.ingest_sync(
            symbol="BTCUSDT",
            timeframe="15m",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )

        # Row count should remain the same
        df = store.read("BTCUSDT", "15m")
        assert result1.stored_rows == result2.stored_rows
        assert len(df) == result1.stored_rows

    def test_ingest_handles_api_error(self, tmp_path) -> None:
        """Ingestion should catch and report API errors."""
        from trading_bot.storage.parquet_store import ParquetStore

        store = ParquetStore(base_dir=tmp_path / "processed")
        mock_client = AsyncMock(spec=BinanceClient)
        mock_client.fetch_klines.side_effect = Exception("Connection refused")

        service = MarketDataService(store=store, client=mock_client)
        result = service.ingest_sync(
            symbol="BTCUSDT",
            timeframe="15m",
        )

        assert not result.is_success
        assert "Connection refused" in result.error
