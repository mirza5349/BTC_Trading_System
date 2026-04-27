"""Tests for the Parquet storage backend."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from trading_bot.storage.parquet_store import ParquetStore


def _make_candle_df(
    n: int = 5,
    start: str = "2024-01-01",
    symbol: str = "BTCUSDT",
    timeframe: str = "15m",
) -> pd.DataFrame:
    """Create a sample candle DataFrame."""
    timestamps = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(42)

    return pd.DataFrame({
        "symbol": symbol,
        "timeframe": timeframe,
        "timestamp": timestamps,
        "open": 42000 + rng.standard_normal(n) * 100,
        "high": 42200 + rng.uniform(0, 100, n),
        "low": 41800 + rng.uniform(0, 100, n),
        "close": 42000 + rng.standard_normal(n) * 100,
        "volume": rng.uniform(10, 500, n),
        "source": "test",
    })


class TestParquetStore:
    """Test suite for ParquetStore."""

    def test_write_and_read(self, tmp_path) -> None:
        """Should write and read back identically."""
        store = ParquetStore(base_dir=tmp_path)
        df = _make_candle_df(10)

        row_count = store.write(df, "BTCUSDT", "15m")
        assert row_count == 10

        result = store.read("BTCUSDT", "15m")
        assert len(result) == 10
        assert "timestamp" in result.columns

    def test_write_creates_directories(self, tmp_path) -> None:
        """Should create symbol/timeframe directories automatically."""
        store = ParquetStore(base_dir=tmp_path)
        df = _make_candle_df(3)

        store.write(df, "BTCUSDT", "15m")

        expected_path = tmp_path / "BTCUSDT" / "15m" / "candles.parquet"
        assert expected_path.exists()

    def test_write_deduplicates_on_merge(self, tmp_path) -> None:
        """Appending overlapping data should not create duplicates."""
        store = ParquetStore(base_dir=tmp_path)

        df1 = _make_candle_df(5, start="2024-01-01")
        df2 = _make_candle_df(5, start="2024-01-01")  # Same timestamps

        store.write(df1, "BTCUSDT", "15m")
        total = store.write(df2, "BTCUSDT", "15m")

        assert total == 5  # Not 10

    def test_write_merges_new_data(self, tmp_path) -> None:
        """Appending non-overlapping data should increase row count."""
        store = ParquetStore(base_dir=tmp_path)

        df1 = _make_candle_df(5, start="2024-01-01")
        df2 = _make_candle_df(5, start="2024-01-02")

        store.write(df1, "BTCUSDT", "15m")
        total = store.write(df2, "BTCUSDT", "15m")

        assert total == 10

    def test_write_empty_df(self, tmp_path) -> None:
        """Writing an empty DataFrame should be a no-op."""
        store = ParquetStore(base_dir=tmp_path)
        result = store.write(pd.DataFrame(), "BTCUSDT", "15m")
        assert result == 0

    def test_read_nonexistent(self, tmp_path) -> None:
        """Reading nonexistent data should return empty DataFrame."""
        store = ParquetStore(base_dir=tmp_path)
        result = store.read("BTCUSDT", "15m")
        assert result.empty

    def test_read_with_time_filter(self, tmp_path) -> None:
        """Should filter by start/end datetime."""
        store = ParquetStore(base_dir=tmp_path)
        df = _make_candle_df(20, start="2024-01-01")
        store.write(df, "BTCUSDT", "15m")

        # Read only a subset
        start = datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 3, 0, tzinfo=timezone.utc)
        result = store.read("BTCUSDT", "15m", start=start, end=end)

        assert len(result) > 0
        assert result["timestamp"].min() >= pd.Timestamp(start)
        assert result["timestamp"].max() <= pd.Timestamp(end)

    def test_exists(self, tmp_path) -> None:
        """exists() should reflect stored state."""
        store = ParquetStore(base_dir=tmp_path)
        assert not store.exists("BTCUSDT", "15m")

        store.write(_make_candle_df(3), "BTCUSDT", "15m")
        assert store.exists("BTCUSDT", "15m")

    def test_get_info(self, tmp_path) -> None:
        """get_info should return metadata about stored data."""
        store = ParquetStore(base_dir=tmp_path)
        store.write(_make_candle_df(10), "BTCUSDT", "15m")

        info = store.get_info("BTCUSDT", "15m")
        assert info["row_count"] == 10
        assert "min_timestamp" in info
        assert "max_timestamp" in info
        assert "file_size_mb" in info

    def test_get_info_nonexistent(self, tmp_path) -> None:
        """get_info for nonexistent data should return empty dict."""
        store = ParquetStore(base_dir=tmp_path)
        info = store.get_info("BTCUSDT", "15m")
        assert info == {}

    def test_get_latest_timestamp(self, tmp_path) -> None:
        """Should return the latest timestamp in stored data."""
        store = ParquetStore(base_dir=tmp_path)
        df = _make_candle_df(10)
        store.write(df, "BTCUSDT", "15m")

        latest = store.get_latest_timestamp("BTCUSDT", "15m")
        assert latest is not None
        assert latest == df["timestamp"].max().to_pydatetime()

    def test_get_latest_timestamp_nonexistent(self, tmp_path) -> None:
        """Should return None for nonexistent data."""
        store = ParquetStore(base_dir=tmp_path)
        assert store.get_latest_timestamp("BTCUSDT", "15m") is None

    def test_multiple_symbols(self, tmp_path) -> None:
        """Should support different symbols in separate paths."""
        store = ParquetStore(base_dir=tmp_path)

        df_btc = _make_candle_df(5, symbol="BTCUSDT")
        df_eth = _make_candle_df(3, symbol="ETHUSDT")

        store.write(df_btc, "BTCUSDT", "15m")
        store.write(df_eth, "ETHUSDT", "15m")

        assert store.read("BTCUSDT", "15m").shape[0] == 5
        assert store.read("ETHUSDT", "15m").shape[0] == 3

    def test_multiple_timeframes(self, tmp_path) -> None:
        """Should support different timeframes for the same symbol."""
        store = ParquetStore(base_dir=tmp_path)

        df_15m = _make_candle_df(5, timeframe="15m")
        df_1h = _make_candle_df(3, timeframe="1h")

        store.write(df_15m, "BTCUSDT", "15m")
        store.write(df_1h, "BTCUSDT", "1h")

        assert store.read("BTCUSDT", "15m").shape[0] == 5
        assert store.read("BTCUSDT", "1h").shape[0] == 3
