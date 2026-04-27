"""Tests for the news Parquet storage backend."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from trading_bot.storage.news_store import NewsStore


def _make_news_df(
    n: int = 5,
    start: str = "2024-01-15",
    asset: str = "BTC",
    provider: str = "cryptocompare",
) -> pd.DataFrame:
    """Create a sample news DataFrame."""
    timestamps = pd.date_range(start, periods=n, freq="1h", tz="UTC")

    return pd.DataFrame({
        "asset": asset,
        "provider": provider,
        "published_at": timestamps,
        "title": [f"Bitcoin article {i}" for i in range(n)],
        "url": [f"https://example.com/article-{i}" for i in range(n)],
        "source_name": [f"CryptoNews{i % 3}" for i in range(n)],
        "provider_article_id": [str(1000 + i) for i in range(n)],
        "summary": [f"Summary {i}" for i in range(n)],
        "body": [""] * n,
        "author": ["Author"] * n,
        "language": ["EN"] * n,
        "source_domain": ["example.com"] * n,
        "category": ["BTC"] * n,
        "tags": ["BTC|Trading"] * n,
        "raw_symbol_refs": ["BTC"] * n,
        "ingested_at": pd.Timestamp.now(tz="UTC"),
    })


class TestNewsStore:
    """Test suite for NewsStore."""

    def test_write_and_read(self, tmp_path) -> None:
        """Should write and read back identically."""
        store = NewsStore(base_dir=tmp_path)
        df = _make_news_df(10)

        row_count = store.write(df, "BTC", "cryptocompare")
        assert row_count == 10

        result = store.read("BTC", "cryptocompare")
        assert len(result) == 10
        assert "published_at" in result.columns

    def test_write_creates_directories(self, tmp_path) -> None:
        """Should create asset/provider directories automatically."""
        store = NewsStore(base_dir=tmp_path)
        df = _make_news_df(3)

        store.write(df, "BTC", "cryptocompare")

        expected_path = tmp_path / "BTC" / "cryptocompare" / "news.parquet"
        assert expected_path.exists()

    def test_write_deduplicates_by_id(self, tmp_path) -> None:
        """Appending duplicate articles should not create duplicates."""
        store = NewsStore(base_dir=tmp_path)

        df1 = _make_news_df(5)
        df2 = _make_news_df(5)  # Same provider_article_ids

        store.write(df1, "BTC", "cryptocompare")
        total = store.write(df2, "BTC", "cryptocompare")

        assert total == 5  # Not 10

    def test_write_merges_new_data(self, tmp_path) -> None:
        """Appending non-overlapping data should increase row count."""
        store = NewsStore(base_dir=tmp_path)

        df1 = _make_news_df(5, start="2024-01-15")
        df2 = _make_news_df(5, start="2024-01-16")
        df2["provider_article_id"] = [str(2000 + i) for i in range(5)]
        df2["url"] = [f"https://example.com/new-{i}" for i in range(5)]

        store.write(df1, "BTC", "cryptocompare")
        total = store.write(df2, "BTC", "cryptocompare")

        assert total == 10

    def test_write_empty_df(self, tmp_path) -> None:
        """Writing an empty DataFrame should be a no-op."""
        store = NewsStore(base_dir=tmp_path)
        result = store.write(pd.DataFrame(), "BTC", "cryptocompare")
        assert result == 0

    def test_read_nonexistent(self, tmp_path) -> None:
        """Reading nonexistent data should return empty DataFrame."""
        store = NewsStore(base_dir=tmp_path)
        result = store.read("BTC", "cryptocompare")
        assert result.empty

    def test_read_with_time_filter(self, tmp_path) -> None:
        """Should filter by start/end datetime."""
        store = NewsStore(base_dir=tmp_path)
        df = _make_news_df(20, start="2024-01-15")
        store.write(df, "BTC", "cryptocompare")

        start = datetime(2024, 1, 15, 5, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        result = store.read("BTC", "cryptocompare", start=start, end=end)

        assert len(result) > 0
        assert result["published_at"].min() >= pd.Timestamp(start)
        assert result["published_at"].max() <= pd.Timestamp(end)

    def test_exists(self, tmp_path) -> None:
        """exists() should reflect stored state."""
        store = NewsStore(base_dir=tmp_path)
        assert not store.exists("BTC", "cryptocompare")

        store.write(_make_news_df(3), "BTC", "cryptocompare")
        assert store.exists("BTC", "cryptocompare")

    def test_get_info(self, tmp_path) -> None:
        """get_info should return metadata about stored data."""
        store = NewsStore(base_dir=tmp_path)
        store.write(_make_news_df(10), "BTC", "cryptocompare")

        info = store.get_info("BTC", "cryptocompare")
        assert info["row_count"] == 10
        assert "min_published_at" in info
        assert "max_published_at" in info
        assert "file_size_mb" in info
        assert "unique_sources" in info
        assert "top_sources" in info

    def test_get_info_nonexistent(self, tmp_path) -> None:
        """get_info for nonexistent data should return empty dict."""
        store = NewsStore(base_dir=tmp_path)
        info = store.get_info("BTC", "cryptocompare")
        assert info == {}

    def test_get_latest_timestamp(self, tmp_path) -> None:
        """Should return the latest published_at in stored data."""
        store = NewsStore(base_dir=tmp_path)
        df = _make_news_df(10)
        store.write(df, "BTC", "cryptocompare")

        latest = store.get_latest_timestamp("BTC", "cryptocompare")
        assert latest is not None
        assert latest == df["published_at"].max().to_pydatetime()

    def test_get_latest_timestamp_nonexistent(self, tmp_path) -> None:
        """Should return None for nonexistent data."""
        store = NewsStore(base_dir=tmp_path)
        assert store.get_latest_timestamp("BTC", "cryptocompare") is None

    def test_multiple_providers(self, tmp_path) -> None:
        """Should support different providers in separate paths."""
        store = NewsStore(base_dir=tmp_path)

        df_cc = _make_news_df(5, provider="cryptocompare")
        df_cg = _make_news_df(3, provider="coingecko")
        df_cg["provider_article_id"] = [str(3000 + i) for i in range(3)]
        df_cg["url"] = [f"https://coingecko.com/news/{i}" for i in range(3)]

        store.write(df_cc, "BTC", "cryptocompare")
        store.write(df_cg, "BTC", "coingecko")

        assert store.read("BTC", "cryptocompare").shape[0] == 5
        assert store.read("BTC", "coingecko").shape[0] == 3

    def test_fallback_dedup_without_provider_id(self, tmp_path) -> None:
        """Should deduplicate by url+timestamp when provider_article_id is empty."""
        store = NewsStore(base_dir=tmp_path)

        df = _make_news_df(3)
        df["provider_article_id"] = ""  # No provider IDs

        store.write(df, "BTC", "cryptocompare")
        total = store.write(df, "BTC", "cryptocompare")

        assert total == 3  # Not 6
