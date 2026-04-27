"""Tests for news data ingestion pipeline.

Tests normalization, validation, BTC relevance filtering, and the ingestion
service with mocked API responses. No live network calls.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from trading_bot.ingestion.news_client import CryptoCompareNewsClient
from trading_bot.ingestion.news_data import NewsIngestionService
from trading_bot.schemas.news_data import (
    REQUIRED_COLUMNS,
    is_btc_relevant,
    normalize_news_df,
    validate_news,
)
from trading_bot.storage.news_store import NewsStore


# ============================================================================
# Fixtures
# ============================================================================
def _make_raw_news(n: int = 10, btc_relevant: bool = True) -> pd.DataFrame:
    """Create a realistic raw news DataFrame mimicking CryptoCompare output."""
    base_ts = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
    records = []
    for i in range(n):
        title = f"Bitcoin price surges past ${42000 + i * 100}" if btc_relevant else f"Ethereum upgrade {i}"
        records.append({
            "provider_article_id": str(1000 + i),
            "published_at": pd.Timestamp(base_ts) + pd.Timedelta(hours=i),
            "title": title,
            "url": f"https://example.com/article-{i}",
            "source_name": f"CryptoNews{i % 3}",
            "summary": f"Summary about {'Bitcoin' if btc_relevant else 'Ethereum'} article {i}.",
            "body": "",
            "author": f"Author {i}",
            "language": "EN",
            "category": "BTC" if btc_relevant else "ETH",
            "tags": "BTC|Trading" if btc_relevant else "ETH|DeFi",
            "raw_symbol_refs": "BTC" if btc_relevant else "ETH",
        })
    return pd.DataFrame(records)


# ============================================================================
# Schema normalization tests
# ============================================================================
class TestNewsNormalization:
    """Test news DataFrame normalization."""

    def test_normalize_adds_metadata_columns(self) -> None:
        """Normalization should add asset, provider columns."""
        raw = _make_raw_news(5)
        result = normalize_news_df(raw, asset="BTC", provider="cryptocompare")

        assert "asset" in result.columns
        assert "provider" in result.columns
        assert (result["asset"] == "BTC").all()
        assert (result["provider"] == "cryptocompare").all()

    def test_normalize_adds_ingested_at(self) -> None:
        """Normalization should add an ingested_at timestamp."""
        raw = _make_raw_news(3)
        result = normalize_news_df(raw)

        assert "ingested_at" in result.columns
        assert result["ingested_at"].notna().all()

    def test_normalize_ensures_utc_timestamps(self) -> None:
        """published_at should be timezone-aware UTC."""
        raw = _make_raw_news(3)
        result = normalize_news_df(raw)

        assert str(result["published_at"].dt.tz) == "UTC"

    def test_normalize_strips_whitespace(self) -> None:
        """Text fields should have whitespace stripped."""
        raw = pd.DataFrame({
            "published_at": [pd.Timestamp("2024-01-01", tz="UTC")],
            "title": ["  Bitcoin rises  "],
            "url": ["  https://example.com  "],
            "source_name": ["  CryptoNews  "],
        })
        result = normalize_news_df(raw)

        assert result["title"].iloc[0] == "Bitcoin rises"
        assert result["url"].iloc[0] == "https://example.com"
        assert result["source_name"].iloc[0] == "CryptoNews"

    def test_normalize_extracts_domain(self) -> None:
        """source_domain should be extracted from URL."""
        raw = pd.DataFrame({
            "published_at": [pd.Timestamp("2024-01-01", tz="UTC")],
            "title": ["Test"],
            "url": ["https://www.coindesk.com/article/123"],
            "source_name": ["CoinDesk"],
        })
        result = normalize_news_df(raw)

        assert result["source_domain"].iloc[0] == "www.coindesk.com"

    def test_normalize_sorts_by_published_at(self) -> None:
        """Output should be sorted by published_at ascending."""
        raw = pd.DataFrame({
            "published_at": pd.to_datetime(
                ["2024-01-03", "2024-01-01", "2024-01-02"], utc=True
            ),
            "title": ["C", "A", "B"],
            "url": ["c.com", "a.com", "b.com"],
            "source_name": ["S"] * 3,
        })
        result = normalize_news_df(raw)

        ts = result["published_at"].tolist()
        assert ts == sorted(ts)

    def test_normalize_adds_missing_optional_columns(self) -> None:
        """Missing optional columns should be added with defaults."""
        raw = pd.DataFrame({
            "published_at": [pd.Timestamp("2024-01-01", tz="UTC")],
            "title": ["Test"],
            "url": ["https://example.com"],
            "source_name": ["Source"],
        })
        result = normalize_news_df(raw)

        assert "summary" in result.columns
        assert "body" in result.columns
        assert "author" in result.columns
        assert "provider_article_id" in result.columns


# ============================================================================
# Validation tests
# ============================================================================
class TestNewsValidation:
    """Test news data validation."""

    def test_validate_passes_clean_data(self) -> None:
        """Clean data should pass validation."""
        df = _make_raw_news(5)
        df["asset"] = "BTC"
        df["provider"] = "cryptocompare"
        clean, result = validate_news(df)

        assert result.is_valid
        assert result.valid_rows == 5
        assert result.duplicate_count == 0

    def test_validate_removes_duplicates_by_id(self) -> None:
        """Duplicate provider_article_ids should be removed."""
        df = _make_raw_news(3)
        df["asset"] = "BTC"
        df["provider"] = "cryptocompare"
        # Duplicate the first row
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        clean, result = validate_news(df)

        assert result.duplicate_count == 1
        assert result.valid_rows == 3

    def test_validate_removes_missing_title(self) -> None:
        """Rows with missing titles should be removed."""
        df = _make_raw_news(3)
        df["asset"] = "BTC"
        df["provider"] = "cryptocompare"
        df.loc[1, "title"] = ""
        clean, result = validate_news(df)

        assert result.missing_required_count == 1
        assert result.valid_rows == 2

    def test_validate_removes_missing_url(self) -> None:
        """Rows with missing URLs should be removed."""
        df = _make_raw_news(3)
        df["asset"] = "BTC"
        df["provider"] = "cryptocompare"
        df.loc[0, "url"] = ""
        clean, result = validate_news(df)

        assert result.missing_required_count == 1

    def test_validate_removes_invalid_timestamps(self) -> None:
        """Rows with impossible timestamps should be removed."""
        df = _make_raw_news(3)
        df["asset"] = "BTC"
        df["provider"] = "cryptocompare"
        df.loc[0, "published_at"] = pd.Timestamp("1990-01-01", tz="UTC")
        clean, result = validate_news(df)

        assert result.invalid_timestamp_count == 1
        assert result.valid_rows == 2

    def test_validate_detects_missing_columns(self) -> None:
        """Missing required columns should fail validation."""
        df = pd.DataFrame({"title": ["Test"]})
        clean, result = validate_news(df)

        assert not result.is_valid
        assert len(result.missing_column_names) > 0

    def test_validate_empty_df(self) -> None:
        """Empty DataFrame should pass validation with 0 rows."""
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        clean, result = validate_news(df)

        assert result.is_valid
        assert result.valid_rows == 0

    def test_validate_sorts_output(self) -> None:
        """Validated output should be sorted by published_at."""
        df = pd.DataFrame({
            "asset": ["BTC"] * 3,
            "provider": ["test"] * 3,
            "published_at": pd.to_datetime(
                ["2024-01-03", "2024-01-01", "2024-01-02"], utc=True
            ),
            "title": ["C", "A", "B"],
            "url": ["c.com", "a.com", "b.com"],
            "source_name": ["S"] * 3,
        })
        clean, _ = validate_news(df)

        ts = clean["published_at"].tolist()
        assert ts == sorted(ts)


# ============================================================================
# BTC relevance filter tests
# ============================================================================
class TestBTCRelevance:
    """Test BTC keyword relevance filter."""

    def test_btc_keyword_match(self) -> None:
        """Should match BTC-related text."""
        assert is_btc_relevant("Bitcoin price hits new ATH")
        assert is_btc_relevant("BTC surges to $50k")
        assert is_btc_relevant("Spot Bitcoin ETF approved")

    def test_non_btc_no_match(self) -> None:
        """Should not match non-BTC text."""
        assert not is_btc_relevant("Ethereum merge complete")
        assert not is_btc_relevant("Solana NFT marketplace launches")
        assert not is_btc_relevant("")

    def test_case_insensitive(self) -> None:
        """Should be case insensitive."""
        assert is_btc_relevant("BITCOIN crashes")
        assert is_btc_relevant("btc dips below support")

    def test_custom_keywords(self) -> None:
        """Should support custom keyword lists."""
        assert is_btc_relevant("dogecoin to the moon", keywords=["dogecoin"])
        assert not is_btc_relevant("dogecoin to the moon", keywords=["bitcoin"])


# ============================================================================
# Ingestion service tests (mocked API)
# ============================================================================
class TestNewsIngestionService:
    """Test the news ingestion service with mocked CryptoCompare client."""

    def _make_service(self, tmp_path, raw_df: pd.DataFrame | None = None):
        """Create a NewsIngestionService with a mocked client."""
        store = NewsStore(base_dir=tmp_path / "processed")
        mock_client = AsyncMock(spec=CryptoCompareNewsClient)

        if raw_df is not None:
            mock_client.fetch_news.return_value = raw_df
        else:
            mock_client.fetch_news.return_value = _make_raw_news(20)

        return NewsIngestionService(store=store, client=mock_client), store

    def test_ingest_success(self, tmp_path) -> None:
        """Ingestion with valid data should succeed."""
        service, store = self._make_service(tmp_path)
        result = service.ingest_sync(
            asset="BTC",
            provider="cryptocompare",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 2, 1, tzinfo=timezone.utc),
        )

        assert result.is_success
        assert result.fetched_rows == 20
        assert result.stored_rows > 0
        assert result.error == ""

    def test_ingest_stores_to_parquet(self, tmp_path) -> None:
        """Ingested data should be persisted as Parquet."""
        service, store = self._make_service(tmp_path)
        service.ingest_sync(
            asset="BTC",
            provider="cryptocompare",
        )

        assert store.exists("BTC", "cryptocompare")
        df = store.read("BTC", "cryptocompare")
        assert len(df) > 0
        assert "published_at" in df.columns
        assert "title" in df.columns

    def test_ingest_handles_empty_response(self, tmp_path) -> None:
        """Ingestion should handle empty API response gracefully."""
        service, store = self._make_service(
            tmp_path,
            raw_df=pd.DataFrame(),
        )
        result = service.ingest_sync(asset="BTC", provider="cryptocompare")

        assert not result.is_success
        assert "No news data" in result.error

    def test_ingest_idempotent(self, tmp_path) -> None:
        """Running ingestion twice should not create duplicates."""
        service, store = self._make_service(tmp_path)

        result1 = service.ingest_sync(asset="BTC", provider="cryptocompare")
        result2 = service.ingest_sync(asset="BTC", provider="cryptocompare")

        df = store.read("BTC", "cryptocompare")
        assert result1.stored_rows == result2.stored_rows
        assert len(df) == result1.stored_rows

    def test_ingest_filters_non_btc(self, tmp_path) -> None:
        """Non-BTC articles should be filtered out."""
        raw = _make_raw_news(10, btc_relevant=False)
        service, store = self._make_service(tmp_path, raw_df=raw)
        result = service.ingest_sync(asset="BTC", provider="cryptocompare")

        # All articles should be filtered because they're about Ethereum
        assert result.btc_filtered_out == 10
        assert not result.is_success

    def test_ingest_handles_api_error(self, tmp_path) -> None:
        """Ingestion should catch and report API errors."""
        store = NewsStore(base_dir=tmp_path / "processed")
        mock_client = AsyncMock(spec=CryptoCompareNewsClient)
        mock_client.fetch_news.side_effect = Exception("Connection refused")

        service = NewsIngestionService(store=store, client=mock_client)
        result = service.ingest_sync(asset="BTC", provider="cryptocompare")

        assert not result.is_success
        assert "Connection refused" in result.error

    def test_ingest_mixed_relevance(self, tmp_path) -> None:
        """Should keep BTC articles and filter non-BTC ones."""
        btc = _make_raw_news(5, btc_relevant=True)
        non_btc = _make_raw_news(5, btc_relevant=False)
        # Avoid ID collisions
        non_btc["provider_article_id"] = [str(2000 + i) for i in range(5)]
        non_btc["url"] = [f"https://example.com/eth-{i}" for i in range(5)]
        mixed = pd.concat([btc, non_btc], ignore_index=True)

        service, store = self._make_service(tmp_path, raw_df=mixed)
        result = service.ingest_sync(asset="BTC", provider="cryptocompare")

        assert result.is_success
        assert result.btc_filtered_out == 5
        assert result.valid_rows == 5
