"""Parquet-based local storage for news data.

Stores normalized news articles as Parquet files organized by
asset and provider. Supports idempotent writes with deduplication
on merge.

Directory layout:
    data/processed/news/BTC/cryptocompare/news.parquet
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from trading_bot.logging_config import get_logger
from trading_bot.schemas.news_data import UNIQUE_KEY_FALLBACK, UNIQUE_KEY_PRIMARY

logger = get_logger(__name__)


class NewsStoreError(Exception):
    """Raised on storage read/write failures."""


class NewsStore:
    """Local Parquet file store for news data.

    Organizes data under a base directory:
        {base_dir}/{asset}/{provider}/news.parquet

    Provides idempotent append behavior — repeated writes with
    overlapping data are merged by deduplicating on provider_article_id
    (or url+published_at as fallback), keeping the latest record.
    """

    FILENAME = "news.parquet"

    def __init__(self, base_dir: str | Path) -> None:
        """Initialize the news store.

        Args:
            base_dir: Root directory for news data storage.
        """
        self.base_dir = Path(base_dir)

    def _path_for(self, asset: str, provider: str) -> Path:
        """Return the Parquet file path for an asset/provider pair."""
        return self.base_dir / asset / provider / self.FILENAME

    def write(
        self,
        df: pd.DataFrame,
        asset: str,
        provider: str,
    ) -> int:
        """Write or append news data, deduplicating on the unique key.

        If a file already exists, the new data is merged with existing data.
        Duplicate articles are resolved by keeping the latest record.

        Args:
            df: DataFrame conforming to the news schema.
            asset: Asset tag (e.g., "BTC").
            provider: Data provider name.

        Returns:
            Total row count after write.

        Raises:
            NewsStoreError: On write failure.
        """
        if df.empty:
            logger.debug("news_write_skip_empty", asset=asset, provider=provider)
            return 0

        path = self._path_for(asset, provider)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if path.exists():
                existing = pd.read_parquet(path)
                logger.debug(
                    "news_merge_existing",
                    existing_rows=len(existing),
                    new_rows=len(df),
                )
                merged = pd.concat([existing, df], ignore_index=True)
                merged = self._deduplicate(merged)
            else:
                merged = self._deduplicate(df)

            merged = merged.sort_values("published_at").reset_index(drop=True)
            merged.to_parquet(path, index=False, engine="pyarrow")

            logger.info(
                "news_write_complete",
                path=str(path),
                rows=len(merged),
                asset=asset,
                provider=provider,
            )
            return len(merged)

        except Exception as exc:
            raise NewsStoreError(f"Failed to write {path}: {exc}") from exc

    @staticmethod
    def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate news articles using the best available key.

        Uses provider_article_id when available and non-empty,
        falls back to url + published_at for records without IDs.
        """
        if df.empty:
            return df

        has_id_col = "provider_article_id" in df.columns
        if has_id_col:
            has_id = df["provider_article_id"].notna() & (df["provider_article_id"].astype(str).str.strip() != "")
            df_with_id = df[has_id].drop_duplicates(subset=UNIQUE_KEY_PRIMARY, keep="last")
            df_without_id = df[~has_id].drop_duplicates(subset=UNIQUE_KEY_FALLBACK, keep="last")
            return pd.concat([df_with_id, df_without_id], ignore_index=True)
        else:
            return df.drop_duplicates(subset=UNIQUE_KEY_FALLBACK, keep="last")

    def read(
        self,
        asset: str,
        provider: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Read news data from the store, optionally filtered by time range.

        Args:
            asset: Asset tag.
            provider: Data provider name.
            start: Optional start datetime filter (inclusive).
            end: Optional end datetime filter (inclusive).

        Returns:
            DataFrame of news data, or empty DataFrame if not found.
        """
        path = self._path_for(asset, provider)

        if not path.exists():
            logger.debug("news_read_not_found", path=str(path))
            return pd.DataFrame()

        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            logger.error("news_read_error", path=str(path), error=str(exc))
            return pd.DataFrame()

        if "published_at" in df.columns:
            df["published_at"] = pd.to_datetime(df["published_at"], utc=True)

        if start is not None:
            start_ts = pd.Timestamp(start)
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            else:
                start_ts = start_ts.tz_convert("UTC")
            df = df[df["published_at"] >= start_ts]

        if end is not None:
            end_ts = pd.Timestamp(end)
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize("UTC")
            else:
                end_ts = end_ts.tz_convert("UTC")
            df = df[df["published_at"] <= end_ts]

        return df.sort_values("published_at").reset_index(drop=True)

    def get_info(self, asset: str, provider: str) -> dict[str, Any]:
        """Get metadata about a stored news dataset.

        Args:
            asset: Asset tag.
            provider: Data provider name.

        Returns:
            Dict with row_count, timestamps, file size. Empty dict if absent.
        """
        path = self._path_for(asset, provider)

        if not path.exists():
            return {}

        try:
            df = pd.read_parquet(path)
            if df.empty:
                return {
                    "asset": asset,
                    "provider": provider,
                    "row_count": 0,
                    "path": str(path),
                }

            df["published_at"] = pd.to_datetime(df["published_at"], utc=True)

            # Source distribution
            source_counts: dict[str, int] = {}
            if "source_name" in df.columns:
                source_counts = df["source_name"].value_counts().head(10).to_dict()

            return {
                "asset": asset,
                "provider": provider,
                "row_count": len(df),
                "min_published_at": df["published_at"].min().isoformat(),
                "max_published_at": df["published_at"].max().isoformat(),
                "unique_sources": int(df["source_name"].nunique()) if "source_name" in df.columns else 0,
                "top_sources": source_counts,
                "file_size_mb": round(path.stat().st_size / (1024 * 1024), 3),
                "path": str(path),
            }
        except Exception as exc:
            logger.error("news_info_error", path=str(path), error=str(exc))
            return {"error": str(exc)}

    def exists(self, asset: str, provider: str) -> bool:
        """Check if data exists for an asset/provider pair."""
        return self._path_for(asset, provider).exists()

    def get_latest_timestamp(
        self, asset: str, provider: str
    ) -> datetime | None:
        """Get the latest published_at in the stored data.

        Useful for incremental ingestion.

        Args:
            asset: Asset tag.
            provider: Data provider name.

        Returns:
            Latest timestamp as datetime, or None if no data exists.
        """
        path = self._path_for(asset, provider)
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path, columns=["published_at"])
            if df.empty:
                return None
            ts = pd.to_datetime(df["published_at"], utc=True).max()
            return ts.to_pydatetime()
        except Exception:
            return None
