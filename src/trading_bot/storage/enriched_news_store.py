"""Parquet-backed local persistence for FinBERT-enriched news records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from trading_bot.logging_config import get_logger
from trading_bot.schemas.nlp import validate_enriched_news_df

logger = get_logger(__name__)


class EnrichedNewsStoreError(Exception):
    """Raised when enriched news artifacts cannot be persisted or loaded."""


class EnrichedNewsStore:
    """Persist enriched news records by asset, provider, model, and version."""

    FILENAME = "enriched_news.parquet"

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def _path_for(
        self,
        *,
        asset: str,
        provider: str,
        model_name: str,
        enrichment_version: str,
    ) -> Path:
        return (
            self.base_dir
            / f"asset={asset}"
            / f"provider={provider}"
            / f"model={_model_slug(model_name)}"
            / f"enrichment_version={enrichment_version}"
            / self.FILENAME
        )

    def write(
        self,
        df: pd.DataFrame,
        *,
        asset: str,
        provider: str,
        model_name: str,
        enrichment_version: str,
    ) -> int:
        """Write or append enriched records with idempotent merge by enrichment_key."""
        if df.empty:
            return 0
        validate_enriched_news_df(df)

        path = self._path_for(
            asset=asset,
            provider=provider,
            model_name=model_name,
            enrichment_version=enrichment_version,
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if path.exists():
                existing = pd.read_parquet(path)
                merged = pd.concat([existing, df], ignore_index=True)
            else:
                merged = df.copy()

            if "enrichment_key" in merged.columns:
                merged = merged.drop_duplicates(subset=["enrichment_key"], keep="last")
            merged["published_at"] = pd.to_datetime(merged["published_at"], utc=True)
            merged = merged.sort_values("published_at").reset_index(drop=True)
            merged.to_parquet(path, index=False, engine="pyarrow")
        except Exception as exc:
            raise EnrichedNewsStoreError(f"Failed to write enriched news to {path}: {exc}") from exc

        logger.info("enriched_news_write_complete", path=str(path), rows=len(merged))
        return len(merged)

    def read(
        self,
        *,
        asset: str,
        provider: str,
        model_name: str,
        enrichment_version: str,
    ) -> pd.DataFrame:
        """Read one enriched-news dataset or return an empty DataFrame."""
        path = self._path_for(
            asset=asset,
            provider=provider,
            model_name=model_name,
            enrichment_version=enrichment_version,
        )
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            raise EnrichedNewsStoreError(f"Failed to read enriched news from {path}: {exc}") from exc
        if "published_at" in df.columns:
            df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
            df = df.sort_values("published_at").reset_index(drop=True)
        return df

    def get_info(
        self,
        *,
        asset: str,
        provider: str,
        model_name: str,
        enrichment_version: str,
    ) -> dict[str, Any]:
        """Return local enriched-news stats and label distribution."""
        df = self.read(
            asset=asset,
            provider=provider,
            model_name=model_name,
            enrichment_version=enrichment_version,
        )
        if df.empty:
            return {}

        label_counts = (
            df["sentiment_label"].value_counts().sort_index().to_dict()
            if "sentiment_label" in df.columns
            else {}
        )
        coverage = float(df["sentiment_label"].notna().mean()) if "sentiment_label" in df.columns else 0.0
        path = self._path_for(
            asset=asset,
            provider=provider,
            model_name=model_name,
            enrichment_version=enrichment_version,
        )
        return {
            "asset": asset,
            "provider": provider,
            "model_name": model_name,
            "enrichment_version": enrichment_version,
            "row_count": int(len(df)),
            "min_published_at": df["published_at"].min().isoformat(),
            "max_published_at": df["published_at"].max().isoformat(),
            "sentiment_label_counts": label_counts,
            "scoring_coverage": round(coverage, 6),
            "path": str(path),
        }


def _model_slug(model_name: str) -> str:
    """Return a stable path-safe model slug."""
    slug = model_name.split("/")[-1].strip().lower().replace("_", "-")
    return slug or "model"
