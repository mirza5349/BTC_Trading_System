"""Normalized news article schema.

Defines the canonical internal representation of crypto news records.
All news ingestion pipelines must normalize to this schema before storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical column names
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS: list[str] = [
    "asset",
    "provider",
    "published_at",
    "title",
    "url",
    "source_name",
]

OPTIONAL_COLUMNS: list[str] = [
    "provider_article_id",
    "summary",
    "body",
    "author",
    "language",
    "source_domain",
    "category",
    "tags",
    "raw_symbol_refs",
    "ingested_at",
]

ALL_COLUMNS: list[str] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

# Unique key for deduplication — prefer provider_article_id when present
UNIQUE_KEY_PRIMARY: list[str] = ["provider", "provider_article_id"]
UNIQUE_KEY_FALLBACK: list[str] = ["provider", "url", "published_at"]

# BTC-relevance keywords for lightweight filtering
BTC_KEYWORDS: list[str] = [
    "btc",
    "bitcoin",
    "spot bitcoin etf",
    "bitcoin etf",
    "bitcoin network",
    "bitcoin miner",
    "bitcoin miners",
    "bitcoin mining",
    "bitcoin halving",
    "satoshi",
    "lightning network",
]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NewsValidationResult:
    """Result of validating a news DataFrame."""

    is_valid: bool
    total_rows: int
    valid_rows: int
    duplicate_count: int
    missing_required_count: int
    invalid_timestamp_count: int
    missing_column_names: list[str]
    errors: list[str]


def validate_news(df: pd.DataFrame) -> tuple[pd.DataFrame, NewsValidationResult]:
    """Validate and clean a DataFrame of news records.

    Checks:
    - Required columns are present
    - No duplicate articles (by provider_article_id or url+published_at)
    - No missing titles or URLs
    - No impossible timestamps (future or before 2000)
    - Sorts by published_at ascending

    Args:
        df: Raw news DataFrame.

    Returns:
        Tuple of (cleaned DataFrame, NewsValidationResult).
    """
    errors: list[str] = []
    original_count = len(df)

    # -- Check required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return df, NewsValidationResult(
            is_valid=False,
            total_rows=original_count,
            valid_rows=0,
            duplicate_count=0,
            missing_required_count=0,
            invalid_timestamp_count=0,
            missing_column_names=missing,
            errors=[f"Missing required columns: {missing}"],
        )

    if df.empty:
        return df, NewsValidationResult(
            is_valid=True,
            total_rows=0,
            valid_rows=0,
            duplicate_count=0,
            missing_required_count=0,
            invalid_timestamp_count=0,
            missing_column_names=[],
            errors=[],
        )

    # -- Drop rows missing critical fields
    critical_mask = (
        df["title"].isna()
        | (df["title"].str.strip() == "")
        | df["url"].isna()
        | (df["url"].str.strip() == "")
        | df["published_at"].isna()
    )
    missing_required_count = int(critical_mask.sum())
    if missing_required_count > 0:
        errors.append(f"Removed {missing_required_count} rows with missing title/url/published_at")
        df = df[~critical_mask].copy()

    if df.empty:
        return df, NewsValidationResult(
            is_valid=False,
            total_rows=original_count,
            valid_rows=0,
            duplicate_count=0,
            missing_required_count=missing_required_count,
            invalid_timestamp_count=0,
            missing_column_names=[],
            errors=errors,
        )

    # -- Reject impossible timestamps (before 2000 or in the future)
    now_utc = pd.Timestamp.now(tz="UTC")
    min_valid = pd.Timestamp("2000-01-01", tz="UTC")
    ts_col = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    invalid_ts_mask = ts_col.isna() | (ts_col < min_valid) | (ts_col > now_utc + pd.Timedelta(days=1))
    invalid_timestamp_count = int(invalid_ts_mask.sum())
    if invalid_timestamp_count > 0:
        errors.append(f"Removed {invalid_timestamp_count} rows with invalid timestamps")
        df = df[~invalid_ts_mask].copy()

    # -- Deduplicate
    duplicate_count = 0
    if "provider_article_id" in df.columns and df["provider_article_id"].notna().any():
        # Use provider_article_id when available
        has_id = df["provider_article_id"].notna() & (df["provider_article_id"] != "")
        df_with_id = df[has_id]
        df_without_id = df[~has_id]

        dup_mask_id = df_with_id.duplicated(subset=UNIQUE_KEY_PRIMARY, keep="last")
        dup_count_id = int(dup_mask_id.sum())

        dup_mask_fallback = df_without_id.duplicated(subset=UNIQUE_KEY_FALLBACK, keep="last")
        dup_count_fallback = int(dup_mask_fallback.sum())

        duplicate_count = dup_count_id + dup_count_fallback

        df = pd.concat(
            [df_with_id[~dup_mask_id], df_without_id[~dup_mask_fallback]],
            ignore_index=True,
        )
    else:
        dup_mask = df.duplicated(subset=UNIQUE_KEY_FALLBACK, keep="last")
        duplicate_count = int(dup_mask.sum())
        df = df[~dup_mask].copy()

    if duplicate_count > 0:
        errors.append(f"Removed {duplicate_count} duplicate rows")

    # -- Sort by published_at ascending
    df = df.sort_values("published_at").reset_index(drop=True)

    valid_count = len(df)

    return df, NewsValidationResult(
        is_valid=valid_count > 0,
        total_rows=original_count,
        valid_rows=valid_count,
        duplicate_count=duplicate_count,
        missing_required_count=missing_required_count,
        invalid_timestamp_count=invalid_timestamp_count,
        missing_column_names=[],
        errors=errors,
    )


def normalize_news_df(
    df: pd.DataFrame,
    asset: str = "BTC",
    provider: str = "cryptocompare",
) -> pd.DataFrame:
    """Ensure a news DataFrame conforms to the canonical schema.

    Adds missing optional columns with defaults, normalizes text fields,
    and ensures published_at is timezone-aware UTC.

    Args:
        df: DataFrame with at least the required columns.
        asset: Asset tag.
        provider: Data source identifier.

    Returns:
        Normalized DataFrame.
    """
    df = df.copy()

    # Ensure metadata columns
    if "asset" not in df.columns:
        df["asset"] = asset
    if "provider" not in df.columns:
        df["provider"] = provider

    # Ensure published_at is UTC datetime
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")

    # Normalize text fields — strip whitespace
    for col in ("title", "summary", "body", "author", "source_name"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Normalize URLs
    if "url" in df.columns:
        df["url"] = df["url"].astype(str).str.strip()

    # Extract source_domain from URL if missing
    if "source_domain" not in df.columns and "url" in df.columns:
        df["source_domain"] = df["url"].apply(_extract_domain)

    # Add ingested_at timestamp
    if "ingested_at" not in df.columns:
        df["ingested_at"] = pd.Timestamp.now(tz="UTC")
    else:
        df["ingested_at"] = pd.to_datetime(df["ingested_at"], utc=True, errors="coerce")

    # Add missing optional columns with defaults
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            if col == "ingested_at":
                df[col] = pd.Timestamp.now(tz="UTC")
            elif col == "provider_article_id":
                df[col] = ""
            else:
                df[col] = ""

    # Sort by published_at
    df = df.sort_values("published_at").reset_index(drop=True)

    # Select and order columns
    available = [c for c in ALL_COLUMNS if c in df.columns]
    return df[available]


def is_btc_relevant(text: str, keywords: list[str] | None = None) -> bool:
    """Check if text is relevant to BTC using keyword matching.

    Args:
        text: Text to check (title, body, categories, etc.).
        keywords: List of keywords to match against. Uses BTC_KEYWORDS default.

    Returns:
        True if any keyword matches.
    """
    if not text:
        return False
    keywords = keywords or BTC_KEYWORDS
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def _extract_domain(url: str) -> str:
    """Extract the domain from a URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc or ""
    except Exception:
        return ""
