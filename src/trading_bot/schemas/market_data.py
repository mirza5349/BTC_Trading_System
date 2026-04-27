"""Normalized market data schema.

Defines the canonical internal representation of OHLCV candle data.
All ingestion pipelines must normalize to this schema before storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical column names and dtypes
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS: list[str] = [
    "symbol",
    "timeframe",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

OPTIONAL_COLUMNS: list[str] = [
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "source",
]

ALL_COLUMNS: list[str] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

# Expected dtypes for validation / coercion
COLUMN_DTYPES: dict[str, str] = {
    "symbol": "object",
    "timeframe": "object",
    "timestamp": "datetime64[ms, UTC]",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "close_time": "datetime64[ms, UTC]",
    "quote_asset_volume": "float64",
    "number_of_trades": "int64",
    "taker_buy_base_volume": "float64",
    "taker_buy_quote_volume": "float64",
    "source": "object",
}

# Unique key for deduplication
UNIQUE_KEY: list[str] = ["symbol", "timeframe", "timestamp"]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ValidationResult:
    """Result of validating a candle DataFrame."""

    is_valid: bool
    total_rows: int
    valid_rows: int
    duplicate_count: int
    invalid_ohlc_count: int
    negative_price_count: int
    missing_column_names: list[str]
    errors: list[str]

    @property
    def dropped_rows(self) -> int:
        return self.total_rows - self.valid_rows


def validate_candles(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationResult]:
    """Validate and clean a DataFrame of candle data.

    Checks:
    - Required columns are present
    - No duplicate timestamps (per symbol+timeframe)
    - No negative prices
    - OHLC relationship: low <= open, close, high; high >= open, close, low
    - Sorts by timestamp ascending

    Args:
        df: Raw candle DataFrame.

    Returns:
        Tuple of (cleaned DataFrame, ValidationResult).
    """
    errors: list[str] = []
    original_count = len(df)

    # -- Check required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return df, ValidationResult(
            is_valid=False,
            total_rows=original_count,
            valid_rows=0,
            duplicate_count=0,
            invalid_ohlc_count=0,
            negative_price_count=0,
            missing_column_names=missing,
            errors=[f"Missing required columns: {missing}"],
        )

    if df.empty:
        return df, ValidationResult(
            is_valid=True,
            total_rows=0,
            valid_rows=0,
            duplicate_count=0,
            invalid_ohlc_count=0,
            negative_price_count=0,
            missing_column_names=[],
            errors=[],
        )

    # -- Remove duplicate timestamps
    dup_mask = df.duplicated(subset=UNIQUE_KEY, keep="last")
    duplicate_count = int(dup_mask.sum())
    if duplicate_count > 0:
        errors.append(f"Removed {duplicate_count} duplicate rows")
        df = df[~dup_mask].copy()

    # -- Flag negative prices
    price_cols = ["open", "high", "low", "close", "volume"]
    neg_mask = (df[price_cols] < 0).any(axis=1)
    negative_price_count = int(neg_mask.sum())
    if negative_price_count > 0:
        errors.append(f"Removed {negative_price_count} rows with negative prices")
        df = df[~neg_mask].copy()

    # -- Flag invalid OHLC relationships (low > high)
    invalid_ohlc_mask = df["low"] > df["high"]
    invalid_ohlc_count = int(invalid_ohlc_mask.sum())
    if invalid_ohlc_count > 0:
        errors.append(f"Removed {invalid_ohlc_count} rows with low > high")
        df = df[~invalid_ohlc_mask].copy()

    # -- Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    valid_count = len(df)

    return df, ValidationResult(
        is_valid=valid_count > 0 and len(errors) == 0 or (valid_count > 0),
        total_rows=original_count,
        valid_rows=valid_count,
        duplicate_count=duplicate_count,
        invalid_ohlc_count=invalid_ohlc_count,
        negative_price_count=negative_price_count,
        missing_column_names=[],
        errors=errors,
    )


def normalize_candle_df(
    df: pd.DataFrame,
    symbol: str = "BTCUSDT",
    timeframe: str = "15m",
    source: str = "binance",
) -> pd.DataFrame:
    """Ensure a candle DataFrame conforms to the canonical schema.

    Adds missing optional columns with defaults, coerces dtypes,
    and ensures timestamp is timezone-aware UTC.

    Args:
        df: DataFrame with at least the price columns.
        symbol: Trading pair symbol.
        timeframe: Candle timeframe string.
        source: Data source identifier.

    Returns:
        Normalized DataFrame.
    """
    df = df.copy()

    # Rename Binance-style column if present
    if "open_time" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"open_time": "timestamp"})

    # Ensure metadata columns
    if "symbol" not in df.columns:
        df["symbol"] = symbol
    if "timeframe" not in df.columns:
        df["timeframe"] = timeframe
    if "source" not in df.columns:
        df["source"] = source

    # Ensure timestamp is datetime and UTC
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    if "close_time" in df.columns:
        df["close_time"] = pd.to_datetime(df["close_time"], utc=True)

    # Coerce numeric columns
    for col in ["open", "high", "low", "close", "volume",
                 "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "number_of_trades" in df.columns:
        df["number_of_trades"] = pd.to_numeric(df["number_of_trades"], errors="coerce").astype(
            "Int64"
        )

    # Add missing optional columns with defaults
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            if col in ("close_time",):
                df[col] = pd.NaT
            elif col == "source":
                df[col] = source
            elif col == "number_of_trades":
                df[col] = pd.array([0] * len(df), dtype="Int64")
            else:
                df[col] = np.float64(0.0)

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Select and order columns
    available = [c for c in ALL_COLUMNS if c in df.columns]
    return df[available]
