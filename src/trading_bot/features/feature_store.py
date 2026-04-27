"""Parquet-backed local storage for engineered feature datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from trading_bot.logging_config import get_logger
from trading_bot.schemas.features import INDEX_COLUMNS, FeatureDatasetName

logger = get_logger(__name__)

DATASET_FILENAMES: dict[FeatureDatasetName, str] = {
    "market_features": "market_features.parquet",
    "news_features": "news_features.parquet",
    "feature_table": "features.parquet",
}


class FeatureStoreError(Exception):
    """Raised when feature datasets cannot be persisted or loaded."""


@dataclass(frozen=True)
class FeatureSet:
    """Metadata for a stored feature dataset."""

    dataset_name: FeatureDatasetName
    asset: str
    symbol: str
    timeframe: str
    version: str
    path: str
    row_count: int
    column_count: int
    feature_names: list[str] = field(default_factory=list)
    start: datetime | None = None
    end: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ParquetFeatureStore:
    """Persist feature datasets under a deterministic local path layout."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def dataset_path(
        self,
        dataset_name: FeatureDatasetName,
        *,
        asset: str,
        symbol: str,
        timeframe: str,
        version: str,
    ) -> Path:
        """Return the on-disk path for a feature dataset."""
        filename = DATASET_FILENAMES[dataset_name]
        return (
            self.base_dir
            / f"asset={asset}"
            / f"symbol={symbol}"
            / f"timeframe={timeframe}"
            / f"version={version}"
            / filename
        )

    def write_dataset(
        self,
        dataset_name: FeatureDatasetName,
        df: pd.DataFrame,
        *,
        asset: str,
        symbol: str,
        timeframe: str,
        version: str,
        metadata: dict[str, Any] | None = None,
    ) -> FeatureSet:
        """Write a feature dataset to a stable Parquet path."""
        path = self.dataset_path(
            dataset_name,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            version=version,
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            dataset = _prepare_for_storage(df)
            dataset.to_parquet(path, index=False, engine="pyarrow")
        except Exception as exc:
            raise FeatureStoreError(f"Failed to write feature dataset to {path}: {exc}") from exc

        feature_set = _build_feature_set(
            dataset_name=dataset_name,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            version=version,
            path=path,
            df=dataset,
            metadata=metadata or {},
        )

        logger.info(
            "feature_store_write_complete",
            dataset_name=dataset_name,
            path=str(path),
            rows=feature_set.row_count,
            columns=feature_set.column_count,
        )

        return feature_set

    def read_dataset(
        self,
        dataset_name: FeatureDatasetName,
        *,
        asset: str,
        symbol: str,
        timeframe: str,
        version: str,
    ) -> pd.DataFrame:
        """Read a stored feature dataset or return an empty DataFrame."""
        path = self.dataset_path(
            dataset_name,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            version=version,
        )
        if not path.exists():
            return pd.DataFrame()

        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            raise FeatureStoreError(f"Failed to read feature dataset from {path}: {exc}") from exc

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def get_dataset_info(
        self,
        dataset_name: FeatureDatasetName,
        *,
        asset: str,
        symbol: str,
        timeframe: str,
        version: str,
    ) -> dict[str, Any]:
        """Return row counts, timestamp range, and columns for a dataset."""
        path = self.dataset_path(
            dataset_name,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            version=version,
        )
        if not path.exists():
            return {}

        df = self.read_dataset(
            dataset_name,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            version=version,
        )
        feature_set = _build_feature_set(
            dataset_name=dataset_name,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            version=version,
            path=path,
            df=df,
            metadata={},
        )
        return {
            "dataset_name": feature_set.dataset_name,
            "asset": feature_set.asset,
            "symbol": feature_set.symbol,
            "timeframe": feature_set.timeframe,
            "version": feature_set.version,
            "row_count": feature_set.row_count,
            "column_count": feature_set.column_count,
            "feature_names": feature_set.feature_names,
            "columns": list(df.columns),
            "min_timestamp": feature_set.start.isoformat() if feature_set.start else None,
            "max_timestamp": feature_set.end.isoformat() if feature_set.end else None,
            "file_size_mb": round(path.stat().st_size / (1024 * 1024), 3),
            "path": str(path),
        }

    def exists(
        self,
        dataset_name: FeatureDatasetName,
        *,
        asset: str,
        symbol: str,
        timeframe: str,
        version: str,
    ) -> bool:
        """Return whether a specific feature dataset exists locally."""
        return self.dataset_path(
            dataset_name,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            version=version,
        ).exists()

    def list_feature_sets(self, symbol: str | None = None) -> list[FeatureSet]:
        """List stored feature datasets under the base directory."""
        feature_sets: list[FeatureSet] = []

        for path in sorted(self.base_dir.glob("asset=*/symbol=*/timeframe=*/version=*/*.parquet")):
            dataset_name = _dataset_name_from_filename(path.name)
            if dataset_name is None:
                continue

            asset = path.parents[3].name.split("=", 1)[1]
            stored_symbol = path.parents[2].name.split("=", 1)[1]
            timeframe = path.parents[1].name.split("=", 1)[1]
            version = path.parent.name.split("=", 1)[1]

            if symbol is not None and stored_symbol != symbol:
                continue

            df = self.read_dataset(
                dataset_name,
                asset=asset,
                symbol=stored_symbol,
                timeframe=timeframe,
                version=version,
            )
            feature_sets.append(
                _build_feature_set(
                    dataset_name=dataset_name,
                    asset=asset,
                    symbol=stored_symbol,
                    timeframe=timeframe,
                    version=version,
                    path=path,
                    df=df,
                    metadata={},
                )
            )

        return feature_sets


def _prepare_for_storage(df: pd.DataFrame) -> pd.DataFrame:
    """Sort and normalize timestamps before persistence."""
    dataset = df.copy()
    if "timestamp" in dataset.columns:
        dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], utc=True)
        dataset = dataset.sort_values("timestamp").reset_index(drop=True)
    return dataset


def _build_feature_set(
    *,
    dataset_name: FeatureDatasetName,
    asset: str,
    symbol: str,
    timeframe: str,
    version: str,
    path: Path,
    df: pd.DataFrame,
    metadata: dict[str, Any],
) -> FeatureSet:
    """Create feature metadata from a DataFrame."""
    start: datetime | None = None
    end: datetime | None = None
    if not df.empty and "timestamp" in df.columns:
        timestamp_series = pd.to_datetime(df["timestamp"], utc=True)
        start = timestamp_series.min().to_pydatetime()
        end = timestamp_series.max().to_pydatetime()

    feature_names = [column for column in df.columns if column not in INDEX_COLUMNS]
    return FeatureSet(
        dataset_name=dataset_name,
        asset=asset,
        symbol=symbol,
        timeframe=timeframe,
        version=version,
        path=str(path),
        row_count=len(df),
        column_count=len(df.columns),
        feature_names=feature_names,
        start=start,
        end=end,
        metadata=metadata,
    )


def _dataset_name_from_filename(filename: str) -> FeatureDatasetName | None:
    """Infer a dataset name from the persisted filename."""
    for dataset_name, expected_filename in DATASET_FILENAMES.items():
        if filename == expected_filename:
            return dataset_name
    return None
