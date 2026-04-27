"""Parquet-backed local storage for labels and supervised datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from trading_bot.logging_config import get_logger
from trading_bot.schemas.datasets import DatasetArtifactName

logger = get_logger(__name__)

DATASET_FILENAMES: dict[DatasetArtifactName, str] = {
    "labels": "labels.parquet",
    "supervised_dataset": "dataset.parquet",
}


class DatasetStoreError(Exception):
    """Raised when dataset artifacts cannot be persisted or loaded."""


@dataclass(frozen=True)
class StoredDatasetArtifact:
    """Metadata for a stored label or supervised dataset artifact."""

    artifact_name: DatasetArtifactName
    asset: str
    symbol: str
    timeframe: str
    label_version: str
    dataset_version: str | None
    path: str
    row_count: int
    column_count: int
    columns: list[str] = field(default_factory=list)
    start: datetime | None = None
    end: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetStore:
    """Persist labels and supervised datasets under deterministic local paths."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def artifact_path(
        self,
        artifact_name: DatasetArtifactName,
        *,
        asset: str,
        symbol: str,
        timeframe: str,
        label_version: str,
        dataset_version: str | None = None,
    ) -> Path:
        """Return the on-disk path for a label or supervised dataset artifact."""
        base_path = (
            self.base_dir
            / f"asset={asset}"
            / f"symbol={symbol}"
            / f"timeframe={timeframe}"
            / f"label_version={label_version}"
        )

        if artifact_name == "supervised_dataset":
            if dataset_version is None:
                raise ValueError("dataset_version is required for supervised_dataset artifacts")
            base_path = base_path / f"dataset_version={dataset_version}"

        return base_path / DATASET_FILENAMES[artifact_name]

    def write_artifact(
        self,
        artifact_name: DatasetArtifactName,
        df: pd.DataFrame,
        *,
        asset: str,
        symbol: str,
        timeframe: str,
        label_version: str,
        dataset_version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StoredDatasetArtifact:
        """Write a dataset artifact to a stable Parquet path."""
        path = self.artifact_path(
            artifact_name,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            label_version=label_version,
            dataset_version=dataset_version,
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            dataset = _prepare_for_storage(df)
            dataset.to_parquet(path, index=False, engine="pyarrow")
        except Exception as exc:
            raise DatasetStoreError(f"Failed to write dataset artifact to {path}: {exc}") from exc

        artifact = _build_artifact_metadata(
            artifact_name=artifact_name,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            label_version=label_version,
            dataset_version=dataset_version,
            path=path,
            df=dataset,
            metadata=metadata or {},
        )

        logger.info(
            "dataset_store_write_complete",
            artifact_name=artifact_name,
            path=str(path),
            rows=artifact.row_count,
            columns=artifact.column_count,
        )

        return artifact

    def read_artifact(
        self,
        artifact_name: DatasetArtifactName,
        *,
        asset: str,
        symbol: str,
        timeframe: str,
        label_version: str,
        dataset_version: str | None = None,
    ) -> pd.DataFrame:
        """Read a stored label or supervised dataset artifact."""
        path = self.artifact_path(
            artifact_name,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            label_version=label_version,
            dataset_version=dataset_version,
        )
        if not path.exists():
            return pd.DataFrame()

        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            raise DatasetStoreError(f"Failed to read dataset artifact from {path}: {exc}") from exc

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def get_artifact_info(
        self,
        artifact_name: DatasetArtifactName,
        *,
        asset: str,
        symbol: str,
        timeframe: str,
        label_version: str,
        dataset_version: str | None = None,
    ) -> dict[str, Any]:
        """Return row counts, timestamp range, and columns for an artifact."""
        path = self.artifact_path(
            artifact_name,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            label_version=label_version,
            dataset_version=dataset_version,
        )
        if not path.exists():
            return {}

        df = self.read_artifact(
            artifact_name,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            label_version=label_version,
            dataset_version=dataset_version,
        )
        artifact = _build_artifact_metadata(
            artifact_name=artifact_name,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            label_version=label_version,
            dataset_version=dataset_version,
            path=path,
            df=df,
            metadata={},
        )

        return {
            "artifact_name": artifact.artifact_name,
            "asset": artifact.asset,
            "symbol": artifact.symbol,
            "timeframe": artifact.timeframe,
            "label_version": artifact.label_version,
            "dataset_version": artifact.dataset_version,
            "row_count": artifact.row_count,
            "column_count": artifact.column_count,
            "columns": artifact.columns,
            "min_timestamp": artifact.start.isoformat() if artifact.start else None,
            "max_timestamp": artifact.end.isoformat() if artifact.end else None,
            "file_size_mb": round(path.stat().st_size / (1024 * 1024), 3),
            "path": str(path),
        }

    def exists(
        self,
        artifact_name: DatasetArtifactName,
        *,
        asset: str,
        symbol: str,
        timeframe: str,
        label_version: str,
        dataset_version: str | None = None,
    ) -> bool:
        """Return whether a specific artifact exists locally."""
        return self.artifact_path(
            artifact_name,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            label_version=label_version,
            dataset_version=dataset_version,
        ).exists()


def _prepare_for_storage(df: pd.DataFrame) -> pd.DataFrame:
    """Sort and normalize timestamps before persistence."""
    dataset = df.copy()
    if "timestamp" in dataset.columns:
        dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], utc=True)
        dataset = dataset.sort_values("timestamp").reset_index(drop=True)
    return dataset


def _build_artifact_metadata(
    *,
    artifact_name: DatasetArtifactName,
    asset: str,
    symbol: str,
    timeframe: str,
    label_version: str,
    dataset_version: str | None,
    path: Path,
    df: pd.DataFrame,
    metadata: dict[str, Any],
) -> StoredDatasetArtifact:
    """Create artifact metadata from a stored DataFrame."""
    start: datetime | None = None
    end: datetime | None = None
    if not df.empty and "timestamp" in df.columns:
        timestamp_series = pd.to_datetime(df["timestamp"], utc=True)
        start = timestamp_series.min().to_pydatetime()
        end = timestamp_series.max().to_pydatetime()

    return StoredDatasetArtifact(
        artifact_name=artifact_name,
        asset=asset,
        symbol=symbol,
        timeframe=timeframe,
        label_version=label_version,
        dataset_version=dataset_version,
        path=str(path),
        row_count=len(df),
        column_count=len(df.columns),
        columns=list(df.columns),
        start=start,
        end=end,
        metadata=metadata,
    )
