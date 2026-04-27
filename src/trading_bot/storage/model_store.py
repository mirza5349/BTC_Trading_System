"""Local persistence for trained model artifacts."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trading_bot.logging_config import get_logger

logger = get_logger(__name__)


class ModelStoreError(Exception):
    """Raised when model artifacts cannot be persisted or loaded."""


@dataclass(frozen=True)
class StoredModelArtifact:
    """Metadata describing one persisted fold model."""

    asset: str
    symbol: str
    timeframe: str
    model_version: str
    run_id: str
    fold_id: int
    path: str
    size_bytes: int


class ModelStore:
    """Persist fold-level model artifacts under deterministic local paths."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def run_dir(
        self,
        *,
        asset: str,
        symbol: str,
        timeframe: str,
        model_version: str,
        run_id: str,
    ) -> Path:
        """Return the storage directory for one model run."""
        return (
            self.base_dir
            / f"asset={asset}"
            / f"symbol={symbol}"
            / f"timeframe={timeframe}"
            / f"model_version={model_version}"
            / f"run_id={run_id}"
        )

    def model_path(
        self,
        *,
        asset: str,
        symbol: str,
        timeframe: str,
        model_version: str,
        run_id: str,
        fold_id: int,
    ) -> Path:
        """Return the persisted file path for one fold model."""
        return self.run_dir(
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=run_id,
        ) / f"fold_{fold_id}_model.pkl"

    def write_fold_model(
        self,
        model: Any,
        *,
        asset: str,
        symbol: str,
        timeframe: str,
        model_version: str,
        run_id: str,
        fold_id: int,
    ) -> StoredModelArtifact:
        """Persist one fold model with pickle serialization."""
        path = self.model_path(
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=run_id,
            fold_id=fold_id,
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with path.open("wb") as file_obj:
                pickle.dump(model, file_obj)
        except Exception as exc:
            raise ModelStoreError(f"Failed to write model artifact to {path}: {exc}") from exc

        artifact = StoredModelArtifact(
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=run_id,
            fold_id=fold_id,
            path=str(path),
            size_bytes=path.stat().st_size,
        )

        logger.info(
            "model_store_write_complete",
            path=str(path),
            fold_id=fold_id,
            run_id=run_id,
            size_bytes=artifact.size_bytes,
        )
        return artifact

    def read_fold_model(
        self,
        *,
        asset: str,
        symbol: str,
        timeframe: str,
        model_version: str,
        run_id: str,
        fold_id: int,
    ) -> Any:
        """Load one stored fold model from disk."""
        path = self.model_path(
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=run_id,
            fold_id=fold_id,
        )
        if not path.exists():
            raise ModelStoreError(f"Model artifact does not exist at {path}")

        try:
            with path.open("rb") as file_obj:
                return pickle.load(file_obj)
        except Exception as exc:
            raise ModelStoreError(f"Failed to read model artifact from {path}: {exc}") from exc
