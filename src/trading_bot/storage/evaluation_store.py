"""Local persistence for walk-forward evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from trading_bot.logging_config import get_logger

logger = get_logger(__name__)

FOLD_METRICS_FILENAME = "fold_metrics.parquet"
PREDICTIONS_FILENAME = "predictions.parquet"
FEATURE_IMPORTANCE_FILENAME = "feature_importance.parquet"
AGGREGATE_FEATURE_IMPORTANCE_FILENAME = "aggregate_feature_importance.parquet"
AGGREGATE_METRICS_FILENAME = "aggregate_metrics.json"
RUN_METADATA_FILENAME = "run_metadata.json"
REPORT_FILENAME = "report.json"


class EvaluationStoreError(Exception):
    """Raised when evaluation artifacts cannot be persisted or loaded."""


class EvaluationStore:
    """Persist training run metrics, predictions, and reports to local storage."""

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
        """Return the storage directory for one evaluation run."""
        return (
            self.base_dir
            / f"asset={asset}"
            / f"symbol={symbol}"
            / f"timeframe={timeframe}"
            / f"model_version={model_version}"
            / f"run_id={run_id}"
        )

    def write_fold_metrics(
        self,
        df: pd.DataFrame,
        *,
        asset: str,
        symbol: str,
        timeframe: str,
        model_version: str,
        run_id: str,
    ) -> Path:
        """Persist per-fold metrics as Parquet."""
        return self._write_parquet(
            df,
            filename=FOLD_METRICS_FILENAME,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=run_id,
        )

    def read_fold_metrics(
        self,
        *,
        asset: str,
        symbol: str,
        timeframe: str,
        model_version: str,
        run_id: str,
    ) -> pd.DataFrame:
        """Load persisted per-fold metrics."""
        return self._read_parquet(
            filename=FOLD_METRICS_FILENAME,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=run_id,
        )

    def write_predictions(
        self,
        df: pd.DataFrame,
        *,
        asset: str,
        symbol: str,
        timeframe: str,
        model_version: str,
        run_id: str,
    ) -> Path:
        """Persist validation predictions as Parquet."""
        return self._write_parquet(
            df,
            filename=PREDICTIONS_FILENAME,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=run_id,
        )

    def read_predictions(
        self,
        *,
        asset: str,
        symbol: str,
        timeframe: str,
        model_version: str,
        run_id: str,
    ) -> pd.DataFrame:
        """Load persisted validation predictions."""
        return self._read_parquet(
            filename=PREDICTIONS_FILENAME,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=run_id,
        )

    def write_feature_importance(
        self,
        df: pd.DataFrame,
        *,
        asset: str,
        symbol: str,
        timeframe: str,
        model_version: str,
        run_id: str,
        aggregate: bool = False,
    ) -> Path:
        """Persist per-fold or aggregate feature importance as Parquet."""
        filename = (
            AGGREGATE_FEATURE_IMPORTANCE_FILENAME if aggregate else FEATURE_IMPORTANCE_FILENAME
        )
        return self._write_parquet(
            df,
            filename=filename,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=run_id,
        )

    def read_feature_importance(
        self,
        *,
        asset: str,
        symbol: str,
        timeframe: str,
        model_version: str,
        run_id: str,
        aggregate: bool = False,
    ) -> pd.DataFrame:
        """Load persisted per-fold or aggregate feature importance."""
        filename = (
            AGGREGATE_FEATURE_IMPORTANCE_FILENAME if aggregate else FEATURE_IMPORTANCE_FILENAME
        )
        return self._read_parquet(
            filename=filename,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=run_id,
        )

    def write_aggregate_metrics(
        self,
        payload: dict[str, Any],
        *,
        asset: str,
        symbol: str,
        timeframe: str,
        model_version: str,
        run_id: str,
    ) -> Path:
        """Persist aggregate metrics as JSON."""
        return self._write_json(
            payload,
            filename=AGGREGATE_METRICS_FILENAME,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=run_id,
        )

    def read_aggregate_metrics(
        self,
        *,
        asset: str,
        symbol: str,
        timeframe: str,
        model_version: str,
        run_id: str,
    ) -> dict[str, Any]:
        """Load aggregate metrics JSON."""
        return self._read_json(
            filename=AGGREGATE_METRICS_FILENAME,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=run_id,
        )

    def write_run_metadata(
        self,
        payload: dict[str, Any],
        *,
        asset: str,
        symbol: str,
        timeframe: str,
        model_version: str,
        run_id: str,
    ) -> Path:
        """Persist run metadata as JSON."""
        return self._write_json(
            payload,
            filename=RUN_METADATA_FILENAME,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=run_id,
        )

    def read_run_metadata(
        self,
        *,
        asset: str,
        symbol: str,
        timeframe: str,
        model_version: str,
        run_id: str,
    ) -> dict[str, Any]:
        """Load run metadata JSON."""
        return self._read_json(
            filename=RUN_METADATA_FILENAME,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=run_id,
        )

    def write_report(
        self,
        payload: dict[str, Any],
        *,
        asset: str,
        symbol: str,
        timeframe: str,
        model_version: str,
        run_id: str,
    ) -> Path:
        """Persist the compact evaluation report as JSON."""
        return self._write_json(
            payload,
            filename=REPORT_FILENAME,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=run_id,
        )

    def read_report(
        self,
        *,
        asset: str,
        symbol: str,
        timeframe: str,
        model_version: str,
        run_id: str,
    ) -> dict[str, Any]:
        """Load the compact evaluation report JSON."""
        return self._read_json(
            filename=REPORT_FILENAME,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=run_id,
        )

    def list_runs(
        self,
        *,
        asset: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        model_version: str | None = None,
    ) -> list[dict[str, Any]]:
        """List stored evaluation runs from their metadata files."""
        if not self.base_dir.exists():
            return []

        pattern = "asset=*/symbol=*/timeframe=*/model_version=*/run_id=*/run_metadata.json"
        runs: list[dict[str, Any]] = []
        for metadata_path in sorted(self.base_dir.glob(pattern)):
            stored_asset = metadata_path.parents[4].name.split("=", 1)[1]
            stored_symbol = metadata_path.parents[3].name.split("=", 1)[1]
            stored_timeframe = metadata_path.parents[2].name.split("=", 1)[1]
            stored_model_version = metadata_path.parents[1].name.split("=", 1)[1]
            stored_run_id = metadata_path.parent.name.split("=", 1)[1]

            if asset is not None and stored_asset != asset:
                continue
            if symbol is not None and stored_symbol != symbol:
                continue
            if timeframe is not None and stored_timeframe != timeframe:
                continue
            if model_version is not None and stored_model_version != model_version:
                continue

            payload = json.loads(metadata_path.read_text())
            payload.setdefault("asset", stored_asset)
            payload.setdefault("symbol", stored_symbol)
            payload.setdefault("timeframe", stored_timeframe)
            payload.setdefault("model_version", stored_model_version)
            payload.setdefault("run_id", stored_run_id)
            runs.append(payload)

        runs.sort(
            key=lambda row: (str(row.get("created_at", "")), str(row.get("run_id", ""))),
            reverse=True,
        )
        return runs

    def _write_parquet(
        self,
        df: pd.DataFrame,
        *,
        filename: str,
        asset: str,
        symbol: str,
        timeframe: str,
        model_version: str,
        run_id: str,
    ) -> Path:
        """Persist a tabular artifact to the run directory."""
        path = self.run_dir(
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=run_id,
        ) / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        dataset = df.copy()
        if "timestamp" in dataset.columns:
            dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], utc=True)
            dataset = dataset.sort_values("timestamp").reset_index(drop=True)

        try:
            dataset.to_parquet(path, index=False, engine="pyarrow")
        except Exception as exc:
            raise EvaluationStoreError(
                f"Failed to write evaluation artifact to {path}: {exc}"
            ) from exc

        logger.info(
            "evaluation_store_write_complete",
            path=str(path),
            rows=len(dataset),
            columns=len(dataset.columns),
        )
        return path

    def _read_parquet(
        self,
        *,
        filename: str,
        asset: str,
        symbol: str,
        timeframe: str,
        model_version: str,
        run_id: str,
    ) -> pd.DataFrame:
        """Load a tabular run artifact or return an empty DataFrame."""
        path = self.run_dir(
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=run_id,
        ) / filename
        if not path.exists():
            return pd.DataFrame()

        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            raise EvaluationStoreError(
                f"Failed to read evaluation artifact from {path}: {exc}"
            ) from exc

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _write_json(
        self,
        payload: dict[str, Any],
        *,
        filename: str,
        asset: str,
        symbol: str,
        timeframe: str,
        model_version: str,
        run_id: str,
    ) -> Path:
        """Persist a JSON artifact to the run directory."""
        path = self.run_dir(
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=run_id,
        ) / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        except Exception as exc:
            raise EvaluationStoreError(f"Failed to write JSON artifact to {path}: {exc}") from exc

        logger.info("evaluation_store_write_json_complete", path=str(path))
        return path

    def _read_json(
        self,
        *,
        filename: str,
        asset: str,
        symbol: str,
        timeframe: str,
        model_version: str,
        run_id: str,
    ) -> dict[str, Any]:
        """Load a JSON run artifact or return an empty dict."""
        path = self.run_dir(
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=run_id,
        ) / filename
        if not path.exists():
            return {}

        try:
            return json.loads(path.read_text())
        except Exception as exc:
            raise EvaluationStoreError(f"Failed to read JSON artifact from {path}: {exc}") from exc
