"""Tests for the walk-forward LightGBM training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_bot.models.train import WalkForwardTrainingPipeline
from trading_bot.storage.dataset_store import DatasetStore
from trading_bot.storage.evaluation_store import EvaluationStore
from trading_bot.storage.model_store import ModelStore


class FakeBooster:
    """Pickleable fake booster exposing feature importance."""

    def __init__(self, feature_count: int) -> None:
        self.feature_count = feature_count

    def feature_importance(self, importance_type: str = "gain") -> np.ndarray:
        """Return a descending deterministic importance vector."""
        del importance_type
        return np.arange(self.feature_count, 0, -1, dtype=float)


class FakeLGBMClassifier:
    """Pickleable fake LightGBM sklearn wrapper for deterministic tests."""

    fail_cuda: bool = False

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.feature_importances_: np.ndarray | None = None
        self.booster_: FakeBooster | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> FakeLGBMClassifier:
        """Record feature importances and optionally fail on CUDA requests."""
        device_type = self.kwargs.get("device_type", "cpu")
        if self.fail_cuda and device_type in {"cuda", "gpu"}:
            raise RuntimeError("CUDA backend unavailable")

        self.feature_importances_ = np.arange(x.shape[1], 0, -1, dtype=float)
        self.booster_ = FakeBooster(x.shape[1])
        self.base_rate_ = float(y.mean())
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Generate deterministic probabilities from the signal feature."""
        signal = pd.to_numeric(x["f_signal"], errors="raise").to_numpy(dtype=float)
        probabilities = np.clip(0.2 + 0.6 * signal, 0.01, 0.99)
        return np.column_stack([1.0 - probabilities, probabilities])


class FakeLightGBMModule:
    """Minimal fake module matching the import contract used by the trainer."""

    __version__ = "test-lightgbm"
    LGBMClassifier = FakeLGBMClassifier


def _make_supervised_dataset(row_count: int = 12) -> pd.DataFrame:
    """Create a deterministic supervised dataset with numeric and non-numeric columns."""
    timestamps = pd.date_range("2024-01-01", periods=row_count, freq="15min", tz="UTC")
    signal = np.linspace(0.05, 0.95, row_count)
    targets = np.array([0, 1] * (row_count // 2), dtype=int)
    future_returns = np.linspace(-0.01, 0.02, row_count)

    return pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "timeframe": "15m",
            "timestamp": timestamps,
            "f_signal": signal,
            "f_volume": np.linspace(10.0, 30.0, row_count),
            "text_tag": ["btc"] * row_count,
            "target_up_2bars": targets,
            "future_return_2bars": future_returns,
        }
    )


def _make_pipeline(
    tmp_path: Path,
    *,
    requested_device: str = "cpu",
    allow_cuda_fallback_to_cpu: bool = True,
) -> tuple[WalkForwardTrainingPipeline, DatasetStore, EvaluationStore, ModelStore]:
    """Create a training pipeline backed by temporary local stores."""
    dataset_store = DatasetStore(tmp_path / "datasets")
    evaluation_store = EvaluationStore(tmp_path / "evaluation")
    model_store = ModelStore(tmp_path / "models")

    pipeline = WalkForwardTrainingPipeline(
        dataset_store=dataset_store,
        model_store=model_store,
        evaluation_store=evaluation_store,
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        label_version="v1",
        dataset_version="v1",
        model_version="v1",
        primary_target="target_up_2bars",
        random_seed=42,
        save_fold_models=True,
        requested_device=requested_device,
        allow_cuda_fallback_to_cpu=allow_cuda_fallback_to_cpu,
        probability_threshold=0.5,
        include_feature_patterns=[],
        exclude_feature_patterns=[],
        walk_forward_mode="expanding_window",
        min_train_rows=6,
        validation_rows=2,
        step_rows=2,
        max_folds=None,
        rolling_train_rows=None,
        lightgbm_params={
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "n_estimators": 50,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "device_type": "cpu",
        },
    )
    return pipeline, dataset_store, evaluation_store, model_store


def _write_supervised_dataset(dataset_store: DatasetStore, df: pd.DataFrame) -> None:
    """Persist a supervised dataset artifact for training tests."""
    dataset_store.write_artifact(
        "supervised_dataset",
        df,
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        label_version="v1",
        dataset_version="v1",
    )


def _make_fake_lightgbm_module(*, fail_cuda: bool = False) -> Any:
    """Create a fake LightGBM module for deterministic tests."""
    FakeLGBMClassifier.fail_cuda = fail_cuda
    return FakeLightGBMModule()


def test_run_training_persists_metrics_predictions_models_and_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """CPU training should produce leak-safe folds and persist evaluation artifacts."""
    pipeline, dataset_store, evaluation_store, _ = _make_pipeline(tmp_path)
    _write_supervised_dataset(dataset_store, _make_supervised_dataset())

    monkeypatch.setattr(
        "trading_bot.models.train._import_lightgbm_module",
        lambda: _make_fake_lightgbm_module(),
    )

    result = pipeline.run_training(run_id="RUNCPU")

    assert result.run_id == "RUNCPU"
    assert result.requested_device == "cpu"
    assert result.effective_device == "cpu"
    assert result.feature_columns == ["f_signal", "f_volume"]
    assert result.fold_metrics["fold_id"].tolist() == [0, 1, 2]
    assert list(result.predictions.columns) == [
        "symbol",
        "timeframe",
        "timestamp",
        "fold_id",
        "y_true",
        "y_pred",
        "y_proba",
    ]
    assert result.metadata["lightgbm_version"] == "test-lightgbm"
    assert result.metadata["requested_device"] == "cpu"
    assert result.metadata["effective_device"] == "cpu"
    assert result.metadata["feature_version"] == "unknown"
    assert result.aggregate_feature_importance.iloc[0]["feature_name"] == "f_signal"
    assert len(result.model_paths) == 3

    fold_metrics_path = (
        tmp_path
        / "evaluation"
        / "asset=BTC"
        / "symbol=BTCUSDT"
        / "timeframe=15m"
        / "model_version=v1"
        / "run_id=RUNCPU"
        / "fold_metrics.parquet"
    )
    predictions_path = fold_metrics_path.with_name("predictions.parquet")
    report_path = fold_metrics_path.with_name("report.json")
    model_path = (
        tmp_path
        / "models"
        / "asset=BTC"
        / "symbol=BTCUSDT"
        / "timeframe=15m"
        / "model_version=v1"
        / "run_id=RUNCPU"
        / "fold_0_model.pkl"
    )

    assert fold_metrics_path.exists()
    assert predictions_path.exists()
    assert report_path.exists()
    assert model_path.exists()
    assert not evaluation_store.read_fold_metrics(
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        model_version="v1",
        run_id="RUNCPU",
    ).empty


def test_run_training_records_requested_feature_version_in_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Training provenance should record the feature version used for the rebuilt dataset."""
    pipeline, dataset_store, _, _ = _make_pipeline(tmp_path)
    _write_supervised_dataset(dataset_store, _make_supervised_dataset())

    monkeypatch.setattr(
        "trading_bot.models.train._import_lightgbm_module",
        lambda: _make_fake_lightgbm_module(),
    )

    result = pipeline.run_training(run_id="RUNFV", feature_version="v2_sentiment")

    assert result.metadata["feature_version"] == "v2_sentiment"


def test_run_training_falls_back_to_cpu_when_cuda_requested_and_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """CUDA requests should fall back to CPU cleanly when LightGBM CUDA fails."""
    pipeline, dataset_store, _, _ = _make_pipeline(tmp_path, requested_device="cuda")
    _write_supervised_dataset(dataset_store, _make_supervised_dataset())

    monkeypatch.setattr(
        "trading_bot.models.train._import_lightgbm_module",
        lambda: _make_fake_lightgbm_module(fail_cuda=True),
    )

    result = pipeline.run_training(run_id="RUNCUDA")

    assert result.requested_device == "cuda"
    assert result.effective_device == "cpu"
    assert result.metadata["requested_device"] == "cuda"
    assert result.metadata["effective_device"] == "cpu"
    assert result.metadata["gpu_available_check_result"] == "cuda_fallback_to_cpu"
    assert result.metadata["lightgbm_device_type"] == "cpu"
