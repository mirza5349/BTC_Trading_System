"""Tests for local experiment tracking and registry refresh."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trading_bot.experiments.registry import ExperimentRegistry
from trading_bot.experiments.tracker import ExperimentTracker
from trading_bot.storage.evaluation_store import EvaluationStore
from trading_bot.storage.experiment_store import ExperimentStore
from trading_bot.storage.report_store import ReportStore


def _write_run_artifacts(base_dir: Path, run_id: str) -> tuple[EvaluationStore, dict[str, str]]:
    """Create one synthetic stored run for experiment-tracker tests."""
    evaluation_store = EvaluationStore(base_dir / "evaluation")
    coordinates = {
        "asset": "BTC",
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "model_version": "v1",
        "run_id": run_id,
    }

    fold_metrics_df = pd.DataFrame(
        [
            {
                "fold_id": 0,
                "train_start": "2024-01-01T00:00:00+00:00",
                "train_end": "2024-01-01T12:00:00+00:00",
                "validation_start": "2024-01-01T12:15:00+00:00",
                "validation_end": "2024-01-01T13:00:00+00:00",
                "n_train": 1000,
                "n_validation": 250,
                "accuracy": 0.61,
                "precision": 0.59,
                "recall": 0.62,
                "f1": 0.605,
                "roc_auc": 0.64,
                "log_loss": 0.67,
                "tp": 80,
                "tn": 72,
                "fp": 48,
                "fn": 50,
            }
        ]
    )
    predictions_df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "timeframe": ["15m", "15m", "15m"],
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"),
            "fold_id": [0, 0, 0],
            "y_true": [0, 1, 1],
            "y_pred": [0, 1, 0],
            "y_proba": [0.21, 0.81, 0.41],
        }
    )
    feature_importance_df = pd.DataFrame(
        {
            "feature_name": ["ret_1", "news_count_1h"],
            "mean_importance": [12.5, 7.5],
            "importance": [12.5, 7.5],
            "folds_present": [1, 1],
            "rank": [1, 2],
        }
    )
    aggregate_metrics = {
        "fold_count": 1,
        "accuracy": 0.61,
        "precision": 0.59,
        "recall": 0.62,
        "f1": 0.605,
        "roc_auc": 0.64,
        "log_loss": 0.67,
        "tp": 80,
        "tn": 72,
        "fp": 48,
        "fn": 50,
    }
    metadata = {
        "run_id": run_id,
        "created_at": "2024-04-21T00:00:00+00:00",
        "asset": "BTC",
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "dataset_version": "v1",
        "feature_version": "v1",
        "label_version": "v1",
        "model_version": "v1",
        "target_column": "target_up_2bars",
        "feature_columns": ["ret_1", "news_count_1h"],
        "requested_device": "cpu",
        "effective_device": "cpu",
        "lightgbm_version": "test-lightgbm",
        "fold_count": 1,
        "extra": {
            "target_summary": {
                "target_up_2bars": {
                    "type": "binary",
                    "count": 250,
                    "zero_count": 120,
                    "one_count": 130,
                    "positive_rate": 0.52,
                }
            }
        },
    }
    report_payload = {
        "run_id": run_id,
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "dataset_version": "v1",
        "label_version": "v1",
        "model_version": "v1",
        "target_column": "target_up_2bars",
        "feature_count": 2,
        "fold_count": 1,
        "requested_device": "cpu",
        "effective_device": "cpu",
        "lightgbm_version": "test-lightgbm",
        "metrics": aggregate_metrics,
    }

    evaluation_store.write_fold_metrics(fold_metrics_df, **coordinates)
    evaluation_store.write_predictions(predictions_df, **coordinates)
    evaluation_store.write_feature_importance(feature_importance_df, aggregate=True, **coordinates)
    evaluation_store.write_aggregate_metrics(aggregate_metrics, **coordinates)
    evaluation_store.write_run_metadata(metadata, **coordinates)
    evaluation_store.write_report(report_payload, **coordinates)
    return evaluation_store, coordinates


def test_refresh_registry_registers_runs_and_persists_reports(tmp_path: Path) -> None:
    """Refreshing the registry should build summaries and rich reports from stored runs."""
    evaluation_store, _ = _write_run_artifacts(tmp_path, "RUN123")
    experiment_store = ExperimentStore(
        base_dir=tmp_path / "experiments",
        registry_path=tmp_path / "experiments" / "run_registry.parquet",
    )
    report_store = ReportStore(tmp_path / "reports")

    tracker = ExperimentTracker(
        evaluation_store=evaluation_store,
        experiment_store=experiment_store,
        report_store=report_store,
        default_status="candidate",
        write_markdown=True,
        top_k_feature_importance=5,
    )

    registry_df = tracker.refresh_registry()

    assert registry_df["run_id"].tolist() == ["RUN123"]
    assert registry_df.iloc[0]["feature_version"] == "v1"
    assert registry_df.iloc[0]["status"] == "candidate"
    assert registry_df.iloc[0]["roc_auc_mean"] == 0.64

    summary_path = experiment_store.summary_path("RUN123")
    report_json_path = report_store.report_json_path("RUN123")
    report_markdown_path = report_store.report_markdown_path("RUN123")
    top_features_path = experiment_store.top_features_path("RUN123")

    assert summary_path.exists()
    assert report_json_path.exists()
    assert report_markdown_path.exists()
    assert top_features_path.exists()

    summary_payload = json.loads(summary_path.read_text())
    report_payload = json.loads(report_json_path.read_text())
    assert summary_payload["prediction_probability_summary"]["row_count"] == 3
    assert report_payload["prediction_probability_summary"]["confusion_matrix"]["tp"] == 1
    assert report_payload["feature_importance_summary"]["top_features"][0]["feature_name"] == "ret_1"


def test_set_run_status_updates_summary_and_registry(tmp_path: Path) -> None:
    """Promotion tagging should update both the registry row and stored run summary."""
    evaluation_store, _ = _write_run_artifacts(tmp_path, "RUNTAG")
    experiment_store = ExperimentStore(
        base_dir=tmp_path / "experiments",
        registry_path=tmp_path / "experiments" / "run_registry.parquet",
    )
    report_store = ReportStore(tmp_path / "reports")

    tracker = ExperimentTracker(
        evaluation_store=evaluation_store,
        experiment_store=experiment_store,
        report_store=report_store,
        default_status="candidate",
        write_markdown=False,
        top_k_feature_importance=5,
    )
    tracker.refresh_registry()

    registry = ExperimentRegistry(
        experiment_store=experiment_store,
        default_sort_metric="roc_auc_mean",
    )
    updated = registry.set_run_status(
        run_id="RUNTAG",
        status="promoted_for_backtest",
        note="best_baseline",
    )

    assert updated["status"] == "promoted_for_backtest"
    assert updated["notes"] == "best_baseline"

    registry_df = experiment_store.read_registry()
    assert registry_df.iloc[0]["status"] == "promoted_for_backtest"
    assert registry_df.iloc[0]["notes"] == "best_baseline"


def test_refresh_registry_handles_scalar_feature_importance_without_crashing(
    tmp_path: Path,
) -> None:
    """Malformed feature-importance artifacts should not crash registry refresh."""
    evaluation_store, coordinates = _write_run_artifacts(tmp_path, "RUNSCALAR")
    experiment_store = ExperimentStore(
        base_dir=tmp_path / "experiments",
        registry_path=tmp_path / "experiments" / "run_registry.parquet",
    )
    report_store = ReportStore(tmp_path / "reports")

    class ScalarImportanceFrame(pd.DataFrame):
        @property
        def _constructor(self):  # type: ignore[override]
            return ScalarImportanceFrame

        def get(self, key, default=None):  # type: ignore[override]
            if key == "importance":
                return pd.Series([1.0], dtype=float).iloc[0]
            return super().get(key, default)

    malformed_feature_importance_df = ScalarImportanceFrame(
        {
            "feature_name": ["ret_1", "news_count_1h"],
            "importance": [12.5, 7.5],
            "mean_importance": [12.5, 7.5],
            "rank": [1, 2],
        }
    )
    evaluation_store.write_feature_importance(
        pd.DataFrame(malformed_feature_importance_df),
        aggregate=True,
        **coordinates,
    )

    tracker = ExperimentTracker(
        evaluation_store=evaluation_store,
        experiment_store=experiment_store,
        report_store=report_store,
        default_status="candidate",
        write_markdown=False,
        top_k_feature_importance=5,
    )

    original_read_feature_importance = evaluation_store.read_feature_importance

    def _read_feature_importance_with_scalar_bug(**kwargs):
        loaded = original_read_feature_importance(**kwargs)
        return ScalarImportanceFrame(loaded)

    evaluation_store.read_feature_importance = _read_feature_importance_with_scalar_bug  # type: ignore[method-assign]

    registry_df = tracker.refresh_registry()

    assert registry_df["run_id"].tolist() == ["RUNSCALAR"]
    summary_payload = json.loads(experiment_store.summary_path("RUNSCALAR").read_text())
    assert summary_payload["feature_importance_summary"]["top_features"] == []
    assert summary_payload["feature_importance_summary"]["feature_count"] == 0
