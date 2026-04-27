"""Tests for run-comparison utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_bot.experiments.comparison import RunComparator
from trading_bot.experiments.registry import ExperimentRegistry
from trading_bot.storage.experiment_store import ExperimentStore


def test_compare_runs_sorts_by_metric_and_preserves_metadata(tmp_path: Path) -> None:
    """Comparisons should keep run identity and sort consistently by the requested metric."""
    experiment_store = ExperimentStore(
        base_dir=tmp_path / "experiments",
        registry_path=tmp_path / "experiments" / "run_registry.parquet",
    )
    registry_df = pd.DataFrame(
        [
            {
                "run_id": "RUN_B",
                "created_at": "2024-04-22T00:00:00+00:00",
                "symbol": "BTCUSDT",
                "timeframe": "15m",
                "dataset_version": "v2",
                "feature_version": "v2",
                "label_version": "v1",
                "model_version": "v2",
                "target_column": "target_up_2bars",
                "requested_device": "cpu",
                "effective_device": "cpu",
                "fold_count": 5,
                "feature_count": 24,
                "accuracy_mean": 0.62,
                "precision_mean": 0.61,
                "recall_mean": 0.60,
                "f1_mean": 0.605,
                "roc_auc_mean": 0.68,
                "log_loss_mean": 0.63,
                "status": "candidate",
                "notes": "",
            },
            {
                "run_id": "RUN_A",
                "created_at": "2024-04-21T00:00:00+00:00",
                "symbol": "BTCUSDT",
                "timeframe": "15m",
                "dataset_version": "v1",
                "feature_version": "v1",
                "label_version": "v1",
                "model_version": "v1",
                "target_column": "target_up_2bars",
                "requested_device": "cpu",
                "effective_device": "cpu",
                "fold_count": 5,
                "feature_count": 22,
                "accuracy_mean": 0.60,
                "precision_mean": 0.58,
                "recall_mean": 0.59,
                "f1_mean": 0.585,
                "roc_auc_mean": 0.64,
                "log_loss_mean": 0.67,
                "status": "candidate",
                "notes": "",
            },
        ]
    )
    experiment_store.write_registry(registry_df)

    experiment_store.write_top_features(
        "RUN_A",
        pd.DataFrame(
            {
                "feature_name": ["ret_1", "volume_zscore_8"],
                "importance": [10.0, 7.5],
                "rank": [1, 2],
            }
        ),
    )
    experiment_store.write_top_features(
        "RUN_B",
        pd.DataFrame(
            {
                "feature_name": ["news_count_1h", "ret_1"],
                "importance": [11.0, 8.5],
                "rank": [1, 2],
            }
        ),
    )

    registry = ExperimentRegistry(
        experiment_store=experiment_store,
        default_sort_metric="roc_auc_mean",
    )
    comparator = RunComparator(registry=registry)

    comparison_df = comparator.compare_runs(["RUN_A", "RUN_B"], sort_by="roc_auc_mean")
    feature_df = comparator.compare_feature_importance(["RUN_A", "RUN_B"], top_k=2)

    assert comparison_df["run_id"].tolist() == ["RUN_B", "RUN_A"]
    assert comparison_df.iloc[0]["feature_version"] == "v2"
    assert comparison_df.iloc[0]["roc_auc_mean"] == 0.68
    assert set(feature_df["run_id"].tolist()) == {"RUN_A", "RUN_B"}
    assert "feature_name" in feature_df.columns


def test_build_sentiment_ablation_summary_returns_metric_deltas(tmp_path: Path) -> None:
    """Ablation summaries should compute explicit metric deltas with provenance."""
    experiment_store = ExperimentStore(
        base_dir=tmp_path / "experiments",
        registry_path=tmp_path / "experiments" / "run_registry.parquet",
    )
    registry_df = pd.DataFrame(
        [
            {
                "run_id": "RUN_BASELINE",
                "created_at": "2024-04-21T00:00:00+00:00",
                "symbol": "BTCUSDT",
                "timeframe": "15m",
                "dataset_version": "v1",
                "feature_version": "v1",
                "label_version": "v1",
                "model_version": "v1",
                "target_column": "target_up_2bars",
                "requested_device": "cpu",
                "effective_device": "cpu",
                "fold_count": 5,
                "feature_count": 22,
                "accuracy_mean": 0.60,
                "precision_mean": 0.58,
                "recall_mean": 0.59,
                "f1_mean": 0.585,
                "roc_auc_mean": 0.64,
                "log_loss_mean": 0.67,
                "status": "candidate",
                "notes": "",
            },
            {
                "run_id": "RUN_SENTIMENT",
                "created_at": "2024-04-22T00:00:00+00:00",
                "symbol": "BTCUSDT",
                "timeframe": "15m",
                "dataset_version": "v2_sentiment",
                "feature_version": "v2_sentiment",
                "label_version": "v1",
                "model_version": "v2_sentiment",
                "target_column": "target_up_2bars",
                "requested_device": "cpu",
                "effective_device": "cpu",
                "fold_count": 5,
                "feature_count": 35,
                "accuracy_mean": 0.63,
                "precision_mean": 0.60,
                "recall_mean": 0.62,
                "f1_mean": 0.61,
                "roc_auc_mean": 0.69,
                "log_loss_mean": 0.61,
                "status": "candidate",
                "notes": "",
            },
        ]
    )
    experiment_store.write_registry(registry_df)
    experiment_store.write_run_summary(
        "RUN_BASELINE",
        {"run_id": "RUN_BASELINE", "artifact_paths": {"report_json_path": "/tmp/base.json"}},
    )
    experiment_store.write_run_summary(
        "RUN_SENTIMENT",
        {"run_id": "RUN_SENTIMENT", "artifact_paths": {"report_json_path": "/tmp/new.json"}},
    )

    registry = ExperimentRegistry(
        experiment_store=experiment_store,
        default_sort_metric="roc_auc_mean",
    )
    comparator = RunComparator(registry=registry)

    summary = comparator.build_sentiment_ablation_summary(
        baseline_run_id="RUN_BASELINE",
        new_run_id="RUN_SENTIMENT",
    )

    assert summary["baseline_run_id"] == "RUN_BASELINE"
    assert summary["new_run_id"] == "RUN_SENTIMENT"
    assert summary["metric_deltas"]["roc_auc_mean"] == 0.05
    assert summary["metric_improvements"]["log_loss_mean"] == "improved"
    assert summary["provenance"]["baseline_summary"]["run_id"] == "RUN_BASELINE"
