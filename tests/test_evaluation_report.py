"""Tests for rich evaluation-report generation."""

from __future__ import annotations

import pandas as pd

from trading_bot.validation.evaluation_report import (
    build_evaluation_report,
    format_evaluation_report_markdown,
)


def test_build_evaluation_report_includes_rich_sections() -> None:
    """Evaluation reports should include prediction and feature summaries."""
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
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "timeframe": ["15m", "15m"],
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="15min", tz="UTC"),
            "fold_id": [0, 0],
            "y_true": [0, 1],
            "y_pred": [0, 1],
            "y_proba": [0.30, 0.80],
        }
    )
    feature_importance_df = pd.DataFrame(
        {
            "feature_name": ["ret_1", "news_count_1h"],
            "importance": [10.0, 7.0],
            "rank": [1, 2],
        }
    )
    metadata = {
        "run_id": "RUNREPORT",
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "dataset_version": "v1",
        "label_version": "v1",
        "model_version": "v1",
        "target_column": "target_up_2bars",
        "feature_columns": ["ret_1", "news_count_1h"],
        "requested_device": "cpu",
        "effective_device": "cpu",
        "lightgbm_version": "test-lightgbm",
        "fold_count": 1,
    }
    aggregate_metrics = {
        "fold_count": 1,
        "accuracy": 0.61,
        "precision": 0.59,
        "recall": 0.62,
        "f1": 0.605,
        "roc_auc": 0.64,
        "log_loss": 0.67,
    }

    report = build_evaluation_report(
        run_metadata=metadata,
        aggregate_metrics=aggregate_metrics,
        fold_metrics_df=fold_metrics_df,
        predictions_df=predictions_df,
        feature_importance_df=feature_importance_df,
        target_distribution_summary={"target_up_2bars": {"positive_rate": 0.5}},
        artifact_paths={"evaluation_dir": "/tmp/eval"},
        status="candidate",
        notes="baseline",
        top_k_feature_importance=2,
    ).to_dict()

    markdown = format_evaluation_report_markdown(report)

    assert report["run_id"] == "RUNREPORT"
    assert report["prediction_probability_summary"]["confusion_matrix"]["tp"] == 1
    assert report["feature_importance_summary"]["top_features"][0]["feature_name"] == "ret_1"
    assert report["target_distribution_summary"]["target_up_2bars"]["positive_rate"] == 0.5
    assert "RUNREPORT" in markdown
    assert "Top Features" in markdown
