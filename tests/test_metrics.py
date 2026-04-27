"""Tests for classification metrics used in walk-forward evaluation."""

from __future__ import annotations

import pytest

from trading_bot.validation.metrics import (
    aggregate_classification_metrics,
    compute_classification_metrics,
)


def test_compute_classification_metrics_returns_expected_values() -> None:
    """Fold metrics should match known binary-classification calculations."""
    metrics = compute_classification_metrics(
        y_true=[0, 1, 1, 0],
        y_pred=[0, 1, 0, 0],
        y_prob=[0.1, 0.9, 0.4, 0.2],
    )

    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(2.0 / 3.0)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.log_loss == pytest.approx(0.33753883)
    assert metrics.tp == 1
    assert metrics.tn == 2
    assert metrics.fp == 0
    assert metrics.fn == 1
    assert metrics.positive_rate == pytest.approx(0.5)
    assert metrics.prediction_positive_rate == pytest.approx(0.25)
    assert metrics.balanced_accuracy == pytest.approx(0.75)


def test_compute_classification_metrics_handles_single_class_auc_gracefully() -> None:
    """ROC AUC and balanced accuracy should be null when only one class is present."""
    metrics = compute_classification_metrics(
        y_true=[1, 1, 1],
        y_pred=[1, 1, 1],
        y_prob=[0.7, 0.8, 0.9],
    )

    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.roc_auc is None
    assert metrics.balanced_accuracy is None
    assert metrics.log_loss is not None


def test_aggregate_classification_metrics_combines_fold_outputs() -> None:
    """Aggregate metrics should average fold scores and sum confusion counts."""
    aggregate = aggregate_classification_metrics(
        [
            {
                "accuracy": 0.8,
                "precision": 0.75,
                "recall": 0.6,
                "f1": 0.6666667,
                "roc_auc": 0.82,
                "log_loss": 0.48,
                "positive_rate": 0.5,
                "prediction_positive_rate": 0.4,
                "balanced_accuracy": 0.7,
                "tp": 6,
                "tn": 10,
                "fp": 2,
                "fn": 4,
            },
            {
                "accuracy": 0.7,
                "precision": 0.5,
                "recall": 0.8,
                "f1": 0.6153846,
                "roc_auc": None,
                "log_loss": 0.6,
                "positive_rate": 0.55,
                "prediction_positive_rate": 0.6,
                "balanced_accuracy": None,
                "tp": 8,
                "tn": 6,
                "fp": 6,
                "fn": 2,
            },
        ]
    )

    assert aggregate["fold_count"] == 2
    assert aggregate["accuracy"] == pytest.approx(0.75)
    assert aggregate["precision"] == pytest.approx(0.625)
    assert aggregate["recall"] == pytest.approx(0.7)
    assert aggregate["roc_auc"] == pytest.approx(0.82)
    assert aggregate["log_loss"] == pytest.approx(0.54)
    assert aggregate["tp"] == 14
    assert aggregate["tn"] == 16
    assert aggregate["fp"] == 8
    assert aggregate["fn"] == 6
