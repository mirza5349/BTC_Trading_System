"""Evaluation metrics for walk-forward classification runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClassificationMetrics:
    """Binary classification metrics for one validation fold."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    log_loss: float | None
    tp: int
    tn: int
    fp: int
    fn: int
    positive_rate: float
    prediction_positive_rate: float
    balanced_accuracy: float | None

    @property
    def f1_score(self) -> float:
        """Backward-compatible alias."""
        return self.f1

    @property
    def auc_roc(self) -> float | None:
        """Backward-compatible alias."""
        return self.roc_auc

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return asdict(self)


@dataclass(frozen=True)
class TradingMetrics:
    """Placeholder trading metrics for future strategy evaluation steps."""

    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_duration_minutes: float = 0.0


def compute_classification_metrics(
    y_true: list[float] | np.ndarray | pd.Series,
    y_pred: list[float] | np.ndarray | pd.Series,
    y_prob: list[float] | np.ndarray | pd.Series | None = None,
) -> ClassificationMetrics:
    """Compute classification metrics from fold predictions."""
    y_true_array = _to_binary_int_array(y_true, name="y_true")
    y_pred_array = _to_binary_int_array(y_pred, name="y_pred")

    if y_true_array.shape[0] != y_pred_array.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    y_prob_array = _to_probability_array(y_prob) if y_prob is not None else None
    if y_prob_array is not None and y_prob_array.shape[0] != y_true_array.shape[0]:
        raise ValueError("y_prob must have the same length as y_true")

    tp = int(np.sum((y_true_array == 1) & (y_pred_array == 1)))
    tn = int(np.sum((y_true_array == 0) & (y_pred_array == 0)))
    fp = int(np.sum((y_true_array == 0) & (y_pred_array == 1)))
    fn = int(np.sum((y_true_array == 1) & (y_pred_array == 0)))

    total_count = int(y_true_array.shape[0])
    accuracy = float((tp + tn) / total_count) if total_count else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    positive_rate = float(np.mean(y_true_array == 1)) if total_count else 0.0
    prediction_positive_rate = float(np.mean(y_pred_array == 1)) if total_count else 0.0

    tpr = float(tp / (tp + fn)) if (tp + fn) else None
    tnr = float(tn / (tn + fp)) if (tn + fp) else None
    balanced_accuracy = (
        float((tpr + tnr) / 2.0) if tpr is not None and tnr is not None else None
    )

    roc_auc = _compute_roc_auc(y_true_array, y_prob_array)
    log_loss = _compute_log_loss(y_true_array, y_prob_array)

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        log_loss=log_loss,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        positive_rate=positive_rate,
        prediction_positive_rate=prediction_positive_rate,
        balanced_accuracy=balanced_accuracy,
    )


def aggregate_classification_metrics(
    fold_metrics: list[dict[str, Any] | ClassificationMetrics],
) -> dict[str, float | int | None]:
    """Aggregate per-fold metrics into a compact summary."""
    if not fold_metrics:
        return {"fold_count": 0}

    rows = [
        metric.to_dict() if isinstance(metric, ClassificationMetrics) else dict(metric)
        for metric in fold_metrics
    ]
    metrics_df = pd.DataFrame(rows)

    aggregate: dict[str, float | int | None] = {
        "fold_count": int(metrics_df.shape[0]),
        "tp": int(metrics_df["tp"].sum()) if "tp" in metrics_df.columns else 0,
        "tn": int(metrics_df["tn"].sum()) if "tn" in metrics_df.columns else 0,
        "fp": int(metrics_df["fp"].sum()) if "fp" in metrics_df.columns else 0,
        "fn": int(metrics_df["fn"].sum()) if "fn" in metrics_df.columns else 0,
    }

    for column in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "log_loss",
        "positive_rate",
        "prediction_positive_rate",
        "balanced_accuracy",
    ):
        aggregate[column] = (
            _mean_or_none(metrics_df[column]) if column in metrics_df.columns else None
        )

    return aggregate


def compute_trading_metrics(
    returns: list[float] | np.ndarray,
    trades: list[dict[str, float]] | None = None,
) -> TradingMetrics:
    """Return placeholder trading metrics for future steps."""
    del returns, trades
    return TradingMetrics()


def _to_binary_int_array(
    values: list[float] | np.ndarray | pd.Series,
    *,
    name: str,
) -> np.ndarray:
    """Coerce binary inputs to an integer numpy array."""
    array = pd.Series(values).astype(int).to_numpy(dtype=int)
    invalid_values = sorted(
        {int(value) for value in np.unique(array).tolist() if value not in (0, 1)}
    )
    if invalid_values:
        raise ValueError(f"{name} must contain only binary 0/1 values, found {invalid_values}")
    return array


def _to_probability_array(
    values: list[float] | np.ndarray | pd.Series,
) -> np.ndarray:
    """Coerce probabilities to a clipped float array."""
    array = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(dtype=float)
    return np.clip(array, 1e-15, 1.0 - 1e-15)


def _compute_log_loss(
    y_true: np.ndarray,
    y_prob: np.ndarray | None,
) -> float | None:
    """Compute binary log loss when probabilities are available."""
    if y_prob is None or y_true.shape[0] == 0:
        return None

    loss = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    return float(np.mean(loss))


def _compute_roc_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray | None,
) -> float | None:
    """Compute ROC AUC via rank statistics without sklearn."""
    if y_prob is None or y_true.shape[0] == 0:
        return None

    positive_mask = y_true == 1
    negative_mask = y_true == 0
    n_pos = int(np.sum(positive_mask))
    n_neg = int(np.sum(negative_mask))
    if n_pos == 0 or n_neg == 0:
        return None

    ranks = pd.Series(y_prob).rank(method="average").to_numpy(dtype=float)
    positive_rank_sum = float(np.sum(ranks[positive_mask]))
    auc = (positive_rank_sum - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def _mean_or_none(series: pd.Series) -> float | None:
    """Return the mean of a metric column, ignoring null values."""
    non_null = pd.to_numeric(series, errors="coerce").dropna()
    if non_null.empty:
        return None
    return float(non_null.mean())
