"""Prediction helpers for stored fold models and validation outputs."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_bot.schemas.features import INDEX_COLUMNS


@dataclass(frozen=True)
class Prediction:
    """A single model prediction."""

    timestamp_ms: int
    probability: float
    predicted_class: int
    model_id: str = ""
    confidence: float = 0.0

    @property
    def is_bullish(self) -> bool:
        """Return True if the model predicts price increase."""
        return self.predicted_class == 1


class Predictor(ABC):
    """Abstract predictor interface."""

    @abstractmethod
    def predict(self, features: Any) -> list[Prediction]:
        """Generate predictions from feature data."""

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load a serialized model from disk."""


class LightGBMPredictor(Predictor):
    """Minimal predictor wrapper for LightGBM-compatible sklearn models."""

    def __init__(self) -> None:
        self.model: Any | None = None

    def load_model(self, model_path: str) -> None:
        """Load a pickled model artifact."""
        with Path(model_path).open("rb") as file_obj:
            self.model = pickle.load(file_obj)

    def predict(self, features: Any) -> list[Prediction]:
        """Predict probability and class outputs for the provided feature frame."""
        if self.model is None:
            raise ValueError("No model has been loaded")

        if not isinstance(features, pd.DataFrame):
            raise ValueError("features must be a pandas DataFrame")

        probabilities, predictions = predict_binary_classifier(self.model, features)
        timestamps = _extract_prediction_timestamps(features)

        return [
            Prediction(
                timestamp_ms=int(timestamp.timestamp() * 1000),
                probability=float(probability),
                predicted_class=int(prediction),
                confidence=float(abs(probability - 0.5) * 2.0),
            )
            for timestamp, probability, prediction in zip(
                timestamps, probabilities, predictions, strict=False
            )
        ]


class StubPredictor(Predictor):
    """Simple no-op predictor used in placeholder or dry-run contexts."""

    def predict(self, features: Any) -> list[Prediction]:
        """Return empty predictions."""
        del features
        return []

    def load_model(self, model_path: str) -> None:
        """No-op model loading."""
        del model_path


def predict_binary_classifier(
    model: Any,
    features: pd.DataFrame,
    *,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate probability and hard-class predictions from a fitted model."""
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
        if probabilities.ndim == 2:
            probabilities = probabilities[:, -1]
    else:
        probabilities = np.asarray(model.predict(features), dtype=float)

    probabilities = np.clip(probabilities.astype(float), 0.0, 1.0)
    predictions = (probabilities >= threshold).astype(int)
    return probabilities, predictions


def build_prediction_frame(
    *,
    base_df: pd.DataFrame,
    fold_id: int,
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Return a validation prediction table aligned to the original rows."""
    missing_identifier_columns = [
        column for column in INDEX_COLUMNS if column not in base_df.columns
    ]
    if missing_identifier_columns:
        raise ValueError(
            f"Prediction frame requires identifier columns {missing_identifier_columns}"
        )

    predictions = base_df.loc[:, INDEX_COLUMNS].copy()
    predictions["fold_id"] = int(fold_id)
    predictions["y_true"] = pd.Series(y_true).astype(int).to_numpy()
    predictions["y_pred"] = pd.Series(y_pred).astype(int).to_numpy()
    predictions["y_proba"] = pd.to_numeric(pd.Series(y_proba), errors="raise").to_numpy(dtype=float)
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True)
    return predictions.sort_values("timestamp").reset_index(drop=True)


def _extract_prediction_timestamps(features: pd.DataFrame) -> pd.Series:
    """Return timestamps for prediction rows when available."""
    if "timestamp" in features.columns:
        return pd.to_datetime(features["timestamp"], utc=True)
    return pd.Series(pd.date_range("1970-01-01", periods=len(features), freq="1s", tz="UTC"))
