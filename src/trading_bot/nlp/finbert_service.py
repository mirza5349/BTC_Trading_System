"""Local FinBERT sentiment inference service with safe CPU/CUDA handling."""

from __future__ import annotations

from typing import Any

import pandas as pd

from trading_bot.logging_config import get_logger
from trading_bot.schemas.nlp import SentimentPrediction, normalize_sentiment_label

logger = get_logger(__name__)


class FinBERTServiceError(Exception):
    """Raised when local FinBERT inference cannot be completed."""


class FinBERTService:
    """Reusable local sentiment service centered on FinBERT-compatible models."""

    def __init__(
        self,
        *,
        model_name: str = "ProsusAI/finbert",
        batch_size: int = 16,
        requested_device: str = "cpu",
        allow_cuda_fallback_to_cpu: bool = True,
        max_length: int = 256,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.requested_device = requested_device.lower()
        self.allow_cuda_fallback_to_cpu = allow_cuda_fallback_to_cpu
        self.max_length = max_length
        self._classifier: Any | None = None
        self._effective_device: str | None = None

    @property
    def effective_device(self) -> str:
        """Return the effective device after lazy model loading."""
        if self._effective_device is None:
            return "cpu" if self.requested_device != "cuda" else "unknown"
        return self._effective_device

    def score_texts(self, texts: list[str]) -> list[SentimentPrediction]:
        """Score a batch of texts with FinBERT locally."""
        if not texts:
            return []

        classifier = self._get_classifier()
        predictions: list[SentimentPrediction] = []
        for start_idx in range(0, len(texts), self.batch_size):
            batch = texts[start_idx : start_idx + self.batch_size]
            raw_outputs = classifier(
                batch,
                truncation=True,
                max_length=self.max_length,
                top_k=None,
            )
            predictions.extend(_normalize_pipeline_outputs(raw_outputs))
        return predictions

    def _get_classifier(self) -> Any:
        """Lazily load the local transformers pipeline."""
        if self._classifier is not None:
            return self._classifier

        transformers_module, torch_module = _import_transformers_and_torch()
        resolved_device = self._resolve_device(torch_module)
        pipeline_device = 0 if resolved_device == "cuda" else -1

        self._classifier = transformers_module.pipeline(
            "text-classification",
            model=self.model_name,
            tokenizer=self.model_name,
            device=pipeline_device,
        )
        self._effective_device = resolved_device
        logger.info(
            "finbert_model_loaded",
            model_name=self.model_name,
            requested_device=self.requested_device,
            effective_device=self._effective_device,
        )
        return self._classifier

    def _resolve_device(self, torch_module: Any) -> str:
        """Resolve the effective runtime device with safe CUDA fallback."""
        requested = self.requested_device
        if requested != "cuda":
            return "cpu"

        cuda_available = bool(torch_module.cuda.is_available())
        if cuda_available:
            return "cuda"
        if self.allow_cuda_fallback_to_cpu:
            logger.warning("finbert_cuda_unavailable_falling_back_to_cpu")
            return "cpu"
        raise FinBERTServiceError("CUDA requested for FinBERT but CUDA is unavailable")


def _normalize_pipeline_outputs(raw_outputs: Any) -> list[SentimentPrediction]:
    """Normalize transformers pipeline outputs into a stable internal format."""
    normalized_predictions: list[SentimentPrediction] = []
    for item in raw_outputs:
        scores = _scores_to_map(item)
        positive = float(scores.get("positive", 0.0))
        neutral = float(scores.get("neutral", 0.0))
        negative = float(scores.get("negative", 0.0))
        label = max(
            ("positive", positive),
            ("neutral", neutral),
            ("negative", negative),
            key=lambda pair: pair[1],
        )[0]
        normalized_predictions.append(
            SentimentPrediction(
                sentiment_label=label,
                prob_positive=positive,
                prob_neutral=neutral,
                prob_negative=negative,
                sentiment_score=positive - negative,
            )
        )
    return normalized_predictions


def _scores_to_map(item: Any) -> dict[str, float]:
    """Convert a pipeline output item into a normalized score map."""
    rows = item if isinstance(item, list) else [item]
    result: dict[str, float] = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    for row in rows:
        label = normalize_sentiment_label(str(row.get("label", "")))
        result[label] = float(row.get("score", 0.0))
    return result


def _import_transformers_and_torch() -> tuple[Any, Any]:
    """Import transformers and torch lazily so base package imports remain light."""
    try:
        import torch  # type: ignore[import-not-found]
        import transformers  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised indirectly
        raise FinBERTServiceError(
            "FinBERT local inference requires the optional NLP extras "
            "(`pip install -e \".[nlp]\"`)."
        ) from exc
    return transformers, torch
