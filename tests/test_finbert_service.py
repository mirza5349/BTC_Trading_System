"""Tests for the local FinBERT sentiment service and preprocessing."""

from __future__ import annotations

from trading_bot.nlp.finbert_service import FinBERTService, _normalize_pipeline_outputs
from trading_bot.nlp.text_preprocessing import assemble_news_text, normalize_text


def test_normalize_text_and_assemble_title_plus_summary() -> None:
    """Whitespace normalization and text assembly should be deterministic."""
    assert normalize_text("  BTC   rallies \n again  ") == "BTC rallies again"
    row = {
        "title": " BTC jumps ",
        "summary": " Market reacts   quickly ",
    }
    assert assemble_news_text(row, text_mode="title_only", max_length=100) == "BTC jumps"
    assert (
        assemble_news_text(row, text_mode="title_plus_summary", max_length=100)
        == "BTC jumps Market reacts quickly"
    )


def test_normalize_pipeline_outputs_returns_stable_sentiment_predictions() -> None:
    """Raw transformers pipeline outputs should map to stable sentiment fields."""
    raw_outputs = [
        [
            {"label": "positive", "score": 0.7},
            {"label": "neutral", "score": 0.2},
            {"label": "negative", "score": 0.1},
        ]
    ]

    predictions = _normalize_pipeline_outputs(raw_outputs)

    assert len(predictions) == 1
    assert predictions[0].sentiment_label == "positive"
    assert predictions[0].prob_positive == 0.7
    assert predictions[0].prob_neutral == 0.2
    assert predictions[0].prob_negative == 0.1
    assert predictions[0].sentiment_score == 0.6


def test_finbert_service_scores_texts_with_mocked_classifier(monkeypatch) -> None:
    """The service should batch texts and honor the mocked local classifier."""
    service = FinBERTService(
        model_name="ProsusAI/finbert",
        batch_size=2,
        requested_device="cpu",
        allow_cuda_fallback_to_cpu=True,
        max_length=256,
    )

    class DummyService(FinBERTService):
        def _get_classifier(self):
            self._effective_device = "cpu"

            def _classifier(texts, truncation=True, max_length=256, top_k=None):
                del truncation, max_length, top_k
                return [
                    [
                        {"label": "positive", "score": 0.8},
                        {"label": "neutral", "score": 0.1},
                        {"label": "negative", "score": 0.1},
                    ]
                    for _ in texts
                ]

            return _classifier

    dummy_service = DummyService(
        model_name=service.model_name,
        batch_size=service.batch_size,
        requested_device=service.requested_device,
        allow_cuda_fallback_to_cpu=service.allow_cuda_fallback_to_cpu,
        max_length=service.max_length,
    )

    predictions = dummy_service.score_texts(["alpha", "beta", "gamma"])

    assert len(predictions) == 3
    assert all(prediction.sentiment_label == "positive" for prediction in predictions)
    assert dummy_service.effective_device == "cpu"
