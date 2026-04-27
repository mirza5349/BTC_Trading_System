"""Tests for the scheduled offline paper-trading loop."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_bot.features.feature_pipeline import FeaturePipeline
from trading_bot.models.model_registry import ModelRegistry
from trading_bot.paper.loop import (
    PaperLoopSelection,
    ScheduledPaperTradingService,
    ScheduledPaperTradingServiceError,
)
from trading_bot.settings import load_settings
from trading_bot.storage.enriched_news_store import EnrichedNewsStore
from trading_bot.storage.model_store import ModelStore
from trading_bot.storage.news_store import NewsStore
from trading_bot.storage.optimization_store import OptimizationStore
from trading_bot.storage.paper_loop_store import PaperLoopStore
from trading_bot.storage.parquet_store import ParquetStore
from trading_bot.storage.evaluation_store import EvaluationStore
from trading_bot.features.feature_store import ParquetFeatureStore


class DummyProbabilityModel:
    """Very small sklearn-like stub model for offline loop tests."""

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        rows = len(features)
        positive = np.full(rows, 0.72, dtype=float)
        negative = 1.0 - positive
        return np.column_stack([negative, positive])


class DummyMarketDataService:
    """No-op market data updater used by tests."""

    def ingest_sync(self, *args, **kwargs):
        return {"is_success": True, "args": args, "kwargs": kwargs}


def test_run_paper_once_updates_prediction_logs_and_account_state(tmp_path: Path) -> None:
    """One-shot loop runs should append a new prediction and update local paper artifacts."""
    settings = load_settings()
    market_store = ParquetStore(tmp_path / "market")
    evaluation_store = EvaluationStore(tmp_path / "evaluation")
    model_store = ModelStore(tmp_path / "models")
    optimization_store = OptimizationStore(tmp_path / "optimization")
    paper_loop_store = PaperLoopStore(tmp_path / "paper_loop")

    feature_pipeline = FeaturePipeline(
        market_store=market_store,
        news_store=NewsStore(tmp_path / "news"),
        enriched_news_store=EnrichedNewsStore(tmp_path / "enriched_news"),
        feature_store=ParquetFeatureStore(tmp_path / "features"),
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        provider="cryptocompare",
        version="v1",
        market_windows=settings.features.market_windows,
        news_lookback_windows=settings.features.news_lookback_windows,
        fill_missing_news_with_zero=settings.features.fill_missing_news_with_zero,
        include_breakout_features=settings.features.include_breakout_features,
        include_calendar_features=settings.features.include_calendar_features,
        include_optional_news_features=settings.features.include_optional_news_features,
        news_burst_threshold_1h=settings.features.news_burst_threshold_1h,
        news_burst_threshold_4h=settings.features.news_burst_threshold_4h,
        sentiment_feature_version=settings.sentiment_features.feature_version,
        sentiment_feature_lookback_windows=settings.sentiment_features.lookback_windows,
        use_enriched_news_store=settings.sentiment_features.use_enriched_news_store,
        finbert_model_name=settings.nlp.model_name,
        enrichment_version=settings.nlp.enrichment_version,
        positive_burst_threshold_1h=settings.sentiment_features.positive_burst_threshold_1h,
        negative_burst_threshold_1h=settings.sentiment_features.negative_burst_threshold_1h,
    )

    base_timestamps = pd.date_range("2024-01-01 00:00:00", periods=12, freq="15min", tz="UTC")
    candles_df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 12,
            "timeframe": ["15m"] * 12,
            "timestamp": base_timestamps,
            "open": np.linspace(100.0, 111.0, 12),
            "high": np.linspace(100.5, 111.5, 12),
            "low": np.linspace(99.5, 110.5, 12),
            "close": np.linspace(100.2, 111.2, 12),
            "volume": np.linspace(10.0, 21.0, 12),
            "close_time": base_timestamps + pd.Timedelta(minutes=14, seconds=59),
        }
    )
    market_store.write(candles_df, "BTCUSDT", "15m")

    latest_feature_row = feature_pipeline.build_latest_feature_row(
        symbol="BTCUSDT",
        timeframe="15m",
        asset="BTC",
        provider="cryptocompare",
        version="v1",
    )
    feature_columns = [
        column
        for column in latest_feature_row.columns
        if column not in {"symbol", "timeframe", "timestamp"}
    ]

    model_store.write_fold_model(
        DummyProbabilityModel(),
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        model_version="fee_v1",
        run_id="RUNPAPER",
        fold_id=2,
    )
    evaluation_store.write_run_metadata(
        {
            "run_id": "RUNPAPER",
            "asset": "BTC",
            "symbol": "BTCUSDT",
            "timeframe": "15m",
            "dataset_version": "fee_h4",
            "label_version": "fee_h4",
            "model_version": "fee_v1",
            "target_column": "target_long_net_positive_4bars",
            "feature_version": "v1",
            "feature_columns": feature_columns,
            "created_at": "2024-01-01T00:00:00+00:00",
        },
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        model_version="fee_v1",
        run_id="RUNPAPER",
    )
    evaluation_store.write_fold_metrics(
        pd.DataFrame({"fold_id": [0, 1, 2]}),
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        model_version="fee_v1",
        run_id="RUNPAPER",
    )
    optimization_store.write_report_json(
        "RUNPAPER",
        "OPTPAPER",
        {
            "optimization_id": "OPTPAPER",
            "model_run_id": "RUNPAPER",
            "best_candidate": {
                "candidate_id": "C0001",
                "entry_threshold": 0.60,
                "exit_threshold": 0.45,
                "minimum_holding_bars": 2,
                "cooldown_bars": 2,
                "max_position_fraction": 0.5,
            },
        },
    )

    current_time = [datetime(2024, 1, 1, 3, 5, tzinfo=timezone.utc)]
    service = ScheduledPaperTradingService(
        settings=settings,
        model_registry=ModelRegistry(evaluation_store=evaluation_store),
        model_store=model_store,
        optimization_store=optimization_store,
        market_store=market_store,
        market_data_service=DummyMarketDataService(),
        feature_pipeline=feature_pipeline,
        paper_loop_store=paper_loop_store,
        now_fn=lambda: current_time[0],
        sleep_fn=lambda seconds: None,
    )
    selection = PaperLoopSelection(
        loop_id="loop_test",
        target_name="target_long_net_positive_4bars",
        target_key="fee_4",
        model_run_id="RUNPAPER",
        optimization_id="OPTPAPER",
        symbol="BTCUSDT",
        timeframe="15m",
        feature_version="v1",
        provider="cryptocompare",
    )

    first_result = service.run_once(selection=selection)

    predictions_df = paper_loop_store.read_predictions("loop_test")
    assert first_result.no_new_candle is False
    assert len(predictions_df) == 1
    assert round(float(predictions_df.iloc[0]["y_proba"]), 2) == 0.72
    assert paper_loop_store.read_status("loop_test")["latest_probability"] == 0.72
    assert paper_loop_store.read_summary("loop_test")["target_key"] == "fee_4"

    next_candle = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "timeframe": ["15m"],
            "timestamp": [pd.Timestamp("2024-01-01 03:00:00+00:00")],
            "open": [112.0],
            "high": [112.5],
            "low": [111.5],
            "close": [112.2],
            "volume": [22.0],
            "close_time": [pd.Timestamp("2024-01-01 03:14:59+00:00")],
        }
    )
    market_store.write(next_candle, "BTCUSDT", "15m")
    current_time[0] = datetime(2024, 1, 1, 3, 20, tzinfo=timezone.utc)

    second_result = service.run_once(selection=selection)

    updated_predictions_df = paper_loop_store.read_predictions("loop_test")
    equity_curve_df = paper_loop_store.read_equity_curve("loop_test")
    orders_df = paper_loop_store.read_orders("loop_test")
    positions_df = paper_loop_store.read_positions("loop_test")

    assert second_result.no_new_candle is False
    assert second_result.prediction_timestamp == "2024-01-01T03:00:00+00:00"
    assert len(updated_predictions_df) == 2
    assert len(equity_curve_df) == 2
    assert not orders_df.empty
    assert not positions_df.empty
    assert second_result.in_position is True

    third_result = service.run_once(selection=selection)
    assert third_result.no_new_candle is True
    assert len(paper_loop_store.read_predictions("loop_test")) == 2


def test_run_paper_once_accepts_equivalent_timestamp_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equivalent candle and feature timestamps should pass after normalization."""
    settings = load_settings()
    market_store = ParquetStore(tmp_path / "market")
    evaluation_store = EvaluationStore(tmp_path / "evaluation")
    model_store = ModelStore(tmp_path / "models")
    optimization_store = OptimizationStore(tmp_path / "optimization")
    paper_loop_store = PaperLoopStore(tmp_path / "paper_loop")

    feature_pipeline = FeaturePipeline(
        market_store=market_store,
        news_store=NewsStore(tmp_path / "news"),
        enriched_news_store=EnrichedNewsStore(tmp_path / "enriched_news"),
        feature_store=ParquetFeatureStore(tmp_path / "features"),
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        provider="cryptocompare",
        version="v1",
        market_windows=settings.features.market_windows,
        news_lookback_windows=settings.features.news_lookback_windows,
        fill_missing_news_with_zero=settings.features.fill_missing_news_with_zero,
        include_breakout_features=settings.features.include_breakout_features,
        include_calendar_features=settings.features.include_calendar_features,
        include_optional_news_features=settings.features.include_optional_news_features,
        news_burst_threshold_1h=settings.features.news_burst_threshold_1h,
        news_burst_threshold_4h=settings.features.news_burst_threshold_4h,
        sentiment_feature_version=settings.sentiment_features.feature_version,
        sentiment_feature_lookback_windows=settings.sentiment_features.lookback_windows,
        use_enriched_news_store=settings.sentiment_features.use_enriched_news_store,
        finbert_model_name=settings.nlp.model_name,
        enrichment_version=settings.nlp.enrichment_version,
        positive_burst_threshold_1h=settings.sentiment_features.positive_burst_threshold_1h,
        negative_burst_threshold_1h=settings.sentiment_features.negative_burst_threshold_1h,
    )

    base_timestamps = pd.date_range("2024-01-01 00:00:00", periods=12, freq="15min", tz="UTC")
    candles_df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 12,
            "timeframe": ["15m"] * 12,
            "timestamp": base_timestamps,
            "open": np.linspace(100.0, 111.0, 12),
            "high": np.linspace(100.5, 111.5, 12),
            "low": np.linspace(99.5, 110.5, 12),
            "close": np.linspace(100.2, 111.2, 12),
            "volume": np.linspace(10.0, 21.0, 12),
            "close_time": base_timestamps + pd.Timedelta(minutes=14, seconds=59),
        }
    )
    market_store.write(candles_df, "BTCUSDT", "15m")

    latest_feature_row = feature_pipeline.build_latest_feature_row(
        symbol="BTCUSDT",
        timeframe="15m",
        asset="BTC",
        provider="cryptocompare",
        version="v1",
    )
    feature_columns = [
        column
        for column in latest_feature_row.columns
        if column not in {"symbol", "timeframe", "timestamp"}
    ]

    original_build_latest_feature_row = feature_pipeline.build_latest_feature_row

    def build_latest_feature_row_with_string_timestamp(*args, **kwargs):
        frame = original_build_latest_feature_row(*args, **kwargs).copy()
        frame = frame.astype({"timestamp": "object"})
        frame.loc[:, "timestamp"] = frame["timestamp"].map(
            lambda value: pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        return frame

    monkeypatch.setattr(
        feature_pipeline,
        "build_latest_feature_row",
        build_latest_feature_row_with_string_timestamp,
    )

    model_store.write_fold_model(
        DummyProbabilityModel(),
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        model_version="fee_v1",
        run_id="RUNPAPERSTR",
        fold_id=1,
    )
    evaluation_store.write_run_metadata(
        {
            "run_id": "RUNPAPERSTR",
            "asset": "BTC",
            "symbol": "BTCUSDT",
            "timeframe": "15m",
            "dataset_version": "fee_h4",
            "label_version": "fee_h4",
            "model_version": "fee_v1",
            "target_column": "target_long_net_positive_4bars",
            "feature_version": "v1",
            "feature_columns": feature_columns,
            "created_at": "2024-01-01T00:00:00+00:00",
        },
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        model_version="fee_v1",
        run_id="RUNPAPERSTR",
    )
    evaluation_store.write_fold_metrics(
        pd.DataFrame({"fold_id": [0, 1]}),
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        model_version="fee_v1",
        run_id="RUNPAPERSTR",
    )
    optimization_store.write_report_json(
        "RUNPAPERSTR",
        "OPTSTR",
        {
            "optimization_id": "OPTSTR",
            "model_run_id": "RUNPAPERSTR",
            "best_candidate": {
                "candidate_id": "C0001",
                "entry_threshold": 0.60,
                "exit_threshold": 0.45,
                "minimum_holding_bars": 2,
                "cooldown_bars": 2,
                "max_position_fraction": 0.5,
            },
        },
    )

    service = ScheduledPaperTradingService(
        settings=settings,
        model_registry=ModelRegistry(evaluation_store=evaluation_store),
        model_store=model_store,
        optimization_store=optimization_store,
        market_store=market_store,
        market_data_service=DummyMarketDataService(),
        feature_pipeline=feature_pipeline,
        paper_loop_store=paper_loop_store,
        now_fn=lambda: datetime(2024, 1, 1, 3, 5, tzinfo=timezone.utc),
        sleep_fn=lambda seconds: None,
    )
    selection = PaperLoopSelection(
        loop_id="loop_test_string_ts",
        target_name="target_long_net_positive_4bars",
        target_key="fee_4",
        model_run_id="RUNPAPERSTR",
        optimization_id="OPTSTR",
        symbol="BTCUSDT",
        timeframe="15m",
        feature_version="v1",
        provider="cryptocompare",
    )

    result = service.run_once(selection=selection)

    assert result.no_new_candle is False
    assert result.prediction_timestamp == "2024-01-01T02:45:00+00:00"


def test_run_paper_once_uses_safety_buffered_canonical_timestamp(
    tmp_path: Path,
) -> None:
    """When safety minutes exclude the newest candle, the loop must build features for the prior one."""
    settings = load_settings()
    settings.paper_loop.candle_close_safety_minutes = 15
    market_store = ParquetStore(tmp_path / "market")
    evaluation_store = EvaluationStore(tmp_path / "evaluation")
    model_store = ModelStore(tmp_path / "models")
    optimization_store = OptimizationStore(tmp_path / "optimization")
    paper_loop_store = PaperLoopStore(tmp_path / "paper_loop")

    feature_pipeline = FeaturePipeline(
        market_store=market_store,
        news_store=NewsStore(tmp_path / "news"),
        enriched_news_store=EnrichedNewsStore(tmp_path / "enriched_news"),
        feature_store=ParquetFeatureStore(tmp_path / "features"),
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        provider="cryptocompare",
        version="v1",
        market_windows=settings.features.market_windows,
        news_lookback_windows=settings.features.news_lookback_windows,
        fill_missing_news_with_zero=settings.features.fill_missing_news_with_zero,
        include_breakout_features=settings.features.include_breakout_features,
        include_calendar_features=settings.features.include_calendar_features,
        include_optional_news_features=settings.features.include_optional_news_features,
        news_burst_threshold_1h=settings.features.news_burst_threshold_1h,
        news_burst_threshold_4h=settings.features.news_burst_threshold_4h,
        sentiment_feature_version=settings.sentiment_features.feature_version,
        sentiment_feature_lookback_windows=settings.sentiment_features.lookback_windows,
        use_enriched_news_store=settings.sentiment_features.use_enriched_news_store,
        finbert_model_name=settings.nlp.model_name,
        enrichment_version=settings.nlp.enrichment_version,
        positive_burst_threshold_1h=settings.sentiment_features.positive_burst_threshold_1h,
        negative_burst_threshold_1h=settings.sentiment_features.negative_burst_threshold_1h,
    )

    base_timestamps = pd.date_range("2024-01-01 00:00:00", periods=13, freq="15min", tz="UTC")
    candles_df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 13,
            "timeframe": ["15m"] * 13,
            "timestamp": base_timestamps,
            "open": np.linspace(100.0, 112.0, 13),
            "high": np.linspace(100.5, 112.5, 13),
            "low": np.linspace(99.5, 111.5, 13),
            "close": np.linspace(100.2, 112.2, 13),
            "volume": np.linspace(10.0, 22.0, 13),
            "close_time": base_timestamps + pd.Timedelta(minutes=14, seconds=59),
        }
    )
    market_store.write(candles_df, "BTCUSDT", "15m")

    latest_feature_row = feature_pipeline.build_latest_feature_row(
        symbol="BTCUSDT",
        timeframe="15m",
        asset="BTC",
        provider="cryptocompare",
        version="v1",
        target_timestamp=pd.Timestamp("2024-01-01 02:45:00+00:00"),
    )
    feature_columns = [
        column
        for column in latest_feature_row.columns
        if column not in {"symbol", "timeframe", "timestamp"}
    ]

    model_store.write_fold_model(
        DummyProbabilityModel(),
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        model_version="fee_v1",
        run_id="RUNPAPERBUF",
        fold_id=1,
    )
    evaluation_store.write_run_metadata(
        {
            "run_id": "RUNPAPERBUF",
            "asset": "BTC",
            "symbol": "BTCUSDT",
            "timeframe": "15m",
            "dataset_version": "fee_h4",
            "label_version": "fee_h4",
            "model_version": "fee_v1",
            "target_column": "target_long_net_positive_4bars",
            "feature_version": "v1",
            "feature_columns": feature_columns,
            "created_at": "2024-01-01T00:00:00+00:00",
        },
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        model_version="fee_v1",
        run_id="RUNPAPERBUF",
    )
    evaluation_store.write_fold_metrics(
        pd.DataFrame({"fold_id": [0, 1]}),
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        model_version="fee_v1",
        run_id="RUNPAPERBUF",
    )
    optimization_store.write_report_json(
        "RUNPAPERBUF",
        "OPTBUF",
        {
            "optimization_id": "OPTBUF",
            "model_run_id": "RUNPAPERBUF",
            "best_candidate": {
                "candidate_id": "C0001",
                "entry_threshold": 0.60,
                "exit_threshold": 0.45,
                "minimum_holding_bars": 2,
                "cooldown_bars": 2,
                "max_position_fraction": 0.5,
            },
        },
    )

    service = ScheduledPaperTradingService(
        settings=settings,
        model_registry=ModelRegistry(evaluation_store=evaluation_store),
        model_store=model_store,
        optimization_store=optimization_store,
        market_store=market_store,
        market_data_service=DummyMarketDataService(),
        feature_pipeline=feature_pipeline,
        paper_loop_store=paper_loop_store,
        now_fn=lambda: datetime(2024, 1, 1, 3, 14, 55, tzinfo=timezone.utc),
        sleep_fn=lambda seconds: None,
    )
    selection = PaperLoopSelection(
        loop_id="loop_test_buffered",
        target_name="target_long_net_positive_4bars",
        target_key="fee_4",
        model_run_id="RUNPAPERBUF",
        optimization_id="OPTBUF",
        symbol="BTCUSDT",
        timeframe="15m",
        feature_version="v1",
        provider="cryptocompare",
    )

    result = service.run_once(selection=selection)

    assert result.prediction_timestamp == "2024-01-01T02:45:00+00:00"
    assert result.latest_candle_timestamp == "2024-01-01T02:45:00+00:00"


def test_run_paper_once_still_rejects_truly_stale_feature_timestamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A truly stale feature row should still fail the timestamp alignment guard."""
    settings = load_settings()
    market_store = ParquetStore(tmp_path / "market")
    evaluation_store = EvaluationStore(tmp_path / "evaluation")
    model_store = ModelStore(tmp_path / "models")
    optimization_store = OptimizationStore(tmp_path / "optimization")
    paper_loop_store = PaperLoopStore(tmp_path / "paper_loop")

    feature_pipeline = FeaturePipeline(
        market_store=market_store,
        news_store=NewsStore(tmp_path / "news"),
        enriched_news_store=EnrichedNewsStore(tmp_path / "enriched_news"),
        feature_store=ParquetFeatureStore(tmp_path / "features"),
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        provider="cryptocompare",
        version="v1",
        market_windows=settings.features.market_windows,
        news_lookback_windows=settings.features.news_lookback_windows,
        fill_missing_news_with_zero=settings.features.fill_missing_news_with_zero,
        include_breakout_features=settings.features.include_breakout_features,
        include_calendar_features=settings.features.include_calendar_features,
        include_optional_news_features=settings.features.include_optional_news_features,
        news_burst_threshold_1h=settings.features.news_burst_threshold_1h,
        news_burst_threshold_4h=settings.features.news_burst_threshold_4h,
        sentiment_feature_version=settings.sentiment_features.feature_version,
        sentiment_feature_lookback_windows=settings.sentiment_features.lookback_windows,
        use_enriched_news_store=settings.sentiment_features.use_enriched_news_store,
        finbert_model_name=settings.nlp.model_name,
        enrichment_version=settings.nlp.enrichment_version,
        positive_burst_threshold_1h=settings.sentiment_features.positive_burst_threshold_1h,
        negative_burst_threshold_1h=settings.sentiment_features.negative_burst_threshold_1h,
    )

    base_timestamps = pd.date_range("2024-01-01 00:00:00", periods=12, freq="15min", tz="UTC")
    candles_df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 12,
            "timeframe": ["15m"] * 12,
            "timestamp": base_timestamps,
            "open": np.linspace(100.0, 111.0, 12),
            "high": np.linspace(100.5, 111.5, 12),
            "low": np.linspace(99.5, 110.5, 12),
            "close": np.linspace(100.2, 111.2, 12),
            "volume": np.linspace(10.0, 21.0, 12),
            "close_time": base_timestamps + pd.Timedelta(minutes=14, seconds=59),
        }
    )
    market_store.write(candles_df, "BTCUSDT", "15m")

    latest_feature_row = feature_pipeline.build_latest_feature_row(
        symbol="BTCUSDT",
        timeframe="15m",
        asset="BTC",
        provider="cryptocompare",
        version="v1",
    )
    feature_columns = [
        column
        for column in latest_feature_row.columns
        if column not in {"symbol", "timeframe", "timestamp"}
    ]

    original_build_latest_feature_row = feature_pipeline.build_latest_feature_row

    def build_latest_feature_row_with_stale_timestamp(*args, **kwargs):
        frame = original_build_latest_feature_row(*args, **kwargs).copy()
        frame.loc[:, "timestamp"] = pd.Timestamp("2024-01-01 02:30:00+00:00")
        return frame

    monkeypatch.setattr(
        feature_pipeline,
        "build_latest_feature_row",
        build_latest_feature_row_with_stale_timestamp,
    )

    model_store.write_fold_model(
        DummyProbabilityModel(),
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        model_version="fee_v1",
        run_id="RUNPAPERSTALE",
        fold_id=1,
    )
    evaluation_store.write_run_metadata(
        {
            "run_id": "RUNPAPERSTALE",
            "asset": "BTC",
            "symbol": "BTCUSDT",
            "timeframe": "15m",
            "dataset_version": "fee_h4",
            "label_version": "fee_h4",
            "model_version": "fee_v1",
            "target_column": "target_long_net_positive_4bars",
            "feature_version": "v1",
            "feature_columns": feature_columns,
            "created_at": "2024-01-01T00:00:00+00:00",
        },
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        model_version="fee_v1",
        run_id="RUNPAPERSTALE",
    )
    evaluation_store.write_fold_metrics(
        pd.DataFrame({"fold_id": [0, 1]}),
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        model_version="fee_v1",
        run_id="RUNPAPERSTALE",
    )
    optimization_store.write_report_json(
        "RUNPAPERSTALE",
        "OPTSTALE",
        {
            "optimization_id": "OPTSTALE",
            "model_run_id": "RUNPAPERSTALE",
            "best_candidate": {
                "candidate_id": "C0001",
                "entry_threshold": 0.60,
                "exit_threshold": 0.45,
                "minimum_holding_bars": 2,
                "cooldown_bars": 2,
                "max_position_fraction": 0.5,
            },
        },
    )

    service = ScheduledPaperTradingService(
        settings=settings,
        model_registry=ModelRegistry(evaluation_store=evaluation_store),
        model_store=model_store,
        optimization_store=optimization_store,
        market_store=market_store,
        market_data_service=DummyMarketDataService(),
        feature_pipeline=feature_pipeline,
        paper_loop_store=paper_loop_store,
        now_fn=lambda: datetime(2024, 1, 1, 3, 5, tzinfo=timezone.utc),
        sleep_fn=lambda seconds: None,
    )
    selection = PaperLoopSelection(
        loop_id="loop_test_stale_ts",
        target_name="target_long_net_positive_4bars",
        target_key="fee_4",
        model_run_id="RUNPAPERSTALE",
        optimization_id="OPTSTALE",
        symbol="BTCUSDT",
        timeframe="15m",
        feature_version="v1",
        provider="cryptocompare",
    )

    with pytest.raises(
        ScheduledPaperTradingServiceError,
        match="Latest feature row timestamp does not match the latest closed candle timestamp.",
    ) as exc_info:
        service.run_once(selection=selection)
    message = str(exc_info.value)
    assert "normalized_latest_closed_candle_timestamp=2024-01-01T02:45:00+00:00" in message
    assert "normalized_feature_timestamp=2024-01-01T02:30:00+00:00" in message
    assert "timeframe=15m" in message
    assert "equality_result=False" in message
