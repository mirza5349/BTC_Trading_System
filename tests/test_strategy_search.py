"""Tests for strategy search-space generation and runner behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trading_bot.models.model_registry import ModelRegistry
from trading_bot.optimization.search_space import build_parameter_grid
from trading_bot.optimization.strategy_search import StrategySearchRunner
from trading_bot.settings import load_settings
from trading_bot.storage.evaluation_store import EvaluationStore
from trading_bot.storage.optimization_store import OptimizationStore
from trading_bot.storage.parquet_store import ParquetStore


def test_build_parameter_grid_rejects_invalid_threshold_combinations() -> None:
    """Search-space generation should skip invalid threshold pairs."""
    grid = build_parameter_grid(
        {
            "entry_threshold": [0.6, 0.5],
            "exit_threshold": [0.55, 0.5],
            "minimum_holding_bars": [2],
            "cooldown_bars": [2],
            "max_position_fraction": [0.5],
        }
    )

    assert len(grid) == 2
    assert grid[0].entry_threshold == 0.6
    assert grid[0].exit_threshold == 0.55


def test_strategy_search_runner_persists_compact_results(tmp_path: Path) -> None:
    """A strategy search should persist candidate rows, top candidates, and a report."""
    evaluation_store = EvaluationStore(tmp_path / "evaluation")
    market_store = ParquetStore(tmp_path / "market")
    optimization_store = OptimizationStore(tmp_path / "optimization")

    coordinates = {
        "asset": "BTC",
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "model_version": "v1",
        "run_id": "RUNOPT",
    }
    metadata = {
        "run_id": "RUNOPT",
        "created_at": "2024-01-01T00:00:00+00:00",
        "asset": "BTC",
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "dataset_version": "v1",
        "feature_version": "v1",
        "label_version": "v1",
        "model_version": "v1",
        "target_column": "target_up_2bars",
        "requested_device": "cpu",
        "effective_device": "cpu",
        "lightgbm_version": "test-lightgbm",
    }
    predictions_df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 8,
            "timeframe": ["15m"] * 8,
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC"),
            "y_true": [1, 1, 0, 1, 0, 1, 0, 0],
            "y_pred": [1, 1, 0, 1, 0, 1, 0, 0],
            "y_proba": [0.72, 0.75, 0.40, 0.70, 0.38, 0.69, 0.35, 0.30],
        }
    )
    candles_df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 8,
            "timeframe": ["15m"] * 8,
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC"),
            "open": [100, 101, 103, 104, 105, 106, 107, 108],
            "high": [101, 104, 104, 105, 106, 107, 108, 109],
            "low": [99, 100, 102, 103, 104, 105, 106, 107],
            "close": [101, 103, 104, 105, 106, 107, 108, 109],
            "volume": [10, 11, 12, 13, 14, 15, 16, 17],
            "open_time": pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC"),
            "close_time": pd.date_range("2024-01-01 00:14:59", periods=8, freq="15min", tz="UTC"),
        }
    )
    evaluation_store.write_predictions(predictions_df, **coordinates)
    evaluation_store.write_run_metadata(metadata, **coordinates)
    market_store.write(candles_df, "BTCUSDT", "15m")

    settings = load_settings()
    settings.optimization_search_space.entry_threshold = [0.6]
    settings.optimization_search_space.exit_threshold = [0.45]
    settings.optimization_search_space.minimum_holding_bars = [2]
    settings.optimization_search_space.cooldown_bars = [2]
    settings.optimization_search_space.max_position_fraction = [0.5, 1.0]
    settings.optimization_selection.top_k = 5

    runner = StrategySearchRunner(
        model_registry=ModelRegistry(evaluation_store=evaluation_store),
        market_store=market_store,
        optimization_store=optimization_store,
        settings=settings,
    )

    report = runner.run_search(model_run_id="RUNOPT", optimization_id="OPT001")

    candidate_path = (
        tmp_path
        / "optimization"
        / "model_run_id=RUNOPT"
        / "optimization_id=OPT001"
        / "candidate_results.parquet"
    )
    report_path = candidate_path.with_name("optimization_report.json")

    assert candidate_path.exists()
    assert report_path.exists()
    assert report["optimization_id"] == "OPT001"
    assert report["candidates_tested"] == 2

    payload = json.loads(report_path.read_text())
    assert "recommendation" in payload
