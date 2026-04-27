"""Tests for backtest orchestration and execution simulation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trading_bot.backtest.execution import simulate_execution
from trading_bot.backtest.signals import generate_threshold_signals
from trading_bot.backtest.simulator import BacktestSimulator
from trading_bot.backtest.strategy import Strategy
from trading_bot.models.model_registry import ModelRegistry
from trading_bot.schemas.backtest import ExecutionConfig, StrategyConfig
from trading_bot.storage.backtest_store import BacktestStore
from trading_bot.storage.evaluation_store import EvaluationStore
from trading_bot.storage.paper_sim_store import PaperSimulationStore
from trading_bot.storage.parquet_store import ParquetStore


def test_simulate_execution_uses_next_bar_open_and_records_trade_log() -> None:
    """Entry and exit should be filled on the following bar open with costs applied."""
    market_predictions_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC"),
            "open": [100.0, 101.0, 103.0, 102.0],
            "high": [101.0, 104.0, 104.0, 103.0],
            "low": [99.5, 100.5, 101.5, 101.0],
            "close": [100.5, 103.0, 102.0, 101.5],
            "volume": [10.0, 11.0, 12.0, 13.0],
            "y_proba": [0.60, 0.58, 0.40, 0.39],
        }
    )
    strategy = StrategyConfig(
        entry_threshold=0.55,
        exit_threshold=0.45,
        minimum_holding_bars=1,
        cooldown_bars=0,
    )
    signals_df = generate_threshold_signals(market_predictions_df, strategy_config=strategy)
    frames = simulate_execution(
        market_predictions_df,
        signals_df=signals_df,
        execution_config=ExecutionConfig(
            starting_cash=100.0,
            fee_rate=0.001,
            slippage_rate=0.0005,
            max_position_fraction=1.0,
        ),
    )

    assert not frames.equity_curve.empty
    assert len(frames.trades) == 1
    assert frames.trades.iloc[0]["entry_timestamp"] == "2024-01-01T00:15:00+00:00"
    assert frames.trades.iloc[0]["exit_timestamp"] == "2024-01-01T00:45:00+00:00"
    assert frames.trades.iloc[0]["fees_paid"] > 0.0
    assert frames.signals.iloc[1]["executed_action"] == "enter_long"


def test_backtest_simulator_persists_artifacts_with_run_provenance(tmp_path: Path) -> None:
    """A backtest should load stored predictions, simulate, and persist local outputs."""
    evaluation_store = EvaluationStore(tmp_path / "evaluation")
    market_store = ParquetStore(tmp_path / "market")

    run_metadata = {
        "run_id": "RUN123",
        "asset": "BTC",
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "model_version": "v2_sentiment",
        "dataset_version": "v2_sentiment",
        "feature_version": "v2_sentiment",
        "target_column": "target_up_2bars",
        "created_at": "2024-01-01T00:00:00+00:00",
    }
    predictions_df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 4,
            "timeframe": ["15m"] * 4,
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC"),
            "y_true": [1, 1, 0, 0],
            "y_pred": [1, 1, 0, 0],
            "y_proba": [0.60, 0.57, 0.42, 0.40],
        }
    )
    candles_df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 4,
            "timeframe": ["15m"] * 4,
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC"),
            "open": [100.0, 101.0, 103.0, 102.0],
            "high": [101.0, 104.0, 104.0, 103.0],
            "low": [99.5, 100.5, 101.5, 101.0],
            "close": [100.5, 103.0, 102.0, 101.5],
            "volume": [10.0, 11.0, 12.0, 13.0],
            "open_time": pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC"),
            "close_time": pd.date_range("2024-01-01 00:14:59", periods=4, freq="15min", tz="UTC"),
        }
    )

    evaluation_store.write_predictions(
        predictions_df,
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        model_version="v2_sentiment",
        run_id="RUN123",
    )
    evaluation_store.write_run_metadata(
        run_metadata,
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        model_version="v2_sentiment",
        run_id="RUN123",
    )
    market_store.write(candles_df, "BTCUSDT", "15m")

    simulator = BacktestSimulator(
        model_registry=ModelRegistry(evaluation_store=evaluation_store),
        market_store=market_store,
        backtest_store=BacktestStore(tmp_path / "backtests"),
        paper_sim_store=PaperSimulationStore(tmp_path / "paper_simulations"),
        write_markdown=True,
    )

    result = simulator.run_simulation(
        simulation_type="backtest",
        model_run_id="RUN123",
        strategy=Strategy(config=StrategyConfig()),
        execution_config=ExecutionConfig(),
        simulation_id="SIM123",
    )

    summary_path = tmp_path / "backtests" / "simulation_id=SIM123" / "summary.json"
    trades_path = tmp_path / "backtests" / "simulation_id=SIM123" / "trades.parquet"
    equity_path = tmp_path / "backtests" / "simulation_id=SIM123" / "equity_curve.parquet"
    signals_path = tmp_path / "backtests" / "simulation_id=SIM123" / "signals.parquet"
    report_path = tmp_path / "backtests" / "simulation_id=SIM123" / "report.md"

    assert result.simulation_id == "SIM123"
    assert result.model_run_id == "RUN123"
    assert result.provenance["feature_version"] == "v2_sentiment"
    assert summary_path.exists()
    assert trades_path.exists()
    assert equity_path.exists()
    assert signals_path.exists()
    assert report_path.exists()

    summary_payload = json.loads(summary_path.read_text())
    assert summary_payload["provenance"]["dataset_version"] == "v2_sentiment"
    assert summary_payload["artifact_paths"]["summary_json_path"].endswith("summary.json")
