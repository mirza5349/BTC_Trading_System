"""Tests for validation-suite diagnostics and reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trading_bot.backtest.simulator import BacktestSimulator
from trading_bot.models.model_registry import ModelRegistry
from trading_bot.storage.backtest_store import BacktestStore
from trading_bot.storage.evaluation_store import EvaluationStore
from trading_bot.storage.paper_sim_store import PaperSimulationStore
from trading_bot.storage.parquet_store import ParquetStore
from trading_bot.storage.validation_store import ValidationStore
from trading_bot.settings import load_settings
from trading_bot.validation.suite import (
    ValidationSuite,
    build_failure_analysis,
    build_prediction_diagnostics,
)


def test_build_prediction_diagnostics_buckets_probabilities() -> None:
    """Prediction diagnostics should bucket probabilities and compute outcomes."""
    predictions_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC"),
            "y_true": [1, 1, 0, 1, 0],
            "y_pred": [1, 1, 0, 0, 1],
            "y_proba": [0.55, 0.62, 0.74, 0.83, 0.91],
            "close": [100.0, 101.0, 102.0, 103.0, 102.0],
        }
    )

    diagnostics_df = build_prediction_diagnostics(
        predictions_df=predictions_df,
        bucket_edges=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )

    assert diagnostics_df["bucket"].tolist()[:2] == ["0.5-0.6", "0.6-0.7"]
    assert diagnostics_df["trade_count"].sum() == 5


def test_build_failure_analysis_returns_losers_and_drawdowns() -> None:
    """Failure analysis should surface the worst trades and deepest drawdown rows."""
    trades_df = pd.DataFrame(
        {
            "trade_id": ["T1", "T2"],
            "pnl": [-4.0, 2.0],
            "return_pct": [-0.04, 0.02],
            "entry_price": [100.0, 101.0],
        }
    )
    equity_curve_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"),
            "drawdown": [0.0, 0.05, 0.02],
        }
    )
    signals_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"),
            "y_proba": [0.55, 0.92, 0.60],
            "signal_action": ["enter_long", "hold_long", "exit_long"],
        }
    )

    analysis = build_failure_analysis(
        trades_df=trades_df,
        equity_curve_df=equity_curve_df,
        signals_df=signals_df,
        top_failure_cases=1,
        drawdown_case_count=1,
    )

    assert analysis["top_losing_trades"][0]["trade_id"] == "T1"
    assert analysis["largest_drawdown_periods"][0]["drawdown"] == 0.05


def test_run_validation_suite_persists_report_and_comparison(tmp_path: Path) -> None:
    """The validation suite should persist a full report and ranked comparison table."""
    evaluation_store = EvaluationStore(tmp_path / "evaluation")
    market_store = ParquetStore(tmp_path / "market")
    backtest_store = BacktestStore(tmp_path / "backtests")
    paper_store = PaperSimulationStore(tmp_path / "paper")
    validation_store = ValidationStore(tmp_path / "validation")

    run_metadata = {
        "run_id": "RUNVAL",
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
            "symbol": ["BTCUSDT"] * 6,
            "timeframe": ["15m"] * 6,
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="15min", tz="UTC"),
            "y_true": [1, 1, 0, 1, 0, 0],
            "y_pred": [1, 1, 0, 1, 0, 0],
            "y_proba": [0.60, 0.65, 0.42, 0.70, 0.40, 0.38],
        }
    )
    candles_df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 6,
            "timeframe": ["15m"] * 6,
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="15min", tz="UTC"),
            "open": [100.0, 101.0, 103.0, 102.0, 104.0, 105.0],
            "high": [101.0, 104.0, 104.0, 103.0, 105.0, 106.0],
            "low": [99.5, 100.5, 101.5, 101.0, 103.0, 104.0],
            "close": [100.5, 103.0, 102.0, 104.0, 105.0, 104.5],
            "volume": [10.0, 11.0, 12.0, 13.0, 12.0, 11.0],
            "open_time": pd.date_range("2024-01-01", periods=6, freq="15min", tz="UTC"),
            "close_time": pd.date_range("2024-01-01 00:14:59", periods=6, freq="15min", tz="UTC"),
        }
    )
    feature_importance_df = pd.DataFrame(
        {
            "feature_name": ["news_sentiment_mean_1h", "ret_1"],
            "mean_importance": [10.0, 8.0],
            "folds_present": [5, 5],
        }
    )

    evaluation_store.write_predictions(
        predictions_df,
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        model_version="v2_sentiment",
        run_id="RUNVAL",
    )
    evaluation_store.write_run_metadata(
        run_metadata,
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        model_version="v2_sentiment",
        run_id="RUNVAL",
    )
    evaluation_store.write_feature_importance(
        feature_importance_df,
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        model_version="v2_sentiment",
        run_id="RUNVAL",
        aggregate=True,
    )
    market_store.write(candles_df, "BTCUSDT", "15m")

    simulator = BacktestSimulator(
        model_registry=ModelRegistry(evaluation_store=evaluation_store),
        market_store=market_store,
        backtest_store=backtest_store,
        paper_sim_store=paper_store,
        write_markdown=True,
    )
    suite = ValidationSuite(
        model_registry=ModelRegistry(evaluation_store=evaluation_store),
        simulator=simulator,
        market_store=market_store,
        backtest_store=backtest_store,
        validation_store=validation_store,
        settings=load_settings(),
    )

    report = suite.run_validation_suite(model_run_id="RUNVAL", validation_id="VAL001")

    report_path = tmp_path / "validation" / "validation_id=VAL001" / "validation_report.json"
    comparison_path = tmp_path / "validation" / "validation_id=VAL001" / "strategy_comparison.parquet"

    assert report["validation_id"] == "VAL001"
    assert report["horizon_bars"] == 2
    assert report["target_column"] == "target_up_2bars"
    assert len(report["strategy_comparison"]) == 5
    assert report_path.exists()
    assert comparison_path.exists()

    payload = json.loads(report_path.read_text())
    assert payload["feature_importance_summary"]["has_sentiment_signal"] is True
