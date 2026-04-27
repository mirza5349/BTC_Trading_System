"""Tests for simulation summary metrics."""

from __future__ import annotations

import pandas as pd

from trading_bot.backtest.metrics import compute_simulation_metrics


def test_compute_simulation_metrics_handles_no_trade_case() -> None:
    """No-trade simulations should report clean zero-like metrics."""
    equity_curve_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"),
            "equity": [100.0, 100.0, 100.0],
            "drawdown": [0.0, 0.0, 0.0],
            "in_position": [False, False, False],
            "turnover_cumulative": [0.0, 0.0, 0.0],
            "realized_pnl": [0.0, 0.0, 0.0],
        }
    )

    metrics = compute_simulation_metrics(equity_curve_df, pd.DataFrame(), starting_cash=100.0)

    assert metrics.total_return == 0.0
    assert metrics.number_of_trades == 0
    assert metrics.win_rate == 0.0
    assert metrics.turnover == 0.0


def test_compute_simulation_metrics_computes_trade_statistics() -> None:
    """Trade and equity outputs should aggregate into stable simulation metrics."""
    equity_curve_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC"),
            "equity": [100.0, 102.0, 101.0, 105.0],
            "drawdown": [0.0, 0.0, 0.00980392, 0.0],
            "in_position": [False, True, True, False],
            "turnover_cumulative": [0.0, 100.0, 100.0, 205.0],
            "realized_pnl": [0.0, 0.0, 0.0, 5.0],
        }
    )
    trades_df = pd.DataFrame(
        {
            "pnl": [5.0, -2.0],
            "return_pct": [0.05, -0.02],
            "holding_bars": [2, 1],
        }
    )

    metrics = compute_simulation_metrics(equity_curve_df, trades_df, starting_cash=100.0)

    assert metrics.total_return == 0.05
    assert metrics.number_of_trades == 2
    assert metrics.win_rate == 0.5
    assert metrics.average_holding_bars == 1.5
