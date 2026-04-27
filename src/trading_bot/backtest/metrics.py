"""Performance metrics for backtests and offline paper simulations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from trading_bot.schemas.backtest import SimulationSummaryMetrics


def compute_simulation_metrics(
    equity_curve_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    *,
    starting_cash: float,
) -> SimulationSummaryMetrics:
    """Compute aggregate performance metrics from simulation outputs."""
    if equity_curve_df.empty:
        return SimulationSummaryMetrics(
            total_return=0.0,
            max_drawdown=0.0,
            sharpe_like=0.0,
            number_of_trades=0,
            win_rate=0.0,
            average_trade_return=0.0,
            exposure_ratio=0.0,
            turnover=0.0,
            fee_impact=0.0,
            profit_factor=0.0,
            average_holding_bars=0.0,
            ending_equity=starting_cash,
            realized_pnl=0.0,
        )

    ending_equity = float(equity_curve_df["equity"].iloc[-1])
    total_return = ((ending_equity / starting_cash) - 1.0) if starting_cash else 0.0
    max_drawdown = (
        float(pd.to_numeric(equity_curve_df["drawdown"], errors="coerce").fillna(0.0).max())
        if "drawdown" in equity_curve_df.columns
        else 0.0
    )
    exposure_ratio = (
        float(pd.to_numeric(equity_curve_df["in_position"], errors="coerce").fillna(0.0).mean())
        if "in_position" in equity_curve_df.columns
        else 0.0
    )
    turnover = (
        float(equity_curve_df["turnover_cumulative"].iloc[-1])
        if "turnover_cumulative" in equity_curve_df.columns
        else 0.0
    )
    fee_impact = (
        float(equity_curve_df["fees_cumulative"].iloc[-1])
        if "fees_cumulative" in equity_curve_df.columns
        else 0.0
    )
    realized_pnl = (
        float(equity_curve_df["realized_pnl"].iloc[-1])
        if "realized_pnl" in equity_curve_df.columns
        else 0.0
    )
    equity_returns = pd.to_numeric(equity_curve_df["equity"], errors="coerce").pct_change().fillna(0.0)
    return_std = float(equity_returns.std(ddof=0))
    sharpe_like = float(equity_returns.mean() / return_std) if return_std > 0.0 else 0.0

    if trades_df.empty:
        return SimulationSummaryMetrics(
            total_return=round(total_return, 8),
            max_drawdown=round(max_drawdown, 8),
            sharpe_like=round(sharpe_like, 8),
            number_of_trades=0,
            win_rate=0.0,
            average_trade_return=0.0,
            exposure_ratio=round(exposure_ratio, 8),
            turnover=round(turnover, 8),
            fee_impact=round(fee_impact, 8),
            profit_factor=0.0,
            average_holding_bars=0.0,
            ending_equity=round(ending_equity, 8),
            realized_pnl=round(realized_pnl, 8),
        )

    number_of_trades = int(len(trades_df))
    pnl_series = pd.to_numeric(trades_df["pnl"], errors="coerce").fillna(0.0)
    return_series = pd.to_numeric(trades_df["return_pct"], errors="coerce").fillna(0.0)
    holding_series = pd.to_numeric(trades_df["holding_bars"], errors="coerce").fillna(0.0)

    wins = pnl_series[pnl_series > 0.0]
    losses = pnl_series[pnl_series < 0.0]
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    if gross_loss == 0.0:
        profit_factor = float("inf") if gross_profit > 0.0 else 0.0
    else:
        profit_factor = gross_profit / gross_loss

    win_rate = float((pnl_series > 0.0).mean()) if number_of_trades else 0.0
    average_trade_return = float(return_series.mean()) if number_of_trades else 0.0
    average_holding_bars = float(holding_series.mean()) if number_of_trades else 0.0

    return SimulationSummaryMetrics(
        total_return=round(total_return, 8),
        max_drawdown=round(max_drawdown, 8),
        sharpe_like=round(sharpe_like, 8),
        number_of_trades=number_of_trades,
        win_rate=round(win_rate, 8),
        average_trade_return=round(average_trade_return, 8),
        exposure_ratio=round(exposure_ratio, 8),
        turnover=round(turnover, 8),
        fee_impact=round(fee_impact, 8),
        profit_factor=round(profit_factor, 8) if profit_factor != float("inf") else float("inf"),
        average_holding_bars=round(average_holding_bars, 8),
        ending_equity=round(ending_equity, 8),
        realized_pnl=round(realized_pnl, 8),
    )


def compare_simulation_summaries(
    summaries: list[dict[str, Any]],
    *,
    sort_by: str = "total_return",
) -> pd.DataFrame:
    """Return a machine-loadable comparison table for simulation summaries."""
    if not summaries:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for summary in summaries:
        metrics = summary.get("metrics", {})
        row = {
            "simulation_id": summary.get("simulation_id"),
            "simulation_type": summary.get("simulation_type"),
            "model_run_id": summary.get("model_run_id"),
            "symbol": summary.get("symbol"),
            "timeframe": summary.get("timeframe"),
            "strategy_version": summary.get("strategy_version"),
            "start_timestamp": summary.get("start_timestamp"),
            "end_timestamp": summary.get("end_timestamp"),
        }
        if isinstance(metrics, dict):
            row.update(metrics)
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    if sort_by in comparison_df.columns:
        ascending = sort_by in {"max_drawdown"}
        comparison_df = comparison_df.sort_values(
            by=[sort_by, "simulation_id"],
            ascending=[ascending, True],
            na_position="last",
        )
    return comparison_df.reset_index(drop=True)
