"""Execution simulation for backtests and offline paper simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from trading_bot.schemas.backtest import ExecutionConfig


class ExecutionSimulationError(Exception):
    """Raised when execution simulation cannot proceed."""


@dataclass
class SimulationFrames:
    """All tabular outputs produced by the execution engine."""

    signals: pd.DataFrame
    equity_curve: pd.DataFrame
    trades: pd.DataFrame


def simulate_execution(
    market_predictions_df: pd.DataFrame,
    *,
    signals_df: pd.DataFrame,
    execution_config: ExecutionConfig,
) -> SimulationFrames:
    """Replay signals through a strict next-bar-open execution engine."""
    required_columns = {"timestamp", "open", "high", "low", "close", "volume"}
    missing_columns = sorted(required_columns - set(market_predictions_df.columns))
    if missing_columns:
        raise ExecutionSimulationError(
            f"Simulation frame is missing required market columns: {missing_columns}"
        )

    if market_predictions_df.empty:
        empty = pd.DataFrame()
        return SimulationFrames(signals=signals_df.copy(), equity_curve=empty, trades=empty)

    frame = market_predictions_df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    signals = signals_df.copy()
    signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True)
    signals = signals.sort_values("timestamp").reset_index(drop=True)

    merged = frame.merge(
        signals[
            [
                "timestamp",
                "signal_action",
                "desired_position",
                "bars_held_signal_state",
                "cooldown_remaining",
            ]
        ],
        on="timestamp",
        how="left",
    )
    merged["signal_action"] = merged["signal_action"].fillna("flat")
    merged["desired_position"] = merged["desired_position"].fillna(0).astype(int)
    merged["bars_held_signal_state"] = merged["bars_held_signal_state"].fillna(0).astype(int)
    merged["cooldown_remaining"] = merged["cooldown_remaining"].fillna(0).astype(int)

    cash = float(execution_config.starting_cash)
    btc_quantity = 0.0
    realized_pnl = 0.0
    peak_equity = cash
    turnover = 0.0
    fees_cumulative = 0.0
    slippage_cumulative = 0.0
    pending_action: str | None = None
    open_trade: dict[str, Any] | None = None
    trade_sequence = 0
    bars_in_position = 0

    trade_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    executed_actions: list[str] = []

    for index, row in merged.iterrows():
        timestamp = pd.Timestamp(row["timestamp"])
        bar_open = float(row["open"])
        bar_close = float(row["close"])
        executed_action = "none"

        if pending_action == "enter_long" and btc_quantity <= 0.0:
            quantity, entry_fee, entry_slippage_cost, entry_notional = _open_long_position(
                cash=cash,
                next_open=bar_open,
                execution_config=execution_config,
            )
            if quantity > 0.0:
                fill_price = _apply_entry_slippage(bar_open, execution_config.slippage_rate)
                cash -= entry_notional + entry_fee
                btc_quantity = quantity
                turnover += entry_notional
                fees_cumulative += entry_fee
                slippage_cumulative += entry_slippage_cost
                trade_sequence += 1
                open_trade = {
                    "trade_id": f"T{trade_sequence:05d}",
                    "side": "long",
                    "entry_timestamp": timestamp.isoformat(),
                    "entry_price": round(fill_price, 8),
                    "entry_notional": round(entry_notional, 8),
                    "quantity": round(quantity, 12),
                    "entry_fee": round(entry_fee, 8),
                    "entry_slippage": round(entry_slippage_cost, 8),
                    "bars_held": 0,
                }
                executed_action = "enter_long"
                bars_in_position = 0

        elif pending_action == "exit_long" and btc_quantity > 0.0 and open_trade is not None:
            fill_price = _apply_exit_slippage(bar_open, execution_config.slippage_rate)
            exit_notional = btc_quantity * fill_price
            exit_fee = exit_notional * execution_config.fee_rate
            exit_slippage_cost = btc_quantity * (bar_open - fill_price)
            cash += exit_notional - exit_fee
            turnover += exit_notional
            fees_cumulative += exit_fee
            slippage_cumulative += exit_slippage_cost

            trade_fees = float(open_trade["entry_fee"]) + exit_fee
            trade_slippage = float(open_trade["entry_slippage"]) + exit_slippage_cost
            pnl = (exit_notional - exit_fee) - (
                float(open_trade["entry_notional"]) + float(open_trade["entry_fee"])
            )
            entry_cost_basis = float(open_trade["entry_notional"]) + float(open_trade["entry_fee"])
            return_pct = pnl / entry_cost_basis if entry_cost_basis else 0.0
            holding_bars = int(open_trade["bars_held"])

            trade_rows.append(
                {
                    "trade_id": open_trade["trade_id"],
                    "side": "long",
                    "entry_timestamp": open_trade["entry_timestamp"],
                    "exit_timestamp": timestamp.isoformat(),
                    "entry_price": float(open_trade["entry_price"]),
                    "exit_price": round(fill_price, 8),
                    "quantity": round(btc_quantity, 12),
                    "entry_notional": float(open_trade["entry_notional"]),
                    "exit_notional": round(exit_notional, 8),
                    "fees_paid": round(trade_fees, 8),
                    "slippage_incurred": round(trade_slippage, 8),
                    "pnl": round(pnl, 8),
                    "return_pct": round(return_pct, 8),
                    "holding_bars": holding_bars,
                }
            )

            realized_pnl += pnl
            btc_quantity = 0.0
            open_trade = None
            executed_action = "exit_long"
            bars_in_position = 0

        if btc_quantity > 0.0 and open_trade is not None:
            bars_in_position += 1
            open_trade["bars_held"] = bars_in_position

        market_value = btc_quantity * bar_close
        unrealized_pnl = (
            market_value - float(open_trade["entry_notional"]) - float(open_trade["entry_fee"])
            if btc_quantity > 0.0 and open_trade is not None
            else 0.0
        )
        equity = cash + market_value
        peak_equity = max(peak_equity, equity)
        drawdown = (peak_equity - equity) / peak_equity if peak_equity else 0.0

        equity_rows.append(
            {
                "timestamp": timestamp,
                "signal_action": row["signal_action"],
                "executed_action": executed_action,
                "open": bar_open,
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": bar_close,
                "volume": float(row["volume"]),
                "y_proba": float(row["y_proba"]) if not pd.isna(row["y_proba"]) else None,
                "cash": round(cash, 8),
                "btc_quantity": round(btc_quantity, 12),
                "market_value": round(market_value, 8),
                "realized_pnl": round(realized_pnl, 8),
                "unrealized_pnl": round(unrealized_pnl, 8),
                "equity": round(equity, 8),
                "drawdown": round(drawdown, 8),
                "bars_in_position": bars_in_position if btc_quantity > 0.0 else 0,
                "in_position": btc_quantity > 0.0,
                "turnover_cumulative": round(turnover, 8),
                "fees_cumulative": round(fees_cumulative, 8),
                "slippage_cumulative": round(slippage_cumulative, 8),
            }
        )
        executed_actions.append(executed_action)

        pending_action = None
        if index < len(merged) - 1 and row["signal_action"] in {"enter_long", "exit_long"}:
            pending_action = str(row["signal_action"])

    signals_out = merged.copy()
    signals_out["executed_action"] = executed_actions
    signals_out["can_execute_next_bar"] = signals_out.index < (len(signals_out) - 1)

    return SimulationFrames(
        signals=signals_out,
        equity_curve=pd.DataFrame(equity_rows),
        trades=pd.DataFrame(trade_rows),
    )


def _open_long_position(
    *,
    cash: float,
    next_open: float,
    execution_config: ExecutionConfig,
) -> tuple[float, float, float, float]:
    """Return quantity, fee, slippage cost, and notional for a long entry."""
    if cash <= 0.0:
        return 0.0, 0.0, 0.0, 0.0

    allocated_cash = cash * execution_config.max_position_fraction
    fill_price = _apply_entry_slippage(next_open, execution_config.slippage_rate)
    gross_unit_cost = fill_price * (1.0 + execution_config.fee_rate)
    quantity = allocated_cash / gross_unit_cost if gross_unit_cost > 0.0 else 0.0
    if not execution_config.allow_fractional_position:
        quantity = float(int(quantity))
    if quantity <= 0.0:
        return 0.0, 0.0, 0.0, 0.0

    notional = quantity * fill_price
    fee = notional * execution_config.fee_rate
    slippage_cost = quantity * (fill_price - next_open)
    return quantity, fee, slippage_cost, notional


def _apply_entry_slippage(open_price: float, slippage_rate: float) -> float:
    """Apply adverse slippage to a long entry."""
    return open_price * (1.0 + slippage_rate)


def _apply_exit_slippage(open_price: float, slippage_rate: float) -> float:
    """Apply adverse slippage to a long exit."""
    return open_price * (1.0 - slippage_rate)
