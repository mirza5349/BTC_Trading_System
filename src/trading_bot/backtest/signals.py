"""Signal generation from stored model prediction probabilities."""

from __future__ import annotations

import pandas as pd

from trading_bot.schemas.backtest import SIGNAL_ACTIONS, StrategyConfig

REQUIRED_PREDICTION_COLUMNS = ("timestamp", "y_proba")


class SignalGenerationError(Exception):
    """Raised when prediction-driven signals cannot be generated."""


def generate_threshold_signals(
    predictions_df: pd.DataFrame,
    *,
    strategy_config: StrategyConfig,
) -> pd.DataFrame:
    """Convert prediction probabilities into a long-only action stream."""
    missing_columns = [
        column for column in REQUIRED_PREDICTION_COLUMNS if column not in predictions_df.columns
    ]
    if missing_columns:
        raise SignalGenerationError(
            f"Predictions are missing required signal columns: {missing_columns}"
        )

    if predictions_df.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "y_proba",
                "signal_action",
                "desired_position",
                "bars_held_signal_state",
                "cooldown_remaining",
            ]
        )

    frame = predictions_df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

    signals: list[dict[str, object]] = []
    in_position = False
    bars_held = 0
    cooldown_remaining = 0

    for row in frame.itertuples(index=False):
        timestamp = pd.Timestamp(row.timestamp)
        y_proba = float(row.y_proba)

        if in_position:
            bars_held += 1

        action = "flat"
        if not in_position:
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
            elif y_proba >= strategy_config.entry_threshold:
                action = "enter_long"
                in_position = True
                bars_held = 0
        else:
            if (
                bars_held >= strategy_config.minimum_holding_bars
                and y_proba <= strategy_config.exit_threshold
            ):
                action = "exit_long"
                in_position = False
                bars_held = 0
                cooldown_remaining = strategy_config.cooldown_bars
            else:
                action = "hold_long"

        if action not in SIGNAL_ACTIONS:
            raise SignalGenerationError(f"Unsupported signal action generated: {action}")

        signals.append(
            {
                "timestamp": timestamp,
                "y_proba": y_proba,
                "signal_action": action,
                "desired_position": 1 if in_position else 0,
                "bars_held_signal_state": bars_held if in_position else 0,
                "cooldown_remaining": cooldown_remaining,
            }
        )

    return pd.DataFrame(signals)
