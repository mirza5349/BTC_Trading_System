"""Operational logger — structured event logging for the trading bot.

Provides domain-specific logging helpers on top of the base logging config.
"""

from __future__ import annotations

from typing import Any

from trading_bot.logging_config import get_logger

_logger = get_logger("trading_bot.monitoring")


def log_trade_event(
    event_type: str,
    symbol: str,
    **kwargs: Any,
) -> None:
    """Log a trade-related event with structured context.

    Args:
        event_type: Type of event (e.g., "signal_generated", "order_filled").
        symbol: Trading pair symbol.
        **kwargs: Additional structured fields.
    """
    _logger.info(
        event_type,
        symbol=symbol,
        **kwargs,
    )


def log_risk_event(
    event_type: str,
    **kwargs: Any,
) -> None:
    """Log a risk-management event.

    Args:
        event_type: Type of risk event (e.g., "daily_loss_limit_hit").
        **kwargs: Additional structured fields.
    """
    _logger.warning(
        event_type,
        category="risk",
        **kwargs,
    )


def log_system_event(
    event_type: str,
    **kwargs: Any,
) -> None:
    """Log a system-level event.

    Args:
        event_type: Type of system event (e.g., "startup", "shutdown").
        **kwargs: Additional structured fields.
    """
    _logger.info(
        event_type,
        category="system",
        **kwargs,
    )
