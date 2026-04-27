"""Signal engine — translates model predictions into trading signals.

Combines model output, risk rules, and market state to produce
actionable trade signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from trading_bot.logging_config import get_logger

logger = get_logger(__name__)


class SignalAction(str, Enum):
    """Possible signal actions."""

    LONG = "long"
    SHORT = "short"
    CLOSE = "close"
    NO_ACTION = "no_action"


@dataclass(frozen=True)
class TradeSignal:
    """An actionable trade signal."""

    timestamp_ms: int
    symbol: str
    action: SignalAction
    confidence: float = 0.0
    suggested_size: float = 0.0
    reason: str = ""
    model_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """Return True if this signal requires action."""
        return self.action != SignalAction.NO_ACTION


class SignalEngine:
    """Translates model predictions and risk constraints into trade signals.

    The signal engine acts as a gatekeeper between predictions and execution,
    applying confidence thresholds and risk checks.
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        max_position_fraction: float = 0.01,
    ) -> None:
        """Initialize the signal engine.

        Args:
            min_confidence: Minimum model confidence to generate a signal.
            max_position_fraction: Maximum portfolio fraction per trade.
        """
        self.min_confidence = min_confidence
        self.max_position_fraction = max_position_fraction

    def generate_signal(
        self,
        prediction: Any = None,
        market_state: Any = None,
    ) -> TradeSignal:
        """Generate a trade signal from a prediction.

        Args:
            prediction: Model prediction (Prediction instance).
            market_state: Current market state context.

        Returns:
            TradeSignal with the recommended action.

        Note:
            Placeholder — full logic in step 2.
        """
        logger.debug("generate_signal_called", status="placeholder")
        return TradeSignal(
            timestamp_ms=0,
            symbol="BTCUSDT",
            action=SignalAction.NO_ACTION,
            reason="placeholder — no live signal generation",
        )
