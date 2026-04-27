"""Broker adapter — abstract interface for exchange connectivity.

Defines the contract for paper and live trading adapters.
Live trading will NOT be implemented in step 1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from trading_bot.execution.order_manager import Order, OrderSide
from trading_bot.logging_config import get_logger

logger = get_logger(__name__)


class BrokerAdapter(ABC):
    """Abstract broker adapter for exchange connectivity.

    Concrete subclasses:
    - PaperBrokerAdapter: Simulated execution (step 2)
    - BinanceBrokerAdapter: Real exchange execution (future)
    """

    @abstractmethod
    async def submit_order(self, order: Order) -> str:
        """Submit an order to the exchange.

        Args:
            order: Order to submit.

        Returns:
            Exchange order ID.
        """

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: Exchange order ID.

        Returns:
            True if cancellation was successful.
        """

    @abstractmethod
    async def get_balance(self, asset: str = "USDT") -> float:
        """Get available balance for an asset.

        Args:
            asset: Asset symbol (e.g., "USDT", "BTC").

        Returns:
            Available balance.
        """

    @abstractmethod
    async def get_position(self, symbol: str) -> dict[str, Any]:
        """Get current position for a symbol.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Position details dict.
        """


class PaperBrokerAdapter(BrokerAdapter):
    """Paper trading broker that simulates execution locally.

    All orders are filled instantly at the requested price.
    No real exchange connectivity.
    """

    def __init__(self, initial_balance: float = 10_000.0) -> None:
        self._balance: float = initial_balance
        self._positions: dict[str, float] = {}

    async def submit_order(self, order: Order) -> str:
        """Simulate order submission with instant fill."""
        logger.info(
            "paper_order_submitted",
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            price=order.price,
        )
        # Simulate instant fill
        if order.side == OrderSide.BUY:
            cost = order.quantity * order.price
            self._balance -= cost
            self._positions[order.symbol] = self._positions.get(order.symbol, 0.0) + order.quantity
        else:
            proceeds = order.quantity * order.price
            self._balance += proceeds
            self._positions[order.symbol] = self._positions.get(order.symbol, 0.0) - order.quantity

        return f"paper_{id(order)}"

    async def cancel_order(self, order_id: str) -> bool:
        """Paper broker: cancellation always succeeds."""
        logger.info("paper_order_cancelled", order_id=order_id)
        return True

    async def get_balance(self, asset: str = "USDT") -> float:
        """Return simulated balance."""
        if asset == "USDT":
            return self._balance
        return self._positions.get(asset, 0.0)

    async def get_position(self, symbol: str) -> dict[str, Any]:
        """Return simulated position."""
        qty = self._positions.get(symbol, 0.0)
        return {"symbol": symbol, "quantity": qty, "side": "long" if qty > 0 else "flat"}
