"""Order manager — tracks and manages order lifecycle.

Handles order creation, status tracking, fill management,
and position reconciliation. Placeholder for step 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from trading_bot.logging_config import get_logger

logger = get_logger(__name__)


class OrderStatus(str, Enum):
    """Order lifecycle states."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(str, Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Represents a trading order."""

    order_id: str = ""
    symbol: str = "BTCUSDT"
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class OrderManager:
    """Manages the order lifecycle.

    Tracks pending and filled orders, validates risk limits before
    submission, and reconciles positions.
    """

    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float = 0.0,
    ) -> Order:
        """Create a new order (does not submit it).

        Args:
            symbol: Trading pair symbol.
            side: Buy or sell.
            quantity: Order quantity.
            price: Limit price (0 = market order).

        Returns:
            The created Order instance.

        Note:
            Placeholder — real order management in step 2.
        """
        logger.info(
            "order_created",
            symbol=symbol,
            side=side.value,
            quantity=quantity,
            status="placeholder",
        )
        order = Order(symbol=symbol, side=side, quantity=quantity, price=price)
        return order

    def get_open_orders(self) -> list[Order]:
        """Return all open (non-terminal) orders."""
        return [
            o for o in self._orders.values()
            if o.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED)
        ]
