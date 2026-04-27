"""Alerting system — sends notifications for critical events.

Placeholder for future integration with Slack, email, PagerDuty, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from trading_bot.logging_config import get_logger

logger = get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Alert:
    """An alert notification."""

    title: str
    message: str
    severity: AlertSeverity = AlertSeverity.INFO
    source: str = "trading_bot"


class AlertSender(ABC):
    """Abstract alert sender interface."""

    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """Send an alert notification.

        Args:
            alert: Alert to send.

        Returns:
            True if the alert was sent successfully.
        """


class LogAlertSender(AlertSender):
    """Sends alerts by logging them (default for development)."""

    async def send(self, alert: Alert) -> bool:
        """Log the alert instead of sending it externally."""
        log_fn = logger.info if alert.severity == AlertSeverity.INFO else logger.warning
        log_fn(
            "alert_sent",
            title=alert.title,
            severity=alert.severity.value,
            message=alert.message,
        )
        return True
