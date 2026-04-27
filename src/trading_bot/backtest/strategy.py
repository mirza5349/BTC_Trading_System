"""Strategy helpers for backtesting and offline paper simulation."""

from __future__ import annotations

from dataclasses import dataclass

from trading_bot.schemas.backtest import StrategyConfig
from trading_bot.settings import AppSettings, load_settings


@dataclass(frozen=True)
class Strategy:
    """Thin wrapper around the configured long-only threshold strategy."""

    config: StrategyConfig

    @property
    def name(self) -> str:
        """Return a stable display name."""
        return f"{self.config.type}:{self.config.strategy_version}"

    @classmethod
    def from_settings(cls, settings: AppSettings | None = None) -> Strategy:
        """Build a strategy from application settings."""
        resolved_settings = settings or load_settings()
        return cls(config=get_default_strategy(resolved_settings))


def get_default_strategy(settings: AppSettings | None = None) -> StrategyConfig:
    """Return the default conservative strategy for BTC 15m research."""
    resolved_settings = settings or load_settings()
    configured = resolved_settings.backtest_strategy
    return StrategyConfig(
        strategy_version=configured.strategy_version,
        type="long_only_threshold",
        entry_threshold=configured.entry_threshold,
        exit_threshold=configured.exit_threshold,
        minimum_holding_bars=configured.minimum_holding_bars,
        cooldown_bars=configured.cooldown_bars,
    )
