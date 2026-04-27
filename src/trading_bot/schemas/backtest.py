"""Schemas for backtesting and offline paper-trading simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SIMULATION_TYPES = ("backtest", "paper_simulation")
SIGNAL_ACTIONS = ("flat", "enter_long", "hold_long", "exit_long")


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration for the prediction-driven strategy."""

    strategy_version: str = "v1"
    type: Literal["long_only_threshold"] = "long_only_threshold"
    entry_threshold: float = 0.55
    exit_threshold: float = 0.45
    minimum_holding_bars: int = 1
    cooldown_bars: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return asdict(self)


@dataclass(frozen=True)
class ExecutionConfig:
    """Configuration for execution and portfolio assumptions."""

    fill_model: Literal["next_bar_open"] = "next_bar_open"
    starting_cash: float = 100.0
    fee_rate: float = 0.001
    slippage_rate: float = 0.0005
    max_position_fraction: float = 1.0
    allow_fractional_position: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return asdict(self)


@dataclass(frozen=True)
class BacktestConfig:
    """Full backtest or offline paper simulation configuration."""

    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "strategy": self.strategy.to_dict(),
            "execution": self.execution.to_dict(),
        }


@dataclass(frozen=True)
class SimulationSummaryMetrics:
    """Aggregate performance metrics for one simulation."""

    total_return: float
    max_drawdown: float
    sharpe_like: float
    number_of_trades: int
    win_rate: float
    average_trade_return: float
    exposure_ratio: float
    turnover: float
    fee_impact: float
    profit_factor: float
    average_holding_bars: float
    ending_equity: float
    realized_pnl: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return asdict(self)


@dataclass(frozen=True)
class BacktestResult:
    """Final result container for a persisted simulation run."""

    simulation_id: str
    simulation_type: Literal["backtest", "paper_simulation"]
    model_run_id: str
    symbol: str
    timeframe: str
    start_timestamp: str
    end_timestamp: str
    strategy_version: str
    config: BacktestConfig
    metrics: SimulationSummaryMetrics
    artifact_paths: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "simulation_id": self.simulation_id,
            "simulation_type": self.simulation_type,
            "model_run_id": self.model_run_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "strategy_version": self.strategy_version,
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
            "artifact_paths": dict(self.artifact_paths),
            "provenance": dict(self.provenance),
        }
