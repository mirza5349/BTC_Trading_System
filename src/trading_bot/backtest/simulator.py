"""Backtest and offline paper-simulation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

from trading_bot.backtest.execution import SimulationFrames, simulate_execution
from trading_bot.backtest.metrics import compute_simulation_metrics
from trading_bot.backtest.reporting import (
    build_simulation_markdown_report,
    build_simulation_summary_payload,
)
from trading_bot.backtest.signals import generate_threshold_signals
from trading_bot.backtest.strategy import Strategy, get_default_strategy
from trading_bot.logging_config import get_logger
from trading_bot.models.model_registry import ModelRegistry
from trading_bot.schemas.backtest import BacktestConfig, BacktestResult, ExecutionConfig
from trading_bot.settings import PROJECT_ROOT, AppSettings, load_settings
from trading_bot.storage.backtest_store import BacktestStore
from trading_bot.storage.paper_sim_store import PaperSimulationStore
from trading_bot.storage.parquet_store import ParquetStore

logger = get_logger(__name__)

SimulationType = Literal["backtest", "paper_simulation"]


class BacktestSimulatorError(Exception):
    """Raised when a simulation run cannot be completed."""


@dataclass(frozen=True)
class SimulationArtifacts:
    """Resolved persisted artifact paths for one simulation."""

    summary_path: Path
    equity_curve_path: Path
    trades_path: Path
    signals_path: Path
    report_path: Path | None


class BacktestSimulator:
    """Orchestrate prediction-driven backtests and offline paper simulations."""

    def __init__(
        self,
        *,
        model_registry: ModelRegistry,
        market_store: ParquetStore,
        backtest_store: BacktestStore,
        paper_sim_store: PaperSimulationStore,
        write_markdown: bool = True,
    ) -> None:
        self.model_registry = model_registry
        self.market_store = market_store
        self.backtest_store = backtest_store
        self.paper_sim_store = paper_sim_store
        self.write_markdown = write_markdown

    @classmethod
    def from_settings(cls, settings: AppSettings | None = None) -> BacktestSimulator:
        """Create a simulator from project settings."""
        resolved_settings = settings or load_settings()
        return cls(
            model_registry=ModelRegistry.from_settings(resolved_settings),
            market_store=ParquetStore(PROJECT_ROOT / resolved_settings.market_data.processed_data_dir),
            backtest_store=BacktestStore(PROJECT_ROOT / resolved_settings.simulations.backtests_dir),
            paper_sim_store=PaperSimulationStore(
                PROJECT_ROOT / resolved_settings.simulations.paper_simulations_dir
            ),
            write_markdown=resolved_settings.reports.write_markdown,
        )

    def run_simulation(
        self,
        *,
        simulation_type: SimulationType,
        model_run_id: str,
        symbol: str | None = None,
        timeframe: str | None = None,
        strategy: Strategy | None = None,
        execution_config: ExecutionConfig | None = None,
        simulation_id: str | None = None,
    ) -> BacktestResult:
        """Run one strict time-ordered simulation and persist all artifacts."""
        resolved_run = self.model_registry.resolve_run(
            model_run_id,
            symbol=symbol,
            timeframe=timeframe,
        )
        predictions_df = self.model_registry.read_predictions(
            model_run_id,
            symbol=resolved_run.symbol,
            timeframe=resolved_run.timeframe,
            model_version=resolved_run.model_version,
            asset=resolved_run.asset,
        )
        market_df = self.market_store.read(resolved_run.symbol, resolved_run.timeframe)
        if predictions_df.empty:
            raise BacktestSimulatorError(f"No stored predictions found for run_id={model_run_id}")
        if market_df.empty:
            raise BacktestSimulatorError(
                f"No stored market candles found for {resolved_run.symbol}/{resolved_run.timeframe}"
            )

        joined_df = _build_simulation_frame(predictions_df, market_df)
        if joined_df.empty:
            raise BacktestSimulatorError(
                f"No overlapping timestamps found for run_id={model_run_id} and market candles"
            )

        resolved_simulation_id = simulation_id or _default_simulation_id(simulation_type)
        active_strategy = strategy or Strategy(config=get_default_strategy())
        execution = execution_config or ExecutionConfig()
        return self.run_simulation_from_inputs(
            simulation_type=simulation_type,
            model_run_id=model_run_id,
            symbol=resolved_run.symbol,
            timeframe=resolved_run.timeframe,
            joined_df=joined_df,
            signals_df=generate_threshold_signals(joined_df, strategy_config=active_strategy.config),
            strategy=active_strategy,
            execution_config=execution,
            simulation_id=resolved_simulation_id,
            provenance={
                "asset": resolved_run.asset,
                "model_version": resolved_run.model_version,
                "dataset_version": resolved_run.metadata.get("dataset_version"),
                "feature_version": resolved_run.metadata.get("feature_version"),
                "target_column": resolved_run.metadata.get("target_column"),
                "prediction_rows": int(len(predictions_df)),
                "simulation_rows": int(len(joined_df)),
            },
        )

    def run_simulation_from_inputs(
        self,
        *,
        simulation_type: SimulationType,
        model_run_id: str,
        symbol: str,
        timeframe: str,
        joined_df: pd.DataFrame,
        signals_df: pd.DataFrame,
        strategy: Strategy,
        execution_config: ExecutionConfig,
        simulation_id: str | None = None,
        provenance: dict[str, object] | None = None,
    ) -> BacktestResult:
        """Run a simulation from a pre-built joined frame and signal stream."""
        config = BacktestConfig(strategy=strategy.config, execution=execution_config)
        frames = simulate_execution(
            joined_df,
            signals_df=signals_df,
            execution_config=execution_config,
        )
        metrics = compute_simulation_metrics(
            frames.equity_curve,
            frames.trades,
            starting_cash=execution_config.starting_cash,
        )

        result = BacktestResult(
            simulation_id=simulation_id or _default_simulation_id(simulation_type),
            simulation_type=simulation_type,
            model_run_id=model_run_id,
            symbol=symbol,
            timeframe=timeframe,
            start_timestamp=str(joined_df["timestamp"].min().isoformat()),
            end_timestamp=str(joined_df["timestamp"].max().isoformat()),
            strategy_version=strategy.config.strategy_version,
            config=config,
            metrics=metrics,
            provenance=dict(provenance or {}),
        )

        artifacts = self._persist_frames(result, frames)
        artifact_paths = {
            "summary_json_path": str(artifacts.summary_path),
            "equity_curve_path": str(artifacts.equity_curve_path),
            "trades_path": str(artifacts.trades_path),
            "signals_path": str(artifacts.signals_path),
        }
        if artifacts.report_path is not None:
            artifact_paths["report_markdown_path"] = str(artifacts.report_path)

        final_result = BacktestResult(
            simulation_id=result.simulation_id,
            simulation_type=result.simulation_type,
            model_run_id=result.model_run_id,
            symbol=result.symbol,
            timeframe=result.timeframe,
            start_timestamp=result.start_timestamp,
            end_timestamp=result.end_timestamp,
            strategy_version=result.strategy_version,
            config=result.config,
            metrics=result.metrics,
            artifact_paths=artifact_paths,
            provenance=result.provenance,
        )
        self._write_summary(final_result, store_type=result.simulation_type)
        return final_result

    def _persist_frames(
        self,
        result: BacktestResult,
        frames: SimulationFrames,
    ) -> SimulationArtifacts:
        """Write simulation frame artifacts to the appropriate local store."""
        store = self.backtest_store if result.simulation_type == "backtest" else self.paper_sim_store
        summary_path = store.simulation_dir(result.simulation_id) / "summary.json"
        equity_curve_path = store.write_equity_curve(result.simulation_id, frames.equity_curve)
        trades_path = store.write_trades(result.simulation_id, frames.trades)
        signals_path = store.write_signals(result.simulation_id, frames.signals)
        report_path: Path | None = None
        if self.write_markdown:
            report_path = store.write_report(
                result.simulation_id,
                build_simulation_markdown_report(result),
            )
        logger.info(
            "simulation_persist_complete",
            simulation_id=result.simulation_id,
            simulation_type=result.simulation_type,
            trade_rows=len(frames.trades),
            equity_rows=len(frames.equity_curve),
        )
        return SimulationArtifacts(
            summary_path=summary_path,
            equity_curve_path=equity_curve_path,
            trades_path=trades_path,
            signals_path=signals_path,
            report_path=report_path,
        )

    def _write_summary(self, result: BacktestResult, *, store_type: SimulationType) -> Path:
        """Persist the final summary payload after artifact paths are known."""
        store = self.backtest_store if store_type == "backtest" else self.paper_sim_store
        return store.write_summary(result.simulation_id, build_simulation_summary_payload(result))


def build_simulation_frame(predictions_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """Join predictions and market candles on timestamp with strict ordering."""
    predictions = predictions_df.copy()
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True)
    predictions = predictions.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

    market = market_df.copy()
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)
    market = market.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

    joined = predictions.merge(
        market[["timestamp", "open", "high", "low", "close", "volume"]],
        on="timestamp",
        how="inner",
    )
    return joined.sort_values("timestamp").reset_index(drop=True)


def _build_simulation_frame(predictions_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper for the public simulation-frame helper."""
    return build_simulation_frame(predictions_df, market_df)


def _default_simulation_id(simulation_type: SimulationType) -> str:
    """Return a deterministic timestamp-based simulation identifier."""
    prefix = "BT" if simulation_type == "backtest" else "PS"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}{timestamp}"
