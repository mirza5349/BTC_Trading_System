"""Scheduled offline paper trading using stored models and the local simulator."""

from __future__ import annotations

import os
import signal
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from trading_bot.backtest.execution import simulate_execution
from trading_bot.backtest.metrics import compute_simulation_metrics
from trading_bot.backtest.signals import generate_threshold_signals
from trading_bot.backtest.simulator import build_simulation_frame
from trading_bot.features.feature_pipeline import FeaturePipeline
from trading_bot.ingestion.market_data import MarketDataService
from trading_bot.logging_config import get_logger
from trading_bot.models.model_registry import ModelRegistry
from trading_bot.models.predict import predict_binary_classifier
from trading_bot.storage.optimization_store import OptimizationStore
from trading_bot.schemas.backtest import ExecutionConfig, StrategyConfig
from trading_bot.settings import PROJECT_ROOT, AppSettings, load_settings
from trading_bot.storage.model_store import ModelStore
from trading_bot.storage.paper_loop_store import PaperLoopStore
from trading_bot.storage.parquet_store import ParquetStore

logger = get_logger(__name__)


class ScheduledPaperTradingServiceError(Exception):
    """Raised when the scheduled paper loop cannot proceed safely."""


@dataclass(frozen=True)
class PaperLoopSelection:
    """Fixed selected model/target/strategy coordinates for the paper loop."""

    loop_id: str
    target_name: str
    target_key: str
    model_run_id: str
    optimization_id: str
    symbol: str
    timeframe: str
    feature_version: str
    provider: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return asdict(self)


@dataclass(frozen=True)
class PaperLoopRunResult:
    """Result of one one-shot paper-loop update."""

    loop_id: str
    latest_candle_timestamp: str | None
    prediction_timestamp: str | None
    y_proba: float | None
    y_pred: int | None
    signal_action: str
    executed_action: str
    current_equity: float
    in_position: bool
    trade_count: int
    no_new_candle: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return asdict(self)


class ScheduledPaperTradingService:
    """Run a selected paper-trading setup as repeated local one-shot cycles."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        model_registry: ModelRegistry,
        model_store: ModelStore,
        optimization_store: OptimizationStore,
        market_store: ParquetStore,
        market_data_service: MarketDataService,
        feature_pipeline: FeaturePipeline,
        paper_loop_store: PaperLoopStore,
        now_fn: Callable[[], datetime] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.settings = settings
        self.model_registry = model_registry
        self.model_store = model_store
        self.optimization_store = optimization_store
        self.market_store = market_store
        self.market_data_service = market_data_service
        self.feature_pipeline = feature_pipeline
        self.paper_loop_store = paper_loop_store
        self.now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self.sleep_fn = sleep_fn or time.sleep
        self._keep_running = True

    @classmethod
    def from_settings(
        cls,
        settings: AppSettings | None = None,
    ) -> ScheduledPaperTradingService:
        """Create the scheduled paper-loop service from app settings."""
        resolved_settings = settings or load_settings()
        market_store = ParquetStore(PROJECT_ROOT / resolved_settings.market_data.processed_data_dir)
        return cls(
            settings=resolved_settings,
            model_registry=ModelRegistry.from_settings(resolved_settings),
            model_store=ModelStore(PROJECT_ROOT / resolved_settings.artifacts.models_dir),
            optimization_store=OptimizationStore(
                PROJECT_ROOT / resolved_settings.optimization_storage.output_dir
            ),
            market_store=market_store,
            market_data_service=MarketDataService(store=market_store),
            feature_pipeline=FeaturePipeline.from_settings(resolved_settings),
            paper_loop_store=PaperLoopStore(PROJECT_ROOT / resolved_settings.paper_loop.output_dir),
        )

    def default_selection(self) -> PaperLoopSelection:
        """Return the configured default selected target/model/strategy."""
        configured = self.settings.paper_loop
        return PaperLoopSelection(
            loop_id=configured.loop_id,
            target_name=configured.target_name,
            target_key=configured.target_key,
            model_run_id=configured.model_run_id,
            optimization_id=configured.optimization_id,
            symbol=configured.symbol,
            timeframe=configured.timeframe,
            feature_version=configured.feature_version,
            provider=configured.provider,
        )

    def run_once(
        self,
        *,
        selection: PaperLoopSelection | None = None,
    ) -> PaperLoopRunResult:
        """Refresh the latest candle, score the latest closed bar, and update the paper account."""
        active_selection = selection or self.default_selection()
        self.paper_loop_store.write_config(active_selection.loop_id, active_selection.to_dict())

        resolved_run = self.model_registry.resolve_run(
            active_selection.model_run_id,
            symbol=active_selection.symbol,
            timeframe=active_selection.timeframe,
        )
        optimization_report = self.optimization_store.read_report_json(
            active_selection.optimization_id
        )
        if not optimization_report:
            raise ScheduledPaperTradingServiceError(
                f"No stored optimization report found for optimization_id={active_selection.optimization_id}"
            )

        best_candidate = dict(optimization_report.get("best_candidate") or {})
        if not best_candidate:
            raise ScheduledPaperTradingServiceError(
                f"No best candidate found for optimization_id={active_selection.optimization_id}"
            )

        if str(optimization_report.get("model_run_id", "")) != active_selection.model_run_id:
            raise ScheduledPaperTradingServiceError(
                "Optimization report model_run_id does not match the selected paper-loop model run."
            )

        self._refresh_latest_market_candle(active_selection.symbol, active_selection.timeframe)
        market_df = self.market_store.read(active_selection.symbol, active_selection.timeframe)
        canonical_candle_timestamp = _latest_closed_candle_timestamp(
            market_df,
            timeframe=active_selection.timeframe,
            now_utc=self.now_fn(),
            candle_close_safety_minutes=self.settings.paper_loop.candle_close_safety_minutes,
        )
        if canonical_candle_timestamp is None:
            raise ScheduledPaperTradingServiceError(
                f"No closed candles are available for {active_selection.symbol}/{active_selection.timeframe}"
            )

        existing_predictions = self.paper_loop_store.read_predictions(active_selection.loop_id)
        if not existing_predictions.empty:
            existing_predictions["timestamp"] = pd.to_datetime(
                existing_predictions["timestamp"],
                utc=True,
            )
            if existing_predictions["timestamp"].max() >= canonical_candle_timestamp:
                result = self._result_from_existing_state(
                    loop_id=active_selection.loop_id,
                    latest_closed_timestamp=canonical_candle_timestamp,
                    no_new_candle=True,
                )
                self._write_status(
                    active_selection,
                    result,
                    pid=os.getpid(),
                    running=False,
                    note="No new closed candle was available.",
                )
                return result

        latest_feature_row = self.feature_pipeline.build_latest_feature_row(
            symbol=active_selection.symbol,
            timeframe=active_selection.timeframe,
            asset=resolved_run.asset,
            provider=active_selection.provider,
            version=active_selection.feature_version,
            market_history_rows=self.settings.paper_loop.market_history_rows,
            target_timestamp=canonical_candle_timestamp,
        )
        raw_latest_closed_timestamp = canonical_candle_timestamp
        raw_latest_feature_timestamp = latest_feature_row.iloc[-1]["timestamp"]
        normalized_latest_closed_timestamp = _normalize_loop_timestamp(
            raw_latest_closed_timestamp,
            timeframe=active_selection.timeframe,
        )
        normalized_latest_feature_timestamp = _normalize_loop_timestamp(
            raw_latest_feature_timestamp,
            timeframe=active_selection.timeframe,
        )
        equality_result = normalized_latest_feature_timestamp == normalized_latest_closed_timestamp
        if not equality_result:
            alignment_details = {
                "raw_latest_closed_candle_timestamp": str(raw_latest_closed_timestamp),
                "raw_latest_closed_candle_timestamp_type": type(raw_latest_closed_timestamp).__name__,
                "raw_feature_timestamp": str(raw_latest_feature_timestamp),
                "raw_feature_timestamp_type": type(raw_latest_feature_timestamp).__name__,
                "normalized_latest_closed_candle_timestamp": (
                    normalized_latest_closed_timestamp.isoformat()
                    if normalized_latest_closed_timestamp is not None
                    else None
                ),
                "normalized_feature_timestamp": (
                    normalized_latest_feature_timestamp.isoformat()
                    if normalized_latest_feature_timestamp is not None
                    else None
                ),
                "normalized_latest_closed_candle_timestamp_type": type(
                    normalized_latest_closed_timestamp
                ).__name__,
                "normalized_feature_timestamp_type": type(
                    normalized_latest_feature_timestamp
                ).__name__,
                "timeframe": active_selection.timeframe,
                "equality_result": equality_result,
            }
            logger.info(
                "paper_loop_timestamp_alignment_mismatch",
                **alignment_details,
            )
            raise ScheduledPaperTradingServiceError(
                "Latest feature row timestamp does not match the latest closed candle timestamp. "
                f"raw_latest_closed_candle_timestamp={alignment_details['raw_latest_closed_candle_timestamp']}; "
                f"raw_latest_closed_candle_timestamp_type={alignment_details['raw_latest_closed_candle_timestamp_type']}; "
                f"raw_feature_timestamp={alignment_details['raw_feature_timestamp']}; "
                f"raw_feature_timestamp_type={alignment_details['raw_feature_timestamp_type']}; "
                "normalized_latest_closed_candle_timestamp="
                f"{alignment_details['normalized_latest_closed_candle_timestamp']}; "
                "normalized_feature_timestamp="
                f"{alignment_details['normalized_feature_timestamp']}; "
                "normalized_latest_closed_candle_timestamp_type="
                f"{alignment_details['normalized_latest_closed_candle_timestamp_type']}; "
                "normalized_feature_timestamp_type="
                f"{alignment_details['normalized_feature_timestamp_type']}; "
                f"timeframe={alignment_details['timeframe']}; "
                f"equality_result={alignment_details['equality_result']}"
            )
        latest_closed_timestamp = normalized_latest_closed_timestamp

        feature_columns = list(resolved_run.metadata.get("feature_columns") or [])
        if not feature_columns:
            raise ScheduledPaperTradingServiceError(
                f"No feature_columns metadata found for run_id={active_selection.model_run_id}"
            )
        missing_feature_columns = [
            column for column in feature_columns if column not in latest_feature_row.columns
        ]
        if missing_feature_columns:
            raise ScheduledPaperTradingServiceError(
                "Latest feature row is missing required model feature columns: "
                f"{missing_feature_columns}"
            )

        fold_id = self._select_inference_fold_id(active_selection.model_run_id, resolved_run)
        model = self.model_store.read_fold_model(
            asset=resolved_run.asset,
            symbol=resolved_run.symbol,
            timeframe=resolved_run.timeframe,
            model_version=resolved_run.model_version,
            run_id=resolved_run.run_id,
            fold_id=fold_id,
        )
        feature_matrix = latest_feature_row.loc[:, feature_columns].copy()
        y_proba_array, y_pred_array = predict_binary_classifier(
            model,
            feature_matrix,
            threshold=self.settings.paper_loop.prediction_threshold,
        )
        y_proba = float(y_proba_array[-1])
        y_pred = int(y_pred_array[-1])

        new_prediction_row = pd.DataFrame(
            [
                {
                    "symbol": resolved_run.symbol,
                    "timeframe": resolved_run.timeframe,
                    "timestamp": latest_closed_timestamp,
                    "y_true": None,
                    "y_pred": y_pred,
                    "y_proba": y_proba,
                    "target_name": active_selection.target_name,
                    "target_key": active_selection.target_key,
                    "feature_version": active_selection.feature_version,
                    "model_run_id": active_selection.model_run_id,
                    "optimization_id": active_selection.optimization_id,
                    "inference_fold_id": fold_id,
                }
            ]
        )
        prediction_log = pd.concat([existing_predictions, new_prediction_row], ignore_index=True)
        prediction_log["timestamp"] = pd.to_datetime(prediction_log["timestamp"], utc=True)
        prediction_log = (
            prediction_log.sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
            .reset_index(drop=True)
        )
        self.paper_loop_store.write_predictions(active_selection.loop_id, prediction_log)

        joined_df = build_simulation_frame(
            prediction_log.loc[:, ["symbol", "timeframe", "timestamp", "y_true", "y_pred", "y_proba"]],
            market_df,
        )
        strategy_config = StrategyConfig(
            strategy_version=str(best_candidate.get("candidate_id", "optimized_paper_loop")),
            type="long_only_threshold",
            entry_threshold=float(best_candidate["entry_threshold"]),
            exit_threshold=float(best_candidate["exit_threshold"]),
            minimum_holding_bars=int(best_candidate["minimum_holding_bars"]),
            cooldown_bars=int(best_candidate["cooldown_bars"]),
        )
        execution_config = ExecutionConfig(
            fill_model=self.settings.backtest_execution.fill_model,
            starting_cash=self.settings.backtest_execution.starting_cash,
            fee_rate=self.settings.backtest_execution.fee_rate,
            slippage_rate=self.settings.backtest_execution.slippage_rate,
            max_position_fraction=float(best_candidate["max_position_fraction"]),
            allow_fractional_position=self.settings.backtest_execution.allow_fractional_position,
        )

        signals_df = generate_threshold_signals(joined_df, strategy_config=strategy_config)
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

        orders_df = _build_order_log(frames.equity_curve)
        positions_df = _build_positions_log(frames.equity_curve)
        self.paper_loop_store.write_signals(active_selection.loop_id, frames.signals)
        self.paper_loop_store.write_trades(active_selection.loop_id, frames.trades)
        self.paper_loop_store.write_orders(active_selection.loop_id, orders_df)
        self.paper_loop_store.write_positions(active_selection.loop_id, positions_df)
        self.paper_loop_store.write_equity_curve(active_selection.loop_id, frames.equity_curve)

        artifact_dir = self.paper_loop_store.loop_dir(active_selection.loop_id)
        latest_signal_action = "flat"
        latest_executed_action = "none"
        current_equity = execution_config.starting_cash
        in_position = False
        if not frames.equity_curve.empty:
            latest_equity_row = frames.equity_curve.iloc[-1]
            latest_signal_action = str(latest_equity_row.get("signal_action", "flat"))
            latest_executed_action = str(latest_equity_row.get("executed_action", "none"))
            current_equity = float(latest_equity_row.get("equity", execution_config.starting_cash))
            in_position = bool(latest_equity_row.get("in_position", False))

        summary = {
            "loop_id": active_selection.loop_id,
            "target_name": active_selection.target_name,
            "target_key": active_selection.target_key,
            "model_run_id": active_selection.model_run_id,
            "optimization_id": active_selection.optimization_id,
            "symbol": active_selection.symbol,
            "timeframe": active_selection.timeframe,
            "feature_version": active_selection.feature_version,
            "run_feature_version": resolved_run.metadata.get("feature_version"),
            "latest_candle_timestamp": latest_closed_timestamp.isoformat(),
            "latest_prediction": {
                "timestamp": latest_closed_timestamp.isoformat(),
                "y_proba": y_proba,
                "y_pred": y_pred,
                "inference_fold_id": fold_id,
            },
            "strategy": strategy_config.to_dict(),
            "execution": execution_config.to_dict(),
            "metrics": metrics.to_dict(),
            "current_state": {
                "equity": current_equity,
                "in_position": in_position,
                "trade_count": int(metrics.number_of_trades),
                "signal_action": latest_signal_action,
                "executed_action": latest_executed_action,
            },
            "artifact_paths": {
                "predictions_path": str(artifact_dir / "predictions.parquet"),
                "signals_path": str(artifact_dir / "signals.parquet"),
                "orders_path": str(artifact_dir / "orders.parquet"),
                "trades_path": str(artifact_dir / "trades.parquet"),
                "positions_path": str(artifact_dir / "positions.parquet"),
                "equity_curve_path": str(artifact_dir / "equity_curve.parquet"),
                "summary_path": str(artifact_dir / "summary.json"),
                "status_path": str(artifact_dir / "status.json"),
            },
        }
        self.paper_loop_store.write_summary(active_selection.loop_id, summary)

        result = PaperLoopRunResult(
            loop_id=active_selection.loop_id,
            latest_candle_timestamp=latest_closed_timestamp.isoformat(),
            prediction_timestamp=latest_closed_timestamp.isoformat(),
            y_proba=y_proba,
            y_pred=y_pred,
            signal_action=latest_signal_action,
            executed_action=latest_executed_action,
            current_equity=current_equity,
            in_position=in_position,
            trade_count=int(metrics.number_of_trades),
            no_new_candle=False,
        )
        self._write_status(
            active_selection,
            result,
            pid=os.getpid(),
            running=False,
            note="Paper loop updated successfully.",
        )
        return result

    def run_forever(
        self,
        *,
        selection: PaperLoopSelection | None = None,
    ) -> None:
        """Run the scheduled loop until interrupted or terminated."""
        active_selection = selection or self.default_selection()
        self._install_signal_handlers(active_selection)
        while self._keep_running:
            try:
                result = self.run_once(selection=active_selection)
                self._write_status(
                    active_selection,
                    result,
                    pid=os.getpid(),
                    running=True,
                    note="Loop sleeping until the next scheduled interval.",
                )
            except Exception as exc:  # pragma: no cover - exercised indirectly in daemon mode
                logger.exception(
                    "paper_loop_cycle_failed",
                    loop_id=active_selection.loop_id,
                    error=str(exc),
                )
                failure_result = PaperLoopRunResult(
                    loop_id=active_selection.loop_id,
                    latest_candle_timestamp=None,
                    prediction_timestamp=None,
                    y_proba=None,
                    y_pred=None,
                    signal_action="flat",
                    executed_action="none",
                    current_equity=self.settings.backtest_execution.starting_cash,
                    in_position=False,
                    trade_count=0,
                )
                self._write_status(
                    active_selection,
                    failure_result,
                    pid=os.getpid(),
                    running=True,
                    note=f"Last cycle failed: {exc}",
                )

            sleep_seconds = _seconds_until_next_interval(
                now_utc=self.now_fn(),
                interval_minutes=self.settings.paper_loop.interval_minutes,
                buffer_seconds=self.settings.paper_loop.poll_buffer_seconds,
            )
            while self._keep_running and sleep_seconds > 0:
                chunk = min(5.0, sleep_seconds)
                self.sleep_fn(chunk)
                sleep_seconds -= chunk

        final_status = self.paper_loop_store.read_status(active_selection.loop_id)
        final_status["running"] = False
        final_status["stopped_at"] = self.now_fn().isoformat()
        final_status["pid"] = None
        self.paper_loop_store.write_status(active_selection.loop_id, final_status)

    def _refresh_latest_market_candle(self, symbol: str, timeframe: str) -> None:
        """Update local stored candles from the latest available timestamp forward."""
        latest_timestamp = self.market_store.get_latest_timestamp(symbol, timeframe)
        start = latest_timestamp - timedelta(minutes=15) if latest_timestamp is not None else None
        self.market_data_service.ingest_sync(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=self.now_fn(),
        )

    def _select_inference_fold_id(self, model_run_id: str, resolved_run) -> int:
        """Use the most recent stored fold model for inference."""
        fold_metrics_df = self.model_registry.read_fold_metrics(
            model_run_id,
            asset=resolved_run.asset,
            symbol=resolved_run.symbol,
            timeframe=resolved_run.timeframe,
            model_version=resolved_run.model_version,
        )
        if fold_metrics_df.empty or "fold_id" not in fold_metrics_df.columns:
            raise ScheduledPaperTradingServiceError(
                f"No stored fold metrics found for run_id={model_run_id}"
            )
        return int(pd.to_numeric(fold_metrics_df["fold_id"], errors="coerce").max())

    def _result_from_existing_state(
        self,
        *,
        loop_id: str,
        latest_closed_timestamp: pd.Timestamp,
        no_new_candle: bool,
    ) -> PaperLoopRunResult:
        """Build a result object from the already-persisted paper account state."""
        summary = self.paper_loop_store.read_summary(loop_id)
        current_state = dict(summary.get("current_state") or {})
        latest_prediction = dict(summary.get("latest_prediction") or {})
        return PaperLoopRunResult(
            loop_id=loop_id,
            latest_candle_timestamp=latest_closed_timestamp.isoformat(),
            prediction_timestamp=latest_prediction.get("timestamp"),
            y_proba=latest_prediction.get("y_proba"),
            y_pred=latest_prediction.get("y_pred"),
            signal_action=str(current_state.get("signal_action", "flat")),
            executed_action=str(current_state.get("executed_action", "none")),
            current_equity=float(
                current_state.get("equity", self.settings.backtest_execution.starting_cash)
            ),
            in_position=bool(current_state.get("in_position", False)),
            trade_count=int(current_state.get("trade_count", 0) or 0),
            no_new_candle=no_new_candle,
        )

    def _write_status(
        self,
        selection: PaperLoopSelection,
        result: PaperLoopRunResult,
        *,
        pid: int | None,
        running: bool,
        note: str,
    ) -> None:
        """Persist a compact runtime status payload."""
        summary = self.paper_loop_store.read_summary(selection.loop_id)
        self.paper_loop_store.write_status(
            selection.loop_id,
            {
                "loop_id": selection.loop_id,
                "running": running,
                "pid": pid,
                "target_name": selection.target_name,
                "target_key": selection.target_key,
                "model_run_id": selection.model_run_id,
                "optimization_id": selection.optimization_id,
                "symbol": selection.symbol,
                "timeframe": selection.timeframe,
                "feature_version": selection.feature_version,
                "last_updated_at": self.now_fn().isoformat(),
                "latest_candle_timestamp": result.latest_candle_timestamp,
                "latest_prediction_timestamp": result.prediction_timestamp,
                "latest_probability": result.y_proba,
                "latest_predicted_class": result.y_pred,
                "signal_action": result.signal_action,
                "executed_action": result.executed_action,
                "current_equity": result.current_equity,
                "in_position": result.in_position,
                "trade_count": result.trade_count,
                "no_new_candle": result.no_new_candle,
                "note": note,
                "artifact_paths": summary.get("artifact_paths", {}),
            },
        )

    def _install_signal_handlers(self, selection: PaperLoopSelection) -> None:
        """Stop the loop cleanly on SIGTERM/SIGINT."""

        def _handle_stop(signum, _frame) -> None:  # pragma: no cover - signal path
            logger.info("paper_loop_stop_requested", loop_id=selection.loop_id, signal=signum)
            self._keep_running = False

        signal.signal(signal.SIGTERM, _handle_stop)
        signal.signal(signal.SIGINT, _handle_stop)


def _latest_closed_candle_timestamp(
    market_df: pd.DataFrame,
    *,
    timeframe: str,
    now_utc: datetime,
    candle_close_safety_minutes: int = 0,
) -> pd.Timestamp | None:
    """Return the canonical candle timestamp allowed for paper-loop inference."""
    if market_df.empty or "timestamp" not in market_df.columns:
        return None

    working_df = market_df.copy()
    working_df["timestamp"] = pd.to_datetime(working_df["timestamp"], utc=True)
    safety_delta = pd.Timedelta(minutes=max(int(candle_close_safety_minutes), 0))
    effective_now = pd.Timestamp(now_utc).tz_convert("UTC") - safety_delta
    latest_allowed_open = effective_now.floor(_timeframe_to_timedelta(timeframe))
    closed_df = working_df[working_df["timestamp"] <= latest_allowed_open]
    if closed_df.empty:
        return None
    return pd.Timestamp(closed_df["timestamp"].max())


def _normalize_loop_timestamp(
    value: object,
    *,
    timeframe: str,
) -> pd.Timestamp | None:
    """Normalize a loop timestamp to a UTC candle-boundary timestamp."""
    if value is None:
        return None

    normalized = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(normalized):
        return None

    timestamp = pd.Timestamp(normalized)
    return timestamp.floor(_timeframe_to_timedelta(timeframe))


def _timeframe_to_timedelta(timeframe: str) -> str:
    """Return a pandas-compatible frequency string for the timeframe."""
    mapping = {
        "1m": "1min",
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
    }
    if timeframe not in mapping:
        raise ScheduledPaperTradingServiceError(f"Unsupported timeframe for paper loop: {timeframe}")
    return mapping[timeframe]


def _seconds_until_next_interval(
    *,
    now_utc: datetime,
    interval_minutes: int,
    buffer_seconds: int,
) -> float:
    """Return sleep seconds until the next scheduled closed-candle boundary."""
    now_ts = pd.Timestamp(now_utc).tz_convert("UTC")
    next_boundary = now_ts.ceil(f"{interval_minutes}min")
    if next_boundary <= now_ts:
        next_boundary += pd.Timedelta(minutes=interval_minutes)
    target = next_boundary + pd.Timedelta(seconds=buffer_seconds)
    return max((target - now_ts).total_seconds(), 1.0)


def _build_order_log(equity_curve_df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact simulated order log from executed account state changes."""
    columns = [
        "timestamp",
        "order_action",
        "price_reference",
        "quantity_delta",
        "cash",
        "btc_quantity",
        "equity",
    ]
    if equity_curve_df.empty:
        return pd.DataFrame(columns=columns)

    working_df = equity_curve_df.copy().reset_index(drop=True)
    working_df["timestamp"] = pd.to_datetime(working_df["timestamp"], utc=True)
    previous_quantity = working_df["btc_quantity"].shift(1).fillna(0.0)
    quantity_delta = (working_df["btc_quantity"] - previous_quantity).abs()
    orders_df = working_df[working_df["executed_action"].isin(["enter_long", "exit_long"])].copy()
    if orders_df.empty:
        return pd.DataFrame(columns=columns)
    orders_df["quantity_delta"] = quantity_delta.loc[orders_df.index].astype(float)
    orders_df = orders_df.rename(
        columns={
            "executed_action": "order_action",
            "open": "price_reference",
        }
    )
    return orders_df.loc[:, columns].reset_index(drop=True)


def _build_positions_log(equity_curve_df: pd.DataFrame) -> pd.DataFrame:
    """Return a lighter-weight position snapshot log from the equity curve."""
    columns = [
        "timestamp",
        "in_position",
        "btc_quantity",
        "cash",
        "market_value",
        "equity",
        "drawdown",
        "bars_in_position",
    ]
    if equity_curve_df.empty:
        return pd.DataFrame(columns=columns)
    return equity_curve_df.loc[:, columns].copy().reset_index(drop=True)
