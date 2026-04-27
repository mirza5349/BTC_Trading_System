"""Storage package — local data persistence backends."""

from trading_bot.storage.backtest_store import BacktestStore, SimulationStoreError
from trading_bot.storage.dataset_store import DatasetStore, DatasetStoreError, StoredDatasetArtifact
from trading_bot.storage.evaluation_store import EvaluationStore, EvaluationStoreError
from trading_bot.storage.enriched_news_store import EnrichedNewsStore, EnrichedNewsStoreError
from trading_bot.storage.experiment_store import ExperimentStore, ExperimentStoreError
from trading_bot.storage.horizon_store import HorizonStore
from trading_bot.storage.model_store import ModelStore, ModelStoreError
from trading_bot.storage.news_store import NewsStore, NewsStoreError
from trading_bot.storage.optimization_store import OptimizationStore
from trading_bot.storage.parquet_store import ParquetStore, ParquetStoreError
from trading_bot.storage.paper_sim_store import PaperSimulationStore
from trading_bot.storage.paper_loop_store import PaperLoopStore
from trading_bot.storage.report_store import ReportStore, ReportStoreError
from trading_bot.storage.target_store import TargetStore
from trading_bot.storage.validation_store import ValidationStore, ValidationStoreError

__all__ = [
    "BacktestStore",
    "SimulationStoreError",
    "DatasetStore",
    "DatasetStoreError",
    "StoredDatasetArtifact",
    "EvaluationStore",
    "EvaluationStoreError",
    "EnrichedNewsStore",
    "EnrichedNewsStoreError",
    "ExperimentStore",
    "ExperimentStoreError",
    "HorizonStore",
    "ModelStore",
    "ModelStoreError",
    "NewsStore",
    "NewsStoreError",
    "OptimizationStore",
    "ParquetStore",
    "ParquetStoreError",
    "PaperSimulationStore",
    "PaperLoopStore",
    "ReportStore",
    "ReportStoreError",
    "TargetStore",
    "ValidationStore",
    "ValidationStoreError",
]
