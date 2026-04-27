"""Tests that verify all package modules import cleanly."""

from __future__ import annotations

import importlib

import pytest

# Every module in the project that must be importable
ALL_MODULES = [
    "trading_bot",
    "trading_bot.settings",
    "trading_bot.logging_config",
    "trading_bot.cli",
    "trading_bot.main",
    "trading_bot.schemas",
    "trading_bot.schemas.market_data",
    "trading_bot.schemas.features",
    "trading_bot.schemas.datasets",
    "trading_bot.schemas.modeling",
    "trading_bot.schemas.experiments",
    "trading_bot.schemas.nlp",
    "trading_bot.schemas.validation",
    "trading_bot.schemas.optimization",
    "trading_bot.schemas.horizons",
    "trading_bot.schemas.targets",
    "trading_bot.ingestion",
    "trading_bot.ingestion.binance_client",
    "trading_bot.ingestion.market_data",
    "trading_bot.ingestion.news_client",
    "trading_bot.ingestion.news_data",
    "trading_bot.storage",
    "trading_bot.storage.parquet_store",
    "trading_bot.storage.news_store",
    "trading_bot.storage.dataset_store",
    "trading_bot.storage.model_store",
    "trading_bot.storage.evaluation_store",
    "trading_bot.storage.backtest_store",
    "trading_bot.storage.paper_sim_store",
    "trading_bot.storage.experiment_store",
    "trading_bot.storage.report_store",
    "trading_bot.storage.validation_store",
    "trading_bot.storage.optimization_store",
    "trading_bot.storage.horizon_store",
    "trading_bot.storage.target_store",
    "trading_bot.storage.enriched_news_store",
    "trading_bot.storage.paper_loop_store",
    "trading_bot.features",
    "trading_bot.features.market_features",
    "trading_bot.features.news_features",
    "trading_bot.features.feature_store",
    "trading_bot.features.feature_pipeline",
    "trading_bot.labeling",
    "trading_bot.labeling.labels",
    "trading_bot.labeling.dataset_builder",
    "trading_bot.models",
    "trading_bot.models.train",
    "trading_bot.models.predict",
    "trading_bot.models.model_registry",
    "trading_bot.models.tuning",
    "trading_bot.nlp",
    "trading_bot.nlp.finbert_service",
    "trading_bot.nlp.text_preprocessing",
    "trading_bot.nlp.news_enrichment",
    "trading_bot.experiments",
    "trading_bot.experiments.tracker",
    "trading_bot.experiments.registry",
    "trading_bot.experiments.comparison",
    "trading_bot.experiments.promotion",
    "trading_bot.optimization",
    "trading_bot.optimization.search_space",
    "trading_bot.optimization.selection",
    "trading_bot.optimization.reporting",
    "trading_bot.optimization.strategy_search",
    "trading_bot.research",
    "trading_bot.research.fee_aware_targets",
    "trading_bot.research.multi_horizon",
    "trading_bot.paper",
    "trading_bot.paper.loop",
    "trading_bot.paper.daemon",
    "trading_bot.validation",
    "trading_bot.validation.walk_forward",
    "trading_bot.validation.metrics",
    "trading_bot.validation.evaluation_report",
    "trading_bot.validation.baselines",
    "trading_bot.validation.suite",
    "trading_bot.backtest",
    "trading_bot.backtest.signals",
    "trading_bot.backtest.execution",
    "trading_bot.backtest.metrics",
    "trading_bot.backtest.reporting",
    "trading_bot.backtest.strategy",
    "trading_bot.backtest.simulator",
    "trading_bot.execution",
    "trading_bot.execution.signal_engine",
    "trading_bot.execution.order_manager",
    "trading_bot.execution.broker_adapter",
    "trading_bot.monitoring",
    "trading_bot.monitoring.logger",
    "trading_bot.monitoring.alerts",
    "trading_bot.schemas.news_data",
    "trading_bot.schemas.backtest",
]


@pytest.mark.parametrize("module_name", ALL_MODULES)
def test_module_imports(module_name: str) -> None:
    """Each module should import without errors."""
    mod = importlib.import_module(module_name)
    assert mod is not None


class TestPackageAttributes:
    """Test that the main package exposes expected attributes."""

    def test_version(self) -> None:
        """trading_bot.__version__ should be defined."""
        import trading_bot

        assert hasattr(trading_bot, "__version__")
        assert trading_bot.__version__ == "0.1.0"

    def test_app_name(self) -> None:
        """trading_bot.__app_name__ should be defined."""
        import trading_bot

        assert hasattr(trading_bot, "__app_name__")
        assert trading_bot.__app_name__ == "btc_ml_trading_bot"
