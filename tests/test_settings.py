"""Tests for settings loading and configuration."""

from __future__ import annotations

from trading_bot.settings import (
    AppSettings,
    ArtifactsConfig,
    AssetConfig,
    DatasetsConfig,
    DataSourcesConfig,
    Environment,
    ExperimentsConfig,
    FeaturesConfig,
    FeeAwareTargetsConfig,
    LabelsConfig,
    LightGBMConfig,
    LogLevel,
    MarketDataConfig,
    ModelConfig,
    MultiHorizonConfig,
    NLPConfig,
    NewsDataConfig,
    ScheduledPaperLoopConfig,
    ReportsConfig,
    RiskConfig,
    SentimentFeaturesConfig,
    StoragePathsConfig,
    TrainingConfig,
    ValidationConfig,
    WalkForwardConfig,
    load_settings,
)


class TestAppSettings:
    """Test suite for AppSettings and config loading."""

    def test_load_settings_returns_app_settings(self) -> None:
        """load_settings() should return a valid AppSettings instance."""
        settings = load_settings()
        assert isinstance(settings, AppSettings)

    def test_default_app_name(self) -> None:
        """Default app name should be 'btc_ml_trading_bot'."""
        settings = load_settings()
        assert settings.app_name == "btc_ml_trading_bot"

    def test_default_environment(self) -> None:
        """Default environment should be development."""
        settings = load_settings()
        assert settings.app_env == Environment.DEVELOPMENT

    def test_default_log_level(self) -> None:
        """Default log level should be INFO."""
        settings = load_settings()
        assert settings.log_level == LogLevel.INFO

    def test_asset_config_loaded(self) -> None:
        """Asset config should be present with BTC defaults."""
        settings = load_settings()
        assert isinstance(settings.asset, AssetConfig)
        assert settings.asset.symbol == "BTCUSDT"
        assert settings.asset.base_asset == "BTC"
        assert settings.asset.quote_asset == "USDT"

    def test_model_config_loaded(self) -> None:
        """Model config should have LightGBM defaults."""
        settings = load_settings()
        assert isinstance(settings.model, ModelConfig)
        assert settings.model.algorithm == "lightgbm"
        assert settings.model.prediction_horizon_minutes == 30

    def test_risk_config_loaded(self) -> None:
        """Risk config should have paper-trading-only enabled."""
        settings = load_settings()
        assert isinstance(settings.risk, RiskConfig)
        assert settings.risk.paper_trading_only is True
        assert settings.risk.allow_live_trading is False

    def test_data_sources_config(self) -> None:
        """Data sources should default to Binance + CoinGecko."""
        settings = load_settings()
        assert isinstance(settings.data_sources, DataSourcesConfig)
        assert settings.data_sources.market == "binance"
        assert settings.data_sources.news == "coingecko"

    def test_validation_config(self) -> None:
        """Validation config should have walk-forward defaults."""
        settings = load_settings()
        assert isinstance(settings.validation, ValidationConfig)
        assert settings.validation.method == "walk_forward"
        assert settings.validation.n_splits == 5

    def test_secrets_default_empty(self) -> None:
        """API keys should default to empty strings."""
        settings = load_settings()
        assert settings.binance_api_key == ""
        assert settings.binance_api_secret == ""
        assert settings.coingecko_api_key == ""

    def test_settings_model_dump(self) -> None:
        """Settings should be serializable via model_dump."""
        settings = load_settings()
        data = settings.model_dump()
        assert isinstance(data, dict)
        assert "asset" in data
        assert "model" in data
        assert "risk" in data
        assert "market_data" in data
        assert "news_data" in data
        assert "features" in data
        assert "labels" in data
        assert "datasets" in data
        assert "training" in data
        assert "walk_forward" in data
        assert "lightgbm" in data
        assert "artifacts" in data
        assert "experiments" in data
        assert "reports" in data
        assert "multi_horizon" in data
        assert "fee_aware_targets" in data
        assert "paper_loop" in data
        assert "nlp" in data
        assert "sentiment_features" in data
        assert "storage" in data

    def test_market_data_config(self) -> None:
        """Market data config should have sensible defaults."""
        settings = load_settings()
        assert isinstance(settings.market_data, MarketDataConfig)
        assert settings.market_data.symbol == "BTCUSDT"
        assert settings.market_data.default_timeframe == "15m"
        assert settings.market_data.storage_format == "parquet"
        assert settings.market_data.request_limit == 1000

    def test_news_data_config(self) -> None:
        """News data config should have sensible defaults."""
        settings = load_settings()
        assert isinstance(settings.news_data, NewsDataConfig)
        assert settings.news_data.asset == "BTC"
        assert settings.news_data.provider == "cryptocompare"
        assert settings.news_data.default_start_date == "2024-01-01"
        assert settings.news_data.processed_data_dir == "data/processed/news"
        assert isinstance(settings.news_data.btc_keywords, list)
        assert len(settings.news_data.btc_keywords) > 0
        assert "bitcoin" in settings.news_data.btc_keywords

    def test_features_config(self) -> None:
        """Feature config should have deterministic BTC defaults."""
        settings = load_settings()
        assert isinstance(settings.features, FeaturesConfig)
        assert settings.features.feature_version == "v1"
        assert settings.features.symbol == "BTCUSDT"
        assert settings.features.asset == "BTC"
        assert settings.features.timeframe == "15m"
        assert settings.features.market_windows == [1, 2, 4, 8]
        assert settings.features.news_lookback_windows == ["1h", "4h"]
        assert settings.features.output_dir == "data/features"
        assert settings.features.fill_missing_news_with_zero is True

    def test_labels_config(self) -> None:
        """Label config should have deterministic BTC label defaults."""
        settings = load_settings()
        assert isinstance(settings.labels, LabelsConfig)
        assert settings.labels.label_version == "v1"
        assert settings.labels.primary_target == "target_up_2bars"
        assert settings.labels.horizon_bars == 2
        assert settings.labels.binary_threshold == 0.0
        assert settings.labels.include_regression_target is True
        assert settings.labels.include_optional_targets is True
        assert settings.labels.optional_horizon_bars == [4]

    def test_datasets_config(self) -> None:
        """Dataset config should point to BTC supervised dataset defaults."""
        settings = load_settings()
        assert isinstance(settings.datasets, DatasetsConfig)
        assert settings.datasets.dataset_version == "v1"
        assert settings.datasets.symbol == "BTCUSDT"
        assert settings.datasets.asset == "BTC"
        assert settings.datasets.timeframe == "15m"
        assert settings.datasets.output_dir == "data/datasets"

    def test_training_config(self) -> None:
        """Training config should expose baseline walk-forward defaults."""
        settings = load_settings()
        assert isinstance(settings.training, TrainingConfig)
        assert settings.training.model_version == "v1"
        assert settings.training.primary_target == "target_up_2bars"
        assert settings.training.random_seed == 42
        assert settings.training.device == "cpu"
        assert settings.training.allow_cuda_fallback_to_cpu is True

    def test_walk_forward_config(self) -> None:
        """Walk-forward config should be suitable for BTC 15m validation."""
        settings = load_settings()
        assert isinstance(settings.walk_forward, WalkForwardConfig)
        assert settings.walk_forward.mode == "expanding_window"
        assert settings.walk_forward.min_train_rows == 1000
        assert settings.walk_forward.validation_rows == 250
        assert settings.walk_forward.step_rows == 250

    def test_lightgbm_config(self) -> None:
        """LightGBM config should expose deterministic baseline parameters."""
        settings = load_settings()
        assert isinstance(settings.lightgbm, LightGBMConfig)
        assert settings.lightgbm.objective == "binary"
        assert settings.lightgbm.metric == "binary_logloss"
        assert settings.lightgbm.learning_rate == 0.05
        assert settings.lightgbm.n_estimators == 200
        assert settings.lightgbm.device_type == "cpu"

    def test_artifacts_config(self) -> None:
        """Artifact config should point to local model and evaluation directories."""
        settings = load_settings()
        assert isinstance(settings.artifacts, ArtifactsConfig)
        assert settings.artifacts.models_dir == "data/models"
        assert settings.artifacts.evaluation_dir == "data/evaluation"

    def test_experiments_config(self) -> None:
        """Experiment config should point to local registry and summary outputs."""
        settings = load_settings()
        assert isinstance(settings.experiments, ExperimentsConfig)
        assert settings.experiments.registry_path == "data/experiments/run_registry.parquet"
        assert settings.experiments.summary_dir == "data/experiments"
        assert settings.experiments.default_sort_metric == "roc_auc_mean"
        assert settings.experiments.default_status_for_new_runs == "candidate"

    def test_reports_config(self) -> None:
        """Report config should point to local JSON/Markdown report outputs."""
        settings = load_settings()
        assert isinstance(settings.reports, ReportsConfig)
        assert settings.reports.output_dir == "data/reports"
        assert settings.reports.write_markdown is True
        assert settings.reports.top_k_feature_importance == 20

    def test_nlp_config(self) -> None:
        """NLP config should default to local FinBERT CPU scoring."""
        settings = load_settings()
        assert isinstance(settings.nlp, NLPConfig)
        assert settings.nlp.model_name == "ProsusAI/finbert"
        assert settings.nlp.enrichment_version == "v1"
        assert settings.nlp.text_mode == "title_plus_summary"
        assert settings.nlp.batch_size == 16
        assert settings.nlp.device == "cpu"

    def test_sentiment_features_config(self) -> None:
        """Sentiment feature config should expose the versioned enriched feature path."""
        settings = load_settings()
        assert isinstance(settings.sentiment_features, SentimentFeaturesConfig)
        assert settings.sentiment_features.enabled is True
        assert settings.sentiment_features.feature_version == "v2_sentiment"
        assert settings.sentiment_features.lookback_windows == ["1h", "4h"]
        assert settings.sentiment_features.use_enriched_news_store is True

    def test_storage_paths_config(self) -> None:
        """Storage path config should expose the enriched-news directory."""
        settings = load_settings()
        assert isinstance(settings.storage, StoragePathsConfig)
        assert settings.storage.enriched_news_dir == "data/enriched_news"

    def test_multi_horizon_config(self) -> None:
        """Multi-horizon config should expose the longer-horizon research defaults."""
        settings = load_settings()
        assert isinstance(settings.multi_horizon, MultiHorizonConfig)
        assert settings.multi_horizon.horizon_bars == [4, 8, 16]
        assert settings.multi_horizon.default_feature_version == "v2_sentiment"
        assert settings.multi_horizon.output_dir == "data/reports/horizon_comparison"

    def test_fee_aware_targets_config(self) -> None:
        """Fee-aware target config should expose the target-research defaults."""
        settings = load_settings()
        assert isinstance(settings.fee_aware_targets, FeeAwareTargetsConfig)
        assert settings.fee_aware_targets.target_keys == [
            "fee_4",
            "fee_8",
            "fee_16",
            "ret025_8",
            "ret050_16",
            "ret075_16",
        ]
        assert settings.fee_aware_targets.default_round_trip_cost == 0.003
        assert settings.fee_aware_targets.output_dir == "data/reports/target_comparison"

    def test_paper_loop_config(self) -> None:
        """Scheduled paper-loop config should expose the selected target/model defaults."""
        settings = load_settings()
        assert isinstance(settings.paper_loop, ScheduledPaperLoopConfig)
        assert settings.paper_loop.loop_id == "paper_fee_4_best"
        assert settings.paper_loop.target_key == "fee_4"
        assert settings.paper_loop.model_run_id == "RUN20260426T163315Z"
        assert settings.paper_loop.optimization_id == "OPT20260426T170429Z"
        assert settings.paper_loop.symbol == "BTCUSDT"
        assert settings.paper_loop.timeframe == "15m"
        assert settings.paper_loop.feature_version == "v1"
        assert settings.paper_loop.candle_close_safety_minutes == 0
