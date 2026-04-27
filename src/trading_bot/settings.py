"""Typed settings and configuration loading.

Loads configuration from three sources (in priority order):
1. Environment variables (highest priority)
2. .env file
3. YAML config files in config/ directory (lowest priority)
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python < 3.11 compatibility
    class StrEnum(str, Enum):  # noqa: UP042
        """Fallback StrEnum for Python versions older than 3.11."""


# ---------------------------------------------------------------------------
# Resolve project root (two levels up from this file: src/trading_bot/)
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _FILE_DIR.parent.parent  # trading_bot/
CONFIG_DIR = PROJECT_ROOT / "config"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class Environment(StrEnum):
    """Application environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(StrEnum):
    """Supported log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Config sub-models (loaded from YAML)
# ---------------------------------------------------------------------------
class AssetConfig(BaseModel):
    """Asset pair configuration."""

    symbol: str = "BTCUSDT"
    base_asset: str = "BTC"
    quote_asset: str = "USDT"
    timeframe: str = "15m"
    exchange: str = "binance"


class DataSourcesConfig(BaseModel):
    """Data source configuration."""

    market: str = "binance"
    news: str = "coingecko"


class ModelParamsConfig(BaseModel):
    """LightGBM training parameters."""

    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8


class ModelConfig(BaseModel):
    """Model configuration."""

    algorithm: str = "lightgbm"
    prediction_horizon_minutes: int = 30
    target_type: str = "binary_return_direction"
    params: ModelParamsConfig = Field(default_factory=ModelParamsConfig)
    feature_groups: list[str] = Field(
        default_factory=lambda: ["market_technical", "market_microstructure", "news_sentiment"]
    )


class ValidationConfig(BaseModel):
    """Walk-forward validation configuration."""

    method: str = "walk_forward"
    n_splits: int = 5
    train_window_days: int = 60
    test_window_days: int = 7
    gap_days: int = 1


class RiskConfig(BaseModel):
    """Risk management configuration."""

    paper_trading_only: bool = True
    allow_live_trading: bool = False
    max_position_fraction: float = 0.01
    max_total_exposure_fraction: float = 0.05
    max_daily_loss_fraction: float = 0.02
    max_drawdown_fraction: float = 0.05
    min_confidence_threshold: float = 0.6
    max_trades_per_day: int = 20
    cooldown_after_loss_minutes: int = 30


class MarketDataConfig(BaseModel):
    """Market data ingestion configuration."""

    symbol: str = "BTCUSDT"
    default_timeframe: str = "15m"
    default_start_date: str = "2024-01-01"
    storage_format: str = "parquet"
    raw_data_dir: str = "data/raw/market"
    processed_data_dir: str = "data/processed/market"
    request_limit: int = 1000
    request_timeout: float = 30.0
    max_retries: int = 3


class NewsDataConfig(BaseModel):
    """News data ingestion configuration."""

    asset: str = "BTC"
    provider: str = "cryptocompare"
    default_start_date: str = "2024-01-01"
    raw_data_dir: str = "data/raw/news"
    processed_data_dir: str = "data/processed/news"
    page_size: int = 100
    request_timeout: float = 30.0
    max_retries: int = 3
    max_pages: int = 500
    btc_keywords: list[str] = Field(
        default_factory=lambda: [
            "btc",
            "bitcoin",
            "spot bitcoin etf",
            "bitcoin etf",
            "bitcoin network",
            "bitcoin miner",
            "bitcoin miners",
            "bitcoin mining",
            "bitcoin halving",
        ]
    )


class FeaturesConfig(BaseModel):
    """Feature engineering configuration."""

    feature_version: str = "v1"
    symbol: str = "BTCUSDT"
    asset: str = "BTC"
    timeframe: str = "15m"
    market_windows: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    news_lookback_windows: list[str] = Field(default_factory=lambda: ["1h", "4h"])
    output_dir: str = "data/features"
    fill_missing_news_with_zero: bool = True
    include_breakout_features: bool = True
    include_calendar_features: bool = True
    include_optional_news_features: bool = True
    news_burst_threshold_1h: int = 3
    news_burst_threshold_4h: int = 8


class NLPConfig(BaseModel):
    """Local NLP enrichment configuration."""

    model_name: str = "ProsusAI/finbert"
    enrichment_version: str = "v1"
    text_mode: str = "title_plus_summary"
    batch_size: int = 16
    device: str = "cpu"
    allow_cuda_fallback_to_cpu: bool = True
    max_length: int = 256


class SentimentFeaturesConfig(BaseModel):
    """Sentiment-aware news feature configuration."""

    enabled: bool = True
    feature_version: str = "v2_sentiment"
    lookback_windows: list[str] = Field(default_factory=lambda: ["1h", "4h"])
    use_enriched_news_store: bool = True
    positive_burst_threshold_1h: int = 2
    negative_burst_threshold_1h: int = 2


class StoragePathsConfig(BaseModel):
    """Local storage path configuration for auxiliary datasets."""

    enriched_news_dir: str = "data/enriched_news"


class LabelsConfig(BaseModel):
    """Label generation configuration."""

    label_version: str = "v1"
    primary_target: str = "target_up_2bars"
    horizon_bars: int = 2
    horizon_minutes: int | None = None
    binary_threshold: float = 0.0
    include_regression_target: bool = True
    include_optional_targets: bool = True
    optional_horizon_bars: list[int] = Field(default_factory=lambda: [4])


class DatasetsConfig(BaseModel):
    """Supervised dataset assembly configuration."""

    dataset_version: str = "v1"
    symbol: str = "BTCUSDT"
    asset: str = "BTC"
    timeframe: str = "15m"
    output_dir: str = "data/datasets"


class TrainingConfig(BaseModel):
    """Baseline training workflow configuration."""

    model_version: str = "v1"
    primary_target: str = "target_up_2bars"
    random_seed: int = 42
    save_fold_models: bool = True
    device: str = "cpu"
    allow_cuda_fallback_to_cpu: bool = True
    probability_threshold: float = 0.5
    include_feature_patterns: list[str] = Field(default_factory=list)
    exclude_feature_patterns: list[str] = Field(default_factory=list)


class WalkForwardConfig(BaseModel):
    """Walk-forward split generation settings."""

    mode: str = "expanding_window"
    min_train_rows: int = 1000
    validation_rows: int = 250
    step_rows: int = 250
    max_folds: int | None = None
    rolling_train_rows: int | None = None


class LightGBMConfig(BaseModel):
    """LightGBM baseline parameter defaults."""

    objective: str = "binary"
    metric: str = "binary_logloss"
    learning_rate: float = 0.05
    n_estimators: int = 200
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    device_type: str = "cpu"
    verbosity: int = -1


class ArtifactsConfig(BaseModel):
    """Artifact output locations for model training runs."""

    models_dir: str = "data/models"
    evaluation_dir: str = "data/evaluation"


class ExperimentsConfig(BaseModel):
    """Local experiment-tracking configuration."""

    registry_path: str = "data/experiments/run_registry.parquet"
    summary_dir: str = "data/experiments"
    default_sort_metric: str = "roc_auc_mean"
    default_status_for_new_runs: str = "candidate"


class ReportsConfig(BaseModel):
    """Evaluation report output configuration."""

    output_dir: str = "data/reports"
    write_markdown: bool = True
    top_k_feature_importance: int = 20


class BacktestStrategyConfig(BaseModel):
    """Probability-threshold strategy defaults for simulation."""

    strategy_version: str = "v1"
    type: str = "long_only_threshold"
    entry_threshold: float = 0.55
    exit_threshold: float = 0.45
    minimum_holding_bars: int = 1
    cooldown_bars: int = 0


class BacktestExecutionConfig(BaseModel):
    """Execution assumptions for offline simulations."""

    fill_model: str = "next_bar_open"
    starting_cash: float = 100.0
    fee_rate: float = 0.001
    slippage_rate: float = 0.0005
    max_position_fraction: float = 1.0
    allow_fractional_position: bool = True


class SimulationStorageConfig(BaseModel):
    """Output directories and defaults for simulations."""

    backtests_dir: str = "data/backtests"
    paper_simulations_dir: str = "data/paper_simulations"
    default_sort_metric: str = "total_return"


class ScheduledPaperLoopConfig(BaseModel):
    """Configuration for the scheduled offline paper-trading loop."""

    output_dir: str = "data/paper_loop"
    loop_id: str = "paper_fee_4_best"
    target_name: str = "target_long_net_positive_4bars"
    target_key: str = "fee_4"
    model_run_id: str = "RUN20260426T163315Z"
    optimization_id: str = "OPT20260426T170429Z"
    symbol: str = "BTCUSDT"
    timeframe: str = "15m"
    feature_version: str = "v1"
    provider: str = "cryptocompare"
    interval_minutes: int = 15
    candle_close_safety_minutes: int = 0
    poll_buffer_seconds: int = 5
    market_history_rows: int = 64
    prediction_threshold: float = 0.5
    write_markdown: bool = True
    daemon_log_filename: str = "paper_loop.log"


class ValidationSuiteConfig(BaseModel):
    """Validation-suite configuration and analysis thresholds."""

    output_dir: str = "data/reports/validation"
    random_seed: int = 42
    bucket_edges: list[float] = Field(default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    top_failure_cases: int = 5
    drawdown_case_count: int = 3
    overtrading_trades_per_day_threshold: float = 10.0
    tiny_edge_after_fees_threshold: float = 0.0


class OptimizationSearchSpaceConfig(BaseModel):
    """Grid-search ranges for strategy-threshold optimization."""

    entry_threshold: list[float] = Field(default_factory=lambda: [0.6, 0.65, 0.7, 0.75])
    exit_threshold: list[float] = Field(default_factory=lambda: [0.45, 0.5, 0.55])
    minimum_holding_bars: list[int] = Field(default_factory=lambda: [2, 4, 8, 16])
    cooldown_bars: list[int] = Field(default_factory=lambda: [2, 4, 8])
    max_position_fraction: list[float] = Field(default_factory=lambda: [0.25, 0.5, 1.0])


class OptimizationSelectionConfig(BaseModel):
    """Selection and filtering defaults for strategy search."""

    max_allowed_drawdown: float = 0.35
    max_allowed_trades: int = 500
    max_allowed_fee_to_starting_cash: float = 2.0
    minimum_total_return: float = 0.0
    top_k: int = 20
    ranking_metric: str = "risk_adjusted_score"


class OptimizationStorageConfig(BaseModel):
    """Output directory for optimization artifacts."""

    output_dir: str = "data/strategy_optimization"


class MultiHorizonConfig(BaseModel):
    """Configuration for multi-horizon research orchestration."""

    horizon_bars: list[int] = Field(default_factory=lambda: [4, 8, 16])
    default_feature_version: str = "v2_sentiment"
    label_version_prefix: str = "v1"
    dataset_version_prefix: str = "v1"
    model_version_prefix: str = "v1"
    output_dir: str = "data/reports/horizon_comparison"
    max_folds_override: int | None = None


class FeeAwareTargetsConfig(BaseModel):
    """Configuration for fee-aware and minimum-return target research."""

    default_feature_version: str = "v2_sentiment"
    target_keys: list[str] = Field(
        default_factory=lambda: [
            "fee_4",
            "fee_8",
            "fee_16",
            "ret025_8",
            "ret050_16",
            "ret075_16",
        ]
    )
    fee_rate: float = 0.001
    slippage_rate: float = 0.0005
    default_round_trip_cost: float = 0.003
    positive_class_rate_min: float = 0.02
    positive_class_rate_max: float = 0.98
    label_version_prefix: str = "fee"
    dataset_version_prefix: str = "fee"
    model_version_prefix: str = "fee"
    output_dir: str = "data/reports/target_comparison"
    max_folds_override: int | None = None


# ---------------------------------------------------------------------------
# YAML loader helper
# ---------------------------------------------------------------------------
def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _load_config_yamls() -> dict[str, Any]:
    """Load and merge all YAML config files."""
    merged: dict[str, Any] = {}
    for name in ("assets", "model", "risk"):
        path = CONFIG_DIR / f"{name}.yaml"
        merged.update(_load_yaml(path))
    return merged


# ---------------------------------------------------------------------------
# Main settings class
# ---------------------------------------------------------------------------
class AppSettings(BaseSettings):
    """Application-wide settings.

    Priority: env vars > .env > YAML defaults > field defaults.
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App-level
    app_name: str = "btc_ml_trading_bot"
    app_env: Environment = Environment.DEVELOPMENT
    log_level: LogLevel = LogLevel.INFO

    # Sub-configs (populated from YAML + env overrides)
    asset: AssetConfig = Field(default_factory=AssetConfig)
    data_sources: DataSourcesConfig = Field(default_factory=DataSourcesConfig)
    market_data: MarketDataConfig = Field(default_factory=MarketDataConfig)
    news_data: NewsDataConfig = Field(default_factory=NewsDataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    datasets: DatasetsConfig = Field(default_factory=DatasetsConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    lightgbm: LightGBMConfig = Field(default_factory=LightGBMConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    experiments: ExperimentsConfig = Field(default_factory=ExperimentsConfig)
    reports: ReportsConfig = Field(default_factory=ReportsConfig)
    backtest_strategy: BacktestStrategyConfig = Field(default_factory=BacktestStrategyConfig)
    backtest_execution: BacktestExecutionConfig = Field(default_factory=BacktestExecutionConfig)
    simulations: SimulationStorageConfig = Field(default_factory=SimulationStorageConfig)
    paper_loop: ScheduledPaperLoopConfig = Field(default_factory=ScheduledPaperLoopConfig)
    validation_suite: ValidationSuiteConfig = Field(default_factory=ValidationSuiteConfig)
    optimization_search_space: OptimizationSearchSpaceConfig = Field(
        default_factory=OptimizationSearchSpaceConfig
    )
    optimization_selection: OptimizationSelectionConfig = Field(
        default_factory=OptimizationSelectionConfig
    )
    optimization_storage: OptimizationStorageConfig = Field(
        default_factory=OptimizationStorageConfig
    )
    multi_horizon: MultiHorizonConfig = Field(default_factory=MultiHorizonConfig)
    fee_aware_targets: FeeAwareTargetsConfig = Field(default_factory=FeeAwareTargetsConfig)
    nlp: NLPConfig = Field(default_factory=NLPConfig)
    sentiment_features: SentimentFeaturesConfig = Field(default_factory=SentimentFeaturesConfig)
    storage: StoragePathsConfig = Field(default_factory=StoragePathsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)

    # API keys (from env only)
    binance_api_key: str = ""
    binance_api_secret: str = ""
    coingecko_api_key: str = ""
    mlflow_tracking_uri: str = "http://localhost:5000"


def load_settings() -> AppSettings:
    """Load settings from YAML configs, .env, and environment variables.

    Returns:
        Fully resolved AppSettings instance.
    """
    yaml_data = _load_config_yamls()

    overrides: dict[str, Any] = {}

    # Map YAML sections to sub-config models
    if "asset" in yaml_data:
        overrides["asset"] = AssetConfig(**yaml_data["asset"])
    if "model" in yaml_data:
        model_raw = yaml_data["model"]
        if "params" in model_raw:
            model_raw["params"] = ModelParamsConfig(**model_raw["params"])
        overrides["model"] = ModelConfig(**model_raw)
    if "validation" in yaml_data:
        overrides["validation"] = ValidationConfig(**yaml_data["validation"])
    if "risk" in yaml_data:
        overrides["risk"] = RiskConfig(**yaml_data["risk"])
    if "data_sources" in yaml_data:
        overrides["data_sources"] = DataSourcesConfig(**yaml_data["data_sources"])
    if "market_data" in yaml_data:
        overrides["market_data"] = MarketDataConfig(**yaml_data["market_data"])
    if "news_data" in yaml_data:
        overrides["news_data"] = NewsDataConfig(**yaml_data["news_data"])
    if "features" in yaml_data:
        overrides["features"] = FeaturesConfig(**yaml_data["features"])
    if "labels" in yaml_data:
        overrides["labels"] = LabelsConfig(**yaml_data["labels"])
    if "datasets" in yaml_data:
        overrides["datasets"] = DatasetsConfig(**yaml_data["datasets"])
    if "training" in yaml_data:
        overrides["training"] = TrainingConfig(**yaml_data["training"])
    if "walk_forward" in yaml_data:
        overrides["walk_forward"] = WalkForwardConfig(**yaml_data["walk_forward"])
    if "lightgbm" in yaml_data:
        overrides["lightgbm"] = LightGBMConfig(**yaml_data["lightgbm"])
    if "artifacts" in yaml_data:
        overrides["artifacts"] = ArtifactsConfig(**yaml_data["artifacts"])
    if "experiments" in yaml_data:
        overrides["experiments"] = ExperimentsConfig(**yaml_data["experiments"])
    if "reports" in yaml_data:
        overrides["reports"] = ReportsConfig(**yaml_data["reports"])
    if "backtest_strategy" in yaml_data:
        overrides["backtest_strategy"] = BacktestStrategyConfig(**yaml_data["backtest_strategy"])
    if "backtest_execution" in yaml_data:
        overrides["backtest_execution"] = BacktestExecutionConfig(
            **yaml_data["backtest_execution"]
        )
    if "simulations" in yaml_data:
        overrides["simulations"] = SimulationStorageConfig(**yaml_data["simulations"])
    if "paper_loop" in yaml_data:
        overrides["paper_loop"] = ScheduledPaperLoopConfig(**yaml_data["paper_loop"])
    if "validation_suite" in yaml_data:
        overrides["validation_suite"] = ValidationSuiteConfig(**yaml_data["validation_suite"])
    if "optimization_search_space" in yaml_data:
        overrides["optimization_search_space"] = OptimizationSearchSpaceConfig(
            **yaml_data["optimization_search_space"]
        )
    if "optimization_selection" in yaml_data:
        overrides["optimization_selection"] = OptimizationSelectionConfig(
            **yaml_data["optimization_selection"]
        )
    if "optimization_storage" in yaml_data:
        overrides["optimization_storage"] = OptimizationStorageConfig(
            **yaml_data["optimization_storage"]
        )
    if "multi_horizon" in yaml_data:
        overrides["multi_horizon"] = MultiHorizonConfig(**yaml_data["multi_horizon"])
    if "fee_aware_targets" in yaml_data:
        overrides["fee_aware_targets"] = FeeAwareTargetsConfig(**yaml_data["fee_aware_targets"])
    if "nlp" in yaml_data:
        overrides["nlp"] = NLPConfig(**yaml_data["nlp"])
    if "sentiment_features" in yaml_data:
        overrides["sentiment_features"] = SentimentFeaturesConfig(
            **yaml_data["sentiment_features"]
        )
    if "storage" in yaml_data:
        overrides["storage"] = StoragePathsConfig(**yaml_data["storage"])

    return AppSettings(**overrides)
