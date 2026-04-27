"""Feature engineering package for market, news, storage, and pipelines."""

from trading_bot.features.feature_pipeline import FeaturePipeline, FeaturePipelineError
from trading_bot.features.feature_store import FeatureSet, ParquetFeatureStore
from trading_bot.features.market_features import compute_market_features
from trading_bot.features.news_features import compute_news_features

__all__ = [
    "FeaturePipeline",
    "FeaturePipelineError",
    "FeatureSet",
    "ParquetFeatureStore",
    "compute_market_features",
    "compute_news_features",
]
