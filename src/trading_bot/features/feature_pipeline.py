"""Reusable feature engineering pipeline for BTC market and news data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_bot.features.feature_store import FeatureSet, ParquetFeatureStore
from trading_bot.features.market_features import compute_market_features
from trading_bot.features.news_features import compute_news_features
from trading_bot.logging_config import get_logger
from trading_bot.schemas.features import (
    INDEX_COLUMNS,
    ZERO_FILL_NEWS_FEATURE_COLUMNS,
    FeatureDatasetName,
    get_expected_market_feature_columns,
    get_expected_news_feature_columns,
    validate_feature_df,
)
from trading_bot.settings import PROJECT_ROOT, AppSettings, load_settings
from trading_bot.storage.enriched_news_store import EnrichedNewsStore
from trading_bot.storage.news_store import NewsStore
from trading_bot.storage.parquet_store import ParquetStore

logger = get_logger(__name__)


class FeaturePipelineError(Exception):
    """Raised when a feature dataset cannot be built safely."""


@dataclass(frozen=True)
class BuildContext:
    """Resolved dataset coordinates used across the pipeline."""

    asset: str
    symbol: str
    timeframe: str
    version: str
    provider: str
    use_sentiment_features: bool


class FeaturePipeline:
    """Build, validate, merge, and persist engineered feature datasets."""

    def __init__(
        self,
        *,
        market_store: ParquetStore,
        news_store: NewsStore,
        enriched_news_store: EnrichedNewsStore,
        feature_store: ParquetFeatureStore,
        asset: str,
        symbol: str,
        timeframe: str,
        provider: str,
        version: str,
        market_windows: list[int],
        news_lookback_windows: list[str],
        fill_missing_news_with_zero: bool,
        include_breakout_features: bool,
        include_calendar_features: bool,
        include_optional_news_features: bool,
        news_burst_threshold_1h: int,
        news_burst_threshold_4h: int,
        sentiment_feature_version: str,
        sentiment_feature_lookback_windows: list[str],
        use_enriched_news_store: bool,
        finbert_model_name: str,
        enrichment_version: str,
        positive_burst_threshold_1h: int,
        negative_burst_threshold_1h: int,
    ) -> None:
        self.market_store = market_store
        self.news_store = news_store
        self.enriched_news_store = enriched_news_store
        self.feature_store = feature_store
        self.default_asset = asset
        self.default_symbol = symbol
        self.default_timeframe = timeframe
        self.default_provider = provider
        self.default_version = version
        self.market_windows = market_windows
        self.news_lookback_windows = news_lookback_windows
        self.fill_missing_news_with_zero = fill_missing_news_with_zero
        self.include_breakout_features = include_breakout_features
        self.include_calendar_features = include_calendar_features
        self.include_optional_news_features = include_optional_news_features
        self.news_burst_threshold_1h = news_burst_threshold_1h
        self.news_burst_threshold_4h = news_burst_threshold_4h
        self.sentiment_feature_version = sentiment_feature_version
        self.sentiment_feature_lookback_windows = sentiment_feature_lookback_windows
        self.use_enriched_news_store = use_enriched_news_store
        self.finbert_model_name = finbert_model_name
        self.enrichment_version = enrichment_version
        self.positive_burst_threshold_1h = positive_burst_threshold_1h
        self.negative_burst_threshold_1h = negative_burst_threshold_1h

    @classmethod
    def from_settings(cls, settings: AppSettings | None = None) -> FeaturePipeline:
        """Build a pipeline instance from project settings."""
        resolved_settings = settings or load_settings()
        market_store = ParquetStore(PROJECT_ROOT / resolved_settings.market_data.processed_data_dir)
        news_store = NewsStore(PROJECT_ROOT / resolved_settings.news_data.processed_data_dir)
        enriched_news_store = EnrichedNewsStore(
            PROJECT_ROOT / resolved_settings.storage.enriched_news_dir
        )
        feature_store = ParquetFeatureStore(PROJECT_ROOT / resolved_settings.features.output_dir)

        return cls(
            market_store=market_store,
            news_store=news_store,
            enriched_news_store=enriched_news_store,
            feature_store=feature_store,
            asset=resolved_settings.features.asset,
            symbol=resolved_settings.features.symbol,
            timeframe=resolved_settings.features.timeframe,
            provider=resolved_settings.news_data.provider,
            version=resolved_settings.features.feature_version,
            market_windows=resolved_settings.features.market_windows,
            news_lookback_windows=resolved_settings.features.news_lookback_windows,
            fill_missing_news_with_zero=resolved_settings.features.fill_missing_news_with_zero,
            include_breakout_features=resolved_settings.features.include_breakout_features,
            include_calendar_features=resolved_settings.features.include_calendar_features,
            include_optional_news_features=resolved_settings.features.include_optional_news_features,
            news_burst_threshold_1h=resolved_settings.features.news_burst_threshold_1h,
            news_burst_threshold_4h=resolved_settings.features.news_burst_threshold_4h,
            sentiment_feature_version=resolved_settings.sentiment_features.feature_version,
            sentiment_feature_lookback_windows=resolved_settings.sentiment_features.lookback_windows,
            use_enriched_news_store=resolved_settings.sentiment_features.use_enriched_news_store,
            finbert_model_name=resolved_settings.nlp.model_name,
            enrichment_version=resolved_settings.nlp.enrichment_version,
            positive_burst_threshold_1h=resolved_settings.sentiment_features.positive_burst_threshold_1h,
            negative_burst_threshold_1h=resolved_settings.sentiment_features.negative_burst_threshold_1h,
        )

    def build_market_features(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        asset: str | None = None,
        version: str | None = None,
    ) -> tuple[pd.DataFrame, FeatureSet]:
        """Build and persist market-only features."""
        context = self._resolve_context(
            symbol=symbol,
            timeframe=timeframe,
            asset=asset,
            version=version,
        )
        market_features = self._build_market_feature_df(context)
        feature_set = self.feature_store.write_dataset(
            "market_features",
            market_features,
            asset=context.asset,
            symbol=context.symbol,
            timeframe=context.timeframe,
            version=context.version,
            metadata={"feature_group": "market"},
        )
        return market_features, feature_set

    def build_news_features(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        asset: str | None = None,
        provider: str | None = None,
        version: str | None = None,
    ) -> tuple[pd.DataFrame, FeatureSet]:
        """Build and persist news-only features aligned to the market timeline."""
        context = self._resolve_context(
            symbol=symbol,
            timeframe=timeframe,
            asset=asset,
            version=version,
            provider=provider,
        )
        market_timeline = self._load_market_data(context)[INDEX_COLUMNS]
        news_features = self._build_news_feature_df(context, market_timeline=market_timeline)
        feature_set = self.feature_store.write_dataset(
            "news_features",
            news_features,
            asset=context.asset,
            symbol=context.symbol,
            timeframe=context.timeframe,
            version=context.version,
            metadata={"feature_group": "news", "provider": context.provider},
        )
        return news_features, feature_set

    def build_feature_table(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        asset: str | None = None,
        provider: str | None = None,
        version: str | None = None,
    ) -> tuple[pd.DataFrame, FeatureSet]:
        """Build and persist the merged model-ready feature table."""
        context = self._resolve_context(
            symbol=symbol,
            timeframe=timeframe,
            asset=asset,
            version=version,
            provider=provider,
        )

        market_features = self._build_market_feature_df(context)
        news_features = self._build_news_feature_df(
            context,
            market_timeline=market_features[INDEX_COLUMNS],
        )

        merged = market_features.merge(
            news_features,
            on=INDEX_COLUMNS,
            how="left",
            validate="one_to_one",
        )
        merged = merged.sort_values("timestamp").reset_index(drop=True)

        if self.fill_missing_news_with_zero:
            for column in ZERO_FILL_NEWS_FEATURE_COLUMNS:
                if column in merged.columns:
                    merged[column] = merged[column].fillna(0.0)

        expected_market_columns = self._expected_market_columns()
        expected_news_columns = self._expected_news_columns(
            include_sentiment_features=context.use_sentiment_features,
            lookback_windows=(
                self.sentiment_feature_lookback_windows
                if context.use_sentiment_features
                else self.news_lookback_windows
            ),
        )
        expected_columns = INDEX_COLUMNS + expected_market_columns + expected_news_columns
        merged = merged.reindex(columns=expected_columns)
        self._validate_output(
            dataset_name="feature_table",
            df=merged,
            expected_feature_columns=expected_market_columns + expected_news_columns,
            expected_row_count=len(market_features),
        )

        feature_set = self.feature_store.write_dataset(
            "feature_table",
            merged,
            asset=context.asset,
            symbol=context.symbol,
            timeframe=context.timeframe,
            version=context.version,
            metadata={"feature_group": "merged", "provider": context.provider},
        )
        return merged, feature_set

    def build_latest_feature_row(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        asset: str | None = None,
        provider: str | None = None,
        version: str | None = None,
        market_history_rows: int = 64,
        target_timestamp: object | None = None,
    ) -> pd.DataFrame:
        """Build only the latest model-ready feature row from recent stored inputs."""
        context = self._resolve_context(
            symbol=symbol,
            timeframe=timeframe,
            asset=asset,
            version=version,
            provider=provider,
        )
        market_df = self._load_market_data(context)
        if market_df.empty:
            raise FeaturePipelineError(
                f"No stored market data found for {context.symbol}/{context.timeframe}"
            )

        if target_timestamp is not None:
            normalized_target_timestamp = pd.to_datetime(target_timestamp, utc=True, errors="coerce")
            if pd.isna(normalized_target_timestamp):
                raise FeaturePipelineError(
                    f"Could not normalize target_timestamp={target_timestamp!r} to UTC"
                )
            normalized_target_timestamp = pd.Timestamp(normalized_target_timestamp)
            market_df = market_df[market_df["timestamp"] <= normalized_target_timestamp].copy()
            if market_df.empty:
                raise FeaturePipelineError(
                    "No stored market rows are available at or before the requested target timestamp."
                )
            if normalized_target_timestamp not in set(market_df["timestamp"]):
                raise FeaturePipelineError(
                    "No stored market row exists for the requested target timestamp."
                )
        else:
            normalized_target_timestamp = None

        recent_market = market_df.tail(max(int(market_history_rows), 9)).reset_index(drop=True)
        market_features = compute_market_features(
            recent_market,
            windows=self.market_windows,
            include_breakout_features=self.include_breakout_features,
            include_calendar_features=self.include_calendar_features,
        )
        if normalized_target_timestamp is not None:
            target_market_row = market_features[
                market_features["timestamp"] == normalized_target_timestamp
            ].copy()
            if target_market_row.empty:
                raise FeaturePipelineError(
                    "Could not build market features for the requested target timestamp."
                )
        else:
            target_market_row = market_features.tail(1).copy()

        market_timeline = target_market_row[INDEX_COLUMNS].reset_index(drop=True)
        news_features = self._build_news_feature_df(
            context,
            market_timeline=market_timeline,
        )

        merged = target_market_row.reset_index(drop=True).merge(
            news_features,
            on=INDEX_COLUMNS,
            how="left",
            validate="one_to_one",
        )
        if self.fill_missing_news_with_zero:
            for column in ZERO_FILL_NEWS_FEATURE_COLUMNS:
                if column in merged.columns:
                    merged[column] = merged[column].fillna(0.0)

        expected_market_columns = self._expected_market_columns()
        expected_news_columns = self._expected_news_columns(
            include_sentiment_features=context.use_sentiment_features,
            lookback_windows=(
                self.sentiment_feature_lookback_windows
                if context.use_sentiment_features
                else self.news_lookback_windows
            ),
        )
        expected_columns = INDEX_COLUMNS + expected_market_columns + expected_news_columns
        merged = merged.reindex(columns=expected_columns)
        self._validate_output(
            dataset_name="feature_table",
            df=merged,
            expected_feature_columns=expected_market_columns + expected_news_columns,
            expected_row_count=1,
        )
        return merged

    def inspect_dataset(
        self,
        dataset_name: FeatureDatasetName,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        asset: str | None = None,
        provider: str | None = None,
        version: str | None = None,
    ) -> dict[str, object]:
        """Return persisted dataset metadata for local inspection."""
        context = self._resolve_context(
            symbol=symbol,
            timeframe=timeframe,
            asset=asset,
            version=version,
            provider=provider,
        )
        return self.feature_store.get_dataset_info(
            dataset_name,
            asset=context.asset,
            symbol=context.symbol,
            timeframe=context.timeframe,
            version=context.version,
        )

    def _build_market_feature_df(self, context: BuildContext) -> pd.DataFrame:
        market_df = self._load_market_data(context)
        features = compute_market_features(
            market_df,
            windows=self.market_windows,
            include_breakout_features=self.include_breakout_features,
            include_calendar_features=self.include_calendar_features,
        )
        self._validate_output(
            dataset_name="market_features",
            df=features,
            expected_feature_columns=self._expected_market_columns(),
            expected_row_count=len(market_df),
        )
        return features

    def _build_news_feature_df(
        self,
        context: BuildContext,
        *,
        market_timeline: pd.DataFrame,
    ) -> pd.DataFrame:
        news_df = self.news_store.read(context.asset, context.provider)
        lookback_windows = self.news_lookback_windows
        if context.use_sentiment_features and self.use_enriched_news_store:
            enriched_df = self.enriched_news_store.read(
                asset=context.asset,
                provider=context.provider,
                model_name=self.finbert_model_name,
                enrichment_version=self.enrichment_version,
            )
            if not enriched_df.empty:
                news_df = enriched_df
                lookback_windows = self.sentiment_feature_lookback_windows
        features = compute_news_features(
            market_timeline,
            news_df,
            lookback_windows=lookback_windows,
            fill_missing_with_zero=self.fill_missing_news_with_zero,
            include_optional_features=self.include_optional_news_features,
            include_sentiment_features=context.use_sentiment_features,
            burst_thresholds={
                "1h": self.news_burst_threshold_1h,
                "4h": self.news_burst_threshold_4h,
            },
            positive_burst_threshold_1h=self.positive_burst_threshold_1h,
            negative_burst_threshold_1h=self.negative_burst_threshold_1h,
        )
        self._validate_output(
            dataset_name="news_features",
            df=features,
            expected_feature_columns=self._expected_news_columns(
                include_sentiment_features=context.use_sentiment_features,
                lookback_windows=lookback_windows,
            ),
            expected_row_count=len(market_timeline),
        )
        return features

    def _load_market_data(self, context: BuildContext) -> pd.DataFrame:
        market_df = self.market_store.read(context.symbol, context.timeframe)
        if market_df.empty:
            raise FeaturePipelineError(
                f"No stored market data found for {context.symbol}/{context.timeframe}"
            )
        market_df["timestamp"] = pd.to_datetime(market_df["timestamp"], utc=True)
        market_df = market_df.sort_values("timestamp").reset_index(drop=True)
        logger.info(
            "feature_pipeline_market_input_loaded",
            rows=len(market_df),
            symbol=context.symbol,
            timeframe=context.timeframe,
            start_timestamp=market_df["timestamp"].min().isoformat(),
            end_timestamp=market_df["timestamp"].max().isoformat(),
        )
        return market_df

    def _resolve_context(
        self,
        *,
        symbol: str | None,
        timeframe: str | None,
        asset: str | None,
        version: str | None,
        provider: str | None = None,
    ) -> BuildContext:
        return BuildContext(
            asset=asset or self.default_asset,
            symbol=symbol or self.default_symbol,
            timeframe=timeframe or self.default_timeframe,
            version=version or self.default_version,
            provider=provider or self.default_provider,
            use_sentiment_features=self._should_use_sentiment_features(
                version or self.default_version
            ),
        )

    def _expected_market_columns(self) -> list[str]:
        return get_expected_market_feature_columns(
            include_breakout_features=self.include_breakout_features,
            include_calendar_features=self.include_calendar_features,
        )

    def _expected_news_columns(
        self,
        *,
        include_sentiment_features: bool,
        lookback_windows: list[str] | None = None,
    ) -> list[str]:
        return get_expected_news_feature_columns(
            lookback_windows=lookback_windows or self.news_lookback_windows,
            include_optional_features=self.include_optional_news_features,
            include_sentiment_features=include_sentiment_features,
        )

    def _validate_output(
        self,
        *,
        dataset_name: FeatureDatasetName,
        df: pd.DataFrame,
        expected_feature_columns: list[str],
        expected_row_count: int,
    ) -> None:
        validation = validate_feature_df(
            df,
            expected_feature_columns,
            expected_row_count=expected_row_count,
        )
        if not validation.is_valid:
            raise FeaturePipelineError(
                f"{dataset_name} validation failed: {'; '.join(validation.errors)}"
            )

        logger.info(
            "feature_pipeline_dataset_validated",
            dataset_name=dataset_name,
            rows=validation.row_count,
            columns=validation.column_count,
            start_timestamp=df["timestamp"].min().isoformat() if not df.empty else None,
            end_timestamp=df["timestamp"].max().isoformat() if not df.empty else None,
        )

    def _should_use_sentiment_features(self, version: str) -> bool:
        """Return whether the requested version should use enriched sentiment features."""
        return version == self.sentiment_feature_version
