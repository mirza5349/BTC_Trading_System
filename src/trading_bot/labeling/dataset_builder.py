"""Reusable supervised dataset builder for BTC feature tables and labels."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_bot.features.feature_store import ParquetFeatureStore
from trading_bot.labeling.labels import generate_custom_target_labels, generate_future_return_labels
from trading_bot.logging_config import get_logger
from trading_bot.schemas.datasets import (
    DatasetArtifactName,
    SupervisedDatasetSchema,
    binary_target_column_name,
    get_expected_target_columns,
    is_binary_target_column,
    is_supported_target_column,
    summarize_missing_values,
    summarize_target_columns,
    validate_dataset_df,
)
from trading_bot.schemas.features import INDEX_COLUMNS
from trading_bot.schemas.targets import TargetConfig
from trading_bot.settings import PROJECT_ROOT, AppSettings, load_settings
from trading_bot.storage.dataset_store import DatasetStore, StoredDatasetArtifact
from trading_bot.storage.parquet_store import ParquetStore

logger = get_logger(__name__)


class DatasetBuilderError(Exception):
    """Raised when labels or supervised datasets cannot be built safely."""


@dataclass(frozen=True)
class DatasetBuildContext:
    """Resolved dataset coordinates used during step-5 assembly."""

    asset: str
    symbol: str
    timeframe: str
    feature_version: str
    label_version: str
    dataset_version: str


class SupervisedDatasetBuilder:
    """Generate labels and assemble leak-safe supervised datasets."""

    def __init__(
        self,
        *,
        market_store: ParquetStore,
        feature_store: ParquetFeatureStore,
        dataset_store: DatasetStore,
        asset: str,
        symbol: str,
        timeframe: str,
        feature_version: str,
        label_version: str,
        dataset_version: str,
        primary_target: str,
        horizon_bars: int,
        horizon_minutes: int | None,
        binary_threshold: float,
        include_regression_target: bool,
        include_optional_targets: bool,
        optional_horizon_bars: list[int],
    ) -> None:
        self.market_store = market_store
        self.feature_store = feature_store
        self.dataset_store = dataset_store
        self.default_asset = asset
        self.default_symbol = symbol
        self.default_timeframe = timeframe
        self.default_feature_version = feature_version
        self.default_label_version = label_version
        self.default_dataset_version = dataset_version
        self.primary_target = primary_target
        self.horizon_bars = horizon_bars
        self.horizon_minutes = horizon_minutes
        self.binary_threshold = binary_threshold
        self.include_regression_target = include_regression_target
        self.include_optional_targets = include_optional_targets
        self.optional_horizon_bars = optional_horizon_bars

    @classmethod
    def from_settings(cls, settings: AppSettings | None = None) -> SupervisedDatasetBuilder:
        """Build a dataset builder instance from project settings."""
        resolved_settings = settings or load_settings()
        market_store = ParquetStore(PROJECT_ROOT / resolved_settings.market_data.processed_data_dir)
        feature_store = ParquetFeatureStore(PROJECT_ROOT / resolved_settings.features.output_dir)
        dataset_store = DatasetStore(PROJECT_ROOT / resolved_settings.datasets.output_dir)

        return cls(
            market_store=market_store,
            feature_store=feature_store,
            dataset_store=dataset_store,
            asset=resolved_settings.datasets.asset,
            symbol=resolved_settings.datasets.symbol,
            timeframe=resolved_settings.datasets.timeframe,
            feature_version=resolved_settings.features.feature_version,
            label_version=resolved_settings.labels.label_version,
            dataset_version=resolved_settings.datasets.dataset_version,
            primary_target=resolved_settings.labels.primary_target,
            horizon_bars=resolved_settings.labels.horizon_bars,
            horizon_minutes=resolved_settings.labels.horizon_minutes,
            binary_threshold=resolved_settings.labels.binary_threshold,
            include_regression_target=resolved_settings.labels.include_regression_target,
            include_optional_targets=resolved_settings.labels.include_optional_targets,
            optional_horizon_bars=resolved_settings.labels.optional_horizon_bars,
        )

    def build_labels(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        asset: str | None = None,
        label_version: str | None = None,
        horizon_bars: int | None = None,
        target_config: TargetConfig | None = None,
    ) -> tuple[pd.DataFrame, StoredDatasetArtifact]:
        """Generate and persist leak-safe future-return labels."""
        context = self._resolve_context(
            symbol=symbol,
            timeframe=timeframe,
            asset=asset,
            label_version=label_version,
        )
        labels_df = self._build_labels_df(
            context,
            horizon_bars=horizon_bars or self.horizon_bars,
            target_config=target_config,
        )
        primary_target = (
            target_config.target_name
            if target_config is not None
            else self._primary_target_column(horizon_bars or self.horizon_bars)
        )
        artifact = self.dataset_store.write_artifact(
            "labels",
            labels_df,
            asset=context.asset,
            symbol=context.symbol,
            timeframe=context.timeframe,
            label_version=context.label_version,
            metadata={
                "primary_target": primary_target,
                "feature_version": context.feature_version,
                "horizon_bars": horizon_bars or self.horizon_bars,
                "target_config": target_config.to_dict() if target_config is not None else None,
            },
        )
        return labels_df, artifact

    def build_supervised_dataset(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        asset: str | None = None,
        feature_version: str | None = None,
        label_version: str | None = None,
        dataset_version: str | None = None,
        horizon_bars: int | None = None,
        target_config: TargetConfig | None = None,
    ) -> tuple[pd.DataFrame, StoredDatasetArtifact]:
        """Build and persist the final supervised dataset."""
        context = self._resolve_context(
            symbol=symbol,
            timeframe=timeframe,
            asset=asset,
            feature_version=feature_version,
            label_version=label_version,
            dataset_version=dataset_version,
        )

        feature_df = self._load_feature_table(context)
        labels_df = self._build_labels_df(
            context,
            horizon_bars=horizon_bars or self.horizon_bars,
            target_config=target_config,
        )
        primary_target = (
            target_config.target_name
            if target_config is not None
            else self._primary_target_column(horizon_bars or self.horizon_bars)
        )

        self.dataset_store.write_artifact(
            "labels",
            labels_df,
            asset=context.asset,
            symbol=context.symbol,
            timeframe=context.timeframe,
            label_version=context.label_version,
            metadata={
                "primary_target": primary_target,
                "feature_version": context.feature_version,
                "horizon_bars": horizon_bars or self.horizon_bars,
                "target_config": target_config.to_dict() if target_config is not None else None,
            },
        )

        merged = feature_df.merge(
            labels_df,
            on=INDEX_COLUMNS,
            how="left",
            validate="one_to_one",
        )
        merged = merged.sort_values("timestamp").reset_index(drop=True)

        target_columns = self._target_columns(
            horizon_bars or self.horizon_bars,
            target_config=target_config,
        )
        feature_columns = [column for column in feature_df.columns if column not in INDEX_COLUMNS]
        schema = self.dataset_schema(
            feature_columns=feature_columns,
            target_columns=target_columns,
            label_version=context.label_version,
            dataset_version=context.dataset_version,
            primary_target=primary_target,
        )

        rows_before_drop = len(merged)
        dataset_df = merged.dropna(subset=target_columns).reset_index(drop=True)
        dropped_rows = rows_before_drop - len(dataset_df)

        for column in target_columns:
            if is_binary_target_column(column) and column in dataset_df.columns:
                dataset_df[column] = dataset_df[column].astype(int)

        dataset_df = dataset_df.reindex(columns=INDEX_COLUMNS + feature_columns + target_columns)

        self._validate_output(
            artifact_name="supervised_dataset",
            df=dataset_df,
            target_columns=target_columns,
            required_target_column=primary_target,
        )

        target_summary = summarize_target_columns(dataset_df, target_columns)
        missing_summary = summarize_missing_values(dataset_df)

        logger.info(
            "supervised_dataset_complete",
            rows=len(dataset_df),
            feature_columns=len(feature_columns),
            target_columns=target_columns,
            dropped_unlabeled_rows=dropped_rows,
            start_timestamp=dataset_df["timestamp"].min().isoformat()
            if not dataset_df.empty
            else None,
            end_timestamp=dataset_df["timestamp"].max().isoformat()
            if not dataset_df.empty
            else None,
            target_summary=target_summary,
            missing_summary=missing_summary,
        )

        artifact = self.dataset_store.write_artifact(
            "supervised_dataset",
            dataset_df,
            asset=context.asset,
            symbol=context.symbol,
            timeframe=context.timeframe,
            label_version=context.label_version,
            dataset_version=context.dataset_version,
            metadata={
                "feature_version": context.feature_version,
                "primary_target": primary_target,
                "target_columns": target_columns,
                "feature_columns": feature_columns,
                "dropped_unlabeled_rows": dropped_rows,
                "target_config": target_config.to_dict() if target_config is not None else None,
                "schema": {
                    "identifier_columns": schema.identifier_columns,
                    "feature_columns": schema.feature_columns,
                    "target_columns": schema.target_columns,
                    "primary_target": schema.primary_target,
                    "label_version": schema.label_version,
                    "dataset_version": schema.dataset_version,
                },
            },
        )
        return dataset_df, artifact

    def inspect_dataset(
        self,
        artifact_name: DatasetArtifactName,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        asset: str | None = None,
        label_version: str | None = None,
        dataset_version: str | None = None,
    ) -> dict[str, object]:
        """Return stored label or dataset stats for local inspection."""
        context = self._resolve_context(
            symbol=symbol,
            timeframe=timeframe,
            asset=asset,
            label_version=label_version,
            dataset_version=dataset_version,
        )

        info = self.dataset_store.get_artifact_info(
            artifact_name,
            asset=context.asset,
            symbol=context.symbol,
            timeframe=context.timeframe,
            label_version=context.label_version,
            dataset_version=(
                context.dataset_version if artifact_name == "supervised_dataset" else None
            ),
        )
        if not info:
            return {}

        dataset_df = self.dataset_store.read_artifact(
            artifact_name,
            asset=context.asset,
            symbol=context.symbol,
            timeframe=context.timeframe,
            label_version=context.label_version,
            dataset_version=(
                context.dataset_version if artifact_name == "supervised_dataset" else None
            ),
        )

        target_columns = [
            column
            for column in dataset_df.columns
            if is_supported_target_column(column)
        ]
        feature_columns = [
            column for column in dataset_df.columns if column not in INDEX_COLUMNS + target_columns
        ]

        info["target_columns"] = target_columns
        info["feature_columns"] = feature_columns
        info["target_summary"] = summarize_target_columns(dataset_df, target_columns)
        info["missing_summary"] = summarize_missing_values(dataset_df)
        return info

    def list_target_columns(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        asset: str | None = None,
        label_version: str | None = None,
        dataset_version: str | None = None,
    ) -> list[str]:
        """List available target columns in the stored supervised dataset."""
        info = self.inspect_dataset(
            "supervised_dataset",
            symbol=symbol,
            timeframe=timeframe,
            asset=asset,
            label_version=label_version,
            dataset_version=dataset_version,
        )
        if not info:
            return []
        return list(info["target_columns"])

    def dataset_schema(
        self,
        *,
        feature_columns: list[str],
        target_columns: list[str],
        label_version: str,
        dataset_version: str,
        primary_target: str,
    ) -> SupervisedDatasetSchema:
        """Build schema metadata for a supervised dataset artifact."""
        return SupervisedDatasetSchema(
            identifier_columns=INDEX_COLUMNS.copy(),
            feature_columns=feature_columns,
            target_columns=target_columns,
            primary_target=primary_target,
            label_version=label_version,
            dataset_version=dataset_version,
        )

    def _build_labels_df(
        self,
        context: DatasetBuildContext,
        *,
        horizon_bars: int,
        target_config: TargetConfig | None = None,
    ) -> pd.DataFrame:
        market_df = self._load_market_data(context)
        if target_config is None:
            labels_df = generate_future_return_labels(
                market_df,
                primary_horizon_bars=horizon_bars,
                horizon_minutes=self.horizon_minutes,
                binary_threshold=self.binary_threshold,
                include_regression_target=self.include_regression_target,
                include_optional_targets=self.include_optional_targets,
                optional_horizon_bars=self.optional_horizon_bars,
                drop_unlabeled_rows=True,
            )
            target_columns = self._target_columns(horizon_bars)
            required_target_column = self._primary_target_column(horizon_bars)
        else:
            generated = generate_custom_target_labels(
                market_df,
                target_configs=[target_config],
                drop_unlabeled_rows=True,
            )
            labels_df = generated.labels_df
            target_columns = self._target_columns(horizon_bars, target_config=target_config)
            required_target_column = target_config.target_name
            self._log_custom_target_balance(target_config, generated.target_metadata)
        self._validate_output(
            artifact_name="labels",
            df=labels_df,
            target_columns=target_columns,
            required_target_column=required_target_column,
        )
        return labels_df

    def _load_market_data(self, context: DatasetBuildContext) -> pd.DataFrame:
        market_df = self.market_store.read(context.symbol, context.timeframe)
        if market_df.empty:
            raise DatasetBuilderError(
                f"No stored market data found for {context.symbol}/{context.timeframe}"
            )
        market_df["timestamp"] = pd.to_datetime(market_df["timestamp"], utc=True)
        market_df = market_df.sort_values("timestamp").reset_index(drop=True)
        return market_df

    def _load_feature_table(self, context: DatasetBuildContext) -> pd.DataFrame:
        feature_df = self.feature_store.read_dataset(
            "feature_table",
            asset=context.asset,
            symbol=context.symbol,
            timeframe=context.timeframe,
            version=context.feature_version,
        )
        if feature_df.empty:
            raise DatasetBuilderError(
                "No stored merged feature table found for "
                f"{context.asset}/{context.symbol}/{context.timeframe}/{context.feature_version}"
            )
        feature_df["timestamp"] = pd.to_datetime(feature_df["timestamp"], utc=True)
        feature_df = feature_df.sort_values("timestamp").reset_index(drop=True)
        return feature_df

    def _resolve_context(
        self,
        *,
        symbol: str | None,
        timeframe: str | None,
        asset: str | None,
        feature_version: str | None = None,
        label_version: str | None = None,
        dataset_version: str | None = None,
    ) -> DatasetBuildContext:
        return DatasetBuildContext(
            asset=asset or self.default_asset,
            symbol=symbol or self.default_symbol,
            timeframe=timeframe or self.default_timeframe,
            feature_version=feature_version or self.default_feature_version,
            label_version=label_version or self.default_label_version,
            dataset_version=dataset_version or self.default_dataset_version,
        )

    def _target_columns(
        self,
        horizon_bars: int,
        *,
        target_config: TargetConfig | None = None,
    ) -> list[str]:
        if target_config is not None:
            return [
                target_config.target_name,
                f"future_return_{target_config.horizon_bars}bars",
            ]
        return get_expected_target_columns(
            primary_horizon_bars=horizon_bars,
            include_regression_target=self.include_regression_target,
            include_optional_targets=self.include_optional_targets,
            optional_horizon_bars=self.optional_horizon_bars,
        )

    def _primary_target_column(self, horizon_bars: int) -> str:
        return binary_target_column_name(horizon_bars)

    def _validate_output(
        self,
        *,
        artifact_name: DatasetArtifactName,
        df: pd.DataFrame,
        target_columns: list[str],
        required_target_column: str,
    ) -> None:
        validation = validate_dataset_df(df, target_columns=target_columns)
        if not validation.is_valid:
            raise DatasetBuilderError(
                f"{artifact_name} validation failed: {'; '.join(validation.errors)}"
            )

        if required_target_column not in target_columns:
            raise DatasetBuilderError(
                f"{artifact_name} validation failed: required target "
                f"{required_target_column} missing"
            )

    def _log_custom_target_balance(
        self,
        target_config: TargetConfig,
        target_metadata: list[dict[str, object]],
    ) -> None:
        """Log class balance and imbalance warnings for one custom target."""
        metadata = next(
            (
                row
                for row in target_metadata
                if str(row.get("target_name")) == target_config.target_name
            ),
            {},
        )
        positive_rate = float(metadata.get("positive_class_rate", 0.0) or 0.0)
        logger.info(
            "custom_target_class_balance",
            target_name=target_config.target_name,
            threshold_type=target_config.threshold_type,
            positive_class_rate=positive_rate,
            horizon_bars=target_config.horizon_bars,
        )
        if positive_rate <= 0.02 or positive_rate >= 0.98:
            logger.warning(
                "custom_target_class_balance_extreme",
                target_name=target_config.target_name,
                positive_class_rate=positive_rate,
            )
