"""Tests for supervised dataset assembly."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trading_bot.features.feature_store import ParquetFeatureStore
from trading_bot.labeling.dataset_builder import DatasetBuilderError, SupervisedDatasetBuilder
from trading_bot.schemas.targets import TargetConfig
from trading_bot.storage.dataset_store import DatasetStore
from trading_bot.storage.parquet_store import ParquetStore


def _make_market_df() -> pd.DataFrame:
    """Create deterministic BTC candles for dataset builder tests."""
    timestamps = pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC")
    close = pd.Series([100.0, 101.0, 102.0, 99.0, 100.0, 104.0, 103.0, 107.0])
    return pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "timeframe": "15m",
            "timestamp": timestamps,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": pd.Series([10, 11, 12, 13, 14, 15, 16, 17], dtype=float),
        }
    )


def _make_feature_df() -> pd.DataFrame:
    """Create a small merged feature table aligned to the market timeline."""
    market_df = _make_market_df()
    return pd.DataFrame(
        {
            "symbol": market_df["symbol"],
            "timeframe": market_df["timeframe"],
            "timestamp": market_df["timestamp"],
            "ret_1": [None, 0.01, 0.0099, -0.0294, 0.0101, 0.04, -0.0096, 0.0388],
            "news_count_1h": [0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0],
        }
    )


def _make_builder(
    tmp_path: Path,
    *,
    include_optional_targets: bool = False,
) -> tuple[SupervisedDatasetBuilder, ParquetStore, ParquetFeatureStore, DatasetStore]:
    """Create a dataset builder backed by temporary local stores."""
    market_store = ParquetStore(tmp_path / "market")
    feature_store = ParquetFeatureStore(tmp_path / "features")
    dataset_store = DatasetStore(tmp_path / "datasets")

    builder = SupervisedDatasetBuilder(
        market_store=market_store,
        feature_store=feature_store,
        dataset_store=dataset_store,
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        feature_version="v1",
        label_version="v1",
        dataset_version="v1",
        primary_target="target_up_2bars",
        horizon_bars=2,
        horizon_minutes=None,
        binary_threshold=0.0,
        include_regression_target=True,
        include_optional_targets=include_optional_targets,
        optional_horizon_bars=[4],
    )
    return builder, market_store, feature_store, dataset_store


def test_build_supervised_dataset_merges_features_and_labels_and_persists(tmp_path: Path) -> None:
    """The final supervised dataset should preserve features and append targets."""
    builder, market_store, feature_store, dataset_store = _make_builder(tmp_path)
    market_store.write(_make_market_df(), "BTCUSDT", "15m")
    feature_store.write_dataset(
        "feature_table",
        _make_feature_df(),
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        version="v1",
    )

    dataset_df, artifact = builder.build_supervised_dataset()

    assert dataset_df.shape[0] == 6
    assert dataset_df["timestamp"].is_monotonic_increasing
    assert list(dataset_df.columns) == [
        "symbol",
        "timeframe",
        "timestamp",
        "ret_1",
        "news_count_1h",
        "target_up_2bars",
        "future_return_2bars",
    ]
    assert dataset_df.iloc[0]["ret_1"] != dataset_df.iloc[1]["ret_1"]
    assert dataset_df.iloc[0]["target_up_2bars"] == 1

    expected_dataset_path = (
        tmp_path
        / "datasets"
        / "asset=BTC"
        / "symbol=BTCUSDT"
        / "timeframe=15m"
        / "label_version=v1"
        / "dataset_version=v1"
        / "dataset.parquet"
    )
    expected_labels_path = (
        tmp_path
        / "datasets"
        / "asset=BTC"
        / "symbol=BTCUSDT"
        / "timeframe=15m"
        / "label_version=v1"
        / "labels.parquet"
    )

    assert Path(artifact.path) == expected_dataset_path
    assert expected_dataset_path.exists()
    assert expected_labels_path.exists()
    assert dataset_store.exists(
        "supervised_dataset",
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        label_version="v1",
        dataset_version="v1",
    )

    info = builder.inspect_dataset("supervised_dataset")
    assert info["row_count"] == 6
    assert info["target_columns"] == ["target_up_2bars", "future_return_2bars"]


def test_build_labels_with_optional_targets_drops_tail_rows_for_longer_horizon(
    tmp_path: Path,
) -> None:
    """Including the optional 4-bar target should shorten the clean label table."""
    builder, market_store, feature_store, _ = _make_builder(
        tmp_path,
        include_optional_targets=True,
    )
    market_store.write(_make_market_df(), "BTCUSDT", "15m")
    feature_store.write_dataset(
        "feature_table",
        _make_feature_df(),
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        version="v1",
    )

    labels_df, _ = builder.build_labels()

    assert labels_df.shape[0] == 4
    assert "target_up_4bars" in labels_df.columns
    assert "future_log_return_2bars" in labels_df.columns


def test_build_supervised_dataset_requires_feature_table(tmp_path: Path) -> None:
    """Dataset assembly should fail loudly when the merged feature table is missing."""
    builder, market_store, _, _ = _make_builder(tmp_path)
    market_store.write(_make_market_df(), "BTCUSDT", "15m")

    with pytest.raises(DatasetBuilderError, match="No stored merged feature table found"):
        builder.build_supervised_dataset()


def test_build_supervised_dataset_uses_requested_feature_version(tmp_path: Path) -> None:
    """Dataset rebuild should load the explicitly requested feature version."""
    builder, market_store, feature_store, _ = _make_builder(tmp_path)
    market_store.write(_make_market_df(), "BTCUSDT", "15m")

    sentiment_feature_df = _make_feature_df().copy()
    sentiment_feature_df["news_sentiment_mean_1h"] = [0.0, 0.1, 0.2, -0.1, 0.05, 0.3, -0.2, 0.4]
    feature_store.write_dataset(
        "feature_table",
        sentiment_feature_df,
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        version="v2_sentiment",
    )

    dataset_df, artifact = builder.build_supervised_dataset(
        feature_version="v2_sentiment",
        dataset_version="v2_sentiment",
    )

    assert "news_sentiment_mean_1h" in dataset_df.columns
    assert "dataset_version=v2_sentiment" in artifact.path


def test_build_supervised_dataset_supports_fee_aware_target_config(tmp_path: Path) -> None:
    """Dataset assembly should support a custom fee-aware target column and metadata."""
    builder, market_store, feature_store, _ = _make_builder(tmp_path)
    market_store.write(_make_market_df(), "BTCUSDT", "15m")
    feature_store.write_dataset(
        "feature_table",
        _make_feature_df(),
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        version="v2_sentiment",
    )

    target_config = TargetConfig(
        key="fee_4",
        target_name="target_long_net_positive_4bars",
        horizon_bars=4,
        threshold_type="fee_aware",
        label_version="fee_h4",
        dataset_version="fee_h4",
        model_version="fee_h4",
        fee_rate=0.001,
        slippage_rate=0.0005,
        round_trip_cost=0.003,
    )

    dataset_df, artifact = builder.build_supervised_dataset(
        feature_version="v2_sentiment",
        label_version="fee_h4",
        dataset_version="fee_h4",
        horizon_bars=4,
        target_config=target_config,
    )

    assert "target_long_net_positive_4bars" in dataset_df.columns
    assert "future_return_4bars" in dataset_df.columns
    assert "dataset_version=fee_h4" in artifact.path
