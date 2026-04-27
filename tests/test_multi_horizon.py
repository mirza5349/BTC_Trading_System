"""Tests for multi-horizon research helpers and workflows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trading_bot.features.feature_store import ParquetFeatureStore
from trading_bot.labeling.dataset_builder import SupervisedDatasetBuilder
from trading_bot.labeling.labels import generate_future_return_labels
from trading_bot.research.multi_horizon import (
    build_horizon_recommendation,
    format_horizon_report_markdown,
    rank_horizon_results,
)
from trading_bot.storage.dataset_store import DatasetStore
from trading_bot.storage.horizon_store import HorizonStore
from trading_bot.storage.parquet_store import ParquetStore


def _make_market_df(row_count: int = 32) -> pd.DataFrame:
    """Create a deterministic BTC candle frame for longer-horizon tests."""
    timestamps = pd.date_range("2024-01-01", periods=row_count, freq="15min", tz="UTC")
    close = pd.Series([100.0 + (index * 0.75) + ((index % 5) - 2) for index in range(row_count)])
    return pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "timeframe": "15m",
            "timestamp": timestamps,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": pd.Series([10.0 + index for index in range(row_count)], dtype=float),
        }
    )


def _make_feature_df(row_count: int = 32) -> pd.DataFrame:
    """Create a small numeric feature table aligned to the market timeline."""
    market_df = _make_market_df(row_count)
    return pd.DataFrame(
        {
            "symbol": market_df["symbol"],
            "timeframe": market_df["timeframe"],
            "timestamp": market_df["timestamp"],
            "ret_1": pd.Series(range(row_count), dtype=float) / 100.0,
            "news_count_1h": pd.Series([index % 3 for index in range(row_count)], dtype=float),
            "news_sentiment_mean_1h": pd.Series(
                [((index % 7) - 3) / 10.0 for index in range(row_count)],
                dtype=float,
            ),
        }
    )


def _make_builder(tmp_path: Path) -> tuple[SupervisedDatasetBuilder, ParquetStore, ParquetFeatureStore]:
    """Create a horizon-aware dataset builder backed by temporary stores."""
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
        feature_version="v2_sentiment",
        label_version="v1",
        dataset_version="v1",
        primary_target="target_up_2bars",
        horizon_bars=2,
        horizon_minutes=None,
        binary_threshold=0.0,
        include_regression_target=True,
        include_optional_targets=False,
        optional_horizon_bars=[4],
    )
    return builder, market_store, feature_store


@pytest.mark.parametrize("horizon_bars", [4, 8, 16])
def test_generate_future_return_labels_supports_longer_horizons(horizon_bars: int) -> None:
    """Label generation should drop the correct tail rows for longer horizons."""
    market_df = _make_market_df()

    result = generate_future_return_labels(
        market_df,
        primary_horizon_bars=horizon_bars,
        include_regression_target=True,
        include_optional_targets=False,
        drop_unlabeled_rows=True,
    )

    assert result.shape[0] == len(market_df) - horizon_bars
    assert f"target_up_{horizon_bars}bars" in result.columns
    assert f"future_return_{horizon_bars}bars" in result.columns


def test_build_supervised_dataset_supports_horizon_aware_versions(tmp_path: Path) -> None:
    """Dataset builds should preserve requested label and dataset versions per horizon."""
    builder, market_store, feature_store = _make_builder(tmp_path)
    market_store.write(_make_market_df(), "BTCUSDT", "15m")
    feature_store.write_dataset(
        "feature_table",
        _make_feature_df(),
        asset="BTC",
        symbol="BTCUSDT",
        timeframe="15m",
        version="v2_sentiment",
    )

    dataset_df, artifact = builder.build_supervised_dataset(
        feature_version="v2_sentiment",
        label_version="v1_h8",
        dataset_version="v1_h8",
        horizon_bars=8,
    )

    assert "target_up_8bars" in dataset_df.columns
    assert "future_return_8bars" in dataset_df.columns
    assert "dataset_version=v1_h8" in artifact.path
    assert "label_version=v1_h8" in artifact.path
    assert dataset_df.shape[0] == 24


def test_rank_horizon_results_and_recommendation_select_best_viable_horizon() -> None:
    """Ranking should prefer a horizon that survives optimization filters."""
    comparison_df = pd.DataFrame(
        [
            {
                "horizon_bars": 4,
                "optimized_has_edge": False,
                "best_strategy_total_return": -0.02,
                "best_strategy_max_drawdown": 0.10,
                "best_strategy_number_of_trades": 150,
                "best_strategy_fee_impact": 5.0,
                "best_strategy_fee_to_starting_cash": 0.05,
                "roc_auc_mean": 0.54,
            },
            {
                "horizon_bars": 8,
                "optimized_has_edge": True,
                "best_strategy_total_return": 0.08,
                "best_strategy_max_drawdown": 0.20,
                "best_strategy_number_of_trades": 120,
                "best_strategy_fee_impact": 3.0,
                "best_strategy_fee_to_starting_cash": 0.03,
                "roc_auc_mean": 0.56,
            },
            {
                "horizon_bars": 16,
                "optimized_has_edge": True,
                "best_strategy_total_return": 0.03,
                "best_strategy_max_drawdown": 0.40,
                "best_strategy_number_of_trades": 90,
                "best_strategy_fee_impact": 2.0,
                "best_strategy_fee_to_starting_cash": 0.02,
                "roc_auc_mean": 0.58,
            },
        ]
    )

    ranked_df = rank_horizon_results(
        comparison_df,
        max_allowed_drawdown=0.35,
        max_allowed_trades=500,
        max_allowed_fee_to_starting_cash=2.0,
    )
    recommendation = build_horizon_recommendation(ranked_df)

    assert int(ranked_df.iloc[0]["horizon_bars"]) == 8
    assert bool(ranked_df.iloc[0]["passes_filters"]) is True
    assert recommendation["best_horizon_bars"] == 8
    assert recommendation["no_viable_horizon"] is False


def test_horizon_store_round_trip_and_markdown(tmp_path: Path) -> None:
    """Horizon comparison artifacts should round-trip through the local store."""
    store = HorizonStore(tmp_path / "reports" / "horizon_comparison")
    comparison_df = pd.DataFrame([{"horizon_bars": 8, "optimized_has_edge": True}])
    report = {
        "comparison_id": "HCOMP001",
        "asset": "BTC",
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "feature_version": "v2_sentiment",
        "generated_at": "2026-04-26T00:00:00+00:00",
        "horizons_tested": [4, 8, 16],
        "rows": comparison_df.to_dict(orient="records"),
        "recommendation": {
            "best_horizon_bars": 8,
            "no_viable_horizon": False,
            "reason": "The 8-bar horizon ranked highest.",
        },
    }

    store.write_comparison_table("HCOMP001", comparison_df)
    store.write_report_json("HCOMP001", report)
    markdown = format_horizon_report_markdown(report)
    store.write_report_markdown("HCOMP001", markdown)

    loaded_df = store.read_comparison_table("HCOMP001")
    loaded_report = store.read_report_json("HCOMP001")

    assert int(loaded_df.iloc[0]["horizon_bars"]) == 8
    assert loaded_report["recommendation"]["best_horizon_bars"] == 8
    assert "best_horizon_bars" in markdown
