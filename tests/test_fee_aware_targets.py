"""Tests for fee-aware target research helpers and storage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_bot.research.fee_aware_targets import (
    build_target_configs,
    build_target_recommendation,
    format_target_report_markdown,
    rank_target_results,
)
from trading_bot.settings import load_settings
from trading_bot.storage.target_store import TargetStore


def test_build_target_configs_exposes_expected_aliases() -> None:
    """Configured fee-aware target aliases should resolve deterministically."""
    settings = load_settings()
    configs = build_target_configs(settings, target_keys=["fee_8", "ret050_16"])

    assert [config.key for config in configs] == ["fee_8", "ret050_16"]
    assert configs[0].target_name == "target_long_net_positive_8bars"
    assert configs[1].threshold_return == 0.005


def test_rank_target_results_prefers_viable_target() -> None:
    """Ranking should prefer a target that passes edge, fee, drawdown, trade, and class filters."""
    comparison_df = pd.DataFrame(
        [
            {
                "target_key": "fee_4",
                "target_name": "target_long_net_positive_4bars",
                "positive_class_rate": 0.50,
                "optimized_has_edge": False,
                "best_strategy_total_return": -0.01,
                "best_strategy_max_drawdown": 0.10,
                "best_strategy_number_of_trades": 150,
                "best_strategy_fee_impact": 5.0,
                "best_strategy_fee_to_starting_cash": 0.05,
                "roc_auc_mean": 0.53,
            },
            {
                "target_key": "ret050_16",
                "target_name": "target_return_gt_050pct_16bars",
                "positive_class_rate": 0.22,
                "optimized_has_edge": True,
                "best_strategy_total_return": 0.09,
                "best_strategy_max_drawdown": 0.18,
                "best_strategy_number_of_trades": 120,
                "best_strategy_fee_impact": 2.5,
                "best_strategy_fee_to_starting_cash": 0.025,
                "roc_auc_mean": 0.57,
            },
        ]
    )

    ranked_df = rank_target_results(
        comparison_df,
        max_allowed_drawdown=0.35,
        max_allowed_trades=500,
        max_allowed_fee_to_starting_cash=2.0,
        positive_class_rate_min=0.02,
        positive_class_rate_max=0.98,
    )
    recommendation = build_target_recommendation(ranked_df)

    assert ranked_df.iloc[0]["target_key"] == "ret050_16"
    assert bool(ranked_df.iloc[0]["passes_filters"]) is True
    assert ranked_df.iloc[0]["rejection_reason"] == "eligible"
    assert recommendation["best_target_name"] == "target_return_gt_050pct_16bars"
    assert recommendation["no_viable_target"] is False
    assert recommendation["validation_has_edge_required"] is False


def test_rank_target_results_explains_exact_fee_aware_vs_threshold_case() -> None:
    """When multiple targets pass optimization, the best one should be selected or clearly explained."""
    comparison_df = pd.DataFrame(
        [
            {
                "target_key": "fee_4",
                "target_name": "target_long_net_positive_4bars",
                "feature_version": "v1",
                "positive_class_rate": 0.18,
                "validation_has_edge": False,
                "optimized_has_edge": True,
                "best_strategy_total_return": 0.0905997,
                "best_strategy_max_drawdown": 0.18,
                "best_strategy_number_of_trades": 36,
                "best_strategy_fee_impact": 7.48140241,
                "best_strategy_fee_to_starting_cash": 0.0748140241,
                "roc_auc_mean": 0.53,
            },
            {
                "target_key": "ret075_16",
                "target_name": "target_return_gt_075pct_16bars",
                "feature_version": "v1",
                "positive_class_rate": 0.10,
                "validation_has_edge": False,
                "optimized_has_edge": True,
                "best_strategy_total_return": 0.05303103,
                "best_strategy_max_drawdown": 0.24,
                "best_strategy_number_of_trades": 112,
                "best_strategy_fee_impact": 11.58945882,
                "best_strategy_fee_to_starting_cash": 0.1158945882,
                "roc_auc_mean": 0.55,
            },
        ]
    )

    ranked_df = rank_target_results(
        comparison_df,
        max_allowed_drawdown=0.35,
        max_allowed_trades=500,
        max_allowed_fee_to_starting_cash=2.0,
        positive_class_rate_min=0.02,
        positive_class_rate_max=0.98,
    )
    recommendation = build_target_recommendation(ranked_df)

    assert ranked_df.iloc[0]["target_key"] == "fee_4"
    assert bool(ranked_df.iloc[0]["optimized_has_edge_passed"]) is True
    assert bool(ranked_df.iloc[0]["return_filter_passed"]) is True
    assert bool(ranked_df.iloc[0]["drawdown_filter_passed"]) is True
    assert bool(ranked_df.iloc[0]["trade_count_filter_passed"]) is True
    assert bool(ranked_df.iloc[0]["fee_filter_passed"]) is True
    assert bool(ranked_df.iloc[0]["class_balance_filter_passed"]) is True
    assert bool(ranked_df.iloc[0]["passes_filters"]) is True
    assert ranked_df.iloc[0]["rejection_reason"] == "eligible"
    assert recommendation["best_target_name"] == "target_long_net_positive_4bars"
    assert recommendation["no_viable_target"] is False


def test_target_store_round_trip_and_markdown(tmp_path: Path) -> None:
    """Target comparison artifacts should round-trip through the local store."""
    store = TargetStore(tmp_path / "reports" / "target_comparison")
    comparison_df = pd.DataFrame([{"target_key": "fee_8", "optimized_has_edge": True}])
    report = {
        "comparison_id": "TCOMP001",
        "asset": "BTC",
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "feature_version": "v1",
        "feature_versions": ["v1"],
        "generated_at": "2026-04-26T00:00:00+00:00",
        "target_keys_tested": ["fee_8"],
        "rows": comparison_df.to_dict(orient="records"),
        "recommendation": {
            "best_target_name": "target_long_net_positive_8bars",
            "no_viable_target": False,
            "reason": "This target ranked highest.",
        },
    }

    store.write_comparison_table("TCOMP001", comparison_df)
    store.write_report_json("TCOMP001", report)
    markdown = format_target_report_markdown(report)
    store.write_report_markdown("TCOMP001", markdown)

    loaded_df = store.read_comparison_table("TCOMP001")
    loaded_report = store.read_report_json("TCOMP001")

    assert loaded_df.iloc[0]["target_key"] == "fee_8"
    assert loaded_report["recommendation"]["best_target_name"] == "target_long_net_positive_8bars"
    assert "best_target_name" in markdown
