"""Tests for optimization candidate selection and filtering."""

from __future__ import annotations

import pandas as pd

from trading_bot.optimization.selection import select_best_candidates


def test_select_best_candidates_filters_drawdown_trade_count_and_fees() -> None:
    """Selection should reject candidates that violate configured filters."""
    candidate_df = pd.DataFrame(
        [
            {
                "candidate_id": "A",
                "total_return": 0.10,
                "max_drawdown": 0.20,
                "number_of_trades": 200,
                "fee_to_starting_cash": 0.50,
            },
            {
                "candidate_id": "B",
                "total_return": 0.12,
                "max_drawdown": 0.50,
                "number_of_trades": 200,
                "fee_to_starting_cash": 0.50,
            },
            {
                "candidate_id": "C",
                "total_return": -0.01,
                "max_drawdown": 0.10,
                "number_of_trades": 50,
                "fee_to_starting_cash": 0.10,
            },
            {
                "candidate_id": "D",
                "total_return": 0.08,
                "max_drawdown": 0.10,
                "number_of_trades": 900,
                "fee_to_starting_cash": 0.10,
            },
        ]
    )

    selected = select_best_candidates(
        candidate_df,
        max_allowed_drawdown=0.35,
        max_allowed_trades=500,
        max_allowed_fee_to_starting_cash=2.0,
        minimum_total_return=0.0,
        top_k=20,
        ranking_metric="risk_adjusted_score",
    )

    assert selected["best_candidate"]["candidate_id"] == "A"
    assert selected["rejection_counts"]["drawdown"] == 1
    assert selected["rejection_counts"]["negative_return"] == 1
    assert selected["rejection_counts"]["trades"] == 1


def test_select_best_candidates_returns_empty_when_all_rejected() -> None:
    """Selection should return no best candidate when every row is rejected."""
    candidate_df = pd.DataFrame(
        [
            {
                "candidate_id": "A",
                "total_return": -0.10,
                "max_drawdown": 0.40,
                "number_of_trades": 900,
                "fee_to_starting_cash": 3.0,
            }
        ]
    )

    selected = select_best_candidates(
        candidate_df,
        max_allowed_drawdown=0.35,
        max_allowed_trades=500,
        max_allowed_fee_to_starting_cash=2.0,
        minimum_total_return=0.0,
        top_k=20,
        ranking_metric="risk_adjusted_score",
    )

    assert selected["best_candidate"] == {}
    assert selected["eligible_candidates"].empty
