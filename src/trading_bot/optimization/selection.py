"""Candidate filtering and selection logic for strategy optimization."""

from __future__ import annotations

from typing import Any

import pandas as pd


def enrich_candidate_metrics(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """Add derived ranking metrics used during candidate selection."""
    if candidate_df.empty:
        return candidate_df.copy()

    working_df = candidate_df.copy()
    working_df["risk_adjusted_score"] = (
        pd.to_numeric(working_df.get("total_return"), errors="coerce").fillna(-1.0)
        - pd.to_numeric(working_df.get("max_drawdown"), errors="coerce").fillna(0.0) * 0.75
        - pd.to_numeric(working_df.get("fee_to_starting_cash"), errors="coerce").fillna(0.0) * 0.15
        - (
            pd.to_numeric(working_df.get("number_of_trades"), errors="coerce").fillna(0.0) / 10000.0
        )
    )
    return working_df


def select_best_candidates(
    candidate_df: pd.DataFrame,
    *,
    max_allowed_drawdown: float,
    max_allowed_trades: int,
    max_allowed_fee_to_starting_cash: float,
    minimum_total_return: float,
    top_k: int,
    ranking_metric: str,
) -> dict[str, Any]:
    """Filter, rank, and return top strategy-search candidates."""
    if candidate_df.empty:
        return {
            "eligible_candidates": pd.DataFrame(),
            "top_candidates": pd.DataFrame(),
            "best_candidate": {},
            "rejection_counts": {},
        }

    working_df = enrich_candidate_metrics(candidate_df)
    rejection_counts = {
        "drawdown": int((working_df["max_drawdown"] > max_allowed_drawdown).sum()),
        "trades": int((working_df["number_of_trades"] > max_allowed_trades).sum()),
        "fees": int((working_df["fee_to_starting_cash"] > max_allowed_fee_to_starting_cash).sum()),
        "negative_return": int((working_df["total_return"] < minimum_total_return).sum()),
    }

    eligible_df = working_df[
        (working_df["max_drawdown"] <= max_allowed_drawdown)
        & (working_df["number_of_trades"] <= max_allowed_trades)
        & (working_df["fee_to_starting_cash"] <= max_allowed_fee_to_starting_cash)
        & (working_df["total_return"] >= minimum_total_return)
    ].copy()

    if eligible_df.empty:
        return {
            "eligible_candidates": eligible_df,
            "top_candidates": eligible_df,
            "best_candidate": {},
            "rejection_counts": rejection_counts,
        }

    sort_by = ranking_metric if ranking_metric in eligible_df.columns else "risk_adjusted_score"
    ascending = sort_by in {"max_drawdown", "number_of_trades", "fee_to_starting_cash"}
    eligible_df = eligible_df.sort_values(
        by=[sort_by, "total_return"],
        ascending=[ascending, False],
        na_position="last",
    ).reset_index(drop=True)
    top_candidates_df = eligible_df.head(top_k).reset_index(drop=True)
    best_candidate = top_candidates_df.iloc[0].to_dict() if not top_candidates_df.empty else {}

    return {
        "eligible_candidates": eligible_df,
        "top_candidates": top_candidates_df,
        "best_candidate": best_candidate,
        "rejection_counts": rejection_counts,
    }
