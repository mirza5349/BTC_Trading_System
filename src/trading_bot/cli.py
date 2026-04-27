"""CLI entrypoint for the BTC ML Trading Bot.

Provides diagnostic, utility, and data ingestion commands.
"""

from __future__ import annotations

import importlib
import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from trading_bot import __version__
from trading_bot.logging_config import get_logger, setup_logging
from trading_bot.settings import PROJECT_ROOT, load_settings

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - Python < 3.11 compatibility
    from datetime import timezone

    UTC = timezone.utc  # noqa: UP017

console = Console()
logger = get_logger(__name__)

# All subpackages that must be importable
_EXPECTED_MODULES = [
    "trading_bot",
    "trading_bot.settings",
    "trading_bot.logging_config",
    "trading_bot.cli",
    "trading_bot.main",
    "trading_bot.schemas",
    "trading_bot.schemas.market_data",
    "trading_bot.schemas.features",
    "trading_bot.schemas.datasets",
    "trading_bot.schemas.modeling",
    "trading_bot.schemas.experiments",
    "trading_bot.schemas.nlp",
    "trading_bot.ingestion",
    "trading_bot.ingestion.binance_client",
    "trading_bot.ingestion.market_data",
    "trading_bot.ingestion.news_client",
    "trading_bot.ingestion.news_data",
    "trading_bot.storage",
    "trading_bot.storage.parquet_store",
    "trading_bot.storage.news_store",
    "trading_bot.storage.dataset_store",
    "trading_bot.storage.model_store",
    "trading_bot.storage.evaluation_store",
    "trading_bot.storage.backtest_store",
    "trading_bot.storage.paper_sim_store",
    "trading_bot.storage.experiment_store",
    "trading_bot.storage.report_store",
    "trading_bot.storage.validation_store",
    "trading_bot.storage.optimization_store",
    "trading_bot.storage.horizon_store",
    "trading_bot.storage.target_store",
    "trading_bot.storage.enriched_news_store",
    "trading_bot.storage.paper_loop_store",
    "trading_bot.features",
    "trading_bot.features.market_features",
    "trading_bot.features.news_features",
    "trading_bot.features.feature_store",
    "trading_bot.features.feature_pipeline",
    "trading_bot.labeling",
    "trading_bot.labeling.labels",
    "trading_bot.labeling.dataset_builder",
    "trading_bot.models",
    "trading_bot.models.train",
    "trading_bot.models.predict",
    "trading_bot.models.model_registry",
    "trading_bot.models.tuning",
    "trading_bot.nlp",
    "trading_bot.nlp.finbert_service",
    "trading_bot.nlp.text_preprocessing",
    "trading_bot.nlp.news_enrichment",
    "trading_bot.experiments",
    "trading_bot.experiments.tracker",
    "trading_bot.experiments.registry",
    "trading_bot.experiments.comparison",
    "trading_bot.experiments.promotion",
    "trading_bot.optimization",
    "trading_bot.optimization.search_space",
    "trading_bot.optimization.selection",
    "trading_bot.optimization.reporting",
    "trading_bot.optimization.strategy_search",
    "trading_bot.research",
    "trading_bot.research.fee_aware_targets",
    "trading_bot.research.multi_horizon",
    "trading_bot.paper",
    "trading_bot.paper.loop",
    "trading_bot.paper.daemon",
    "trading_bot.validation",
    "trading_bot.validation.walk_forward",
    "trading_bot.validation.metrics",
    "trading_bot.validation.evaluation_report",
    "trading_bot.validation.baselines",
    "trading_bot.validation.suite",
    "trading_bot.backtest",
    "trading_bot.backtest.signals",
    "trading_bot.backtest.execution",
    "trading_bot.backtest.metrics",
    "trading_bot.backtest.reporting",
    "trading_bot.backtest.strategy",
    "trading_bot.backtest.simulator",
    "trading_bot.execution",
    "trading_bot.execution.signal_engine",
    "trading_bot.execution.order_manager",
    "trading_bot.execution.broker_adapter",
    "trading_bot.monitoring",
    "trading_bot.monitoring.logger",
    "trading_bot.monitoring.alerts",
    "trading_bot.schemas.news_data",
    "trading_bot.schemas.backtest",
    "trading_bot.schemas.validation",
    "trading_bot.schemas.optimization",
    "trading_bot.schemas.horizons",
    "trading_bot.schemas.targets",
]

_FEATURE_DATASET_NAME_MAP = {
    "market": "market_features",
    "news": "news_features",
    "merged": "feature_table",
}

_DATASET_ARTIFACT_NAME_MAP = {
    "labels": "labels",
    "supervised": "supervised_dataset",
}


@click.group()
@click.version_option(version=__version__, prog_name="trading-bot")
def cli() -> None:
    """BTC ML Trading Bot — CLI tools."""


def _print_feature_build_success(title: str, feature_set) -> None:
    """Render a consistent success panel for feature builds."""
    console.print(
        Panel(
            f"[bold green]{title} complete[/bold green]\n"
            f"  Rows:     {feature_set.row_count:,}\n"
            f"  Columns:  {feature_set.column_count:,}\n"
            f"  Range:    {feature_set.start} → {feature_set.end}\n"
            f"  Path:     {feature_set.path}",
            title="✅ Success",
            style="green",
        )
    )


def _print_feature_stats(info: dict[str, object], dataset_label: str) -> None:
    """Render stored feature dataset metadata."""
    table = Table(title=f"🧪 Feature Dataset Stats — {dataset_label}")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="white")

    table.add_row("Dataset", str(info["dataset_name"]))
    table.add_row("Asset", str(info["asset"]))
    table.add_row("Symbol", str(info["symbol"]))
    table.add_row("Timeframe", str(info["timeframe"]))
    table.add_row("Version", str(info["version"]))
    table.add_row("Row Count", f"{info['row_count']:,}")
    table.add_row("Column Count", f"{info['column_count']:,}")
    table.add_row("First Timestamp", str(info["min_timestamp"]))
    table.add_row("Last Timestamp", str(info["max_timestamp"]))
    table.add_row("Path", str(info["path"]))
    table.add_row("Columns", ", ".join(info["columns"]))

    console.print(table)


def _print_dataset_build_success(title: str, artifact) -> None:
    """Render a consistent success panel for label and dataset builds."""
    console.print(
        Panel(
            f"[bold green]{title} complete[/bold green]\n"
            f"  Rows:     {artifact.row_count:,}\n"
            f"  Columns:  {artifact.column_count:,}\n"
            f"  Range:    {artifact.start} → {artifact.end}\n"
            f"  Path:     {artifact.path}",
            title="✅ Success",
            style="green",
        )
    )


def _print_dataset_stats(info: dict[str, object], artifact_label: str) -> None:
    """Render stored label or supervised dataset metadata."""
    table = Table(title=f"🎯 Dataset Stats — {artifact_label}")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="white")

    table.add_row("Artifact", str(info["artifact_name"]))
    table.add_row("Asset", str(info["asset"]))
    table.add_row("Symbol", str(info["symbol"]))
    table.add_row("Timeframe", str(info["timeframe"]))
    table.add_row("Label Version", str(info["label_version"]))
    table.add_row("Dataset Version", str(info["dataset_version"]))
    table.add_row("Row Count", f"{info['row_count']:,}")
    table.add_row("Column Count", f"{info['column_count']:,}")
    table.add_row("Feature Columns", f"{len(info['feature_columns']):,}")
    table.add_row("Target Columns", f"{len(info['target_columns']):,}")
    table.add_row("First Timestamp", str(info["min_timestamp"]))
    table.add_row("Last Timestamp", str(info["max_timestamp"]))
    table.add_row("Path", str(info["path"]))

    target_summary = info.get("target_summary", {})
    if isinstance(target_summary, dict):
        for column, summary in target_summary.items():
            table.add_row(column, json.dumps(summary, sort_keys=True))

    console.print(table)


def _print_training_run_success(result) -> None:
    """Render a consistent success panel for walk-forward training runs."""
    accuracy = result.aggregate_metrics.get("accuracy")
    roc_auc = result.aggregate_metrics.get("roc_auc")
    console.print(
        Panel(
            f"[bold green]Walk-forward training complete[/bold green]\n"
            f"  Run ID:           {result.run_id}\n"
            f"  Target:           {result.target_column}\n"
            f"  Features:         {len(result.feature_columns):,}\n"
            f"  Folds:            {len(result.fold_metrics):,}\n"
            f"  Requested Device: {result.requested_device}\n"
            f"  Effective Device: {result.effective_device}\n"
            f"  Accuracy:         {accuracy if accuracy is not None else 'n/a'}\n"
            f"  ROC AUC:          {roc_auc if roc_auc is not None else 'n/a'}\n"
            f"  Models Dir:       {result.models_dir}\n"
            f"  Eval Dir:         {result.evaluation_dir}",
            title="✅ Success",
            style="green",
        )
    )


def _print_run_metrics(
    report: dict[str, object],
    aggregate_metrics: dict[str, object],
    fold_metrics_df,
) -> None:
    """Render stored aggregate and per-fold run metrics."""
    summary = Table(title=f"📈 Run Metrics — {report.get('run_id', 'unknown')}")
    summary.add_column("Metric", style="bold cyan")
    summary.add_column("Value", style="white")

    for key in (
        "symbol",
        "timeframe",
        "dataset_version",
        "label_version",
        "model_version",
        "target_column",
        "feature_count",
        "fold_count",
        "requested_device",
        "effective_device",
        "lightgbm_version",
        "validation_start",
        "validation_end",
    ):
        summary.add_row(key, str(report.get(key)))

    for metric_name, metric_value in aggregate_metrics.items():
        summary.add_row(f"aggregate_{metric_name}", str(metric_value))

    console.print(summary)

    if fold_metrics_df.empty:
        return

    fold_table = Table(title="Per-Fold Metrics")
    for column in (
        "fold_id",
        "n_train",
        "n_validation",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "log_loss",
        "tp",
        "tn",
        "fp",
        "fn",
    ):
        fold_table.add_column(column, style="white")

    for _, row in fold_metrics_df.iterrows():
        fold_table.add_row(
            *[
                str(row.get(column))
                for column in (
                    "fold_id",
                    "n_train",
                    "n_validation",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "log_loss",
                    "tp",
                    "tn",
                    "fp",
                    "fn",
                )
            ]
        )

    console.print(fold_table)


def _print_model_runs(runs: list[dict[str, object]]) -> None:
    """Render a list of stored model runs."""
    table = Table(title="🗂 Stored Model Runs")
    for column in (
        "run_id",
        "symbol",
        "timeframe",
        "model_version",
        "target_column",
        "fold_count",
        "requested_device",
        "effective_device",
        "created_at",
    ):
        table.add_column(column, style="white")

    for run in runs:
        table.add_row(
            str(run.get("run_id")),
            str(run.get("symbol")),
            str(run.get("timeframe")),
            str(run.get("model_version")),
            str(run.get("target_column")),
            str(run.get("fold_count")),
            str(run.get("requested_device")),
            str(run.get("effective_device")),
            str(run.get("created_at")),
        )

    console.print(table)


def _print_feature_importance(run_id: str, feature_importance_df, top_k: int) -> None:
    """Render the top aggregate feature importance rows for a stored run."""
    table = Table(title=f"🌟 Feature Importance — {run_id}")
    table.add_column("Rank", style="bold cyan")
    table.add_column("Feature", style="white")
    table.add_column("Mean Importance", style="white")
    table.add_column("Folds", style="white")

    head_df = feature_importance_df.head(top_k).reset_index(drop=True)
    for index, row in head_df.iterrows():
        table.add_row(
            str(index + 1),
            str(row.get("feature_name")),
            str(row.get("mean_importance")),
            str(row.get("folds_present")),
        )

    console.print(table)


def _print_enriched_news_stats(info: dict[str, object]) -> None:
    """Render enriched-news coverage and sentiment distribution stats."""
    table = Table(title="🧠 Enriched News Stats")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="white")

    for key in (
        "asset",
        "provider",
        "model_name",
        "enrichment_version",
        "row_count",
        "min_published_at",
        "max_published_at",
        "scoring_coverage",
        "path",
    ):
        if key in info:
            table.add_row(key, str(info[key]))

    label_counts = info.get("sentiment_label_counts", {})
    if isinstance(label_counts, dict):
        for label, count in label_counts.items():
            table.add_row(f"label_{label}", str(count))

    console.print(table)


def _print_experiment_runs(registry_df) -> None:
    """Render tracked experiment runs with stable key metrics."""
    table = Table(title="🧪 Experiment Runs")
    for column in (
        "run_id",
        "dataset_version",
        "feature_version",
        "model_version",
        "target_column",
        "effective_device",
        "fold_count",
        "feature_count",
        "roc_auc_mean",
        "f1_mean",
        "log_loss_mean",
        "status",
    ):
        table.add_column(column, style="white")

    for _, row in registry_df.iterrows():
        table.add_row(
            *[
                str(row.get(column))
                for column in (
                    "run_id",
                    "dataset_version",
                    "feature_version",
                    "model_version",
                    "target_column",
                    "effective_device",
                    "fold_count",
                    "feature_count",
                    "roc_auc_mean",
                    "f1_mean",
                    "log_loss_mean",
                    "status",
                )
            ]
        )

    console.print(table)


def _print_run_report(report: dict[str, object]) -> None:
    """Render the main sections of a stored evaluation report."""
    metadata_table = Table(title=f"🧾 Run Report — {report.get('run_id', 'unknown')}")
    metadata_table.add_column("Field", style="bold cyan")
    metadata_table.add_column("Value", style="white")

    for key in (
        "symbol",
        "timeframe",
        "dataset_version",
        "label_version",
        "model_version",
        "target_column",
        "feature_count",
        "fold_count",
        "requested_device",
        "effective_device",
        "status",
    ):
        metadata_table.add_row(key, str(report.get(key)))

    console.print(metadata_table)

    aggregate_metrics = report.get("aggregate_metrics", {})
    if isinstance(aggregate_metrics, dict):
        metrics_table = Table(title="Aggregate Metrics")
        metrics_table.add_column("Metric", style="bold cyan")
        metrics_table.add_column("Value", style="white")
        for key, value in aggregate_metrics.items():
            metrics_table.add_row(str(key), str(value))
        console.print(metrics_table)

    prediction_summary = report.get("prediction_probability_summary", {})
    if isinstance(prediction_summary, dict):
        prediction_table = Table(title="Prediction Summary")
        prediction_table.add_column("Metric", style="bold cyan")
        prediction_table.add_column("Value", style="white")
        for key, value in prediction_summary.items():
            if key == "confidence_buckets":
                continue
            prediction_table.add_row(str(key), str(value))
        console.print(prediction_table)

    feature_summary = report.get("feature_importance_summary", {})
    if isinstance(feature_summary, dict):
        top_features = feature_summary.get("top_features", [])
        if isinstance(top_features, list) and top_features:
            feature_table = Table(title="Top Features")
            feature_table.add_column("Feature", style="bold cyan")
            feature_table.add_column("Importance", style="white")
            feature_table.add_column("Rank", style="white")
            for row in top_features:
                if not isinstance(row, dict):
                    continue
                feature_table.add_row(
                    str(row.get("feature_name")),
                    str(row.get("importance", row.get("mean_importance"))),
                    str(row.get("rank")),
                )
            console.print(feature_table)


def _print_run_comparison(comparison_df) -> None:
    """Render a comparison table for two or more runs."""
    table = Table(title="📊 Run Comparison")
    for column in comparison_df.columns:
        table.add_column(str(column), style="white")

    for _, row in comparison_df.iterrows():
        table.add_row(*[str(row.get(column)) for column in comparison_df.columns])

    console.print(table)


def _print_sentiment_ablation(summary: dict[str, object]) -> None:
    """Render a compact baseline-vs-sentiment ablation summary."""
    overview = Table(title="🧪 Sentiment Ablation")
    overview.add_column("Field", style="bold cyan")
    overview.add_column("Value", style="white")

    for key in ("baseline_run_id", "new_run_id"):
        overview.add_row(key, str(summary.get(key)))
    console.print(overview)

    metric_deltas = summary.get("metric_deltas", {})
    metric_improvements = summary.get("metric_improvements", {})
    if isinstance(metric_deltas, dict):
        delta_table = Table(title="Metric Deltas")
        delta_table.add_column("Metric", style="bold cyan")
        delta_table.add_column("Delta", style="white")
        delta_table.add_column("Status", style="white")
        for metric_name, delta in metric_deltas.items():
            status = (
                metric_improvements.get(metric_name, "unknown")
                if isinstance(metric_improvements, dict)
                else "unknown"
            )
            delta_table.add_row(str(metric_name), str(delta), str(status))
        console.print(delta_table)


def _print_simulation_success(result) -> None:
    """Render a consistent success panel for a stored simulation."""
    metrics = result.metrics.to_dict() if hasattr(result.metrics, "to_dict") else {}
    console.print(
        Panel(
            f"[bold green]{result.simulation_type.replace('_', ' ').title()} complete[/bold green]\n"
            f"  Simulation ID:    {result.simulation_id}\n"
            f"  Model Run ID:     {result.model_run_id}\n"
            f"  Strategy Version: {result.strategy_version}\n"
            f"  Total Return:     {metrics.get('total_return', 'n/a')}\n"
            f"  Max Drawdown:     {metrics.get('max_drawdown', 'n/a')}\n"
            f"  Trades:           {metrics.get('number_of_trades', 'n/a')}\n"
            f"  Ending Equity:    {metrics.get('ending_equity', 'n/a')}",
            title="✅ Success",
            style="green",
        )
    )


def _print_simulation_summary(summary: dict[str, object]) -> None:
    """Render the main fields from one stored simulation summary."""
    metrics = summary.get("metrics", {})
    config = summary.get("config", {})

    overview = Table(title=f"📘 Simulation Summary — {summary.get('simulation_id', 'unknown')}")
    overview.add_column("Field", style="bold cyan")
    overview.add_column("Value", style="white")
    for key in (
        "simulation_type",
        "model_run_id",
        "symbol",
        "timeframe",
        "strategy_version",
        "start_timestamp",
        "end_timestamp",
    ):
        overview.add_row(key, str(summary.get(key)))
    console.print(overview)

    if isinstance(metrics, dict):
        metrics_table = Table(title="Simulation Metrics")
        metrics_table.add_column("Metric", style="bold cyan")
        metrics_table.add_column("Value", style="white")
        for key, value in metrics.items():
            metrics_table.add_row(str(key), str(value))
        console.print(metrics_table)

    if isinstance(config, dict):
        strategy_cfg = config.get("strategy", {})
        execution_cfg = config.get("execution", {})
        params_table = Table(title="Simulation Parameters")
        params_table.add_column("Parameter", style="bold cyan")
        params_table.add_column("Value", style="white")
        if isinstance(strategy_cfg, dict):
            for key, value in strategy_cfg.items():
                params_table.add_row(f"strategy.{key}", str(value))
        if isinstance(execution_cfg, dict):
            for key, value in execution_cfg.items():
                params_table.add_row(f"execution.{key}", str(value))
        console.print(params_table)


def _print_simulation_list(summaries_df) -> None:
    """Render a compact table of stored simulations."""
    table = Table(title="🗂 Stored Simulations")
    for column in (
        "simulation_id",
        "simulation_type",
        "model_run_id",
        "symbol",
        "timeframe",
        "strategy_version",
        "total_return",
        "max_drawdown",
        "number_of_trades",
    ):
        table.add_column(column, style="white")

    for _, row in summaries_df.iterrows():
        table.add_row(
            *[
                str(row.get(column))
                for column in (
                    "simulation_id",
                    "simulation_type",
                    "model_run_id",
                    "symbol",
                    "timeframe",
                    "strategy_version",
                    "total_return",
                    "max_drawdown",
                    "number_of_trades",
                )
            ]
        )
    console.print(table)


def _print_simulation_comparison(comparison_df) -> None:
    """Render a comparison table for stored simulations."""
    table = Table(title="📊 Simulation Comparison")
    for column in comparison_df.columns:
        table.add_column(str(column), style="white")
    for _, row in comparison_df.iterrows():
        table.add_row(*[str(row.get(column)) for column in comparison_df.columns])
    console.print(table)


def _print_validation_report(report: dict[str, object]) -> None:
    """Render the top-level fields and conclusion from a validation report."""
    overview = Table(title=f"🧾 Validation Report — {report.get('validation_id', 'unknown')}")
    overview.add_column("Field", style="bold cyan")
    overview.add_column("Value", style="white")
    for key in ("model_run_id", "symbol", "timeframe", "generated_at"):
        overview.add_row(key, str(report.get(key)))
    console.print(overview)

    conclusion = report.get("conclusion", {})
    if isinstance(conclusion, dict):
        conclusion_table = Table(title="Final Conclusion")
        conclusion_table.add_column("Field", style="bold cyan")
        conclusion_table.add_column("Value", style="white")
        for key, value in conclusion.items():
            conclusion_table.add_row(str(key), str(value))
        console.print(conclusion_table)


def _print_prediction_diagnostics(diagnostics_df) -> None:
    """Render bucketed prediction diagnostics."""
    table = Table(title="🎯 Prediction Diagnostics")
    for column in diagnostics_df.columns:
        table.add_column(str(column), style="white")
    for _, row in diagnostics_df.iterrows():
        table.add_row(*[str(row.get(column)) for column in diagnostics_df.columns])
    console.print(table)


def _print_failure_cases(failure_analysis: dict[str, object]) -> None:
    """Render top losing trades and largest drawdown rows."""
    top_losing_trades = failure_analysis.get("top_losing_trades", [])
    if isinstance(top_losing_trades, list) and top_losing_trades:
        trades_table = Table(title="💥 Top Losing Trades")
        for column in top_losing_trades[0].keys():
            trades_table.add_column(str(column), style="white")
        for row in top_losing_trades:
            if isinstance(row, dict):
                trades_table.add_row(*[str(row.get(column)) for column in row.keys()])
        console.print(trades_table)

    drawdowns = failure_analysis.get("largest_drawdown_periods", [])
    if isinstance(drawdowns, list) and drawdowns:
        drawdown_table = Table(title="📉 Largest Drawdown Periods")
        for column in drawdowns[0].keys():
            drawdown_table.add_column(str(column), style="white")
        for row in drawdowns:
            if isinstance(row, dict):
                drawdown_table.add_row(*[str(row.get(column)) for column in row.keys()])
        console.print(drawdown_table)


def _print_baseline_comparison(comparison_df) -> None:
    """Render the ranked model-vs-baselines comparison table."""
    table = Table(title="⚖️ Strategy Comparison")
    for column in comparison_df.columns:
        table.add_column(str(column), style="white")
    for _, row in comparison_df.iterrows():
        table.add_row(*[str(row.get(column)) for column in comparison_df.columns])
    console.print(table)


def _print_optimization_report(report: dict[str, object]) -> None:
    """Render the top-level optimization report and recommendation."""
    overview = Table(title=f"🔍 Strategy Search — {report.get('optimization_id', 'unknown')}")
    overview.add_column("Field", style="bold cyan")
    overview.add_column("Value", style="white")
    for key in ("model_run_id", "generated_at", "candidates_tested"):
        overview.add_row(key, str(report.get(key)))
    console.print(overview)

    recommendation = report.get("recommendation", {})
    if isinstance(recommendation, dict):
        recommendation_table = Table(title="Recommendation")
        recommendation_table.add_column("Field", style="bold cyan")
        recommendation_table.add_column("Value", style="white")
        for key, value in recommendation.items():
            recommendation_table.add_row(str(key), str(value))
        console.print(recommendation_table)


def _print_strategy_searches(searches_df) -> None:
    """Render stored strategy-search reports."""
    table = Table(title="🗂 Strategy Searches")
    for column in ("optimization_id", "model_run_id", "generated_at", "candidates_tested"):
        table.add_column(column, style="white")
    for _, row in searches_df.iterrows():
        table.add_row(
            str(row.get("optimization_id")),
            str(row.get("model_run_id")),
            str(row.get("generated_at")),
            str(row.get("candidates_tested")),
        )
    console.print(table)


def _print_top_strategies(top_candidates_df) -> None:
    """Render the top strategy-search candidates."""
    table = Table(title="🏆 Top Strategies")
    for column in top_candidates_df.columns:
        table.add_column(str(column), style="white")
    for _, row in top_candidates_df.iterrows():
        table.add_row(*[str(row.get(column)) for column in top_candidates_df.columns])
    console.print(table)


def _parse_horizons_option(horizons: str | None, default_horizons: list[int]) -> list[int]:
    """Parse a comma-separated horizon list into a stable integer list."""
    if horizons is None or not horizons.strip():
        return list(default_horizons)
    return sorted({int(value.strip()) for value in horizons.split(",") if value.strip()})


def _parse_targets_option(targets: str | None, default_targets: list[str]) -> list[str]:
    """Parse a comma-separated target-key list into stable source order."""
    if targets is None or not targets.strip():
        return list(default_targets)
    return [value.strip() for value in targets.split(",") if value.strip()]


def _resolve_paper_loop_selection(
    *,
    settings,
    loop_id: str | None,
    target_name: str | None,
    target_key: str | None,
    model_run_id: str | None,
    optimization_id: str | None,
    symbol: str | None,
    timeframe: str | None,
    feature_version: str | None,
    provider: str | None,
):
    """Resolve CLI overrides against the configured scheduled paper-loop selection."""
    from trading_bot.paper.loop import PaperLoopSelection

    configured = settings.paper_loop
    return PaperLoopSelection(
        loop_id=loop_id or configured.loop_id,
        target_name=target_name or configured.target_name,
        target_key=target_key or configured.target_key,
        model_run_id=model_run_id or configured.model_run_id,
        optimization_id=optimization_id or configured.optimization_id,
        symbol=symbol or configured.symbol,
        timeframe=timeframe or configured.timeframe,
        feature_version=feature_version or configured.feature_version,
        provider=provider or configured.provider,
    )


def _print_multi_horizon_rows(title: str, rows: list[dict[str, object]]) -> None:
    """Render a compact table for multi-horizon workflow outputs."""
    if not rows:
        console.print(f"No rows available for {title.lower()}", style="yellow")
        return
    table = Table(title=title)
    for column in rows[0].keys():
        table.add_column(str(column), style="white")
    for row in rows:
        table.add_row(*[str(row.get(column)) for column in rows[0].keys()])
    console.print(table)


def _print_horizon_comparison_report(report: dict[str, object]) -> None:
    """Render the top-level multi-horizon recommendation and ranked rows."""
    overview = Table(title=f"🕒 Horizon Comparison — {report.get('comparison_id', 'unknown')}")
    overview.add_column("Field", style="bold cyan")
    overview.add_column("Value", style="white")
    for key in ("asset", "symbol", "timeframe", "feature_version", "generated_at"):
        overview.add_row(key, str(report.get(key)))
    console.print(overview)

    recommendation = report.get("recommendation", {})
    if isinstance(recommendation, dict):
        recommendation_table = Table(title="Recommendation")
        recommendation_table.add_column("Field", style="bold cyan")
        recommendation_table.add_column("Value", style="white")
        for key, value in recommendation.items():
            recommendation_table.add_row(str(key), str(value))
        console.print(recommendation_table)

    rows = report.get("rows", [])
    if isinstance(rows, list) and rows:
        _print_multi_horizon_rows("Ranked Horizons", rows)


def _print_target_comparison_report(report: dict[str, object]) -> None:
    """Render the top-level fee-aware target recommendation and ranked rows."""
    overview = Table(title=f"🎯 Target Comparison — {report.get('comparison_id', 'unknown')}")
    overview.add_column("Field", style="bold cyan")
    overview.add_column("Value", style="white")
    for key in ("asset", "symbol", "timeframe", "feature_version", "generated_at"):
        overview.add_row(key, str(report.get(key)))
    console.print(overview)

    recommendation = report.get("recommendation", {})
    if isinstance(recommendation, dict):
        recommendation_table = Table(title="Recommendation")
        recommendation_table.add_column("Field", style="bold cyan")
        recommendation_table.add_column("Value", style="white")
        for key, value in recommendation.items():
            recommendation_table.add_row(str(key), str(value))
        console.print(recommendation_table)

    rows = report.get("rows", [])
    if isinstance(rows, list) and rows:
        _print_multi_horizon_rows("Ranked Targets", rows)


def _print_paper_loop_result(result) -> None:
    """Render a compact success panel for a one-shot paper-loop update."""
    title = "Paper loop checked" if getattr(result, "no_new_candle", False) else "Paper loop updated"
    console.print(
        Panel(
            f"[bold green]{title}[/bold green]\n"
            f"  Loop ID:          {result.loop_id}\n"
            f"  Candle:           {result.latest_candle_timestamp}\n"
            f"  Prediction Time:  {result.prediction_timestamp}\n"
            f"  Probability:      {result.y_proba}\n"
            f"  Predicted Class:  {result.y_pred}\n"
            f"  Signal Action:    {result.signal_action}\n"
            f"  Executed Action:  {result.executed_action}\n"
            f"  Equity:           {result.current_equity}\n"
            f"  In Position:      {result.in_position}\n"
            f"  Trades:           {result.trade_count}",
            title="✅ Success",
            style="green",
        )
    )


def _print_paper_loop_status(status: dict[str, object]) -> None:
    """Render the stored scheduled paper-loop status."""
    table = Table(title=f"📝 Paper Loop Status — {status.get('loop_id', 'unknown')}")
    table.add_column("Field", style="bold cyan")
    table.add_column("Value", style="white")
    for key in (
        "running",
        "pid",
        "target_name",
        "target_key",
        "model_run_id",
        "optimization_id",
        "symbol",
        "timeframe",
        "feature_version",
        "last_updated_at",
        "latest_candle_timestamp",
        "latest_prediction_timestamp",
        "latest_probability",
        "signal_action",
        "executed_action",
        "current_equity",
        "in_position",
        "trade_count",
        "no_new_candle",
        "note",
    ):
        table.add_row(key, str(status.get(key)))
    console.print(table)


def _print_paper_trades(trades_df: pd.DataFrame) -> None:
    """Render stored closed paper trades."""
    table = Table(title="📒 Paper Trades")
    for column in trades_df.columns:
        table.add_column(str(column), style="white")
    for _, row in trades_df.iterrows():
        table.add_row(*[str(row.get(column)) for column in trades_df.columns])
    console.print(table)


# ============================================================================
# Diagnostic commands
# ============================================================================
@cli.command()
def health_check() -> None:
    """Validate config loading and package imports."""
    setup_logging("INFO")
    console.print(Panel("🏥 Running health check …", style="bold cyan"))

    # 1. Config loading
    try:
        settings = load_settings()
        console.print("  ✅ Configuration loaded successfully", style="green")
    except Exception as exc:
        console.print(f"  ❌ Config loading failed: {exc}", style="red")
        sys.exit(1)

    # 2. Module imports
    failed: list[str] = []
    for module_name in _EXPECTED_MODULES:
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            failed.append(f"{module_name}: {exc}")

    if failed:
        console.print(f"  ❌ {len(failed)} module(s) failed to import:", style="red")
        for msg in failed:
            console.print(f"     • {msg}", style="red")
        sys.exit(1)
    else:
        console.print(
            f"  ✅ All {len(_EXPECTED_MODULES)} modules imported successfully", style="green"
        )

    # 3. Summary
    console.print(
        Panel(
            f"[bold green]Health check passed[/bold green]\n"
            f"  App:     {settings.app_name}\n"
            f"  Env:     {settings.app_env.value}\n"
            f"  Asset:   {settings.asset.symbol}\n"
            f"  Version: {__version__}",
            title="✅ All systems go",
            style="green",
        )
    )


@cli.command()
@click.option(
    "--show-secrets",
    is_flag=True,
    default=False,
    help="Show API keys (redacted by default)",
)
def show_config(show_secrets: bool) -> None:
    """Print the resolved configuration."""
    setup_logging("INFO")
    settings = load_settings()

    data = settings.model_dump()

    # Redact secrets by default
    if not show_secrets:
        for key in ("binance_api_key", "binance_api_secret", "coingecko_api_key"):
            value = data.get(key, "")
            if value:
                data[key] = value[:4] + "****" + value[-4:] if len(value) > 8 else "****"
            else:
                data[key] = "(not set)"

    formatted = json.dumps(data, indent=2, default=str)
    console.print(Panel(formatted, title="📋 Resolved Configuration", style="cyan"))


@cli.command()
def project_tree() -> None:
    """Print a simplified project module overview."""
    setup_logging("INFO")

    tree = Tree(f"🤖 [bold cyan]trading_bot[/bold cyan] v{__version__}")

    src_dir = PROJECT_ROOT / "src" / "trading_bot"

    def _add_subtree(parent: Tree, directory: Path, prefix: str = "") -> None:
        """Recursively build a rich tree from the package directory."""
        items = sorted(directory.iterdir())
        dirs = [p for p in items if p.is_dir() and not p.name.startswith(("__", "."))]
        files = [p for p in items if p.is_file() and p.suffix == ".py" and p.name != "__init__.py"]

        for d in dirs:
            branch = parent.add(f"📦 [bold]{d.name}[/bold]")
            _add_subtree(branch, d, prefix=f"{prefix}{d.name}.")

        for f in files:
            parent.add(f"📄 {f.name}")

    _add_subtree(tree, src_dir)
    console.print(tree)


# ============================================================================
# Market data ingestion commands
# ============================================================================
@cli.command()
@click.option("--symbol", default=None, help="Trading pair symbol (default: from config)")
@click.option("--timeframe", default=None, help="Candle timeframe (default: from config)")
@click.option("--start", default=None, help="Start date, YYYY-MM-DD (default: from config)")
@click.option("--end", default=None, help="End date, YYYY-MM-DD (default: now)")
def ingest_historical_market(
    symbol: str | None,
    timeframe: str | None,
    start: str | None,
    end: str | None,
) -> None:
    """Download, normalize, validate, and persist historical BTCUSDT candles."""
    setup_logging("INFO")
    settings = load_settings()

    # Resolve parameters from config or CLI overrides
    symbol = symbol or settings.market_data.symbol
    timeframe = timeframe or settings.market_data.default_timeframe

    start_dt: datetime
    if start:
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
    else:
        start_dt = datetime.strptime(
            settings.market_data.default_start_date, "%Y-%m-%d"
        ).replace(tzinfo=UTC)

    end_dt: datetime = (
        datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC)
        if end
        else datetime.now(UTC)
    )

    console.print(
        Panel(
            f"Symbol:    {symbol}\n"
            f"Timeframe: {timeframe}\n"
            f"Start:     {start_dt.strftime('%Y-%m-%d')}\n"
            f"End:       {end_dt.strftime('%Y-%m-%d')}",
            title="📥 Ingesting Historical Market Data",
            style="bold cyan",
        )
    )

    # Build service
    from trading_bot.ingestion.binance_client import BinanceClient
    from trading_bot.ingestion.market_data import MarketDataService
    from trading_bot.storage.parquet_store import ParquetStore

    store_path = PROJECT_ROOT / settings.market_data.processed_data_dir
    store = ParquetStore(base_dir=store_path)
    client = BinanceClient(
        timeout=settings.market_data.request_timeout,
        max_retries=settings.market_data.max_retries,
        request_limit=settings.market_data.request_limit,
    )
    service = MarketDataService(store=store, client=client)

    # Run ingestion
    result = service.ingest_sync(
        symbol=symbol,
        timeframe=timeframe,
        start=start_dt,
        end=end_dt,
    )

    if result.is_success:
        console.print(
            Panel(
                f"[bold green]Ingestion complete[/bold green]\n"
                f"  Fetched:  {result.fetched_rows:,} rows\n"
                f"  Valid:    {result.valid_rows:,} rows\n"
                f"  Stored:   {result.stored_rows:,} total rows\n"
                f"  Deduped:  {result.duplicates_removed:,}\n"
                f"  Invalid:  {result.invalid_rows_removed:,}\n"
                f"  Range:    {result.start_timestamp} → {result.end_timestamp}",
                title="✅ Success",
                style="green",
            )
        )
    else:
        console.print(
            Panel(
                f"[bold red]Ingestion failed[/bold red]\n"
                f"  Error: {result.error}",
                title="❌ Failed",
                style="red",
            )
        )
        sys.exit(1)


@cli.command()
@click.option("--symbol", default=None, help="Trading pair symbol (default: from config)")
@click.option("--timeframe", default=None, help="Candle timeframe (default: from config)")
def show_market_data_range(symbol: str | None, timeframe: str | None) -> None:
    """Print local dataset metadata for a symbol and timeframe."""
    setup_logging("INFO")
    settings = load_settings()

    symbol = symbol or settings.market_data.symbol
    timeframe = timeframe or settings.market_data.default_timeframe

    from trading_bot.storage.parquet_store import ParquetStore

    store_path = PROJECT_ROOT / settings.market_data.processed_data_dir
    store = ParquetStore(base_dir=store_path)
    info = store.get_info(symbol, timeframe)

    if not info:
        console.print(
            f"  No data found for {symbol}/{timeframe}", style="yellow"
        )
        return

    console.print(
        Panel(
            json.dumps(info, indent=2, default=str),
            title=f"📊 Dataset Info — {symbol}/{timeframe}",
            style="cyan",
        )
    )


@cli.command()
@click.option("--symbol", default=None, help="Trading pair symbol (default: from config)")
@click.option("--timeframe", default=None, help="Candle timeframe (default: from config)")
def show_market_data_stats(symbol: str | None, timeframe: str | None) -> None:
    """Print row count, timestamp range, and gap analysis for stored data."""
    setup_logging("INFO")
    settings = load_settings()

    symbol = symbol or settings.market_data.symbol
    timeframe = timeframe or settings.market_data.default_timeframe

    from trading_bot.storage.parquet_store import ParquetStore

    store_path = PROJECT_ROOT / settings.market_data.processed_data_dir
    store = ParquetStore(base_dir=store_path)

    df = store.read(symbol, timeframe)

    if df.empty:
        console.print(f"  No data found for {symbol}/{timeframe}", style="yellow")
        return

    # Basic stats
    min_ts = df["timestamp"].min()
    max_ts = df["timestamp"].max()
    row_count = len(df)

    # Gap analysis — find missing intervals
    from trading_bot.ingestion.binance_client import TIMEFRAME_MS

    interval_ms = TIMEFRAME_MS.get(timeframe, 900_000)
    expected_count = int((max_ts - min_ts).total_seconds() * 1000 / interval_ms) + 1
    missing_count = expected_count - row_count

    # Build output table
    table = Table(title=f"📊 Market Data Stats — {symbol}/{timeframe}")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="white")

    table.add_row("Symbol", symbol)
    table.add_row("Timeframe", timeframe)
    table.add_row("Row Count", f"{row_count:,}")
    table.add_row("First Candle", str(min_ts))
    table.add_row("Last Candle", str(max_ts))
    table.add_row("Expected Candles", f"{expected_count:,}")
    table.add_row(
        "Missing Candles",
        f"{max(0, missing_count):,}" + (" ⚠️" if missing_count > 0 else " ✅"),
    )
    table.add_row(
        "Coverage",
        f"{row_count / max(expected_count, 1) * 100:.1f}%",
    )

    # Price summary
    table.add_row("Min Close", f"${df['close'].min():,.2f}")
    table.add_row("Max Close", f"${df['close'].max():,.2f}")
    table.add_row("Mean Close", f"${df['close'].mean():,.2f}")
    table.add_row("Total Volume", f"{df['volume'].sum():,.2f}")

    console.print(table)


# ============================================================================
# News data ingestion commands
# ============================================================================
@cli.command()
@click.option("--asset", default=None, help="Asset tag (default: from config)")
@click.option("--provider", default=None, help="News provider (default: from config)")
@click.option("--start", default=None, help="Start date, YYYY-MM-DD (default: from config)")
@click.option("--end", default=None, help="End date, YYYY-MM-DD (default: now)")
def ingest_historical_news(
    asset: str | None,
    provider: str | None,
    start: str | None,
    end: str | None,
) -> None:
    """Download, normalize, validate, and persist historical BTC news."""
    setup_logging("INFO")
    settings = load_settings()

    asset = asset or settings.news_data.asset
    provider = provider or settings.news_data.provider

    start_dt: datetime
    if start:
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
    else:
        start_dt = datetime.strptime(
            settings.news_data.default_start_date, "%Y-%m-%d"
        ).replace(tzinfo=UTC)

    end_dt: datetime = (
        datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC)
        if end
        else datetime.now(UTC)
    )

    console.print(
        Panel(
            f"Asset:    {asset}\n"
            f"Provider: {provider}\n"
            f"Start:    {start_dt.strftime('%Y-%m-%d')}\n"
            f"End:      {end_dt.strftime('%Y-%m-%d')}",
            title="📰 Ingesting Historical News",
            style="bold cyan",
        )
    )

    from trading_bot.ingestion.news_client import CryptoCompareNewsClient
    from trading_bot.ingestion.news_data import NewsIngestionService
    from trading_bot.storage.news_store import NewsStore

    store_path = PROJECT_ROOT / settings.news_data.processed_data_dir
    store = NewsStore(base_dir=store_path)
    client = CryptoCompareNewsClient(
        api_key=settings.coingecko_api_key,  # Reuse for CryptoCompare if set
        timeout=settings.news_data.request_timeout,
        max_retries=settings.news_data.max_retries,
        max_pages=settings.news_data.max_pages,
    )
    service = NewsIngestionService(
        store=store,
        client=client,
        btc_keywords=settings.news_data.btc_keywords,
    )

    result = service.ingest_sync(
        asset=asset,
        provider=provider,
        start=start_dt,
        end=end_dt,
    )

    if result.is_success:
        console.print(
            Panel(
                f"[bold green]News ingestion complete[/bold green]\n"
                f"  Fetched:       {result.fetched_rows:,} articles\n"
                f"  BTC filtered:  {result.btc_filtered_out:,} removed\n"
                f"  Valid:         {result.valid_rows:,} articles\n"
                f"  Stored:        {result.stored_rows:,} total articles\n"
                f"  Deduped:       {result.duplicates_removed:,}\n"
                f"  Invalid:       {result.invalid_rows_removed:,}\n"
                f"  Range:         {result.start_timestamp} → {result.end_timestamp}",
                title="✅ Success",
                style="green",
            )
        )
    else:
        console.print(
            Panel(
                f"[bold red]News ingestion failed[/bold red]\n"
                f"  Error: {result.error}",
                title="❌ Failed",
                style="red",
            )
        )
        sys.exit(1)


@cli.command()
@click.option("--asset", default=None, help="Asset tag (default: from config)")
@click.option("--provider", default=None, help="News provider (default: from config)")
def show_news_data_range(asset: str | None, provider: str | None) -> None:
    """Print local dataset metadata for BTC news."""
    setup_logging("INFO")
    settings = load_settings()

    asset = asset or settings.news_data.asset
    provider = provider or settings.news_data.provider

    from trading_bot.storage.news_store import NewsStore

    store_path = PROJECT_ROOT / settings.news_data.processed_data_dir
    store = NewsStore(base_dir=store_path)
    info = store.get_info(asset, provider)

    if not info:
        console.print(f"  No news data found for {asset}/{provider}", style="yellow")
        return

    console.print(
        Panel(
            json.dumps(info, indent=2, default=str),
            title=f"📰 News Dataset Info — {asset}/{provider}",
            style="cyan",
        )
    )


@cli.command()
@click.option("--asset", default=None, help="Asset tag (default: from config)")
@click.option("--provider", default=None, help="News provider (default: from config)")
def show_news_data_stats(asset: str | None, provider: str | None) -> None:
    """Print article count, source distribution, and timestamp range for stored BTC news."""
    setup_logging("INFO")
    settings = load_settings()

    asset = asset or settings.news_data.asset
    provider = provider or settings.news_data.provider

    from trading_bot.storage.news_store import NewsStore

    store_path = PROJECT_ROOT / settings.news_data.processed_data_dir
    store = NewsStore(base_dir=store_path)

    df = store.read(asset, provider)

    if df.empty:
        console.print(f"  No news data found for {asset}/{provider}", style="yellow")
        return

    min_ts = df["published_at"].min()
    max_ts = df["published_at"].max()
    row_count = len(df)

    table = Table(title=f"📰 News Data Stats — {asset}/{provider}")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="white")

    table.add_row("Asset", asset)
    table.add_row("Provider", provider)
    table.add_row("Total Articles", f"{row_count:,}")
    table.add_row("First Article", str(min_ts))
    table.add_row("Last Article", str(max_ts))

    if "source_name" in df.columns:
        unique_sources = df["source_name"].nunique()
        table.add_row("Unique Sources", f"{unique_sources:,}")

        top_sources = df["source_name"].value_counts().head(5)
        for src, count in top_sources.items():
            table.add_row(f"  └ {src}", f"{count:,}")

    if "category" in df.columns:
        unique_cats = df["category"].nunique()
        table.add_row("Unique Categories", f"{unique_cats:,}")

    console.print(table)


@cli.command(name="enrich-news-sentiment")
@click.option("--asset", default=None, help="Asset tag (default: from config)")
@click.option("--provider", default=None, help="News provider (default: from config)")
@click.option(
    "--enrichment-version",
    default=None,
    help="Enrichment version (default: from config)",
)
@click.option(
    "--text-mode",
    type=click.Choice(["title_only", "title_plus_summary"]),
    default=None,
    help="Text assembly mode (default: from config)",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"]),
    default=None,
    help="FinBERT inference device request (default: from config)",
)
def enrich_news_sentiment_cmd(
    asset: str | None,
    provider: str | None,
    enrichment_version: str | None,
    text_mode: str | None,
    device: str | None,
) -> None:
    """Run local FinBERT sentiment scoring on stored BTC news records."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.nlp.finbert_service import FinBERTService
    from trading_bot.nlp.news_enrichment import NewsEnrichmentService
    from trading_bot.storage.enriched_news_store import EnrichedNewsStore
    from trading_bot.storage.news_store import NewsStore

    service = NewsEnrichmentService(
        news_store=NewsStore(PROJECT_ROOT / settings.news_data.processed_data_dir),
        enriched_news_store=EnrichedNewsStore(PROJECT_ROOT / settings.storage.enriched_news_dir),
        finbert_service=FinBERTService(
            model_name=settings.nlp.model_name,
            batch_size=settings.nlp.batch_size,
            requested_device=device or settings.nlp.device,
            allow_cuda_fallback_to_cpu=settings.nlp.allow_cuda_fallback_to_cpu,
            max_length=settings.nlp.max_length,
        ),
        default_asset=settings.news_data.asset,
        default_provider=settings.news_data.provider,
        default_text_mode=settings.nlp.text_mode,
        default_enrichment_version=settings.nlp.enrichment_version,
    )

    result = service.enrich_news(
        asset=asset or settings.news_data.asset,
        provider=provider or settings.news_data.provider,
        enrichment_version=enrichment_version or settings.nlp.enrichment_version,
        text_mode=text_mode or settings.nlp.text_mode,
    )

    console.print(
        Panel(
            f"[bold green]News sentiment enrichment complete[/bold green]\n"
            f"  Asset:                 {result.asset}\n"
            f"  Provider:              {result.provider}\n"
            f"  Model:                 {result.model_name}\n"
            f"  Enrichment Version:    {result.enrichment_version}\n"
            f"  Requested Device:      {result.requested_device}\n"
            f"  Effective Device:      {result.effective_device}\n"
            f"  Input Rows:            {result.input_rows:,}\n"
            f"  Scored Rows:           {result.scored_rows:,}\n"
            f"  Skipped Existing:      {result.skipped_existing_rows:,}\n"
            f"  Skipped Empty Text:    {result.skipped_empty_text_rows:,}\n"
            f"  Failure Rows:          {result.failure_rows:,}\n"
            f"  Stored Rows:           {result.stored_rows:,}",
            title="✅ Success",
            style="green",
        )
    )


@cli.command(name="show-enriched-news-stats")
@click.option("--asset", default=None, help="Asset tag (default: from config)")
@click.option("--provider", default=None, help="News provider (default: from config)")
@click.option(
    "--enrichment-version",
    default=None,
    help="Enrichment version (default: from config)",
)
def show_enriched_news_stats_cmd(
    asset: str | None,
    provider: str | None,
    enrichment_version: str | None,
) -> None:
    """Print coverage, label distribution, and timestamp stats for enriched BTC news."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.storage.enriched_news_store import EnrichedNewsStore

    store = EnrichedNewsStore(PROJECT_ROOT / settings.storage.enriched_news_dir)
    info = store.get_info(
        asset=asset or settings.news_data.asset,
        provider=provider or settings.news_data.provider,
        model_name=settings.nlp.model_name,
        enrichment_version=enrichment_version or settings.nlp.enrichment_version,
    )

    if not info:
        console.print("No enriched news data found", style="yellow")
        return

    _print_enriched_news_stats(info)


# ============================================================================
# Feature engineering commands
# ============================================================================
@cli.command(name="build-market-features")
@click.option("--symbol", default=None, help="Trading pair symbol (default: from config)")
@click.option("--timeframe", default=None, help="Candle timeframe (default: from config)")
@click.option("--asset", default=None, help="Asset tag (default: from config)")
@click.option(
    "--version",
    "feature_version",
    default=None,
    help="Feature version (default: from config)",
)
def build_market_features_cmd(
    symbol: str | None,
    timeframe: str | None,
    asset: str | None,
    feature_version: str | None,
) -> None:
    """Build market-only features from stored BTC candle data."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.features.feature_pipeline import FeaturePipeline, FeaturePipelineError

    pipeline = FeaturePipeline.from_settings(settings)

    try:
        _, feature_set = pipeline.build_market_features(
            symbol=symbol or settings.features.symbol,
            timeframe=timeframe or settings.features.timeframe,
            asset=asset or settings.features.asset,
            version=feature_version or settings.features.feature_version,
        )
    except FeaturePipelineError as exc:
        console.print(
            Panel(
                f"[bold red]Market feature build failed[/bold red]\n  Error: {exc}",
                title="❌ Failed",
                style="red",
            )
        )
        sys.exit(1)

    _print_feature_build_success("Market feature build", feature_set)


@cli.command(name="build-news-features")
@click.option("--symbol", default=None, help="Trading pair symbol (default: from config)")
@click.option("--timeframe", default=None, help="Candle timeframe (default: from config)")
@click.option("--asset", default=None, help="Asset tag (default: from config)")
@click.option("--provider", default=None, help="News provider (default: from config)")
@click.option(
    "--version",
    "feature_version",
    default=None,
    help="Feature version (default: from config)",
)
def build_news_features_cmd(
    symbol: str | None,
    timeframe: str | None,
    asset: str | None,
    provider: str | None,
    feature_version: str | None,
) -> None:
    """Build news-only aggregate features aligned to the market timeline."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.features.feature_pipeline import FeaturePipeline, FeaturePipelineError

    pipeline = FeaturePipeline.from_settings(settings)

    try:
        _, feature_set = pipeline.build_news_features(
            symbol=symbol or settings.features.symbol,
            timeframe=timeframe or settings.features.timeframe,
            asset=asset or settings.features.asset,
            provider=provider or settings.news_data.provider,
            version=feature_version or settings.features.feature_version,
        )
    except FeaturePipelineError as exc:
        console.print(
            Panel(
                f"[bold red]News feature build failed[/bold red]\n  Error: {exc}",
                title="❌ Failed",
                style="red",
            )
        )
        sys.exit(1)

    _print_feature_build_success("News feature build", feature_set)


@cli.command(name="build-sentiment-news-features")
@click.option("--symbol", default=None, help="Trading pair symbol (default: from config)")
@click.option("--timeframe", default=None, help="Candle timeframe (default: from config)")
@click.option("--asset", default=None, help="Asset tag (default: from config)")
@click.option("--provider", default=None, help="News provider (default: from config)")
@click.option(
    "--feature-version",
    default=None,
    help="Sentiment-aware feature version (default: from config)",
)
def build_sentiment_news_features_cmd(
    symbol: str | None,
    timeframe: str | None,
    asset: str | None,
    provider: str | None,
    feature_version: str | None,
) -> None:
    """Build sentiment-aware news aggregate features aligned to the market timeline."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.features.feature_pipeline import FeaturePipeline, FeaturePipelineError

    pipeline = FeaturePipeline.from_settings(settings)

    try:
        _, feature_set = pipeline.build_news_features(
            symbol=symbol or settings.features.symbol,
            timeframe=timeframe or settings.features.timeframe,
            asset=asset or settings.features.asset,
            provider=provider or settings.news_data.provider,
            version=feature_version or settings.sentiment_features.feature_version,
        )
    except FeaturePipelineError as exc:
        console.print(
            Panel(
                f"[bold red]Sentiment news feature build failed[/bold red]\n  Error: {exc}",
                title="❌ Failed",
                style="red",
            )
        )
        sys.exit(1)

    _print_feature_build_success("Sentiment news feature build", feature_set)


@cli.command(name="build-feature-table")
@click.option("--symbol", default=None, help="Trading pair symbol (default: from config)")
@click.option("--timeframe", default=None, help="Candle timeframe (default: from config)")
@click.option("--asset", default=None, help="Asset tag (default: from config)")
@click.option("--provider", default=None, help="News provider (default: from config)")
@click.option(
    "--version",
    "feature_version",
    default=None,
    help="Feature version (default: from config)",
)
def build_feature_table_cmd(
    symbol: str | None,
    timeframe: str | None,
    asset: str | None,
    provider: str | None,
    feature_version: str | None,
) -> None:
    """Build the merged market plus news feature table."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.features.feature_pipeline import FeaturePipeline, FeaturePipelineError

    pipeline = FeaturePipeline.from_settings(settings)

    try:
        _, feature_set = pipeline.build_feature_table(
            symbol=symbol or settings.features.symbol,
            timeframe=timeframe or settings.features.timeframe,
            asset=asset or settings.features.asset,
            provider=provider or settings.news_data.provider,
            version=feature_version or settings.features.feature_version,
        )
    except FeaturePipelineError as exc:
        console.print(
            Panel(
                f"[bold red]Merged feature build failed[/bold red]\n  Error: {exc}",
                title="❌ Failed",
                style="red",
            )
        )
        sys.exit(1)

    _print_feature_build_success("Merged feature build", feature_set)


@cli.command(name="show-feature-stats")
@click.option(
    "--dataset",
    type=click.Choice(sorted(_FEATURE_DATASET_NAME_MAP.keys())),
    default="merged",
    show_default=True,
    help="Stored feature dataset to inspect.",
)
@click.option("--symbol", default=None, help="Trading pair symbol (default: from config)")
@click.option("--timeframe", default=None, help="Candle timeframe (default: from config)")
@click.option("--asset", default=None, help="Asset tag (default: from config)")
@click.option("--provider", default=None, help="News provider (default: from config)")
@click.option(
    "--version",
    "feature_version",
    default=None,
    help="Feature version (default: from config)",
)
def show_feature_stats_cmd(
    dataset: str,
    symbol: str | None,
    timeframe: str | None,
    asset: str | None,
    provider: str | None,
    feature_version: str | None,
) -> None:
    """Print row count, timestamp range, and column list for a stored feature dataset."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.features.feature_pipeline import FeaturePipeline

    pipeline = FeaturePipeline.from_settings(settings)
    dataset_name = _FEATURE_DATASET_NAME_MAP[dataset]
    info = pipeline.inspect_dataset(
        dataset_name,
        symbol=symbol or settings.features.symbol,
        timeframe=timeframe or settings.features.timeframe,
        asset=asset or settings.features.asset,
        provider=provider or settings.news_data.provider,
        version=feature_version or settings.features.feature_version,
    )

    if not info:
        console.print(
            f"  No stored {dataset_name} dataset found for "
            f"{asset or settings.features.asset}/"
            f"{symbol or settings.features.symbol}/"
            f"{timeframe or settings.features.timeframe}/"
            f"{feature_version or settings.features.feature_version}",
            style="yellow",
        )
        return

    _print_feature_stats(info, dataset)


# ============================================================================
# Label generation and supervised dataset commands
# ============================================================================
@cli.command(name="build-labels")
@click.option("--symbol", default=None, help="Trading pair symbol (default: from config)")
@click.option("--timeframe", default=None, help="Candle timeframe (default: from config)")
@click.option("--asset", default=None, help="Asset tag (default: from config)")
@click.option("--horizon-bars", default=None, type=int, help="Future label horizon in bars.")
@click.option(
    "--label-version",
    default=None,
    help="Label version (default: from config)",
)
def build_labels_cmd(
    symbol: str | None,
    timeframe: str | None,
    asset: str | None,
    horizon_bars: int | None,
    label_version: str | None,
) -> None:
    """Generate future-return target columns from stored BTC market data."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.labeling.dataset_builder import DatasetBuilderError, SupervisedDatasetBuilder

    builder = SupervisedDatasetBuilder.from_settings(settings)

    try:
        _, artifact = builder.build_labels(
            symbol=symbol or settings.datasets.symbol,
            timeframe=timeframe or settings.datasets.timeframe,
            asset=asset or settings.datasets.asset,
            label_version=label_version or settings.labels.label_version,
            horizon_bars=horizon_bars or settings.labels.horizon_bars,
        )
    except DatasetBuilderError as exc:
        console.print(
            Panel(
                f"[bold red]Label build failed[/bold red]\n  Error: {exc}",
                title="❌ Failed",
                style="red",
            )
        )
        sys.exit(1)

    _print_dataset_build_success("Label build", artifact)


@cli.command(name="build-supervised-dataset")
@click.option("--symbol", default=None, help="Trading pair symbol (default: from config)")
@click.option("--timeframe", default=None, help="Candle timeframe (default: from config)")
@click.option("--asset", default=None, help="Asset tag (default: from config)")
@click.option(
    "--feature-version",
    default=None,
    help="Feature version to load from step 4 (default: from config)",
)
@click.option(
    "--label-version",
    default=None,
    help="Label version (default: from config)",
)
@click.option(
    "--dataset-version",
    default=None,
    help="Dataset version (default: from config)",
)
def build_supervised_dataset_cmd(
    symbol: str | None,
    timeframe: str | None,
    asset: str | None,
    feature_version: str | None,
    label_version: str | None,
    dataset_version: str | None,
) -> None:
    """Join stored features with generated labels and persist a final training dataset."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.labeling.dataset_builder import DatasetBuilderError, SupervisedDatasetBuilder

    builder = SupervisedDatasetBuilder.from_settings(settings)

    try:
        _, artifact = builder.build_supervised_dataset(
            symbol=symbol or settings.datasets.symbol,
            timeframe=timeframe or settings.datasets.timeframe,
            asset=asset or settings.datasets.asset,
            feature_version=feature_version or settings.features.feature_version,
            label_version=label_version or settings.labels.label_version,
            dataset_version=dataset_version or settings.datasets.dataset_version,
        )
    except DatasetBuilderError as exc:
        console.print(
            Panel(
                f"[bold red]Supervised dataset build failed[/bold red]\n  Error: {exc}",
                title="❌ Failed",
                style="red",
            )
        )
        sys.exit(1)

    _print_dataset_build_success("Supervised dataset build", artifact)


@cli.command(name="show-dataset-stats")
@click.option(
    "--artifact",
    type=click.Choice(sorted(_DATASET_ARTIFACT_NAME_MAP.keys())),
    default="supervised",
    show_default=True,
    help="Stored dataset artifact to inspect.",
)
@click.option("--symbol", default=None, help="Trading pair symbol (default: from config)")
@click.option("--timeframe", default=None, help="Candle timeframe (default: from config)")
@click.option("--asset", default=None, help="Asset tag (default: from config)")
@click.option(
    "--label-version",
    default=None,
    help="Label version (default: from config)",
)
@click.option(
    "--dataset-version",
    default=None,
    help="Dataset version (default: from config)",
)
def show_dataset_stats_cmd(
    artifact: str,
    symbol: str | None,
    timeframe: str | None,
    asset: str | None,
    label_version: str | None,
    dataset_version: str | None,
) -> None:
    """Print row count, timestamp range, target distribution, and column counts."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.labeling.dataset_builder import SupervisedDatasetBuilder

    builder = SupervisedDatasetBuilder.from_settings(settings)
    artifact_name = _DATASET_ARTIFACT_NAME_MAP[artifact]
    info = builder.inspect_dataset(
        artifact_name,
        symbol=symbol or settings.datasets.symbol,
        timeframe=timeframe or settings.datasets.timeframe,
        asset=asset or settings.datasets.asset,
        label_version=label_version or settings.labels.label_version,
        dataset_version=dataset_version or settings.datasets.dataset_version,
    )

    if not info:
        console.print(
            f"  No stored {artifact_name} found for "
            f"{asset or settings.datasets.asset}/"
            f"{symbol or settings.datasets.symbol}/"
            f"{timeframe or settings.datasets.timeframe}/"
            f"{label_version or settings.labels.label_version}",
            style="yellow",
        )
        return

    _print_dataset_stats(info, artifact)


@cli.command(name="show-target-columns")
@click.option("--symbol", default=None, help="Trading pair symbol (default: from config)")
@click.option("--timeframe", default=None, help="Candle timeframe (default: from config)")
@click.option("--asset", default=None, help="Asset tag (default: from config)")
@click.option(
    "--label-version",
    default=None,
    help="Label version (default: from config)",
)
@click.option(
    "--dataset-version",
    default=None,
    help="Dataset version (default: from config)",
)
def show_target_columns_cmd(
    symbol: str | None,
    timeframe: str | None,
    asset: str | None,
    label_version: str | None,
    dataset_version: str | None,
) -> None:
    """List available target columns in the stored supervised dataset."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.labeling.dataset_builder import SupervisedDatasetBuilder

    builder = SupervisedDatasetBuilder.from_settings(settings)
    target_columns = builder.list_target_columns(
        symbol=symbol or settings.datasets.symbol,
        timeframe=timeframe or settings.datasets.timeframe,
        asset=asset or settings.datasets.asset,
        label_version=label_version or settings.labels.label_version,
        dataset_version=dataset_version or settings.datasets.dataset_version,
    )

    if not target_columns:
        console.print(
            "  No stored supervised dataset target columns found",
            style="yellow",
        )
        return

    table = Table(title="🎯 Target Columns")
    table.add_column("Target", style="bold cyan")
    for target_column in target_columns:
        table.add_row(target_column)

    console.print(table)


# ============================================================================
# Walk-forward validation and baseline training commands
# ============================================================================
@cli.command(name="run-walk-forward-training")
@click.option("--symbol", default=None, help="Trading pair symbol (default: from config)")
@click.option("--timeframe", default=None, help="Candle timeframe (default: from config)")
@click.option("--asset", default=None, help="Asset tag (default: from config)")
@click.option(
    "--feature-version",
    default=None,
    help="Feature version provenance for the supervised dataset (default: from config)",
)
@click.option(
    "--label-version",
    default=None,
    help="Label version of the supervised dataset (default: from config)",
)
@click.option(
    "--dataset-version",
    default=None,
    help="Dataset version of the supervised dataset (default: from config)",
)
@click.option(
    "--model-version",
    default=None,
    help="Model version for artifact persistence (default: from config)",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"]),
    default=None,
    help="Training device request (default: from config)",
)
@click.option(
    "--run-id",
    default=None,
    help="Optional explicit run id. Defaults to a UTC timestamp-based id.",
)
def run_walk_forward_training_cmd(
    symbol: str | None,
    timeframe: str | None,
    asset: str | None,
    feature_version: str | None,
    label_version: str | None,
    dataset_version: str | None,
    model_version: str | None,
    device: str | None,
    run_id: str | None,
) -> None:
    """Train and evaluate the baseline LightGBM model across walk-forward folds."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.models.train import TrainingPipelineError, WalkForwardTrainingPipeline

    pipeline = WalkForwardTrainingPipeline.from_settings(settings)

    try:
        result = pipeline.run_training(
            symbol=symbol or settings.datasets.symbol,
            timeframe=timeframe or settings.datasets.timeframe,
            asset=asset or settings.datasets.asset,
            feature_version=feature_version or settings.features.feature_version,
            label_version=label_version or settings.labels.label_version,
            dataset_version=dataset_version or settings.datasets.dataset_version,
            model_version=model_version or settings.training.model_version,
            requested_device=device or settings.training.device,
            run_id=run_id,
        )
    except TrainingPipelineError as exc:
        console.print(
            Panel(
                f"[bold red]Walk-forward training failed[/bold red]\n  Error: {exc}",
                title="❌ Failed",
                style="red",
            )
        )
        sys.exit(1)

    _print_training_run_success(result)


@cli.command(name="show-run-metrics")
@click.option("--run-id", required=True, help="Stored training run id.")
@click.option("--symbol", default=None, help="Optional symbol filter.")
@click.option("--timeframe", default=None, help="Optional timeframe filter.")
@click.option("--model-version", default=None, help="Optional model version filter.")
@click.option("--asset", default=None, help="Optional asset filter.")
def show_run_metrics_cmd(
    run_id: str,
    symbol: str | None,
    timeframe: str | None,
    model_version: str | None,
    asset: str | None,
) -> None:
    """Print aggregate and per-fold metrics for a stored training run."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.models.model_registry import ModelRegistry, ModelRegistryError

    registry = ModelRegistry.from_settings(settings)
    try:
        report = registry.read_report(
            run_id,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
        )
        aggregate_metrics = registry.read_aggregate_metrics(
            run_id,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
        )
        fold_metrics_df = registry.read_fold_metrics(
            run_id,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
        )
    except ModelRegistryError as exc:
        console.print(str(exc), style="yellow")
        return

    if not report:
        console.print(f"No stored report found for run_id={run_id}", style="yellow")
        return

    _print_run_metrics(report, aggregate_metrics, fold_metrics_df)


@cli.command(name="list-model-runs")
@click.option("--symbol", default=None, help="Optional symbol filter.")
@click.option("--timeframe", default=None, help="Optional timeframe filter.")
@click.option("--model-version", default=None, help="Optional model version filter.")
@click.option("--asset", default=None, help="Optional asset filter.")
def list_model_runs_cmd(
    symbol: str | None,
    timeframe: str | None,
    model_version: str | None,
    asset: str | None,
) -> None:
    """List available stored model runs with basic metadata."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.models.model_registry import ModelRegistry

    registry = ModelRegistry.from_settings(settings)
    runs = registry.list_runs(
        asset=asset,
        symbol=symbol,
        timeframe=timeframe,
        model_version=model_version,
    )
    if not runs:
        console.print("No stored model runs found", style="yellow")
        return

    _print_model_runs(runs)


@cli.command(name="show-feature-importance")
@click.option("--run-id", required=True, help="Stored training run id.")
@click.option("--top-k", default=20, show_default=True, type=int, help="Number of rows to print.")
@click.option("--symbol", default=None, help="Optional symbol filter.")
@click.option("--timeframe", default=None, help="Optional timeframe filter.")
@click.option("--model-version", default=None, help="Optional model version filter.")
@click.option("--asset", default=None, help="Optional asset filter.")
def show_feature_importance_cmd(
    run_id: str,
    top_k: int,
    symbol: str | None,
    timeframe: str | None,
    model_version: str | None,
    asset: str | None,
) -> None:
    """Print top aggregate feature importance values for a stored training run."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.models.model_registry import ModelRegistry, ModelRegistryError

    registry = ModelRegistry.from_settings(settings)
    try:
        feature_importance_df = registry.read_feature_importance(
            run_id,
            aggregate=True,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
        )
    except ModelRegistryError as exc:
        console.print(str(exc), style="yellow")
        return

    if feature_importance_df.empty:
        console.print(f"No stored feature importance found for run_id={run_id}", style="yellow")
        return

    _print_feature_importance(run_id, feature_importance_df, top_k)


# ============================================================================
# Experiment tracking and reporting commands
# ============================================================================
@cli.command(name="refresh-experiment-registry")
def refresh_experiment_registry_cmd() -> None:
    """Scan stored step-6 run artifacts and refresh the local experiment registry."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.experiments.tracker import ExperimentTracker

    tracker = ExperimentTracker.from_settings(settings)
    registry_df = tracker.refresh_registry()

    if registry_df.empty:
        console.print("No stored evaluation runs found to register", style="yellow")
        return

    console.print(
        Panel(
            f"[bold green]Experiment registry refreshed[/bold green]\n"
            f"  Runs tracked: {len(registry_df):,}\n"
            f"  Registry:     {PROJECT_ROOT / settings.experiments.registry_path}",
            title="✅ Success",
            style="green",
        )
    )


@cli.command(name="list-experiment-runs")
@click.option("--status", default=None, help="Optional status filter.")
@click.option("--model-version", default=None, help="Optional model version filter.")
@click.option("--dataset-version", default=None, help="Optional dataset version filter.")
@click.option("--target-column", default=None, help="Optional target column filter.")
@click.option("--device", default=None, help="Optional requested/effective device filter.")
@click.option("--sort-by", default=None, help="Sort metric or column (default: config).")
def list_experiment_runs_cmd(
    status: str | None,
    model_version: str | None,
    dataset_version: str | None,
    target_column: str | None,
    device: str | None,
    sort_by: str | None,
) -> None:
    """List tracked experiment runs with key metadata and aggregate metrics."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.experiments.registry import ExperimentRegistry

    registry = ExperimentRegistry.from_settings(settings)
    runs_df = registry.list_runs(
        status=status,
        model_version=model_version,
        dataset_version=dataset_version,
        target_column=target_column,
        device=device,
        sort_by=sort_by,
    )

    if runs_df.empty:
        console.print("No tracked experiment runs found", style="yellow")
        return

    _print_experiment_runs(runs_df)


@cli.command(name="show-run-report")
@click.option("--run-id", required=True, help="Stored training run id.")
def show_run_report_cmd(run_id: str) -> None:
    """Generate and print a rich evaluation report for one run."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.experiments.tracker import ExperimentTracker

    tracker = ExperimentTracker.from_settings(settings)
    try:
        report = tracker.generate_run_report(run_id, write_report=True)
    except ValueError as exc:
        console.print(str(exc), style="red")
        sys.exit(1)

    if not report:
        console.print(f"No stored report found for run_id={run_id}", style="yellow")
        return

    _print_run_report(report)


@cli.command(name="compare-runs")
@click.option(
    "--run-ids",
    multiple=True,
    required=True,
    help="One or more stored run ids to compare.",
)
@click.option("--sort-by", default=None, help="Metric or column to sort by.")
def compare_runs_cmd(run_ids: tuple[str, ...], sort_by: str | None) -> None:
    """Compare two or more runs on metrics and metadata."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.experiments.comparison import RunComparator
    from trading_bot.storage.report_store import ReportStore

    comparator = RunComparator.from_settings(settings)
    comparison_df = comparator.compare_runs(list(run_ids), sort_by=sort_by)

    if comparison_df.empty:
        console.print("No matching runs found for comparison", style="yellow")
        return

    report_store = ReportStore(PROJECT_ROOT / settings.reports.output_dir)
    comparison_name = "__".join(list(run_ids))
    report_store.write_comparison_table(f"compare_{comparison_name}", comparison_df)
    report_store.write_comparison_json(
        f"compare_{comparison_name}",
        {
            "run_ids": list(run_ids),
            "sort_by": sort_by or settings.experiments.default_sort_metric,
            "rows": comparison_df.to_dict(orient="records"),
        },
    )
    _print_run_comparison(comparison_df)


@cli.command(name="show-sentiment-ablation")
@click.option("--baseline-run-id", required=True, help="The non-sentiment baseline run id.")
@click.option("--new-run-id", required=True, help="The sentiment-enhanced run id.")
def show_sentiment_ablation_cmd(baseline_run_id: str, new_run_id: str) -> None:
    """Compare a baseline run against a sentiment-enhanced run and persist deltas."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.experiments.comparison import RunComparator
    from trading_bot.storage.report_store import ReportStore

    comparator = RunComparator.from_settings(settings)
    summary = comparator.build_sentiment_ablation_summary(
        baseline_run_id=baseline_run_id,
        new_run_id=new_run_id,
    )
    if not summary:
        console.print("No matching runs found for sentiment ablation", style="yellow")
        return

    report_store = ReportStore(PROJECT_ROOT / settings.reports.output_dir)
    comparison_name = f"sentiment_ablation_{baseline_run_id}__{new_run_id}"
    report_store.write_comparison_json(comparison_name, summary)
    metric_deltas = summary.get("metric_deltas", {})
    if isinstance(metric_deltas, dict):
        delta_df = pd.DataFrame(
            [
                {
                    "baseline_run_id": baseline_run_id,
                    "new_run_id": new_run_id,
                    "metric_name": metric_name,
                    "delta": delta,
                    "status": summary.get("metric_improvements", {}).get(metric_name),
                }
                for metric_name, delta in metric_deltas.items()
            ]
        )
        report_store.write_comparison_table(comparison_name, delta_df)

    _print_sentiment_ablation(summary)


@cli.command(name="run-backtest")
@click.option("--model-run-id", required=True, help="Stored model run id to simulate.")
@click.option("--symbol", default=None, help="Optional symbol override.")
@click.option("--timeframe", default=None, help="Optional timeframe override.")
@click.option("--strategy-version", default=None, help="Optional strategy version label.")
@click.option("--simulation-id", default=None, help="Optional explicit simulation id.")
def run_backtest_cmd(
    model_run_id: str,
    symbol: str | None,
    timeframe: str | None,
    strategy_version: str | None,
    simulation_id: str | None,
) -> None:
    """Run a prediction-driven backtest from stored model outputs."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.backtest.simulator import BacktestSimulator, BacktestSimulatorError
    from trading_bot.backtest.strategy import Strategy, get_default_strategy
    from trading_bot.schemas.backtest import ExecutionConfig, StrategyConfig

    simulator = BacktestSimulator.from_settings(settings)
    configured_strategy = get_default_strategy(settings)
    strategy = Strategy(
        config=StrategyConfig(
            strategy_version=strategy_version or configured_strategy.strategy_version,
            type=configured_strategy.type,
            entry_threshold=configured_strategy.entry_threshold,
            exit_threshold=configured_strategy.exit_threshold,
            minimum_holding_bars=configured_strategy.minimum_holding_bars,
            cooldown_bars=configured_strategy.cooldown_bars,
        )
    )
    execution = ExecutionConfig(
        fill_model=settings.backtest_execution.fill_model,
        starting_cash=settings.backtest_execution.starting_cash,
        fee_rate=settings.backtest_execution.fee_rate,
        slippage_rate=settings.backtest_execution.slippage_rate,
        max_position_fraction=settings.backtest_execution.max_position_fraction,
        allow_fractional_position=settings.backtest_execution.allow_fractional_position,
    )

    try:
        result = simulator.run_simulation(
            simulation_type="backtest",
            model_run_id=model_run_id,
            symbol=symbol,
            timeframe=timeframe,
            strategy=strategy,
            execution_config=execution,
            simulation_id=simulation_id,
        )
    except BacktestSimulatorError as exc:
        console.print(str(exc), style="red")
        sys.exit(1)

    _print_simulation_success(result)


@cli.command(name="run-paper-simulation")
@click.option("--model-run-id", required=True, help="Stored model run id to simulate.")
@click.option("--symbol", default=None, help="Optional symbol override.")
@click.option("--timeframe", default=None, help="Optional timeframe override.")
@click.option("--strategy-version", default=None, help="Optional strategy version label.")
@click.option("--simulation-id", default=None, help="Optional explicit simulation id.")
def run_paper_simulation_cmd(
    model_run_id: str,
    symbol: str | None,
    timeframe: str | None,
    strategy_version: str | None,
    simulation_id: str | None,
) -> None:
    """Run an offline paper simulation using the same core engine as backtesting."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.backtest.simulator import BacktestSimulator, BacktestSimulatorError
    from trading_bot.backtest.strategy import Strategy, get_default_strategy
    from trading_bot.schemas.backtest import ExecutionConfig, StrategyConfig

    simulator = BacktestSimulator.from_settings(settings)
    configured_strategy = get_default_strategy(settings)
    strategy = Strategy(
        config=StrategyConfig(
            strategy_version=strategy_version or configured_strategy.strategy_version,
            type=configured_strategy.type,
            entry_threshold=configured_strategy.entry_threshold,
            exit_threshold=configured_strategy.exit_threshold,
            minimum_holding_bars=configured_strategy.minimum_holding_bars,
            cooldown_bars=configured_strategy.cooldown_bars,
        )
    )
    execution = ExecutionConfig(
        fill_model=settings.backtest_execution.fill_model,
        starting_cash=settings.backtest_execution.starting_cash,
        fee_rate=settings.backtest_execution.fee_rate,
        slippage_rate=settings.backtest_execution.slippage_rate,
        max_position_fraction=settings.backtest_execution.max_position_fraction,
        allow_fractional_position=settings.backtest_execution.allow_fractional_position,
    )

    try:
        result = simulator.run_simulation(
            simulation_type="paper_simulation",
            model_run_id=model_run_id,
            symbol=symbol,
            timeframe=timeframe,
            strategy=strategy,
            execution_config=execution,
            simulation_id=simulation_id,
        )
    except BacktestSimulatorError as exc:
        console.print(str(exc), style="red")
        sys.exit(1)

    _print_simulation_success(result)


@cli.command(name="show-simulation-summary")
@click.option(
    "--simulation-type",
    required=True,
    type=click.Choice(["backtest", "paper_simulation"]),
    help="Simulation output family to inspect.",
)
@click.option("--simulation-id", required=True, help="Stored simulation id.")
def show_simulation_summary_cmd(simulation_type: str, simulation_id: str) -> None:
    """Print a stored simulation summary."""
    setup_logging("INFO")
    settings = load_settings()

    if simulation_type == "backtest":
        from trading_bot.storage.backtest_store import BacktestStore

        store = BacktestStore(PROJECT_ROOT / settings.simulations.backtests_dir)
    else:
        from trading_bot.storage.paper_sim_store import PaperSimulationStore

        store = PaperSimulationStore(PROJECT_ROOT / settings.simulations.paper_simulations_dir)

    summary = store.read_summary(simulation_id)
    if not summary:
        console.print("No stored simulation summary found", style="yellow")
        return

    _print_simulation_summary(summary)


@cli.command(name="list-simulations")
@click.option(
    "--simulation-type",
    required=True,
    type=click.Choice(["backtest", "paper_simulation"]),
    help="Simulation output family to inspect.",
)
@click.option("--sort-by", default=None, help="Metric column to sort by.")
def list_simulations_cmd(simulation_type: str, sort_by: str | None) -> None:
    """List stored backtests or offline paper simulations."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.backtest.metrics import compare_simulation_summaries

    if simulation_type == "backtest":
        from trading_bot.storage.backtest_store import BacktestStore

        store = BacktestStore(PROJECT_ROOT / settings.simulations.backtests_dir)
    else:
        from trading_bot.storage.paper_sim_store import PaperSimulationStore

        store = PaperSimulationStore(PROJECT_ROOT / settings.simulations.paper_simulations_dir)

    summaries = store.list_summaries()
    comparison_df = compare_simulation_summaries(
        summaries,
        sort_by=sort_by or settings.simulations.default_sort_metric,
    )
    if comparison_df.empty:
        console.print("No stored simulations found", style="yellow")
        return

    _print_simulation_list(comparison_df)


@cli.command(name="compare-simulations")
@click.option(
    "--simulation-type",
    required=True,
    type=click.Choice(["backtest", "paper_simulation"]),
    help="Simulation output family to inspect.",
)
@click.option(
    "--simulation-ids",
    multiple=True,
    required=True,
    help="One or more stored simulation ids to compare.",
)
@click.option("--sort-by", default=None, help="Metric column to sort by.")
def compare_simulations_cmd(
    simulation_type: str,
    simulation_ids: tuple[str, ...],
    sort_by: str | None,
) -> None:
    """Compare two or more stored simulations on summary metrics."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.backtest.metrics import compare_simulation_summaries

    if simulation_type == "backtest":
        from trading_bot.storage.backtest_store import BacktestStore

        store = BacktestStore(PROJECT_ROOT / settings.simulations.backtests_dir)
    else:
        from trading_bot.storage.paper_sim_store import PaperSimulationStore

        store = PaperSimulationStore(PROJECT_ROOT / settings.simulations.paper_simulations_dir)

    summaries = [
        summary
        for simulation_id in simulation_ids
        if (summary := store.read_summary(simulation_id))
    ]
    comparison_df = compare_simulation_summaries(
        summaries,
        sort_by=sort_by or settings.simulations.default_sort_metric,
    )
    if comparison_df.empty:
        console.print("No matching simulations found for comparison", style="yellow")
        return

    _print_simulation_comparison(comparison_df)


@cli.command(name="run-paper-once")
@click.option("--loop-id", default=None, help="Optional paper-loop id override.")
@click.option("--target-name", default=None, help="Optional target name override.")
@click.option("--target-key", default=None, help="Optional target key override.")
@click.option("--model-run-id", default=None, help="Optional stored model run id override.")
@click.option("--optimization-id", default=None, help="Optional optimization id override.")
@click.option("--symbol", default=None, help="Optional symbol override.")
@click.option("--timeframe", default=None, help="Optional timeframe override.")
@click.option("--feature-version", default=None, help="Optional feature version override.")
@click.option("--provider", default=None, help="Optional news provider override.")
def run_paper_once_cmd(
    loop_id: str | None,
    target_name: str | None,
    target_key: str | None,
    model_run_id: str | None,
    optimization_id: str | None,
    symbol: str | None,
    timeframe: str | None,
    feature_version: str | None,
    provider: str | None,
) -> None:
    """Run one scheduled offline paper-trading update cycle."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.paper.loop import (
        ScheduledPaperTradingService,
        ScheduledPaperTradingServiceError,
    )

    service = ScheduledPaperTradingService.from_settings(settings)
    selection = _resolve_paper_loop_selection(
        settings=settings,
        loop_id=loop_id,
        target_name=target_name,
        target_key=target_key,
        model_run_id=model_run_id,
        optimization_id=optimization_id,
        symbol=symbol,
        timeframe=timeframe,
        feature_version=feature_version,
        provider=provider,
    )
    try:
        result = service.run_once(selection=selection)
    except (ScheduledPaperTradingServiceError, Exception) as exc:
        console.print(str(exc), style="red")
        sys.exit(1)

    _print_paper_loop_result(result)


@cli.command(name="start-paper-loop")
@click.option("--loop-id", default=None, help="Optional paper-loop id override.")
@click.option("--target-name", default=None, help="Optional target name override.")
@click.option("--target-key", default=None, help="Optional target key override.")
@click.option("--model-run-id", default=None, help="Optional stored model run id override.")
@click.option("--optimization-id", default=None, help="Optional optimization id override.")
@click.option("--symbol", default=None, help="Optional symbol override.")
@click.option("--timeframe", default=None, help="Optional timeframe override.")
@click.option("--feature-version", default=None, help="Optional feature version override.")
@click.option("--provider", default=None, help="Optional news provider override.")
def start_paper_loop_cmd(
    loop_id: str | None,
    target_name: str | None,
    target_key: str | None,
    model_run_id: str | None,
    optimization_id: str | None,
    symbol: str | None,
    timeframe: str | None,
    feature_version: str | None,
    provider: str | None,
) -> None:
    """Start the scheduled offline paper-trading loop as a background process."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.storage.paper_loop_store import PaperLoopStore

    selection = _resolve_paper_loop_selection(
        settings=settings,
        loop_id=loop_id,
        target_name=target_name,
        target_key=target_key,
        model_run_id=model_run_id,
        optimization_id=optimization_id,
        symbol=symbol,
        timeframe=timeframe,
        feature_version=feature_version,
        provider=provider,
    )
    store = PaperLoopStore(PROJECT_ROOT / settings.paper_loop.output_dir)
    existing_status = store.read_status(selection.loop_id)
    existing_pid = existing_status.get("pid")
    if existing_status.get("running") and isinstance(existing_pid, int):
        try:
            os.kill(existing_pid, 0)
            console.print(
                f"Paper loop is already running with pid={existing_pid}",
                style="yellow",
            )
            return
        except OSError:
            pass

    loop_dir = store.loop_dir(selection.loop_id)
    loop_dir.mkdir(parents=True, exist_ok=True)
    log_path = loop_dir / settings.paper_loop.daemon_log_filename
    command = [
        sys.executable,
        "-m",
        "trading_bot.paper.daemon",
        "--loop-id",
        selection.loop_id,
        "--target-name",
        selection.target_name,
        "--target-key",
        selection.target_key,
        "--model-run-id",
        selection.model_run_id,
        "--optimization-id",
        selection.optimization_id,
        "--symbol",
        selection.symbol,
        "--timeframe",
        selection.timeframe,
        "--feature-version",
        selection.feature_version,
        "--provider",
        selection.provider,
    ]
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
    store.write_config(selection.loop_id, selection.to_dict())
    store.write_status(
        selection.loop_id,
        {
            "loop_id": selection.loop_id,
            "running": True,
            "pid": process.pid,
            "target_name": selection.target_name,
            "target_key": selection.target_key,
            "model_run_id": selection.model_run_id,
            "optimization_id": selection.optimization_id,
            "symbol": selection.symbol,
            "timeframe": selection.timeframe,
            "feature_version": selection.feature_version,
            "last_updated_at": datetime.now(UTC).isoformat(),
            "note": f"Background paper loop started. Log file: {log_path}",
        },
    )
    console.print(
        Panel(
            f"[bold green]Paper loop started[/bold green]\n"
            f"  Loop ID:   {selection.loop_id}\n"
            f"  PID:       {process.pid}\n"
            f"  Log:       {log_path}",
            title="✅ Success",
            style="green",
        )
    )


@cli.command(name="show-paper-status")
@click.option("--loop-id", default=None, help="Optional paper-loop id override.")
def show_paper_status_cmd(loop_id: str | None) -> None:
    """Print the stored status for the scheduled offline paper-trading loop."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.storage.paper_loop_store import PaperLoopStore

    resolved_loop_id = loop_id or settings.paper_loop.loop_id
    store = PaperLoopStore(PROJECT_ROOT / settings.paper_loop.output_dir)
    status = store.read_status(resolved_loop_id)
    if not status:
        console.print("No stored paper-loop status found", style="yellow")
        return
    pid = status.get("pid")
    running = bool(status.get("running"))
    if running and isinstance(pid, int):
        try:
            os.kill(pid, 0)
        except OSError:
            status["running"] = False
            status["note"] = "Stored PID is no longer active."
            store.write_status(resolved_loop_id, status)
    _print_paper_loop_status(status)


@cli.command(name="show-paper-trades")
@click.option("--loop-id", default=None, help="Optional paper-loop id override.")
@click.option("--limit", default=20, show_default=True, help="Maximum trades to display.")
def show_paper_trades_cmd(loop_id: str | None, limit: int) -> None:
    """Show stored closed trades for the scheduled paper loop."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.storage.paper_loop_store import PaperLoopStore

    resolved_loop_id = loop_id or settings.paper_loop.loop_id
    store = PaperLoopStore(PROJECT_ROOT / settings.paper_loop.output_dir)
    trades_df = store.read_trades(resolved_loop_id)
    if trades_df.empty:
        console.print("No stored paper trades found", style="yellow")
        return
    _print_paper_trades(trades_df.tail(limit).reset_index(drop=True))


@cli.command(name="stop-paper-loop")
@click.option("--loop-id", default=None, help="Optional paper-loop id override.")
def stop_paper_loop_cmd(loop_id: str | None) -> None:
    """Stop a background scheduled paper-trading loop when one is running."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.storage.paper_loop_store import PaperLoopStore

    resolved_loop_id = loop_id or settings.paper_loop.loop_id
    store = PaperLoopStore(PROJECT_ROOT / settings.paper_loop.output_dir)
    status = store.read_status(resolved_loop_id)
    pid = status.get("pid")
    if not status or not isinstance(pid, int):
        console.print("No active paper loop found to stop", style="yellow")
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        status["running"] = False
        status["pid"] = None
        status["note"] = "Paper loop process was already stopped."
        store.write_status(resolved_loop_id, status)
        console.print("Paper loop process was already stopped", style="yellow")
        return

    status["running"] = False
    status["pid"] = None
    status["last_updated_at"] = datetime.now(UTC).isoformat()
    status["note"] = "Stop signal sent to background paper loop."
    store.write_status(resolved_loop_id, status)
    console.print(f"Stop signal sent to paper loop pid={pid}", style="green")


@cli.command(name="run-validation-suite")
@click.option("--model-run-id", required=True, help="Stored model run id to validate.")
@click.option("--validation-id", default=None, help="Optional explicit validation id.")
def run_validation_suite_cmd(model_run_id: str, validation_id: str | None) -> None:
    """Run the full model-vs-baselines validation suite and persist the report."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.validation.suite import ValidationSuite, ValidationSuiteError

    suite = ValidationSuite.from_settings(settings)
    try:
        report = suite.run_validation_suite(
            model_run_id=model_run_id,
            validation_id=validation_id,
        )
    except ValidationSuiteError as exc:
        console.print(str(exc), style="red")
        sys.exit(1)

    _print_validation_report(report)


@cli.command(name="compare-baselines")
@click.option("--validation-id", required=True, help="Stored validation report id.")
def compare_baselines_cmd(validation_id: str) -> None:
    """Print the ranked model-vs-baselines comparison table from a validation report."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.storage.validation_store import ValidationStore

    store = ValidationStore(PROJECT_ROOT / settings.validation_suite.output_dir)
    comparison_df = store.read_table(validation_id, "strategy_comparison")
    if comparison_df.empty:
        console.print("No stored strategy comparison found for validation report", style="yellow")
        return

    _print_baseline_comparison(comparison_df)


@cli.command(name="show-validation-report")
@click.option("--validation-id", required=True, help="Stored validation report id.")
def show_validation_report_cmd(validation_id: str) -> None:
    """Print the top-level validation report and final conclusion."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.storage.validation_store import ValidationStore

    store = ValidationStore(PROJECT_ROOT / settings.validation_suite.output_dir)
    report = store.read_report_json(validation_id)
    if not report:
        console.print("No stored validation report found", style="yellow")
        return

    _print_validation_report(report)


@cli.command(name="analyze-predictions")
@click.option("--model-run-id", required=True, help="Stored model run id to analyze.")
def analyze_predictions_cmd(model_run_id: str) -> None:
    """Print prediction-confidence diagnostics for one stored model run."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.models.model_registry import ModelRegistry, ModelRegistryError
    from trading_bot.storage.parquet_store import ParquetStore
    from trading_bot.validation.suite import build_prediction_diagnostics
    from trading_bot.backtest.simulator import build_simulation_frame

    registry = ModelRegistry.from_settings(settings)
    market_store = ParquetStore(PROJECT_ROOT / settings.market_data.processed_data_dir)
    try:
        resolved = registry.resolve_run(model_run_id)
    except ModelRegistryError as exc:
        console.print(str(exc), style="yellow")
        return

    predictions_df = registry.read_predictions(model_run_id)
    market_df = market_store.read(resolved.symbol, resolved.timeframe)
    diagnostics_df = build_prediction_diagnostics(
        predictions_df=build_simulation_frame(predictions_df, market_df),
        bucket_edges=settings.validation_suite.bucket_edges,
    )
    if diagnostics_df.empty:
        console.print("No prediction diagnostics available for this run", style="yellow")
        return

    _print_prediction_diagnostics(diagnostics_df)


@cli.command(name="show-failure-cases")
@click.option("--validation-id", required=True, help="Stored validation report id.")
def show_failure_cases_cmd(validation_id: str) -> None:
    """Print the stored failure-case analysis for one validation report."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.storage.validation_store import ValidationStore

    store = ValidationStore(PROJECT_ROOT / settings.validation_suite.output_dir)
    report = store.read_report_json(validation_id)
    if not report:
        console.print("No stored validation report found", style="yellow")
        return

    failure_analysis = report.get("failure_analysis", {})
    if not isinstance(failure_analysis, dict) or not failure_analysis:
        console.print("No stored failure analysis found for validation report", style="yellow")
        return

    _print_failure_cases(failure_analysis)


@cli.command(name="run-strategy-search")
@click.option("--model-run-id", required=True, help="Stored model run id to optimize.")
@click.option("--symbol", default=None, help="Optional symbol override.")
@click.option("--timeframe", default=None, help="Optional timeframe override.")
@click.option("--optimization-id", default=None, help="Optional explicit optimization id.")
def run_strategy_search_cmd(
    model_run_id: str,
    symbol: str | None,
    timeframe: str | None,
    optimization_id: str | None,
) -> None:
    """Run a grid search over strategy thresholds and execution rules."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.optimization.strategy_search import (
        StrategySearchRunner,
        StrategySearchRunnerError,
    )

    runner = StrategySearchRunner.from_settings(settings)
    try:
        report = runner.run_search(
            model_run_id=model_run_id,
            symbol=symbol,
            timeframe=timeframe,
            optimization_id=optimization_id,
        )
    except StrategySearchRunnerError as exc:
        console.print(str(exc), style="red")
        sys.exit(1)

    _print_optimization_report(report)


@cli.command(name="show-strategy-search-report")
@click.option("--optimization-id", required=True, help="Stored optimization id.")
def show_strategy_search_report_cmd(optimization_id: str) -> None:
    """Print the optimization report for a strategy search."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.storage.optimization_store import OptimizationStore

    store = OptimizationStore(PROJECT_ROOT / settings.optimization_storage.output_dir)
    report = store.read_report_json(optimization_id)
    if not report:
        console.print("No stored strategy-search report found", style="yellow")
        return

    _print_optimization_report(report)


@cli.command(name="list-strategy-searches")
def list_strategy_searches_cmd() -> None:
    """List stored strategy-search reports."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.storage.optimization_store import OptimizationStore

    store = OptimizationStore(PROJECT_ROOT / settings.optimization_storage.output_dir)
    searches = store.list_reports()
    if not searches:
        console.print("No stored strategy searches found", style="yellow")
        return

    _print_strategy_searches(pd.DataFrame(searches))


@cli.command(name="show-top-strategies")
@click.option("--optimization-id", required=True, help="Stored optimization id.")
@click.option("--top-k", default=20, show_default=True, type=int, help="Rows to print.")
def show_top_strategies_cmd(optimization_id: str, top_k: int) -> None:
    """Print the top strategy-search candidates from one optimization run."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.storage.optimization_store import OptimizationStore

    store = OptimizationStore(PROJECT_ROOT / settings.optimization_storage.output_dir)
    model_run_id = store.find_model_run_id(optimization_id)
    if model_run_id is None:
        console.print("No stored strategy-search report found", style="yellow")
        return

    top_candidates_df = store.read_top_candidates(model_run_id, optimization_id).head(top_k)
    if top_candidates_df.empty:
        console.print("No stored top-candidate table found", style="yellow")
        return

    _print_top_strategies(top_candidates_df)


@cli.command(name="compare-optimized-strategy")
@click.option("--optimization-id", required=True, help="Stored optimization id.")
def compare_optimized_strategy_cmd(optimization_id: str) -> None:
    """Compare the best optimized strategy against baselines and the original model strategy."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.storage.optimization_store import OptimizationStore

    store = OptimizationStore(PROJECT_ROOT / settings.optimization_storage.output_dir)
    model_run_id = store.find_model_run_id(optimization_id)
    if model_run_id is None:
        console.print("No stored strategy-search report found", style="yellow")
        return

    comparison_df = store.read_baseline_comparison(model_run_id, optimization_id)
    if comparison_df.empty:
        console.print("No stored optimized strategy comparison found", style="yellow")
        return

    _print_baseline_comparison(comparison_df)


@cli.command(name="build-multi-horizon-datasets")
@click.option("--symbol", default=None, help="Trading pair symbol (default: from config)")
@click.option("--timeframe", default=None, help="Candle timeframe (default: from config)")
@click.option("--asset", default=None, help="Asset tag (default: from config)")
@click.option(
    "--feature-version",
    default=None,
    help="Feature version to use across all tested horizons.",
)
@click.option(
    "--horizons",
    default=None,
    help="Comma-separated horizons in bars, for example 4,8,16.",
)
def build_multi_horizon_datasets_cmd(
    symbol: str | None,
    timeframe: str | None,
    asset: str | None,
    feature_version: str | None,
    horizons: str | None,
) -> None:
    """Build one supervised dataset per requested prediction horizon."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.research.multi_horizon import (
        MultiHorizonResearchError,
        MultiHorizonResearchRunner,
    )

    runner = MultiHorizonResearchRunner.from_settings(settings)
    try:
        rows = runner.build_datasets(
            symbol=symbol,
            timeframe=timeframe,
            asset=asset,
            feature_version=feature_version or settings.multi_horizon.default_feature_version,
            horizons=_parse_horizons_option(horizons, settings.multi_horizon.horizon_bars),
        )
    except MultiHorizonResearchError as exc:
        console.print(str(exc), style="red")
        sys.exit(1)

    _print_multi_horizon_rows("Multi-Horizon Datasets", rows)


@cli.command(name="train-multi-horizon-models")
@click.option("--symbol", default=None, help="Trading pair symbol (default: from config)")
@click.option("--timeframe", default=None, help="Candle timeframe (default: from config)")
@click.option("--asset", default=None, help="Asset tag (default: from config)")
@click.option(
    "--feature-version",
    default=None,
    help="Feature version to use across all tested horizons.",
)
@click.option(
    "--horizons",
    default=None,
    help="Comma-separated horizons in bars, for example 4,8,16.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"]),
    default=None,
    help="Training device request (default: from config).",
)
def train_multi_horizon_models_cmd(
    symbol: str | None,
    timeframe: str | None,
    asset: str | None,
    feature_version: str | None,
    horizons: str | None,
    device: str | None,
) -> None:
    """Train one baseline walk-forward model per requested horizon."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.research.multi_horizon import (
        MultiHorizonResearchError,
        MultiHorizonResearchRunner,
    )

    runner = MultiHorizonResearchRunner.from_settings(settings)
    try:
        rows = runner.train_models(
            symbol=symbol,
            timeframe=timeframe,
            asset=asset,
            feature_version=feature_version or settings.multi_horizon.default_feature_version,
            horizons=_parse_horizons_option(horizons, settings.multi_horizon.horizon_bars),
            requested_device=device or settings.training.device,
        )
    except MultiHorizonResearchError as exc:
        console.print(str(exc), style="red")
        sys.exit(1)

    _print_multi_horizon_rows("Multi-Horizon Training Runs", rows)


@cli.command(name="run-multi-horizon-validation")
@click.option("--symbol", default=None, help="Trading pair symbol (default: from config)")
@click.option("--timeframe", default=None, help="Candle timeframe (default: from config)")
@click.option(
    "--horizons",
    default=None,
    help="Comma-separated horizons in bars, for example 4,8,16.",
)
def run_multi_horizon_validation_cmd(
    symbol: str | None,
    timeframe: str | None,
    horizons: str | None,
) -> None:
    """Run the validation suite for the latest model run at each requested horizon."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.research.multi_horizon import (
        MultiHorizonResearchError,
        MultiHorizonResearchRunner,
    )

    runner = MultiHorizonResearchRunner.from_settings(settings)
    try:
        rows = runner.run_validation(
            symbol=symbol,
            timeframe=timeframe,
            horizons=_parse_horizons_option(horizons, settings.multi_horizon.horizon_bars),
        )
    except MultiHorizonResearchError as exc:
        console.print(str(exc), style="red")
        sys.exit(1)

    _print_multi_horizon_rows("Multi-Horizon Validation", rows)


@cli.command(name="run-multi-horizon-strategy-search")
@click.option("--symbol", default=None, help="Trading pair symbol (default: from config)")
@click.option("--timeframe", default=None, help="Candle timeframe (default: from config)")
@click.option(
    "--horizons",
    default=None,
    help="Comma-separated horizons in bars, for example 4,8,16.",
)
def run_multi_horizon_strategy_search_cmd(
    symbol: str | None,
    timeframe: str | None,
    horizons: str | None,
) -> None:
    """Run strategy search for the latest model run at each requested horizon."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.research.multi_horizon import (
        MultiHorizonResearchError,
        MultiHorizonResearchRunner,
    )

    runner = MultiHorizonResearchRunner.from_settings(settings)
    try:
        rows = runner.run_strategy_search(
            symbol=symbol,
            timeframe=timeframe,
            horizons=_parse_horizons_option(horizons, settings.multi_horizon.horizon_bars),
        )
    except MultiHorizonResearchError as exc:
        console.print(str(exc), style="red")
        sys.exit(1)

    _print_multi_horizon_rows("Multi-Horizon Strategy Search", rows)


@cli.command(name="show-horizon-comparison-report")
@click.option("--symbol", default=None, help="Trading pair symbol (default: from config)")
@click.option("--timeframe", default=None, help="Candle timeframe (default: from config)")
@click.option("--asset", default=None, help="Asset tag (default: from config)")
@click.option(
    "--feature-version",
    default=None,
    help="Feature version to include in the comparison report.",
)
@click.option(
    "--horizons",
    default=None,
    help="Comma-separated horizons in bars, for example 4,8,16.",
)
@click.option("--comparison-id", default=None, help="Optional explicit comparison id.")
def show_horizon_comparison_report_cmd(
    symbol: str | None,
    timeframe: str | None,
    asset: str | None,
    feature_version: str | None,
    horizons: str | None,
    comparison_id: str | None,
) -> None:
    """Build and print the ranked multi-horizon comparison report."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.research.multi_horizon import (
        MultiHorizonResearchError,
        MultiHorizonResearchRunner,
    )

    runner = MultiHorizonResearchRunner.from_settings(settings)
    try:
        report = runner.build_horizon_comparison_report(
            symbol=symbol or settings.datasets.symbol,
            timeframe=timeframe or settings.datasets.timeframe,
            asset=asset or settings.datasets.asset,
            feature_version=feature_version or settings.multi_horizon.default_feature_version,
            horizons=_parse_horizons_option(horizons, settings.multi_horizon.horizon_bars),
            comparison_id=comparison_id,
        )
    except MultiHorizonResearchError as exc:
        console.print(str(exc), style="red")
        sys.exit(1)

    _print_horizon_comparison_report(report)


@cli.command(name="build-fee-aware-datasets")
@click.option("--symbol", default=None, help="Trading pair symbol (default: from config)")
@click.option("--timeframe", default=None, help="Candle timeframe (default: from config)")
@click.option("--asset", default=None, help="Asset tag (default: from config)")
@click.option(
    "--feature-version",
    default=None,
    help="Feature version to use across all tested target definitions.",
)
@click.option(
    "--targets",
    default=None,
    help="Comma-separated target keys, for example fee_4,fee_8,ret050_16.",
)
def build_fee_aware_datasets_cmd(
    symbol: str | None,
    timeframe: str | None,
    asset: str | None,
    feature_version: str | None,
    targets: str | None,
) -> None:
    """Build one supervised dataset per fee-aware or minimum-return target definition."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.research.fee_aware_targets import (
        FeeAwareTargetResearchError,
        FeeAwareTargetResearchRunner,
    )

    runner = FeeAwareTargetResearchRunner.from_settings(settings)
    try:
        rows = runner.build_datasets(
            symbol=symbol,
            timeframe=timeframe,
            asset=asset,
            feature_version=feature_version or settings.fee_aware_targets.default_feature_version,
            target_keys=_parse_targets_option(targets, settings.fee_aware_targets.target_keys),
        )
    except FeeAwareTargetResearchError as exc:
        console.print(str(exc), style="red")
        sys.exit(1)

    _print_multi_horizon_rows("Fee-Aware Datasets", rows)


@cli.command(name="train-fee-aware-target-models")
@click.option("--symbol", default=None, help="Trading pair symbol (default: from config)")
@click.option("--timeframe", default=None, help="Candle timeframe (default: from config)")
@click.option("--asset", default=None, help="Asset tag (default: from config)")
@click.option(
    "--feature-version",
    default=None,
    help="Feature version to use across all tested target definitions.",
)
@click.option(
    "--targets",
    default=None,
    help="Comma-separated target keys, for example fee_4,fee_8,ret050_16.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"]),
    default=None,
    help="Training device request (default: from config).",
)
def train_fee_aware_target_models_cmd(
    symbol: str | None,
    timeframe: str | None,
    asset: str | None,
    feature_version: str | None,
    targets: str | None,
    device: str | None,
) -> None:
    """Train one baseline walk-forward model per fee-aware target definition."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.research.fee_aware_targets import (
        FeeAwareTargetResearchError,
        FeeAwareTargetResearchRunner,
    )

    runner = FeeAwareTargetResearchRunner.from_settings(settings)
    try:
        rows = runner.train_models(
            symbol=symbol,
            timeframe=timeframe,
            asset=asset,
            feature_version=feature_version or settings.fee_aware_targets.default_feature_version,
            target_keys=_parse_targets_option(targets, settings.fee_aware_targets.target_keys),
            requested_device=device or settings.training.device,
        )
    except FeeAwareTargetResearchError as exc:
        console.print(str(exc), style="red")
        sys.exit(1)

    _print_multi_horizon_rows("Fee-Aware Training Runs", rows)


@cli.command(name="run-fee-aware-validation")
@click.option("--symbol", default=None, help="Trading pair symbol (default: from config)")
@click.option("--timeframe", default=None, help="Candle timeframe (default: from config)")
@click.option(
    "--targets",
    default=None,
    help="Comma-separated target keys, for example fee_4,fee_8,ret050_16.",
)
def run_fee_aware_validation_cmd(
    symbol: str | None,
    timeframe: str | None,
    targets: str | None,
) -> None:
    """Run validation for the latest stored model run at each fee-aware target definition."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.research.fee_aware_targets import (
        FeeAwareTargetResearchError,
        FeeAwareTargetResearchRunner,
    )

    runner = FeeAwareTargetResearchRunner.from_settings(settings)
    try:
        rows = runner.run_validation(
            symbol=symbol,
            timeframe=timeframe,
            target_keys=_parse_targets_option(targets, settings.fee_aware_targets.target_keys),
        )
    except FeeAwareTargetResearchError as exc:
        console.print(str(exc), style="red")
        sys.exit(1)

    _print_multi_horizon_rows("Fee-Aware Validation", rows)


@cli.command(name="run-fee-aware-strategy-search")
@click.option("--symbol", default=None, help="Trading pair symbol (default: from config)")
@click.option("--timeframe", default=None, help="Candle timeframe (default: from config)")
@click.option(
    "--targets",
    default=None,
    help="Comma-separated target keys, for example fee_4,fee_8,ret050_16.",
)
def run_fee_aware_strategy_search_cmd(
    symbol: str | None,
    timeframe: str | None,
    targets: str | None,
) -> None:
    """Run strategy search for the latest stored model run at each fee-aware target definition."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.research.fee_aware_targets import (
        FeeAwareTargetResearchError,
        FeeAwareTargetResearchRunner,
    )

    runner = FeeAwareTargetResearchRunner.from_settings(settings)
    try:
        rows = runner.run_strategy_search(
            symbol=symbol,
            timeframe=timeframe,
            target_keys=_parse_targets_option(targets, settings.fee_aware_targets.target_keys),
        )
    except FeeAwareTargetResearchError as exc:
        console.print(str(exc), style="red")
        sys.exit(1)

    _print_multi_horizon_rows("Fee-Aware Strategy Search", rows)


@cli.command(name="show-target-comparison-report")
@click.option("--symbol", default=None, help="Trading pair symbol (default: from config)")
@click.option("--timeframe", default=None, help="Candle timeframe (default: from config)")
@click.option("--asset", default=None, help="Asset tag (default: from config)")
@click.option(
    "--feature-version",
    default=None,
    help="Feature version to include in the comparison report.",
)
@click.option(
    "--targets",
    default=None,
    help="Comma-separated target keys, for example fee_4,fee_8,ret050_16.",
)
@click.option("--comparison-id", default=None, help="Optional explicit comparison id.")
def show_target_comparison_report_cmd(
    symbol: str | None,
    timeframe: str | None,
    asset: str | None,
    feature_version: str | None,
    targets: str | None,
    comparison_id: str | None,
) -> None:
    """Build and print the ranked fee-aware target comparison report."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.research.fee_aware_targets import (
        FeeAwareTargetResearchError,
        FeeAwareTargetResearchRunner,
    )

    runner = FeeAwareTargetResearchRunner.from_settings(settings)
    try:
        report = runner.build_target_comparison_report(
            symbol=symbol or settings.datasets.symbol,
            timeframe=timeframe or settings.datasets.timeframe,
            asset=asset or settings.datasets.asset,
            feature_version=feature_version or settings.fee_aware_targets.default_feature_version,
            target_keys=_parse_targets_option(targets, settings.fee_aware_targets.target_keys),
            comparison_id=comparison_id,
        )
    except FeeAwareTargetResearchError as exc:
        console.print(str(exc), style="red")
        sys.exit(1)

    _print_target_comparison_report(report)


@cli.command(name="tag-run-status")
@click.option("--run-id", required=True, help="Stored training run id.")
@click.option(
    "--status",
    required=True,
    type=click.Choice(
        ["candidate", "rejected", "archived", "promoted_for_backtest"]
    ),
    help="Promotion status to assign.",
)
@click.option("--note", default=None, help="Optional human-readable note.")
def tag_run_status_cmd(run_id: str, status: str, note: str | None) -> None:
    """Update the local promotion status and note for one tracked run."""
    setup_logging("INFO")
    settings = load_settings()

    from trading_bot.experiments.registry import ExperimentRegistry, ExperimentRegistryError

    registry = ExperimentRegistry.from_settings(settings)
    try:
        updated = registry.set_run_status(run_id=run_id, status=status, note=note)
    except ExperimentRegistryError as exc:
        console.print(str(exc), style="yellow")
        return

    console.print(
        Panel(
            f"[bold green]Run status updated[/bold green]\n"
            f"  Run ID: {run_id}\n"
            f"  Status: {updated.get('status')}\n"
            f"  Note:   {updated.get('notes', '')}",
            title="✅ Success",
            style="green",
        )
    )


if __name__ == "__main__":
    cli()
