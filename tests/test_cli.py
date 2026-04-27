"""Tests for CLI commands."""

from __future__ import annotations

from click.testing import CliRunner

from trading_bot.cli import cli
from trading_bot.experiments.registry import ExperimentRegistry, ExperimentRegistryError
from trading_bot.experiments.tracker import ExperimentTracker
from trading_bot.features.feature_pipeline import FeaturePipeline
from trading_bot.labeling.dataset_builder import SupervisedDatasetBuilder
from trading_bot.models.model_registry import ModelRegistry, ModelRegistryError
from trading_bot.models.train import TrainingPipelineError, WalkForwardTrainingPipeline
from trading_bot.nlp.news_enrichment import NewsEnrichmentService
from trading_bot.research.fee_aware_targets import (
    FeeAwareTargetResearchError,
    FeeAwareTargetResearchRunner,
)
from trading_bot.research.multi_horizon import MultiHorizonResearchError, MultiHorizonResearchRunner


class TestCli:
    """Test suite for CLI commands."""

    def setup_method(self) -> None:
        """Set up a Click test runner."""
        self.runner = CliRunner()

    def test_cli_version(self) -> None:
        """--version should print version string."""
        result = self.runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_health_check(self) -> None:
        """health-check should pass without errors."""
        result = self.runner.invoke(cli, ["health-check"])
        assert result.exit_code == 0
        assert "Health check passed" in result.output or "All systems go" in result.output

    def test_show_config(self) -> None:
        """show-config should display config without crashing."""
        result = self.runner.invoke(cli, ["show-config"])
        assert result.exit_code == 0
        assert "BTCUSDT" in result.output

    def test_show_config_redacts_secrets(self) -> None:
        """show-config should redact API keys by default."""
        result = self.runner.invoke(cli, ["show-config"])
        assert result.exit_code == 0
        # Secrets should be redacted or show "(not set)"
        assert "not set" in result.output or "****" in result.output

    def test_project_tree(self) -> None:
        """project-tree should display module structure."""
        result = self.runner.invoke(cli, ["project-tree"])
        assert result.exit_code == 0
        # Should contain known subpackages
        assert "ingestion" in result.output or "models" in result.output

    def test_help(self) -> None:
        """--help should show available commands."""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "health-check" in result.output
        assert "show-config" in result.output
        assert "project-tree" in result.output
        assert "build-market-features" in result.output
        assert "build-news-features" in result.output
        assert "build-sentiment-news-features" in result.output
        assert "build-feature-table" in result.output
        assert "show-feature-stats" in result.output
        assert "enrich-news-sentiment" in result.output
        assert "show-enriched-news-stats" in result.output
        assert "build-labels" in result.output
        assert "build-supervised-dataset" in result.output
        assert "show-dataset-stats" in result.output
        assert "show-target-columns" in result.output
        assert "run-walk-forward-training" in result.output
        assert "show-run-metrics" in result.output
        assert "list-model-runs" in result.output
        assert "show-feature-importance" in result.output
        assert "refresh-experiment-registry" in result.output
        assert "list-experiment-runs" in result.output
        assert "show-run-report" in result.output
        assert "compare-runs" in result.output
        assert "tag-run-status" in result.output
        assert "show-sentiment-ablation" in result.output
        assert "run-backtest" in result.output
        assert "run-paper-simulation" in result.output
        assert "show-simulation-summary" in result.output
        assert "list-simulations" in result.output
        assert "compare-simulations" in result.output
        assert "run-validation-suite" in result.output
        assert "compare-baselines" in result.output
        assert "show-validation-report" in result.output
        assert "analyze-predictions" in result.output
        assert "show-failure-cases" in result.output
        assert "run-strategy-search" in result.output
        assert "show-strategy-search-report" in result.output
        assert "list-strategy-searches" in result.output
        assert "show-top-strategies" in result.output
        assert "compare-optimized-strategy" in result.output
        assert "build-multi-horizon-datasets" in result.output
        assert "train-multi-horizon-models" in result.output
        assert "run-multi-horizon-validation" in result.output
        assert "run-multi-horizon-strategy-search" in result.output
        assert "show-horizon-comparison-report" in result.output
        assert "build-fee-aware-datasets" in result.output
        assert "train-fee-aware-target-models" in result.output
        assert "run-fee-aware-validation" in result.output
        assert "run-fee-aware-strategy-search" in result.output
        assert "show-target-comparison-report" in result.output
        assert "start-paper-loop" in result.output
        assert "run-paper-once" in result.output
        assert "show-paper-status" in result.output
        assert "show-paper-trades" in result.output
        assert "stop-paper-loop" in result.output

    def test_show_feature_stats_handles_missing_dataset(self, monkeypatch) -> None:
        """show-feature-stats should handle a missing stored dataset cleanly."""

        class DummyPipeline:
            def inspect_dataset(self, *args, **kwargs):
                return {}

        monkeypatch.setattr(
            FeaturePipeline,
            "from_settings",
            classmethod(lambda cls, settings=None: DummyPipeline()),
        )

        result = self.runner.invoke(cli, ["show-feature-stats"])
        assert result.exit_code == 0
        assert "No stored feature_table dataset found" in result.output

    def test_show_dataset_stats_handles_missing_dataset(self, monkeypatch) -> None:
        """show-dataset-stats should handle a missing stored dataset cleanly."""

        class DummyBuilder:
            def inspect_dataset(self, *args, **kwargs):
                return {}

        monkeypatch.setattr(
            SupervisedDatasetBuilder,
            "from_settings",
            classmethod(lambda cls, settings=None: DummyBuilder()),
        )

        result = self.runner.invoke(cli, ["show-dataset-stats"])
        assert result.exit_code == 0
        assert "No stored supervised_dataset found" in result.output

    def test_show_target_columns_handles_missing_dataset(self, monkeypatch) -> None:
        """show-target-columns should handle a missing supervised dataset cleanly."""

        class DummyBuilder:
            def list_target_columns(self, *args, **kwargs):
                return []

        monkeypatch.setattr(
            SupervisedDatasetBuilder,
            "from_settings",
            classmethod(lambda cls, settings=None: DummyBuilder()),
        )

        result = self.runner.invoke(cli, ["show-target-columns"])
        assert result.exit_code == 0
        assert "No stored supervised dataset target columns found" in result.output

    def test_run_walk_forward_training_handles_failure(self, monkeypatch) -> None:
        """run-walk-forward-training should surface pipeline failures cleanly."""

        class DummyPipeline:
            def run_training(self, *args, **kwargs):
                raise TrainingPipelineError("missing supervised dataset")

        monkeypatch.setattr(
            WalkForwardTrainingPipeline,
            "from_settings",
            classmethod(lambda cls, settings=None: DummyPipeline()),
        )

        result = self.runner.invoke(cli, ["run-walk-forward-training"])
        assert result.exit_code == 1
        assert "Walk-forward training failed" in result.output

    def test_list_model_runs_handles_no_runs(self, monkeypatch) -> None:
        """list-model-runs should handle empty evaluation storage cleanly."""

        class DummyRegistry:
            def list_runs(self, *args, **kwargs):
                return []

        monkeypatch.setattr(
            ModelRegistry,
            "from_settings",
            classmethod(lambda cls, settings=None: DummyRegistry()),
        )

        result = self.runner.invoke(cli, ["list-model-runs"])
        assert result.exit_code == 0
        assert "No stored model runs found" in result.output

    def test_show_run_metrics_handles_missing_run(self, monkeypatch) -> None:
        """show-run-metrics should handle unknown run ids cleanly."""

        class DummyRegistry:
            def read_report(self, *args, **kwargs):
                raise ModelRegistryError("No stored model run found for run_id=RUN404")

        monkeypatch.setattr(
            ModelRegistry,
            "from_settings",
            classmethod(lambda cls, settings=None: DummyRegistry()),
        )

        result = self.runner.invoke(cli, ["show-run-metrics", "--run-id", "RUN404"])
        assert result.exit_code == 0
        assert "No stored model run found" in result.output

    def test_show_feature_importance_handles_missing_run(self, monkeypatch) -> None:
        """show-feature-importance should handle unknown run ids cleanly."""

        class DummyRegistry:
            def read_feature_importance(self, *args, **kwargs):
                raise ModelRegistryError("No stored model run found for run_id=RUN404")

        monkeypatch.setattr(
            ModelRegistry,
            "from_settings",
            classmethod(lambda cls, settings=None: DummyRegistry()),
        )

        result = self.runner.invoke(cli, ["show-feature-importance", "--run-id", "RUN404"])
        assert result.exit_code == 0
        assert "No stored model run found" in result.output

    def test_refresh_experiment_registry_handles_no_runs(self, monkeypatch) -> None:
        """refresh-experiment-registry should handle empty evaluation storage cleanly."""

        class DummyTracker:
            def refresh_registry(self):
                import pandas as pd

                return pd.DataFrame()

        monkeypatch.setattr(
            ExperimentTracker,
            "from_settings",
            classmethod(lambda cls, settings=None: DummyTracker()),
        )

        result = self.runner.invoke(cli, ["refresh-experiment-registry"])
        assert result.exit_code == 0
        assert "No stored evaluation runs found" in result.output

    def test_list_experiment_runs_handles_no_runs(self, monkeypatch) -> None:
        """list-experiment-runs should handle an empty registry cleanly."""

        class DummyRegistry:
            def list_runs(self, *args, **kwargs):
                import pandas as pd

                return pd.DataFrame()

        monkeypatch.setattr(
            ExperimentRegistry,
            "from_settings",
            classmethod(lambda cls, settings=None: DummyRegistry()),
        )

        result = self.runner.invoke(cli, ["list-experiment-runs"])
        assert result.exit_code == 0
        assert "No tracked experiment runs found" in result.output

    def test_show_run_report_handles_missing_run(self, monkeypatch) -> None:
        """show-run-report should handle unknown run ids cleanly."""

        class DummyTracker:
            def generate_run_report(self, *args, **kwargs):
                raise ValueError("No stored evaluation metadata found for run_id=RUN404")

        monkeypatch.setattr(
            ExperimentTracker,
            "from_settings",
            classmethod(lambda cls, settings=None: DummyTracker()),
        )

        result = self.runner.invoke(cli, ["show-run-report", "--run-id", "RUN404"])
        assert result.exit_code == 1
        assert "No stored evaluation metadata found" in result.output

    def test_tag_run_status_handles_missing_run(self, monkeypatch) -> None:
        """tag-run-status should handle unknown run ids cleanly."""

        class DummyRegistry:
            def set_run_status(self, *args, **kwargs):
                raise ExperimentRegistryError("No experiment summary found for run_id=RUN404")

        monkeypatch.setattr(
            ExperimentRegistry,
            "from_settings",
            classmethod(lambda cls, settings=None: DummyRegistry()),
        )

        result = self.runner.invoke(
            cli,
            ["tag-run-status", "--run-id", "RUN404", "--status", "candidate"],
        )
        assert result.exit_code == 0
        assert "No experiment summary found" in result.output

    def test_show_sentiment_ablation_handles_missing_runs(self, monkeypatch) -> None:
        """show-sentiment-ablation should handle unknown run ids cleanly."""
        from trading_bot.experiments.comparison import RunComparator

        class DummyComparator:
            def build_sentiment_ablation_summary(self, *args, **kwargs):
                return {}

        monkeypatch.setattr(
            RunComparator,
            "from_settings",
            classmethod(lambda cls, settings=None: DummyComparator()),
        )

        result = self.runner.invoke(
            cli,
            [
                "show-sentiment-ablation",
                "--baseline-run-id",
                "RUN_BASE",
                "--new-run-id",
                "RUN_NEW",
            ],
        )
        assert result.exit_code == 0
        assert "No matching runs found for sentiment ablation" in result.output

    def test_show_simulation_summary_handles_missing_summary(self) -> None:
        """show-simulation-summary should handle missing simulation ids cleanly."""
        result = self.runner.invoke(
            cli,
            [
                "show-simulation-summary",
                "--simulation-type",
                "backtest",
                "--simulation-id",
                "SIM404",
            ],
        )
        assert result.exit_code == 0
        assert "No stored simulation summary found" in result.output

    def test_show_validation_report_handles_missing_report(self) -> None:
        """show-validation-report should handle missing validation ids cleanly."""
        result = self.runner.invoke(
            cli,
            ["show-validation-report", "--validation-id", "VAL404"],
        )
        assert result.exit_code == 0
        assert "No stored validation report found" in result.output

    def test_show_strategy_search_report_handles_missing_report(self) -> None:
        """show-strategy-search-report should handle missing optimization ids cleanly."""
        result = self.runner.invoke(
            cli,
            ["show-strategy-search-report", "--optimization-id", "OPT404"],
        )
        assert result.exit_code == 0
        assert "No stored strategy-search report found" in result.output

    def test_show_horizon_comparison_report_handles_missing_runs(self, monkeypatch) -> None:
        """show-horizon-comparison-report should surface multi-horizon lookup failures."""

        class DummyRunner:
            def build_horizon_comparison_report(self, *args, **kwargs):
                raise MultiHorizonResearchError("No stored model run found for horizon=4bars")

        monkeypatch.setattr(
            MultiHorizonResearchRunner,
            "from_settings",
            classmethod(lambda cls, settings=None: DummyRunner()),
        )

        result = self.runner.invoke(
            cli,
            ["show-horizon-comparison-report", "--symbol", "BTCUSDT", "--timeframe", "15m"],
        )
        assert result.exit_code == 1
        assert "No stored model run found" in result.output

    def test_build_multi_horizon_datasets_renders_rows(self, monkeypatch) -> None:
        """build-multi-horizon-datasets should render the returned horizon rows."""

        class DummyRunner:
            def build_datasets(self, *args, **kwargs):
                return [
                    {
                        "horizon_bars": 4,
                        "target_column": "target_up_4bars",
                        "dataset_version": "v1_h4",
                    }
                ]

        monkeypatch.setattr(
            MultiHorizonResearchRunner,
            "from_settings",
            classmethod(lambda cls, settings=None: DummyRunner()),
        )

        result = self.runner.invoke(cli, ["build-multi-horizon-datasets"])
        assert result.exit_code == 0
        assert "target_up_4bars" in result.output

    def test_show_target_comparison_report_handles_missing_runs(self, monkeypatch) -> None:
        """show-target-comparison-report should surface fee-aware target lookup failures."""

        class DummyRunner:
            def build_target_comparison_report(self, *args, **kwargs):
                raise FeeAwareTargetResearchError("No stored model run found for target=fee_4")

        monkeypatch.setattr(
            FeeAwareTargetResearchRunner,
            "from_settings",
            classmethod(lambda cls, settings=None: DummyRunner()),
        )

        result = self.runner.invoke(
            cli,
            ["show-target-comparison-report", "--symbol", "BTCUSDT", "--timeframe", "15m"],
        )
        assert result.exit_code == 1
        assert "No stored model run found for target=fee_4" in result.output

    def test_build_fee_aware_datasets_renders_rows(self, monkeypatch) -> None:
        """build-fee-aware-datasets should render returned fee-aware target rows."""

        class DummyRunner:
            def build_datasets(self, *args, **kwargs):
                return [
                    {
                        "target_key": "fee_8",
                        "target_name": "target_long_net_positive_8bars",
                        "dataset_version": "fee_h8",
                    }
                ]

        monkeypatch.setattr(
            FeeAwareTargetResearchRunner,
            "from_settings",
            classmethod(lambda cls, settings=None: DummyRunner()),
        )

        result = self.runner.invoke(cli, ["build-fee-aware-datasets"])
        assert result.exit_code == 0
        assert "target_long_net_positive_8bars" in result.output

    def test_show_enriched_news_stats_handles_missing_data(self, monkeypatch) -> None:
        """show-enriched-news-stats should handle missing enriched data cleanly."""
        from trading_bot.storage.enriched_news_store import EnrichedNewsStore

        class DummyStore:
            def get_info(self, *args, **kwargs):
                return {}

        monkeypatch.setattr(EnrichedNewsStore, "__init__", lambda self, base_dir: None)
        monkeypatch.setattr(EnrichedNewsStore, "get_info", DummyStore.get_info)

        result = self.runner.invoke(cli, ["show-enriched-news-stats"])
        assert result.exit_code == 0
        assert "No enriched news data found" in result.output

    def test_enrich_news_sentiment_runs_successfully(self, monkeypatch) -> None:
        """enrich-news-sentiment should render a success summary from the service result."""
        from trading_bot.nlp.news_enrichment import NewsEnrichmentResult

        class DummyService:
            def enrich_news(self, *args, **kwargs):
                return NewsEnrichmentResult(
                    asset="BTC",
                    provider="cryptocompare",
                    model_name="ProsusAI/finbert",
                    enrichment_version="v1",
                    requested_device="cpu",
                    effective_device="cpu",
                    input_rows=10,
                    scored_rows=8,
                    skipped_existing_rows=1,
                    skipped_empty_text_rows=1,
                    failure_rows=0,
                    stored_rows=10,
                )

        monkeypatch.setattr(
            NewsEnrichmentService,
            "__init__",
            lambda self, **kwargs: None,
        )
        monkeypatch.setattr(NewsEnrichmentService, "enrich_news", DummyService.enrich_news)

        result = self.runner.invoke(cli, ["enrich-news-sentiment"])
        assert result.exit_code == 0
        assert "News sentiment enrichment complete" in result.output

    def test_show_paper_status_handles_missing_status(self) -> None:
        """show-paper-status should handle missing loop status cleanly."""
        result = self.runner.invoke(cli, ["show-paper-status"])
        assert result.exit_code == 0
        assert "No stored paper-loop status found" in result.output

    def test_show_paper_trades_handles_missing_trades(self) -> None:
        """show-paper-trades should handle missing stored trade logs cleanly."""
        result = self.runner.invoke(cli, ["show-paper-trades"])
        assert result.exit_code == 0
        assert "No stored paper trades found" in result.output

    def test_run_paper_once_handles_failure(self, monkeypatch) -> None:
        """run-paper-once should surface loop-service failures cleanly."""
        from trading_bot.paper.loop import ScheduledPaperTradingService

        class DummyService:
            def default_selection(self):
                return None

            def run_once(self, *args, **kwargs):
                raise RuntimeError("mock paper-loop failure")

        monkeypatch.setattr(
            ScheduledPaperTradingService,
            "from_settings",
            classmethod(lambda cls, settings=None: DummyService()),
        )

        result = self.runner.invoke(cli, ["run-paper-once"])
        assert result.exit_code == 1
        assert "mock paper-loop failure" in result.output
