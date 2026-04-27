# BTC ML Trading Bot

A **BTC-only machine learning trading research system** built for systematic price prediction and paper trading. Uses public market data and BTC news to assemble deterministic, leak-safe feature tables, label tables, supervised datasets, baseline walk-forward LightGBM training runs, local experiment-tracking artifacts, FinBERT-enriched news sentiment features, and a scheduled offline paper-trading loop for local end-to-end inference rehearsal.

> ⚠️ **Current Step: 15 — Scheduled Offline Paper Trading**. The project now supports a local 15-minute paper loop for the selected `fee_4` target and optimized model configuration. The loop updates BTCUSDT candles, rebuilds only the latest required feature row, loads the chosen stored model artifact, generates one fresh probability prediction, applies the optimized threshold strategy, updates a simulated account, and persists signals, paper orders, positions, equity, and summary state locally. It never places real exchange orders.

---

## Architecture Overview

```
┌─────────────┐    ┌──────────────┐    ┌────────────┐
│  Ingestion   │───▶│   Schemas    │───▶│  Storage   │
│ BinanceClient│    │ Validation   │    │  Parquet   │
│ NewsClient   │    │ BTC Filter   │    │ news_store │
└─────────────┘    └──────────────┘    └────────────┘
       │
       ▼
┌─────────────┐    ┌──────────────┐    ┌────────────┐
│  Features   │───▶│   Labels     │───▶│ Validation  │
│ (step 4)    │    │  (step 5)    │    │ Walk-Fwd   │
└─────────────┘    └──────────────┘    └────────────┘
```

### Data Pipelines

**Market Data (Step 2):**
```
Binance API ──▶ BinanceClient ──▶ normalize_candle_df() ──▶ validate_candles() ──▶ ParquetStore.write()
```

**News Data (Step 3):**
```
CryptoCompare API ──▶ CryptoCompareNewsClient ──▶ normalize_news_df() ──▶ BTC filter ──▶ validate_news() ──▶ NewsStore.write()
                       (paginated via lTs)         (schema coercion)      (keyword)      (dedup, clean)       (idempotent merge)
```

**Feature Engineering (Step 4):**
```
ParquetStore.read() + NewsStore.read()
        └──▶ compute_market_features()
        └──▶ compute_news_features() aligned on market candle timestamps
        └──▶ FeaturePipeline.build_feature_table()
        └──▶ ParquetFeatureStore.write_dataset()
```

**Label & Dataset Assembly (Step 5):**
```
ParquetStore.read() + ParquetFeatureStore.read_dataset("feature_table")
        └──▶ generate_future_return_labels()
        └──▶ SupervisedDatasetBuilder.build_supervised_dataset()
        └──▶ DatasetStore.write_artifact()
```

**Walk-Forward Training (Step 6):**
```
DatasetStore.read_artifact("supervised_dataset")
        └──▶ WalkForwardSplitter.split()
        └──▶ WalkForwardTrainingPipeline.run_training()
                └──▶ LightGBM binary classifier (CPU by default, optional CUDA request)
                └──▶ per-fold predictions + metrics + feature importance
                └──▶ ModelStore.write_fold_model()
                └──▶ EvaluationStore.write_*(...)
```

**Experiment Tracking & Reporting (Step 7):**
```
EvaluationStore.list_runs()
        └──▶ ExperimentTracker.refresh_registry()
                └──▶ run_summary.json per run
                └──▶ run_registry.parquet
                └──▶ evaluation_report.json / .md
                └──▶ comparison tables for tracked runs
```

**NLP Enrichment & Sentiment Features (Step 8):**
```
NewsStore.read()
        └──▶ NewsEnrichmentService.enrich_news()
                └──▶ FinBERTService.score_texts()
                └──▶ EnrichedNewsStore.write()
        └──▶ FeaturePipeline.build_news_features(version="v2_sentiment")
                └──▶ compute_news_features(... include_sentiment_features=True)
                └──▶ ParquetFeatureStore.write_dataset()
```

**Dataset Rebuild & Sentiment Ablation (Step 9):**
```
ParquetFeatureStore.read_dataset("feature_table", version="v2_sentiment")
        └──▶ SupervisedDatasetBuilder.build_supervised_dataset(dataset_version="v2_sentiment")
        └──▶ WalkForwardTrainingPipeline.run_training(feature_version="v2_sentiment")
        └──▶ RunComparator.build_sentiment_ablation_summary(...)
                └──▶ comparison parquet + json outputs
```

**Backtesting & Offline Paper Simulation (Step 10):**
```
ModelRegistry.read_predictions(run_id=...)
        + ParquetStore.read(symbol="BTCUSDT", timeframe="15m")
        └──▶ generate_threshold_signals(...)
        └──▶ simulate_execution(... fill_model="next_bar_open")
        └──▶ compute_simulation_metrics(...)
        └──▶ BacktestStore / PaperSimulationStore.write_*(...)
```

**Validation & Full Pipeline Audit (Step 11):**
```
ValidationSuite.run_validation_suite(run_id=...)
        └──▶ standardized backtests for model + baselines
        └──▶ probability-bucket diagnostics
        └──▶ failure-case and drawdown analysis
        └──▶ feature-importance audit
        └──▶ simulation cost / overtrading audit
        └──▶ validation_report.json / .md with has_edge flag
```

**Strategy Search & Execution Optimization Audit (Step 12):**
```
StrategySearchRunner.run_search(run_id=...)
        └──▶ deterministic threshold / holding / cooldown grid search
        └──▶ compact candidate backtest results
        └──▶ best-candidate selection with drawdown / fee / trade-count filters
        └──▶ optimized-vs-baseline comparison
        └──▶ optimization_report.json / .md with optimized_has_edge flag
```

**Multi-Horizon Research & Comparison (Step 13):**
```
MultiHorizonResearchRunner
        └──▶ build_datasets(horizons=[4,8,16])
        └──▶ train_models(...)
        └──▶ run_validation(...)
        └──▶ run_strategy_search(...)
        └──▶ build_horizon_comparison_report(...)
                └──▶ best_horizon_bars or no_viable_horizon
```

**Fee-Aware Target Research & Comparison (Step 14):**
```
FeeAwareTargetResearchRunner
        └──▶ build_datasets(targets=[fee_4, fee_8, fee_16, ...])
        └──▶ train_models(...)
        └──▶ run_validation(...)
        └──▶ run_strategy_search(...)
        └──▶ build_target_comparison_report(...)
                └──▶ best_target_name or no_viable_target
```

**Scheduled Offline Paper Trading (Step 15):**
```
ScheduledPaperTradingService
        └──▶ MarketDataService.ingest_sync(...) refreshes latest BTCUSDT 15m candles
        └──▶ FeaturePipeline.build_latest_feature_row(version="v1")
        └──▶ load latest stored fold model for run_id=RUN20260426T163315Z
        └──▶ predict latest y_proba for target_long_net_positive_4bars
        └──▶ apply OPT20260426T170429Z best-candidate strategy parameters
        └──▶ replay accumulated paper predictions through simulate_execution(...)
        └──▶ persist predictions, signals, orders, trades, positions, equity, status, summary
```

---

## Project Structure

```
trading_bot/
├── config/
│   ├── assets.yaml             # Asset, market data, and news config
│   ├── model.yaml              # Model & validation parameters
│   └── risk.yaml               # Risk management rules
├── data/
│   ├── raw/
│   │   ├── market/             # Raw market data (future use)
│   │   └── news/               # Raw news data (future use)
│   ├── features/               # ★ Engineered feature datasets
│   │   └── asset=BTC/symbol=BTCUSDT/timeframe=15m/version=v1/
│   │       ├── market_features.parquet
│   │       ├── news_features.parquet
│   │       └── features.parquet
│   ├── datasets/               # ★ Labels + supervised datasets
│   │   └── asset=BTC/symbol=BTCUSDT/timeframe=15m/label_version=v1/
│   │       ├── labels.parquet
│   │       └── dataset_version=v1/dataset.parquet
│   ├── models/                 # ★ Fold-level model artifacts
│   │   └── asset=BTC/symbol=BTCUSDT/timeframe=15m/model_version=v1/run_id=RUN.../
│   │       └── fold_0_model.pkl
│   ├── evaluation/             # ★ Metrics, predictions, reports, feature importance
│   │   └── asset=BTC/symbol=BTCUSDT/timeframe=15m/model_version=v1/run_id=RUN.../
│   │       ├── fold_metrics.parquet
│   │       ├── predictions.parquet
│   │       ├── feature_importance.parquet
│   │       ├── aggregate_feature_importance.parquet
│   │       ├── aggregate_metrics.json
│   │       ├── run_metadata.json
│   │       └── report.json
│   ├── experiments/            # ★ Local experiment registry + per-run summaries
│   │   ├── run_registry.parquet
│   │   └── run_id=RUN.../
│   │       ├── run_summary.json
│   │       └── top_features.parquet
│   ├── reports/                # ★ Generated rich evaluation reports + comparisons
│   │   ├── run_id=RUN.../
│   │   │   ├── evaluation_report.json
│   │   │   └── evaluation_report.md
│   │   ├── comparisons/
│   │   └── validation/
│   │   └── horizon_comparison/
│   ├── strategy_optimization/ # ★ Compact threshold-search candidate results + reports
│   ├── backtests/              # ★ Stored backtest summaries, signals, equity curves, trades
│   ├── paper_simulations/      # ★ Stored offline paper-simulation outputs
│   ├── paper_loop/             # ★ Scheduled offline paper-loop state, logs, and account artifacts
│   ├── enriched_news/          # ★ Local FinBERT-enriched BTC news records
│   │   └── asset=BTC/provider=cryptocompare/model=finbert/enrichment_version=v1/
│   │       └── enriched_news.parquet
│   └── processed/
│       ├── market/             # Stored Parquet candle files
│       │   └── BTCUSDT/15m/candles.parquet
│       └── news/               # Stored Parquet news files
│           └── BTC/cryptocompare/news.parquet
├── src/trading_bot/
│   ├── __init__.py
│   ├── main.py                 # Application entrypoint
│   ├── settings.py             # Typed config (Pydantic)
│   ├── logging_config.py       # Structured logging (structlog)
│   ├── cli.py                  # CLI commands (Click)
│   ├── schemas/
│   │   ├── market_data.py      # Candle schema + validation
│   │   ├── news_data.py        # News schema + validation + BTC filter
│   │   ├── features.py         # Canonical feature columns + validation
│   │   ├── datasets.py         # ★ Label + supervised dataset schema metadata
│   │   └── modeling.py         # ★ Training/run schema metadata
│   ├── ingestion/
│   │   ├── binance_client.py   # Binance REST client (paginated)
│   │   ├── news_client.py      # CryptoCompare news client (paginated)
│   │   ├── market_data.py      # Market ingestion service
│   │   └── news_data.py        # News ingestion service (fetch→filter→validate→store)
│   ├── storage/
│   │   ├── parquet_store.py    # Market data Parquet store
│   │   ├── news_store.py       # News Parquet store with dedup
│   │   ├── dataset_store.py    # ★ Labels + dataset Parquet persistence
│   │   ├── model_store.py      # ★ Fold model persistence
│   │   ├── evaluation_store.py # ★ Metrics/report/prediction persistence
│   │   ├── experiment_store.py # ★ Experiment registry + summary persistence
│   │   ├── report_store.py     # ★ Generated report persistence
│   │   └── enriched_news_store.py # ★ FinBERT-enriched news persistence
│   ├── experiments/
│   │   ├── tracker.py          # ★ Run registration from stored step-6 artifacts
│   │   ├── registry.py         # ★ Local run registry query/update helpers
│   │   ├── comparison.py       # ★ Cross-run comparison utilities
│   │   └── promotion.py        # ★ Promotion status tagging helpers
│   ├── features/
│   │   ├── market_features.py  # ★ OHLCV feature builder
│   │   ├── news_features.py    # ★ News event aggregation builder
│   │   ├── feature_store.py    # ★ Feature Parquet persistence
│   │   └── feature_pipeline.py # ★ Market+news merged build pipeline
│   ├── labeling/
│   │   ├── labels.py           # ★ Future-return target generation
│   │   └── dataset_builder.py  # ★ Supervised dataset builder
│   ├── models/
│   │   ├── train.py            # ★ Walk-forward LightGBM training pipeline
│   │   ├── predict.py          # ★ Prediction helpers
│   │   ├── model_registry.py   # ★ Stored run inspection helpers
│   │   └── tuning.py           # Placeholder for later tuning work
│   ├── validation/
│   │   ├── walk_forward.py     # ★ Expanding/rolling walk-forward splits
│   │   ├── metrics.py          # ★ Classification metrics
│   │   └── evaluation_report.py# ★ Rich evaluation report builder
│   ├── research/
│   │   └── multi_horizon.py    # ★ Multi-horizon orchestration + comparison
│   ├── paper/
│   │   ├── loop.py             # ★ Scheduled offline paper-loop service
│   │   └── daemon.py           # ★ Background paper-loop entrypoint
│   ├── nlp/
│   │   ├── finbert_service.py  # ★ Local FinBERT loading + inference
│   │   ├── text_preprocessing.py # ★ Deterministic scoring text assembly
│   │   └── news_enrichment.py  # ★ News enrichment pipeline
│   ├── backtest/               # ★ Signal generation + execution simulation
│   ├── execution/              # Placeholder
│   └── monitoring/             # Placeholder
├── tests/
│   ├── test_settings.py
│   ├── test_cli.py
│   ├── test_project_imports.py
│   ├── test_market_data_ingestion.py
│   ├── test_parquet_store.py
│   ├── test_news_ingestion.py
│   ├── test_news_store.py
│   ├── test_market_features.py        # ★ Market feature math + leakage checks
│   ├── test_news_features.py          # ★ News alignment + aggregation checks
│   ├── test_feature_pipeline.py       # ★ End-to-end feature persistence tests
│   ├── test_labels.py                 # ★ Label math + end-of-sample checks
│   ├── test_dataset_builder.py        # ★ Supervised dataset assembly tests
│   ├── test_walk_forward.py           # ★ Split boundary and leakage checks
│   ├── test_metrics.py                # ★ Fold metric calculations
│   ├── test_training_pipeline.py      # ★ Baseline training + CUDA fallback tests
│   ├── test_experiment_tracker.py     # ★ Local registry/report generation tests
│   ├── test_run_comparison.py         # ★ Cross-run comparison tests
│   ├── test_evaluation_report.py      # ★ Rich evaluation report tests
│   ├── test_finbert_service.py        # ★ Local FinBERT service tests
│   ├── test_news_enrichment.py        # ★ Enriched news idempotency tests
│   └── test_sentiment_features.py     # ★ Sentiment-aware feature aggregation tests
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

Files marked with ★ are new or significantly changed through step 6.

---

## Setup

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
cd trading_bot

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install build tools + package
pip install --upgrade pip setuptools wheel hatchling editables
pip install --no-build-isolation -e ".[dev]"

# Install the ML extras when you want to run step-6 training locally
pip install --no-build-isolation -e ".[ml]"

# Copy environment template
cp .env.example .env
```

---

## CLI Usage

### Diagnostic Commands

```bash
trading-bot health-check       # Validate config + imports
trading-bot show-config        # Show resolved config
trading-bot project-tree       # Show module tree
```

### Market Data Ingestion (Step 2)

```bash
trading-bot ingest-historical-market                                     # Default: BTCUSDT/15m from 2024-01-01
trading-bot ingest-historical-market --timeframe 1h --start 2024-01-01   # 1h candles
trading-bot show-market-data-range   --timeframe 15m                     # Inspect metadata
trading-bot show-market-data-stats   --timeframe 15m                     # Detailed stats
```

### News Data Ingestion (Step 3)

```bash
# Download historical BTC news (default: from 2024-01-01 to now)
trading-bot ingest-historical-news

# With custom date range
trading-bot ingest-historical-news --start 2024-01-01 --end 2024-03-01

# Inspect stored news data
trading-bot show-news-data-range

# Detailed stats (article count, source distribution)
trading-bot show-news-data-stats
```

### Feature Engineering (Step 4)

```bash
# Build market-only features from stored BTC candles
trading-bot build-market-features --symbol BTCUSDT --timeframe 15m

# Build BTC news aggregate features aligned to the BTCUSDT 15m candle timeline
trading-bot build-news-features --symbol BTCUSDT --timeframe 15m --asset BTC

# Build the merged market + news feature table
trading-bot build-feature-table --symbol BTCUSDT --timeframe 15m --asset BTC

# Inspect a stored feature dataset
trading-bot show-feature-stats --dataset merged
trading-bot show-feature-stats --dataset market
trading-bot show-feature-stats --dataset news
```

### Labels & Supervised Datasets (Step 5)

```bash
# Build future-return labels from stored BTC candles
trading-bot build-labels --symbol BTCUSDT --timeframe 15m --horizon-bars 2

# Build the final supervised dataset from step-4 features + step-5 labels
trading-bot build-supervised-dataset --symbol BTCUSDT --timeframe 15m --label-version v1 --feature-version v1

# Rebuild the supervised dataset from the sentiment-enhanced feature version
trading-bot build-supervised-dataset --symbol BTCUSDT --timeframe 15m --feature-version v2_sentiment --dataset-version v2_sentiment

# Inspect stored label or dataset artifacts
trading-bot show-dataset-stats --artifact labels
trading-bot show-dataset-stats --artifact supervised

# List the available target columns in the stored supervised dataset
trading-bot show-target-columns
```

### Walk-Forward Validation & Baseline Training (Step 6)

```bash
# Run baseline LightGBM walk-forward training on CPU (default)
trading-bot run-walk-forward-training --symbol BTCUSDT --timeframe 15m --dataset-version v1 --model-version v1 --device cpu

# Request CUDA training; the pipeline falls back to CPU if CUDA LightGBM is unavailable
trading-bot run-walk-forward-training --symbol BTCUSDT --timeframe 15m --device cuda

# Retrain on the sentiment-enhanced supervised dataset
trading-bot run-walk-forward-training --symbol BTCUSDT --timeframe 15m --feature-version v2_sentiment --dataset-version v2_sentiment --model-version v2_sentiment --device cpu

# Inspect a stored training run
trading-bot list-model-runs
trading-bot show-run-metrics --run-id RUN20260421T000000Z
trading-bot show-feature-importance --run-id RUN20260421T000000Z --top-k 20
```

### Experiment Tracking & Reporting (Step 7)

```bash
# Refresh the local experiment registry from stored training artifacts
trading-bot refresh-experiment-registry

# List tracked runs with key metrics
trading-bot list-experiment-runs --sort-by roc_auc_mean
trading-bot list-experiment-runs --status candidate

# Generate and print a rich evaluation report for one run
trading-bot show-run-report --run-id RUN20260421T000000Z

# Compare two or more tracked runs
trading-bot compare-runs --run-ids RUN20260421T000000Z --run-ids RUN20260422T000000Z --sort-by f1_mean

# Show a focused baseline-vs-sentiment ablation summary
trading-bot show-sentiment-ablation --baseline-run-id RUN20260421T000000Z --new-run-id RUN20260422T000000Z

# Tag a run for promotion decisions
trading-bot tag-run-status --run-id RUN20260421T000000Z --status promoted_for_backtest --note baseline_model_best_so_far
```

### NLP Enrichment & Sentiment Features (Step 8)

```bash
# Install NLP extras before local FinBERT scoring
pip install --no-build-isolation -e ".[nlp]"

# Enrich stored BTC news locally with FinBERT sentiment
trading-bot enrich-news-sentiment --asset BTC --provider cryptocompare --enrichment-version v1 --device cpu --text-mode title_plus_summary

# Inspect stored enriched-news coverage and label distribution
trading-bot show-enriched-news-stats --asset BTC --provider cryptocompare --enrichment-version v1

# Build sentiment-aware news features with a version distinct from the original step-4 feature table
trading-bot build-sentiment-news-features --symbol BTCUSDT --timeframe 15m --feature-version v2_sentiment

# Rebuild the merged feature table using the sentiment-aware version
trading-bot build-feature-table --symbol BTCUSDT --timeframe 15m --feature-version v2_sentiment
```

### Backtesting & Offline Paper Simulation (Step 10)

```bash
# Run a strict next-bar-open backtest from one stored model run
trading-bot run-backtest --model-run-id RUN20260422T000000Z --symbol BTCUSDT --timeframe 15m --strategy-version v1

# Run the same core engine under the offline paper-simulation output path
trading-bot run-paper-simulation --model-run-id RUN20260422T000000Z --symbol BTCUSDT --timeframe 15m --strategy-version v1

# Inspect and compare stored simulations
trading-bot show-simulation-summary --simulation-type backtest --simulation-id BT20260426T000000Z
trading-bot list-simulations --simulation-type backtest --sort-by total_return
trading-bot compare-simulations --simulation-type backtest --simulation-ids BT20260426T000000Z --simulation-ids BT20260426T010000Z --sort-by total_return
```

### Validation, Baselines & Audit (Step 11)

```bash
# Run the full audit suite for one trained model run
trading-bot run-validation-suite --model-run-id RUN20260422T000000Z

# Inspect the ranked model-vs-baselines table
trading-bot compare-baselines --validation-id VAL20260426T000000Z

# Show the final validation report and conclusion
trading-bot show-validation-report --validation-id VAL20260426T000000Z

# Inspect confidence buckets and failure cases
trading-bot analyze-predictions --model-run-id RUN20260422T000000Z
trading-bot show-failure-cases --validation-id VAL20260426T000000Z
```

### Strategy Threshold Search & Execution Audit (Step 12)

```bash
# Run the strategy search for one stored model run
trading-bot run-strategy-search --model-run-id RUN20260426T053553Z --symbol BTCUSDT --timeframe 15m

# Inspect the optimization report and the top candidate table
trading-bot show-strategy-search-report --optimization-id OPT20260426T000000Z
trading-bot show-top-strategies --optimization-id OPT20260426T000000Z --top-k 20

# List stored searches and compare the best optimized strategy against baselines
trading-bot list-strategy-searches
trading-bot compare-optimized-strategy --optimization-id OPT20260426T000000Z
```

### Multi-Horizon Research (Step 13)

```bash
# Build one supervised dataset per horizon on the same feature version
trading-bot build-multi-horizon-datasets --symbol BTCUSDT --timeframe 15m --feature-version v2_sentiment --horizons 4,8,16

# Train one baseline model per horizon
trading-bot train-multi-horizon-models --symbol BTCUSDT --timeframe 15m --feature-version v2_sentiment --horizons 4,8,16 --device cpu

# Run the validation suite for the latest run at each horizon
trading-bot run-multi-horizon-validation --symbol BTCUSDT --timeframe 15m --horizons 4,8,16

# Run strategy search for the latest run at each horizon
trading-bot run-multi-horizon-strategy-search --symbol BTCUSDT --timeframe 15m --horizons 4,8,16

# Build and print the final horizon comparison report
trading-bot show-horizon-comparison-report --symbol BTCUSDT --timeframe 15m --horizons 4,8,16
```

### Fee-Aware Target Research (Step 14)

```bash
# Build one supervised dataset per fee-aware or minimum-return target definition
trading-bot build-fee-aware-datasets --symbol BTCUSDT --timeframe 15m --feature-version v2_sentiment --targets fee_4,fee_8,fee_16,ret025_8,ret050_16,ret075_16

# Train one baseline model per target definition
trading-bot train-fee-aware-target-models --symbol BTCUSDT --timeframe 15m --feature-version v2_sentiment --targets fee_4,fee_8,fee_16,ret025_8,ret050_16,ret075_16 --device cpu

# Run validation and strategy search for the latest stored run at each target
trading-bot run-fee-aware-validation --symbol BTCUSDT --timeframe 15m --targets fee_4,fee_8,fee_16,ret025_8,ret050_16,ret075_16
trading-bot run-fee-aware-strategy-search --symbol BTCUSDT --timeframe 15m --targets fee_4,fee_8,fee_16,ret025_8,ret050_16,ret075_16

# Build and print the final target comparison report
trading-bot show-target-comparison-report --symbol BTCUSDT --timeframe 15m --targets fee_4,fee_8,fee_16,ret025_8,ret050_16,ret075_16
```

### Scheduled Offline Paper Trading (Step 15)

```bash
# Run one immediate update cycle for the selected best target/model
trading-bot run-paper-once

# Start the background 15-minute offline paper loop
trading-bot start-paper-loop

# Inspect current runtime status and account state
trading-bot show-paper-status

# Inspect the most recent closed paper trades
trading-bot show-paper-trades --limit 20

# Stop the background paper loop
trading-bot stop-paper-loop
```

### All CLI Commands

| Command | Description |
|---|---|
| `health-check` | Validates config loading and all module imports |
| `show-config` | Prints resolved config (secrets redacted) |
| `project-tree` | Displays package module hierarchy |
| `ingest-historical-market` | Download + normalize + validate + persist BTCUSDT candles |
| `show-market-data-range` | Show dataset metadata (timestamps, file size) |
| `show-market-data-stats` | Row count, coverage, price stats, gap analysis |
| `ingest-historical-news` | Download + normalize + BTC filter + validate + persist news |
| `show-news-data-range` | Show news dataset metadata |
| `show-news-data-stats` | Article count, source distribution, timestamp range |
| `build-market-features` | Build market-only deterministic BTC candle features |
| `build-news-features` | Build leak-safe BTC news aggregate features on the market timeline |
| `build-feature-table` | Build the merged model-ready market + news feature dataset |
| `show-feature-stats` | Show row count, timestamp range, and columns for stored feature datasets |
| `build-labels` | Generate future-return target columns from stored BTC candles |
| `build-supervised-dataset` | Build the final supervised dataset from stored features and labels |
| `show-dataset-stats` | Show row count, timestamp range, column counts, and target stats for stored dataset artifacts |
| `show-target-columns` | List available target columns in the stored supervised dataset |
| `run-walk-forward-training` | Train and evaluate the baseline LightGBM model across walk-forward folds |
| `show-run-metrics` | Print aggregate and per-fold metrics for a stored training run |
| `list-model-runs` | List stored model runs with basic metadata |
| `show-feature-importance` | Print top aggregate feature importance rows for a stored run |
| `refresh-experiment-registry` | Scan stored training artifacts and refresh the local experiment registry |
| `list-experiment-runs` | List tracked runs with key metadata and aggregate metrics |
| `show-run-report` | Generate and print a rich evaluation report for one run |
| `compare-runs` | Compare two or more tracked runs on stable metrics and metadata |
| `show-sentiment-ablation` | Show metric deltas between a baseline run and a sentiment-enhanced run |
| `tag-run-status` | Set a local promotion status and note for a tracked run |
| `enrich-news-sentiment` | Score stored BTC news locally with FinBERT and persist enriched news records |
| `show-enriched-news-stats` | Show sentiment coverage, label distribution, and timestamp range for enriched BTC news |
| `build-sentiment-news-features` | Build sentiment-aware news aggregates aligned to the BTC market timeline |
| `run-backtest` | Run a strict next-bar-open backtest from stored prediction artifacts |
| `run-paper-simulation` | Run an offline paper simulation using the same execution engine |
| `show-simulation-summary` | Print the stored summary for one backtest or paper simulation |
| `list-simulations` | List stored simulations with headline performance metrics |
| `compare-simulations` | Compare two or more stored simulations on summary metrics |
| `run-validation-suite` | Run the full model-vs-baselines audit and persist the final validation report |
| `compare-baselines` | Print the ranked strategy comparison table from a stored validation report |
| `show-validation-report` | Print the stored validation report and final `has_edge` conclusion |
| `analyze-predictions` | Bucket stored model probabilities and inspect confidence quality |
| `show-failure-cases` | Print the worst trades and largest drawdown periods from a stored validation report |
| `run-strategy-search` | Run a deterministic threshold and execution-rule search for one stored model run |
| `show-strategy-search-report` | Print the stored optimization report and final `optimized_has_edge` recommendation |
| `list-strategy-searches` | List stored strategy-search reports |
| `show-top-strategies` | Print the top candidate strategies from one optimization run |
| `compare-optimized-strategy` | Compare the best optimized strategy against the original model strategy and baselines |
| `build-multi-horizon-datasets` | Build one supervised dataset per requested prediction horizon |
| `train-multi-horizon-models` | Train one baseline LightGBM walk-forward run per requested horizon |
| `run-multi-horizon-validation` | Run the validation suite for the latest stored run at each horizon |
| `run-multi-horizon-strategy-search` | Run strategy search for the latest stored run at each horizon |
| `show-horizon-comparison-report` | Build and print the final best-horizon or no-viable-horizon report |
| `build-fee-aware-datasets` | Build one supervised dataset per fee-aware or minimum-return target definition |
| `train-fee-aware-target-models` | Train one baseline LightGBM walk-forward run per target definition |
| `run-fee-aware-validation` | Run validation for the latest stored run at each fee-aware target |
| `run-fee-aware-strategy-search` | Run strategy search for the latest stored run at each fee-aware target |
| `show-target-comparison-report` | Build and print the final best-target or no-viable-target report |
| `start-paper-loop` | Start the background scheduled offline paper-trading loop for the selected model and optimized strategy |
| `run-paper-once` | Run one immediate paper-loop cycle without starting the background scheduler |
| `show-paper-status` | Show the stored paper-loop runtime status, latest prediction, and current account state |
| `show-paper-trades` | Show the latest stored closed trades from the scheduled paper loop |
| `stop-paper-loop` | Stop a running background paper loop by its stored PID |

---

## Feature Engineering

### Storage Layout

```
data/features/
└── asset=BTC/
    └── symbol=BTCUSDT/
        └── timeframe=15m/
            └── version=v1/
                ├── market_features.parquet
                ├── news_features.parquet
                └── features.parquet
```

### Implemented Feature Groups

- **Market features**: returns (`ret_1`, `ret_2`, `ret_4`, `ret_8`, `log_ret_1`), candle anatomy, rolling volatility, rolling mean return, rolling volume, volume z-score, SMA distance, breakout distance, and calendar features.
- **News features**: lookback counts (`1h`, `4h`, optional `12h`), unique source counts, minutes since last BTC news item, title-length mean, unique URLs, and simple burst flags.
- **Merged table**: one row per market candle timestamp with the market timeline as the master index and sensible zero-fills for missing news count-style features.

### Time Alignment Rules

- All timestamps are normalized to UTC before feature computation.
- The market candle timeline is the master index for both news-only and merged datasets.
- For a market timestamp `t`, only articles with `published_at <= t` contribute to the aligned news feature row.
- Rolling market features use only current and historical candles; no future candles are referenced.
- Missing count-style news features are filled with zero, while recency and title-length aggregates stay `NaN` when no qualifying history exists.

### Current Scope Limits

- No social or tweet features yet.
- No LLM summarization.
- No model retraining in this step.
- No walk-forward split logic inside the feature step itself.
- No model tuning, calibration, or backtesting logic in the feature step.

### Sentiment-Aware News Features

- The original step-4 news features remain available for ablation studies and are still produced by feature version `v1`.
- Sentiment-aware feature generation is activated by building the sentiment feature version, which defaults to `v2_sentiment`.
- Enriched-news inputs are read from `data/enriched_news/.../enriched_news.parquet` and aligned to market timestamps using the same no-future-leakage rule as the original news feature step.
- Count-like sentiment features are zero-filled when no qualifying history exists, while recency features such as `minutes_since_last_positive_news` remain `NaN` when no prior matching article exists.
- The default scalar sentiment score is `prob_positive - prob_negative`.

---

## Labels & Supervised Datasets

### Storage Layout

```
data/datasets/
└── asset=BTC/
    └── symbol=BTCUSDT/
        └── timeframe=15m/
            └── label_version=v1/
                ├── labels.parquet
                └── dataset_version=v1/
                    └── dataset.parquet
```

### Implemented Targets

- **Primary target**: `target_up_2bars` = `1` if close at `t+2` bars is greater than close at `t`, else `0`.
- **Regression target**: `future_return_2bars` = percentage return from close at `t` to close at `t+2` bars.
- **Optional targets**: `target_up_4bars` and `future_log_return_2bars`.

### Label Alignment Rules

- All timestamps are normalized to UTC before label generation and dataset assembly.
- Labels are generated from future candle closes only; the label row at timestamp `t` never uses market observations from `t-1` or earlier as the target.
- The stored step-4 merged feature table remains the master index for supervised dataset assembly.
- Rows are dropped near the end of the sample when any included target column cannot be computed cleanly.
- Feature columns are preserved as-is during dataset assembly, including early rolling-feature `NaN` values.

### Current Scope Limits

- No walk-forward split generation inside the dataset builder itself.
- No probability calibration, backtesting, or execution logic yet.
- No sentiment or embedding-based targets.

---

## Walk-Forward Validation & Baseline Training

### Storage Layout

```
data/models/
└── asset=BTC/
    └── symbol=BTCUSDT/
        └── timeframe=15m/
            └── model_version=v1/
                └── run_id=RUN20260421T000000Z/
                    └── fold_0_model.pkl

data/evaluation/
└── asset=BTC/
    └── symbol=BTCUSDT/
        └── timeframe=15m/
            └── model_version=v1/
                └── run_id=RUN20260421T000000Z/
                    ├── fold_metrics.parquet
                    ├── predictions.parquet
                    ├── feature_importance.parquet
                    ├── aggregate_feature_importance.parquet
                    ├── aggregate_metrics.json
                    ├── run_metadata.json
                    └── report.json
```

### Implemented Validation & Training Behavior

- **Splitter**: expanding-window walk-forward validation with configurable `min_train_rows`, `validation_rows`, and `step_rows`. A rolling-window mode is also available for later experiments.
- **Baseline model**: LightGBM binary classifier trained on numeric non-identifier, non-target feature columns only.
- **Primary target**: `target_up_2bars`.
- **Predictions**: one validation prediction row per original dataset row with `timestamp`, `fold_id`, `y_true`, `y_pred`, and `y_proba`.
- **Metrics**: accuracy, precision, recall, F1, ROC AUC, log loss, confusion counts, base rate, and prediction positive rate.
- **Feature importance**: stored per fold and as a simple aggregate mean across folds.

### Device Handling

- CPU is the default for portability.
- CUDA can be requested explicitly from config or CLI.
- If CUDA LightGBM is unavailable, unsupported, or raises during initialization/training, the pipeline logs a warning and retries on CPU when fallback is enabled.
- Stored run metadata records both `requested_device` and `effective_device`.

### Current Scope Limits

- No hyperparameter tuning yet.
- No probability calibration yet.
- No threshold optimization yet.
- No backtesting or execution yet.
- No live inference service yet.

---

## News Data

### Source: CryptoCompare

The free [CryptoCompare News API](https://min-api.cryptocompare.com/documentation?key=News) provides:
- Historical BTC news articles with backward pagination via `lTs`
- Category-based filtering (`categories=BTC`)
- No API key required for basic usage (key optional for higher limits)
- Fields: title, URL, source, published timestamp, body/summary, categories

### Storage Layout

```
data/processed/news/
└── BTC/
    └── cryptocompare/
        └── news.parquet
```

### Canonical Schema

| Column | Type | Required | Description |
|---|---|---|---|
| `asset` | string | ✅ | Asset tag (BTC) |
| `provider` | string | ✅ | Data source (cryptocompare) |
| `published_at` | datetime UTC | ✅ | Article publish time |
| `title` | string | ✅ | Article headline |
| `url` | string | ✅ | Article URL |
| `source_name` | string | ✅ | News outlet name |
| `provider_article_id` | string | | Provider's unique ID |
| `summary` | string | | Article summary/excerpt |
| `body` | string | | Full article body |
| `author` | string | | Author name |
| `language` | string | | Language code |
| `source_domain` | string | | Extracted from URL |
| `category` | string | | Provider categories |
| `tags` | string | | Provider tags |
| `raw_symbol_refs` | string | | Raw symbol/coin references |
| `ingested_at` | datetime UTC | | When record was ingested |

### Validation & Dedup Rules

- **Deduplication**: Primary key = `(provider, provider_article_id)`. Fallback = `(provider, url, published_at)`.
- **Missing fields**: Rows with empty title, URL, or published_at are dropped.
- **Timestamps**: Records before 2000 or in the future are rejected.
- **BTC filter**: Keyword matching on title + category + tags (configurable).
- **Idempotent**: Repeated runs merge and deduplicate safely.

---

## Running Tests

```bash
# Run all tests (177 tests)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src .venv/bin/python -m pytest tests

# With coverage
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src .venv/bin/python -m pytest tests --cov=trading_bot --cov-report=term-missing
```

> **Note**: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids unrelated globally installed ROS pytest plugins in this environment.

---

## Configuration

### News Data Config (`config/assets.yaml`)

```yaml
news_data:
  asset: "BTC"
  provider: "cryptocompare"
  default_start_date: "2024-01-01"
  raw_data_dir: "data/raw/news"
  processed_data_dir: "data/processed/news"
  page_size: 100
  request_timeout: 30.0
  max_retries: 3
  max_pages: 500
  btc_keywords:
    - "btc"
    - "bitcoin"
    - "spot bitcoin etf"
    - "bitcoin etf"
    - "bitcoin network"
    - "bitcoin miner"
    - "bitcoin miners"
    - "bitcoin mining"
    - "bitcoin halving"
```

### Feature Config (`config/assets.yaml`)

```yaml
features:
  feature_version: "v1"
  symbol: "BTCUSDT"
  asset: "BTC"
  timeframe: "15m"
  market_windows: [1, 2, 4, 8]
  news_lookback_windows: ["1h", "4h"]
  output_dir: "data/features"
  fill_missing_news_with_zero: true
  include_breakout_features: true
  include_calendar_features: true
  include_optional_news_features: true
  news_burst_threshold_1h: 3
  news_burst_threshold_4h: 8
```

### Labels Config (`config/assets.yaml`)

```yaml
labels:
  label_version: "v1"
  primary_target: "target_up_2bars"
  horizon_bars: 2
  binary_threshold: 0.0
  include_regression_target: true
  include_optional_targets: true
  optional_horizon_bars: [4]
```

### Datasets Config (`config/assets.yaml`)

```yaml
datasets:
  dataset_version: "v1"
  symbol: "BTCUSDT"
  asset: "BTC"
  timeframe: "15m"
  output_dir: "data/datasets"
```

### Training Config (`config/model.yaml`)

```yaml
training:
  model_version: "v1"
  primary_target: "target_up_2bars"
  random_seed: 42
  save_fold_models: true
  device: "cpu"
  allow_cuda_fallback_to_cpu: true
  probability_threshold: 0.5
```

### Walk-Forward Config (`config/model.yaml`)

```yaml
walk_forward:
  mode: "expanding_window"
  min_train_rows: 1000
  validation_rows: 250
  step_rows: 250
```

### LightGBM Config (`config/model.yaml`)

```yaml
lightgbm:
  objective: "binary"
  metric: "binary_logloss"
  learning_rate: 0.05
  n_estimators: 200
  num_leaves: 31
  subsample: 0.8
  colsample_bytree: 0.8
  random_state: 42
  device_type: "cpu"
```

### Artifact Config (`config/model.yaml`)

```yaml
artifacts:
  models_dir: "data/models"
  evaluation_dir: "data/evaluation"
```

---

## Roadmap

### ✅ Step 1 — Project Scaffold
- [x] Project structure, typed config, logging, CLI, placeholder modules

### ✅ Step 2 — Historical Market Data Ingestion
- [x] Binance REST client with pagination and retry
- [x] Canonical candle schema with validation
- [x] Parquet storage with idempotent merge

### ✅ Step 3 — Historical News Ingestion
- [x] CryptoCompare news client with backward pagination
- [x] Canonical news schema with validation and BTC keyword filtering
- [x] News Parquet store with dedup by provider_article_id
- [x] CLI commands for ingestion and inspection
- [x] Historical BTC news stored locally for later feature engineering

### ✅ Step 4 — Feature Engineering
- [x] Deterministic market feature generation from stored BTC candles
- [x] Leak-safe BTC news aggregation aligned to market timestamps
- [x] Merged feature table creation on the market master index
- [x] Parquet feature persistence with stable path naming
- [x] CLI commands for build + inspection
- [x] Feature tables ready for supervised dataset assembly

### ✅ Step 5 — Label Generation & Dataset Builder
- [x] Leak-safe future-return label generation from stored BTC candles
- [x] Stored label tables with explicit target names and versions
- [x] Final supervised dataset assembly from step-4 features + step-5 labels
- [x] CLI commands for build + inspection
- [x] Dataset ready for walk-forward training

### ✅ Step 6 — Walk-Forward Validation & Baseline Training
- [x] Leak-safe walk-forward dataset splits
- [x] Baseline LightGBM training and evaluation
- [x] CPU default with optional CUDA request and safe fallback
- [x] Persisted models, predictions, metrics, reports, and feature importance

### ✅ Step 7 — Experiment Tracking & Richer Evaluation
- [x] Local experiment registry and per-run summaries
- [x] Rich JSON and Markdown evaluation reports
- [x] Cross-run metric comparison utilities
- [x] Lightweight promotion status tagging
- [x] CLI commands for refresh, list, inspect, compare, and tag

### ✅ Step 8 — FinBERT Sentiment Scoring & Enhanced News Features
- [x] Local FinBERT enrichment service with CPU-first execution and safe CUDA fallback
- [x] Versioned enriched-news persistence with idempotent rescoring avoidance
- [x] Sentiment-aware news feature aggregates aligned to BTC market timestamps
- [x] CLI commands for enrichment, enriched-news stats, and sentiment feature rebuilding

### ✅ Step 9 — Dataset Rebuild, Retraining & Sentiment Ablation
- [x] Rebuild supervised datasets from `v2_sentiment`
- [x] Retrain walk-forward LightGBM runs with explicit feature-version provenance
- [x] Persist comparison parquet and JSON outputs for tracked runs
- [x] Generate focused baseline-vs-sentiment ablation summaries

### ✅ Step 10 — Backtesting & Offline Paper Simulation
- [x] Convert probabilities into long-only threshold signals
- [x] Simulate next-bar-open execution with fees and slippage
- [x] Persist signals, trade logs, equity curves, and summary reports
- [x] Support both `backtest` and `paper_simulation` local output families
- [x] CLI commands for run, inspect, list, and compare simulation outputs

### ✅ Step 11 — Validation, Baseline Comparison & Pipeline Audit
- [x] Standardized backtests for model, buy-and-hold, always-flat, random-frequency, and SMA crossover
- [x] Prediction diagnostics by confidence bucket
- [x] Failure-case analysis for losing trades and worst drawdowns
- [x] Feature-importance audit with sentiment-feature visibility
- [x] Final validation report with `has_edge` conclusion

### ✅ Step 12 — Strategy Threshold Search & Execution Optimization Audit
- [x] Deterministic search space for thresholds, holding rules, cooldowns, and position sizing
- [x] Compact candidate-result persistence without full per-candidate simulation dumps
- [x] Best-candidate selection using drawdown, fee, return, and trade-count filters
- [x] Optimized-vs-baseline comparison table and final `optimized_has_edge` recommendation
- [x] CLI commands for search, list, inspect, and compare optimized strategies

### ✅ Step 13 — Multi-Horizon Label Generation, Retraining & Strategy Comparison
- [x] Horizon-aware datasets for 4, 8, and 16-bar targets
- [x] One baseline walk-forward LightGBM run per horizon with stable provenance
- [x] Validation-suite execution per horizon
- [x] Strategy-search execution per horizon
- [x] Final horizon-comparison report with `best_horizon_bars` or `no_viable_horizon`
- [x] CLI commands for dataset build, training, validation, optimization, and comparison

### ✅ Step 14 — Fee-Aware and Trade-Worthy Target Redesign
- [x] Fee-aware target definitions for 4, 8, and 16-bar horizons
- [x] Minimum-return threshold targets for higher-conviction moves
- [x] Target-aware dataset rebuilds and walk-forward model retraining
- [x] Validation-suite execution and strategy-search execution per target definition
- [x] Final target-comparison report with `best_target_name` or `no_viable_target`
- [x] CLI commands for build, train, validate, optimize, and compare targets

### ✅ Step 15 — Scheduled Offline Paper Trading (current)
- [x] Local one-shot inference cycle for the selected `fee_4` target and model run
- [x] Background 15-minute scheduler with start, stop, and status commands
- [x] Latest-candle refresh, latest-row feature rebuild, and stored-model inference
- [x] Reuse of the existing signal generator, execution simulator, and cost model
- [x] Persistent prediction logs, signals, orders, trades, positions, equity, status, and summaries
- [x] CLI commands for one-shot runs, background control, and trade inspection

---

## ⚠️ Important Notes

- **FinBERT runs locally** — install the optional NLP extras with `pip install -e ".[nlp]"`.
- **CPU first** — optional CUDA can be requested for enrichment, but GPU is not required.
- **Retraining uses the existing baseline LightGBM pipeline** — this step validates whether sentiment features help before any tuning work.
- **Baseline model only** — no tuning, calibration, or threshold optimization yet.
- **Strict next-bar-open fill model** — signals are generated on bar `t` and executed at the next bar open only.
- **Long-only for now** — no short selling, leverage, or multi-asset portfolio logic in this step.
- **Offline only** — `run-paper-simulation` is not connected to any broker or exchange.
- **Still offline only** — `start-paper-loop` and `run-paper-once` never place real orders and only update local files under `data/paper_loop/`.
- **`has_edge` is intentionally conservative** — the model only passes when it beats simple baselines after costs and the confidence-bucket analysis supports the signal.
- **`optimized_has_edge` is also conservative** — optimization only passes when the best candidate stays profitable after costs, survives the drawdown and fee filters, and does not look like a random-fee-churn strategy.
- **`best_horizon_bars` is conservative too** — a horizon is only recommended when its optimized strategy survives the same return, drawdown, trade-count, and fee filters.
- **`best_target_name` is conservative too** — a target is only recommended when its optimized strategy survives the return, drawdown, trade-count, fee, and class-balance filters.
- **Simple up/down direction was not enough** — step 14 exists because directional labels did not produce a tradable edge after fees and slippage, so the labels now focus on moves that are large enough to justify taking risk.
- **No live trading** — `paper_trading_only = True`.
- **No social signals yet** — tweet and social ingestion are deferred to later steps.
- **News source**: CryptoCompare free tier. Additional providers can be added later via the existing storage and pipeline abstractions.

---

## Author

Mirza Anas
