"""Baseline walk-forward LightGBM training pipeline."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from trading_bot.logging_config import get_logger
from trading_bot.models.predict import build_prediction_frame, predict_binary_classifier
from trading_bot.schemas.datasets import summarize_missing_values
from trading_bot.schemas.modeling import (
    ModelRunMetadata,
    is_target_column,
    select_numeric_feature_columns,
    summarize_target_distribution,
    summarize_training_dataset,
    validate_training_dataset,
)
from trading_bot.settings import PROJECT_ROOT, AppSettings, load_settings
from trading_bot.storage.dataset_store import DatasetStore
from trading_bot.storage.evaluation_store import EvaluationStore
from trading_bot.storage.model_store import ModelStore
from trading_bot.validation.evaluation_report import build_evaluation_report
from trading_bot.validation.metrics import (
    aggregate_classification_metrics,
    compute_classification_metrics,
)
from trading_bot.validation.walk_forward import WalkForwardSplitter

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - Python < 3.11 compatibility
    from datetime import timezone

    UTC = timezone.utc  # noqa: UP017

logger = get_logger(__name__)


class TrainingPipelineError(Exception):
    """Raised when walk-forward training cannot complete safely."""


@dataclass
class TrainingResult:
    """Result of a single-model training call."""

    model_id: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    is_success: bool = False
    model: Any | None = None


@dataclass(frozen=True)
class FoldTrainingArtifacts:
    """Fold-level outputs retained during one walk-forward run."""

    metrics_row: dict[str, Any]
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    model: Any
    lightgbm_version: str
    lightgbm_device_type: str
    effective_device: str
    gpu_available_check_result: str


@dataclass(frozen=True)
class TrainingRunResult:
    """Outputs of one completed walk-forward training run."""

    run_id: str
    requested_device: str
    effective_device: str
    feature_columns: list[str]
    target_column: str
    fold_metrics: pd.DataFrame
    aggregate_metrics: dict[str, Any]
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    aggregate_feature_importance: pd.DataFrame
    metadata: dict[str, Any]
    report: dict[str, Any]
    model_paths: list[str]
    evaluation_dir: str
    models_dir: str


class WalkForwardTrainingPipeline:
    """Train and evaluate a baseline LightGBM classifier across walk-forward folds."""

    def __init__(
        self,
        *,
        dataset_store: DatasetStore,
        model_store: ModelStore,
        evaluation_store: EvaluationStore,
        asset: str,
        symbol: str,
        timeframe: str,
        label_version: str,
        dataset_version: str,
        model_version: str,
        primary_target: str,
        random_seed: int,
        save_fold_models: bool,
        requested_device: str,
        allow_cuda_fallback_to_cpu: bool,
        probability_threshold: float,
        include_feature_patterns: list[str],
        exclude_feature_patterns: list[str],
        walk_forward_mode: str,
        min_train_rows: int,
        validation_rows: int,
        step_rows: int,
        max_folds: int | None,
        rolling_train_rows: int | None,
        lightgbm_params: dict[str, Any],
        feature_version: str = "unknown",
    ) -> None:
        self.dataset_store = dataset_store
        self.model_store = model_store
        self.evaluation_store = evaluation_store
        self.default_asset = asset
        self.default_symbol = symbol
        self.default_timeframe = timeframe
        self.default_feature_version = feature_version
        self.default_label_version = label_version
        self.default_dataset_version = dataset_version
        self.default_model_version = model_version
        self.primary_target = primary_target
        self.random_seed = random_seed
        self.save_fold_models = save_fold_models
        self.requested_device = requested_device
        self.allow_cuda_fallback_to_cpu = allow_cuda_fallback_to_cpu
        self.probability_threshold = probability_threshold
        self.include_feature_patterns = include_feature_patterns
        self.exclude_feature_patterns = exclude_feature_patterns
        self.walk_forward_mode = walk_forward_mode
        self.min_train_rows = min_train_rows
        self.validation_rows = validation_rows
        self.step_rows = step_rows
        self.max_folds = max_folds
        self.rolling_train_rows = rolling_train_rows
        self.lightgbm_params = dict(lightgbm_params)

    @classmethod
    def from_settings(cls, settings: AppSettings | None = None) -> WalkForwardTrainingPipeline:
        """Create a training pipeline from project settings."""
        resolved_settings = settings or load_settings()
        dataset_store = DatasetStore(PROJECT_ROOT / resolved_settings.datasets.output_dir)
        model_store = ModelStore(PROJECT_ROOT / resolved_settings.artifacts.models_dir)
        evaluation_store = EvaluationStore(
            PROJECT_ROOT / resolved_settings.artifacts.evaluation_dir
        )

        return cls(
            dataset_store=dataset_store,
            model_store=model_store,
            evaluation_store=evaluation_store,
            asset=resolved_settings.datasets.asset,
            symbol=resolved_settings.datasets.symbol,
            timeframe=resolved_settings.datasets.timeframe,
            feature_version=resolved_settings.features.feature_version,
            label_version=resolved_settings.labels.label_version,
            dataset_version=resolved_settings.datasets.dataset_version,
            model_version=resolved_settings.training.model_version,
            primary_target=resolved_settings.training.primary_target,
            random_seed=resolved_settings.training.random_seed,
            save_fold_models=resolved_settings.training.save_fold_models,
            requested_device=resolved_settings.training.device,
            allow_cuda_fallback_to_cpu=resolved_settings.training.allow_cuda_fallback_to_cpu,
            probability_threshold=resolved_settings.training.probability_threshold,
            include_feature_patterns=resolved_settings.training.include_feature_patterns,
            exclude_feature_patterns=resolved_settings.training.exclude_feature_patterns,
            walk_forward_mode=resolved_settings.walk_forward.mode,
            min_train_rows=resolved_settings.walk_forward.min_train_rows,
            validation_rows=resolved_settings.walk_forward.validation_rows,
            step_rows=resolved_settings.walk_forward.step_rows,
            max_folds=resolved_settings.walk_forward.max_folds,
            rolling_train_rows=resolved_settings.walk_forward.rolling_train_rows,
            lightgbm_params=resolved_settings.lightgbm.model_dump(),
        )

    def run_training(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        asset: str | None = None,
        feature_version: str | None = None,
        label_version: str | None = None,
        dataset_version: str | None = None,
        model_version: str | None = None,
        target_column: str | None = None,
        requested_device: str | None = None,
        run_id: str | None = None,
    ) -> TrainingRunResult:
        """Run walk-forward LightGBM training and persist all artifacts."""
        asset = asset or self.default_asset
        symbol = symbol or self.default_symbol
        timeframe = timeframe or self.default_timeframe
        feature_version = feature_version or self.default_feature_version
        label_version = label_version or self.default_label_version
        dataset_version = dataset_version or self.default_dataset_version
        model_version = model_version or self.default_model_version
        target_column = target_column or self.primary_target
        requested_device = (requested_device or self.requested_device).lower()

        dataset_df = self.dataset_store.read_artifact(
            "supervised_dataset",
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            label_version=label_version,
            dataset_version=dataset_version,
        )
        if dataset_df.empty:
            raise TrainingPipelineError(
                "No stored supervised dataset found. Build the step-5 dataset before training."
            )

        dataset_df = dataset_df.dropna(subset=[target_column]).copy()
        dataset_df["timestamp"] = pd.to_datetime(dataset_df["timestamp"], utc=True)
        dataset_df = dataset_df.sort_values("timestamp").reset_index(drop=True)

        target_columns = [column for column in dataset_df.columns if _is_target_column(column)]
        feature_columns = select_numeric_feature_columns(
            dataset_df,
            target_columns=target_columns,
            include_patterns=self.include_feature_patterns,
            exclude_patterns=self.exclude_feature_patterns,
        )
        validate_training_dataset(
            dataset_df,
            target_column=target_column,
            feature_columns=feature_columns,
        )

        splitter = WalkForwardSplitter(
            mode=self.walk_forward_mode,
            min_train_rows=self.min_train_rows,
            validation_rows=self.validation_rows,
            step_rows=self.step_rows,
            max_folds=self.max_folds,
            rolling_train_rows=self.rolling_train_rows,
        )
        splits = splitter.split(dataset_df)
        if not splits:
            raise TrainingPipelineError(
                "Walk-forward splitter produced no folds. "
                "Increase dataset size or reduce split sizes."
            )

        resolved_run_id = run_id or _generate_run_id()
        model_paths: list[str] = []
        fold_metric_rows: list[dict[str, Any]] = []
        prediction_frames: list[pd.DataFrame] = []
        feature_importance_frames: list[pd.DataFrame] = []

        active_requested_device = requested_device
        effective_device = "cpu"
        lightgbm_version = "unknown"
        lightgbm_device_type = "cpu"
        gpu_available_check_result = "not_requested"

        logger.info(
            "walk_forward_training_started",
            run_id=resolved_run_id,
            symbol=symbol,
            timeframe=timeframe,
            target_column=target_column,
            feature_count=len(feature_columns),
            fold_count=len(splits),
            requested_device=requested_device,
        )

        for split in splits:
            train_df = dataset_df.iloc[split.train_start_idx : split.train_end_idx + 1].copy()
            validation_df = dataset_df.iloc[
                split.validation_start_idx : split.validation_end_idx + 1
            ].copy()

            y_train = train_df[target_column].astype(int)
            if y_train.nunique() < 2:
                raise TrainingPipelineError(
                    f"Fold {split.fold_id} training labels contain only one class; "
                    "cannot fit LightGBM safely."
                )

            fold_artifacts = self._train_fold(
                train_df=train_df,
                validation_df=validation_df,
                feature_columns=feature_columns,
                target_column=target_column,
                split=split,
                requested_device=active_requested_device,
            )

            if requested_device == "cuda" and fold_artifacts.effective_device == "cpu":
                active_requested_device = "cpu"

            effective_device = fold_artifacts.effective_device
            lightgbm_version = fold_artifacts.lightgbm_version
            lightgbm_device_type = fold_artifacts.lightgbm_device_type
            if gpu_available_check_result in {"not_requested", "cpu_requested"}:
                gpu_available_check_result = fold_artifacts.gpu_available_check_result

            fold_metric_rows.append(fold_artifacts.metrics_row)
            prediction_frames.append(fold_artifacts.predictions)
            feature_importance_frames.append(fold_artifacts.feature_importance)

            if self.save_fold_models:
                artifact = self.model_store.write_fold_model(
                    fold_artifacts.model,
                    asset=asset,
                    symbol=symbol,
                    timeframe=timeframe,
                    model_version=model_version,
                    run_id=resolved_run_id,
                    fold_id=split.fold_id,
                )
                model_paths.append(artifact.path)

        fold_metrics_df = (
            pd.DataFrame(fold_metric_rows).sort_values("fold_id").reset_index(drop=True)
        )
        predictions_df = pd.concat(prediction_frames, ignore_index=True).sort_values(
            ["timestamp", "fold_id"]
        ).reset_index(drop=True)
        feature_importance_df = pd.concat(feature_importance_frames, ignore_index=True)
        aggregate_feature_importance_df = _aggregate_feature_importance(feature_importance_df)
        aggregate_metrics = aggregate_classification_metrics(fold_metric_rows)

        dataset_summary = summarize_training_dataset(
            dataset_df,
            feature_columns=feature_columns,
            target_columns=target_columns,
        )
        missing_summary = summarize_missing_values(dataset_df)
        target_summary = summarize_target_distribution(dataset_df[target_column])

        metadata = ModelRunMetadata(
            run_id=resolved_run_id,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            dataset_version=dataset_version,
            label_version=label_version,
            model_version=model_version,
            target_column=target_column,
            feature_columns=feature_columns,
            requested_device=requested_device,
            effective_device=effective_device,
            lightgbm_version=lightgbm_version,
            random_seed=self.random_seed,
            probability_threshold=self.probability_threshold,
            walk_forward_mode=self.walk_forward_mode,
            min_train_rows=self.min_train_rows,
            validation_rows=self.validation_rows,
            step_rows=self.step_rows,
            fold_count=len(splits),
            dataset_row_count=dataset_summary.row_count,
            created_at=datetime.now(UTC).isoformat(),
            lightgbm_device_type=lightgbm_device_type,
            dataset_path=str(
                self.dataset_store.artifact_path(
                    "supervised_dataset",
                    asset=asset,
                    symbol=symbol,
                    timeframe=timeframe,
                    label_version=label_version,
                    dataset_version=dataset_version,
                )
            ),
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            gpu_available_check_result=gpu_available_check_result,
            extra={
                "feature_version": feature_version,
                "target_summary": target_summary,
                "missing_summary": missing_summary,
                "dataset_min_timestamp": dataset_summary.min_timestamp,
                "dataset_max_timestamp": dataset_summary.max_timestamp,
            },
        ).to_dict()
        metadata["feature_version"] = feature_version

        artifact_paths = {
            "dataset_path": str(
                self.dataset_store.artifact_path(
                    "supervised_dataset",
                    asset=asset,
                    symbol=symbol,
                    timeframe=timeframe,
                    label_version=label_version,
                    dataset_version=dataset_version,
                )
            ),
            "evaluation_dir": str(
                self.evaluation_store.run_dir(
                    asset=asset,
                    symbol=symbol,
                    timeframe=timeframe,
                    model_version=model_version,
                    run_id=resolved_run_id,
                )
            ),
            "models_dir": str(
                self.model_store.run_dir(
                    asset=asset,
                    symbol=symbol,
                    timeframe=timeframe,
                    model_version=model_version,
                    run_id=resolved_run_id,
                )
            ),
        }

        report = build_evaluation_report(
            run_metadata=metadata,
            aggregate_metrics=aggregate_metrics,
            fold_metrics_df=fold_metrics_df,
            predictions_df=predictions_df,
            feature_importance_df=aggregate_feature_importance_df,
            target_distribution_summary=target_summary,
            artifact_paths=artifact_paths,
        ).to_dict()

        self.evaluation_store.write_fold_metrics(
            fold_metrics_df,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=resolved_run_id,
        )
        self.evaluation_store.write_predictions(
            predictions_df,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=resolved_run_id,
        )
        self.evaluation_store.write_feature_importance(
            feature_importance_df,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=resolved_run_id,
        )
        self.evaluation_store.write_feature_importance(
            aggregate_feature_importance_df,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=resolved_run_id,
            aggregate=True,
        )
        self.evaluation_store.write_aggregate_metrics(
            aggregate_metrics,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=resolved_run_id,
        )
        self.evaluation_store.write_run_metadata(
            metadata,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=resolved_run_id,
        )
        self.evaluation_store.write_report(
            report,
            asset=asset,
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            run_id=resolved_run_id,
        )

        logger.info(
            "walk_forward_training_complete",
            run_id=resolved_run_id,
            fold_count=len(splits),
            feature_count=len(feature_columns),
            requested_device=requested_device,
            effective_device=effective_device,
            aggregate_metrics=aggregate_metrics,
        )

        return TrainingRunResult(
            run_id=resolved_run_id,
            requested_device=requested_device,
            effective_device=effective_device,
            feature_columns=feature_columns,
            target_column=target_column,
            fold_metrics=fold_metrics_df,
            aggregate_metrics=aggregate_metrics,
            predictions=predictions_df,
            feature_importance=feature_importance_df,
            aggregate_feature_importance=aggregate_feature_importance_df,
            metadata=metadata,
            report=report,
            model_paths=model_paths,
            evaluation_dir=str(
                self.evaluation_store.run_dir(
                    asset=asset,
                    symbol=symbol,
                    timeframe=timeframe,
                    model_version=model_version,
                    run_id=resolved_run_id,
                )
            ),
            models_dir=str(
                self.model_store.run_dir(
                    asset=asset,
                    symbol=symbol,
                    timeframe=timeframe,
                    model_version=model_version,
                    run_id=resolved_run_id,
                )
            ),
        )

    def _train_fold(
        self,
        *,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        split,
        requested_device: str,
    ) -> FoldTrainingArtifacts:
        """Train one fold model and compute validation outputs."""
        x_train = train_df.loc[:, feature_columns]
        x_validation = validation_df.loc[:, feature_columns]
        y_train = train_df[target_column].astype(int)
        y_validation = validation_df[target_column].astype(int)

        model, effective_device, lightgbm_device_type, lightgbm_version, gpu_check_result = (
            _fit_lightgbm_classifier(
                x_train=x_train,
                y_train=y_train,
                requested_device=requested_device,
                allow_cuda_fallback_to_cpu=self.allow_cuda_fallback_to_cpu,
                lightgbm_params=self.lightgbm_params,
                random_seed=self.random_seed,
            )
        )

        y_proba, y_pred = predict_binary_classifier(
            model,
            x_validation,
            threshold=self.probability_threshold,
        )
        metrics = compute_classification_metrics(y_validation, y_pred, y_proba)

        metrics_row = {
            "fold_id": split.fold_id,
            "train_start": split.train_start.isoformat() if split.train_start is not None else None,
            "train_end": split.train_end.isoformat() if split.train_end is not None else None,
            "validation_start": (
                split.validation_start.isoformat() if split.validation_start is not None else None
            ),
            "validation_end": (
                split.validation_end.isoformat() if split.validation_end is not None else None
            ),
            "n_train": split.n_train,
            "n_validation": split.n_validation,
            **metrics.to_dict(),
        }

        predictions = build_prediction_frame(
            base_df=validation_df,
            fold_id=split.fold_id,
            y_true=y_validation,
            y_proba=y_proba,
            y_pred=y_pred,
        )
        feature_importance = _extract_feature_importance(
            model,
            feature_columns=feature_columns,
            fold_id=split.fold_id,
        )

        return FoldTrainingArtifacts(
            metrics_row=metrics_row,
            predictions=predictions,
            feature_importance=feature_importance,
            model=model,
            lightgbm_version=lightgbm_version,
            lightgbm_device_type=lightgbm_device_type,
            effective_device=effective_device,
            gpu_available_check_result=gpu_check_result,
        )


def train_model(
    features: Any = None,
    labels: Any = None,
    params: dict[str, Any] | None = None,
) -> TrainingResult:
    """Train a single baseline LightGBM binary classifier for compatibility."""
    if features is None or labels is None:
        return TrainingResult(
            model_id="missing_inputs",
            metrics={},
            feature_importance={},
            params=params or {},
            is_success=False,
        )

    feature_df = pd.DataFrame(features).copy()
    label_series = pd.Series(labels).astype(int)
    try:
        model, effective_device, _, _, _ = _fit_lightgbm_classifier(
            x_train=feature_df,
            y_train=label_series,
            requested_device="cpu",
            allow_cuda_fallback_to_cpu=True,
            lightgbm_params=params or {},
            random_seed=int((params or {}).get("random_state", 42)),
        )
    except Exception as exc:
        logger.error("train_model_failed", error=str(exc))
        return TrainingResult(
            model_id="train_model_failed",
            metrics={},
            feature_importance={},
            params=params or {},
            is_success=False,
        )

    importance_df = _extract_feature_importance(
        model,
        feature_columns=list(feature_df.columns),
        fold_id=0,
    )
    return TrainingResult(
        model_id=f"single_model_{effective_device}",
        metrics={},
        feature_importance=dict(
            zip(
                importance_df["feature_name"].tolist(),
                importance_df["importance"].tolist(),
                strict=False,
            )
        ),
        params=params or {},
        is_success=True,
        model=model,
    )


def _fit_lightgbm_classifier(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    requested_device: str,
    allow_cuda_fallback_to_cpu: bool,
    lightgbm_params: dict[str, Any],
    random_seed: int,
) -> tuple[Any, str, str, str, str]:
    """Train one LightGBM classifier with safe CUDA fallback."""
    lightgbm_module = _import_lightgbm_module()
    lightgbm_version = str(getattr(lightgbm_module, "__version__", "unknown"))
    requested_device = requested_device.lower()

    if requested_device not in {"cpu", "cuda"}:
        raise TrainingPipelineError(
            f"Unsupported training device {requested_device!r}. Expected 'cpu' or 'cuda'."
        )

    if requested_device == "cpu":
        model = _fit_with_device(
            lightgbm_module=lightgbm_module,
            x_train=x_train,
            y_train=y_train,
            lightgbm_params=lightgbm_params,
            random_seed=random_seed,
            lightgbm_device_type="cpu",
        )
        return model, "cpu", "cpu", lightgbm_version, "cpu_requested"

    cuda_errors: list[str] = []
    for candidate_device_type in ("cuda", "gpu"):
        try:
            model = _fit_with_device(
                lightgbm_module=lightgbm_module,
                x_train=x_train,
                y_train=y_train,
                lightgbm_params=lightgbm_params,
                random_seed=random_seed,
                lightgbm_device_type=candidate_device_type,
            )
            return (
                model,
                "cuda",
                candidate_device_type,
                lightgbm_version,
                f"cuda_success:{candidate_device_type}",
            )
        except Exception as exc:
            message = f"{candidate_device_type}:{exc}"
            cuda_errors.append(message)
            logger.warning(
                "lightgbm_cuda_attempt_failed",
                device_type=candidate_device_type,
                error=str(exc),
            )

    if not allow_cuda_fallback_to_cpu:
        raise TrainingPipelineError(
            "CUDA training was requested but failed and CPU fallback is disabled: "
            + " | ".join(cuda_errors)
        )

    logger.warning(
        "lightgbm_cuda_fallback_to_cpu",
        error=" | ".join(cuda_errors),
    )
    model = _fit_with_device(
        lightgbm_module=lightgbm_module,
        x_train=x_train,
        y_train=y_train,
        lightgbm_params=lightgbm_params,
        random_seed=random_seed,
        lightgbm_device_type="cpu",
    )
    return model, "cpu", "cpu", lightgbm_version, "cuda_fallback_to_cpu"


def _fit_with_device(
    *,
    lightgbm_module: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    lightgbm_params: dict[str, Any],
    random_seed: int,
    lightgbm_device_type: str,
) -> Any:
    """Fit one LightGBM sklearn classifier for a specific device setting."""
    params = dict(lightgbm_params)
    params.setdefault("objective", "binary")
    params.setdefault("metric", "binary_logloss")
    params["device_type"] = lightgbm_device_type
    params["random_state"] = random_seed

    try:
        classifier = lightgbm_module.LGBMClassifier(**params)
        classifier.fit(x_train, y_train)
    except Exception as exc:
        raise TrainingPipelineError(
            f"LightGBM training failed for device_type={lightgbm_device_type}: {exc}"
        ) from exc
    return classifier


def _import_lightgbm_module() -> Any:
    """Import lightgbm lazily so CPU-only environments still import this module."""
    try:
        import lightgbm as lightgbm_module
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise TrainingPipelineError(
            "lightgbm is not installed. Install the ML extras with `pip install -e \".[ml]\"`."
        ) from exc

    return lightgbm_module


def _extract_feature_importance(
    model: Any,
    *,
    feature_columns: list[str],
    fold_id: int,
) -> pd.DataFrame:
    """Extract feature importance from a fitted LightGBM model."""
    importance = pd.Series([0.0] * len(feature_columns), index=feature_columns, dtype=float)

    if hasattr(model, "booster_") and hasattr(model.booster_, "feature_importance"):
        try:
            importance = pd.Series(
                model.booster_.feature_importance(importance_type="gain"),
                index=feature_columns,
                dtype=float,
            )
        except Exception:
            importance = pd.Series(
                getattr(model, "feature_importances_", [0.0] * len(feature_columns)),
                index=feature_columns,
                dtype=float,
            )
    elif hasattr(model, "feature_importances_"):
        importance = pd.Series(
            getattr(model, "feature_importances_", [0.0] * len(feature_columns)),
            index=feature_columns,
            dtype=float,
        )

    frame = importance.reset_index()
    frame.columns = ["feature_name", "importance"]
    frame["fold_id"] = int(fold_id)
    frame = frame.sort_values(["importance", "feature_name"], ascending=[False, True]).reset_index(
        drop=True
    )
    return frame.loc[:, ["fold_id", "feature_name", "importance"]]


def _aggregate_feature_importance(feature_importance_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate feature importance across folds with a simple mean."""
    if feature_importance_df.empty:
        return pd.DataFrame(columns=["feature_name", "mean_importance", "folds_present"])

    aggregate = (
        feature_importance_df.groupby("feature_name", as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            folds_present=("fold_id", "nunique"),
        )
        .sort_values(["mean_importance", "feature_name"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return aggregate


def _generate_run_id() -> str:
    """Return a timestamp-based run id suitable for local storage paths."""
    return datetime.now(UTC).strftime("RUN%Y%m%dT%H%M%SZ")


def _is_target_column(column_name: str) -> bool:
    """Return whether a column belongs to the supervised target set."""
    return is_target_column(column_name)
