"""Labeling package for future-return targets and supervised datasets."""

from trading_bot.labeling.dataset_builder import DatasetBuilderError, SupervisedDatasetBuilder
from trading_bot.labeling.labels import (
    LabelDefinition,
    LabelType,
    generate_future_return_labels,
    get_label_definitions,
)

__all__ = [
    "DatasetBuilderError",
    "SupervisedDatasetBuilder",
    "LabelDefinition",
    "LabelType",
    "generate_future_return_labels",
    "get_label_definitions",
]
