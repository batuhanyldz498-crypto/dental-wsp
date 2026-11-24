"""Deterministic label-budget training and WSP warm-start engine."""

from wsp_core.training.trainer import (
    TrainingConfig,
    sample_nested_group_budgets,
    train_classification_budget,
    train_segmentation_budget,
)

__all__ = [
    "TrainingConfig",
    "sample_nested_group_budgets",
    "train_segmentation_budget",
    "train_classification_budget",
]
