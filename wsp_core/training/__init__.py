"""Deterministic label-budget training, WSP warm-start engine, and 3-arm runner."""

from wsp_core.training.runner import (
    plot_classification_learning_curve,
    run_classification_three_arm_validation,
)
from wsp_core.training.trainer import (
    TrainingConfig,
    sample_nested_group_budgets,
    sample_stratified_cls_budgets,
    train_classification_budget,
    train_segmentation_budget,
)

__all__ = [
    "TrainingConfig",
    "sample_nested_group_budgets",
    "sample_stratified_cls_budgets",
    "train_segmentation_budget",
    "train_classification_budget",
    "run_classification_three_arm_validation",
    "plot_classification_learning_curve",
]
