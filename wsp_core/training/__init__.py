"""Deterministic label-budget training, WSP warm-start engine, and 3-arm runner."""

from wsp_core.training.controls import (
    ScheduleSpec,
    bootstrap_group_mean_ci,
    default_factorial_schedules,
    plot_factorial_controls_svg,
    run_segmentation_factorial_controls,
    summarize_factorial_controls,
)
from wsp_core.training.runner import (
    plot_classification_learning_curve,
    plot_segmentation_learning_curve,
    run_classification_three_arm_validation,
    run_segmentation_three_arm_validation,
)
from wsp_core.training.trainer import (
    TrainingConfig,
    compute_prior_output_bias,
    sample_nested_group_budgets,
    sample_stratified_cls_budgets,
    train_classification_budget,
    train_segmentation_budget,
)

__all__ = [
    "ScheduleSpec",
    "TrainingConfig",
    "bootstrap_group_mean_ci",
    "compute_prior_output_bias",
    "default_factorial_schedules",
    "plot_classification_learning_curve",
    "plot_factorial_controls_svg",
    "plot_segmentation_learning_curve",
    "run_classification_three_arm_validation",
    "run_segmentation_factorial_controls",
    "run_segmentation_three_arm_validation",
    "sample_nested_group_budgets",
    "sample_stratified_cls_budgets",
    "summarize_factorial_controls",
    "train_classification_budget",
    "train_segmentation_budget",
]
