"""Standardized segmentation and classification evaluation suites."""

from wsp_core.evaluation.classification import (
    compute_pr_auc,
    compute_roc_auc,
    evaluate_classifier,
)
from wsp_core.evaluation.segmentation import (
    bounding_slices,
    contour_and_band,
    evaluate_segmentation_ablation,
    evaluate_segmentation_paired_audit,
    evaluate_segmentation_swaps,
    group_segmentation_scores,
)

__all__ = [
    "bounding_slices",
    "contour_and_band",
    "group_segmentation_scores",
    "evaluate_segmentation_ablation",
    "evaluate_segmentation_swaps",
    "evaluate_segmentation_paired_audit",
    "compute_roc_auc",
    "compute_pr_auc",
    "evaluate_classifier",
]
