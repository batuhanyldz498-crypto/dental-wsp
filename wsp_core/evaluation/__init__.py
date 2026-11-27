"""Standardized segmentation and classification evaluation suites."""

from wsp_core.evaluation.baselines import (
    GlobalIntensityLogReg,
    build_human_written_seg82_payload,
    compute_otsu_threshold,
    denpar_crown_edge_rule_score,
    evaluate_classification_baselines,
    evaluate_denpar_crown_edge_baseline,
    evaluate_segmentation_baselines,
    extract_global_intensity_features,
    fit_global_intensity_logreg,
    fit_global_intensity_threshold,
)
from wsp_core.evaluation.classification import (
    compute_pr_auc,
    compute_roc_auc,
    evaluate_classifier,
    hanley_mcneil_auc_ci,
)
from wsp_core.evaluation.segmentation import (
    BOX_VALID_FRAC_COLUMN,
    OUTSIDE_FP_SHARE_COLUMN,
    bounding_slices,
    contour_and_band,
    evaluate_segmentation_ablation,
    evaluate_segmentation_paired_audit,
    evaluate_segmentation_swaps,
    group_segmentation_scores,
    score_segmentation_masks,
)

__all__ = [
    "BOX_VALID_FRAC_COLUMN",
    "OUTSIDE_FP_SHARE_COLUMN",
    "GlobalIntensityLogReg",
    "bounding_slices",
    "build_human_written_seg82_payload",
    "compute_otsu_threshold",
    "compute_pr_auc",
    "compute_roc_auc",
    "contour_and_band",
    "denpar_crown_edge_rule_score",
    "evaluate_classification_baselines",
    "evaluate_classifier",
    "evaluate_denpar_crown_edge_baseline",
    "evaluate_segmentation_ablation",
    "evaluate_segmentation_baselines",
    "evaluate_segmentation_paired_audit",
    "evaluate_segmentation_swaps",
    "extract_global_intensity_features",
    "fit_global_intensity_logreg",
    "fit_global_intensity_threshold",
    "group_segmentation_scores",
    "hanley_mcneil_auc_ci",
    "score_segmentation_masks",
]
