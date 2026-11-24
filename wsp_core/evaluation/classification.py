"""Classification evaluation suite with sign-flip and margin-crop shortcut checks."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from wsp_core.architectures.cls_lenet2d import cls_lenet2d_logits, cls_lenet2d_margin_ablation
from wsp_core.guards import require


@dataclass(frozen=True)
class ClassificationSample:
    """Single standardized 2D classification sample."""

    sample_id: str
    group_id: str
    image: np.ndarray  # (H, W) float32 in [0, 1]
    label: int  # 0 or 1


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact ROC-AUC via mid-rank Mann-Whitney U statistic (handles ties exactly)."""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    require(n_pos > 0 and n_neg > 0, "ROC-AUC requires both positive and negative examples.")

    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(sorted_s):
        j = i + 1
        while j < len(sorted_s) and sorted_s[j] == sorted_s[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j

    pos_rank_sum = float(np.sum(ranks[y == 1]))
    return (pos_rank_sum - 0.5 * n_pos * (n_pos + 1)) / (n_pos * n_neg)


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Step-function Average Precision (exact PR-AUC)."""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y == 1))
    require(n_pos > 0, "PR-AUC requires at least one positive example.")

    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - recall_prev) * precision))


def evaluate_classifier(
    payload: dict[str, Any],
    samples: list[ClassificationSample],
    threshold: float = 0.0,
    border_fraction: float = 0.10,
    sign_flip_threshold: float = 0.35,
) -> dict[str, Any]:
    """Evaluate a frozen or trained `cls-lenet2d-v1` model with shortcut diagnostics."""
    require(len(samples) > 0, "Classification evaluation requires at least one sample.")
    y_true = np.array([s.label for s in samples], dtype=np.int64)
    logits = np.array([cls_lenet2d_logits(s.image, payload) for s in samples], dtype=np.float64)
    require(bool(np.isfinite(logits).all()), "Non-finite classification logits.")

    roc_auc = compute_roc_auc(y_true, logits)
    pr_auc = compute_pr_auc(y_true, logits)
    preds = (logits >= threshold).astype(np.int64)
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / len(y_true)

    # Margin-crop shortcut check (zero outer border)
    margin_logits = np.array(
        [cls_lenet2d_logits(cls_lenet2d_margin_ablation(s.image, border_fraction), payload) for s in samples],
        dtype=np.float64,
    )
    margin_roc_auc = compute_roc_auc(y_true, margin_logits)

    # Optional group-level ROC-AUC (e.g., per-scan mean AUC for DentVoxel axial slices)
    by_group: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for s, logit in zip(samples, logits):
        by_group[s.group_id].append((s.label, float(logit)))
    group_aucs = []
    for rows in by_group.values():
        gy = np.array([r[0] for r in rows], dtype=np.int64)
        gs = np.array([r[1] for r in rows], dtype=np.float64)
        if np.any(gy == 1) and np.any(gy == 0):
            group_aucs.append(compute_roc_auc(gy, gs))

    return {
        "n_samples": len(samples),
        "n_groups": len(by_group),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "margin_ablated_roc_auc": float(margin_roc_auc),
        "margin_auc_drop": float(roc_auc - margin_roc_auc),
        "sign_flip_flagged": bool(roc_auc < sign_flip_threshold),
        "mean_within_group_roc_auc": float(np.mean(group_aucs)) if group_aucs else None,
    }
