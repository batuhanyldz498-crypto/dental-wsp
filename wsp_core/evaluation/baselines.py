"""Trivial and non-LLM baseline suite for `dental-wsp` validation controls.

All baselines use the exact same preprocessing, valid masks, group weighting, and
evaluation metrics as the WSP and learned arms, and operate strictly on train/val
splits (never loading held-out test splits).

Segmentation baselines (Track 2):
1. All-foreground: predict every valid pixel/voxel.
2. Per-image Otsu threshold on valid normalized intensities in [0, 1].
3. Fixed global intensity threshold `t*` chosen by maximizing equal-group-weighted
   mean Dice on the TRAIN split only, reported on VAL.
4. Human-written 82-parameter non-LLM prior: filter 0 (scale 1) is a pure center-tap
   intensity detector (`center = 1`, `feature_bias[0] = -t*`, positive readout, all
   other filters/grid zero, plus `-1e-4` output-bias tie-breaker for inclusive
   `logit >= 0.0`). Evaluated frozen (`K = 0`) and optionally as a warm-start.

Classification baselines (Track 1):
5a. Constant-score baseline (`ROC-AUC = 0.500`, `PR-AUC = class prevalence`).
5b. L2-regularized logistic regression on 3 global intensity features
    `(mean, frac < 0.1, frac > 0.9)`, fit on each K-budget train subset (pure NumPy/SciPy).
5c. Zero-parameter DenPAR hand-coded vertical crown-edge rule (`denpar_cls`):
    signed squared asymmetry of a 5x5 bright-above -> dark-below horizontal-edge filter.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from wsp_core.architectures.cls_lenet2d import cls_lenet2d_margin_ablation
from wsp_core.evaluation.classification import (
    ClassificationSample,
    compute_pr_auc,
    compute_roc_auc,
    evaluate_classifier,
)
from wsp_core.evaluation.segmentation import (
    SegmentationSample,
    predict_cohort,
    score_segmentation_masks,
)
from wsp_core.guards import require
from wsp_core.schemas import strict_json_loads, validate_parameters, verify_response_lock, zero_payload
from wsp_core.training.trainer import (
    TrainingConfig,
    sample_nested_group_budgets,
    sample_stratified_cls_budgets,
    train_segmentation_budget,
)

DEFAULT_THRESHOLD_GRID = np.round(np.linspace(0.01, 0.99, 99), 4)


def compute_otsu_threshold(image: np.ndarray, valid: np.ndarray, *, n_bins: int = 256) -> float:
    """Deterministic histogram-based Otsu threshold computed strictly over valid pixels/voxels."""
    vals = np.asarray(image, dtype=np.float32)[valid > 0]
    require(vals.size > 0, "Cannot compute Otsu threshold on an empty valid mask.")
    vmin = float(vals.min())
    vmax = float(vals.max())
    if vmax <= vmin + 1e-12:
        return vmin

    hist, bin_edges = np.histogram(vals, bins=n_bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0.0:
        return 0.5
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    w0 = np.cumsum(hist)
    w1 = total - w0
    cum_mean = np.cumsum(hist * centers)
    global_mean = cum_mean[-1]

    valid_splits = (w0[:-1] > 0.0) & (w1[:-1] > 0.0)
    if not np.any(valid_splits):
        return float(vals.mean())

    m0 = np.zeros(n_bins - 1, dtype=np.float64)
    m1 = np.zeros(n_bins - 1, dtype=np.float64)
    m0[valid_splits] = cum_mean[:-1][valid_splits] / w0[:-1][valid_splits]
    m1[valid_splits] = (global_mean - cum_mean[:-1][valid_splits]) / w1[:-1][valid_splits]
    between_var = np.where(valid_splits, w0[:-1] * w1[:-1] * (m0 - m1) ** 2, -1.0)
    max_var = float(between_var.max())
    best_indices = np.flatnonzero(between_var >= max_var - 1e-12 * max(abs(max_var), 1.0))
    # Average across tied maximal splits so bimodal gaps center the threshold between modes.
    return float(np.mean(bin_edges[best_indices + 1]))


def fit_global_intensity_threshold(
    train_samples: list[SegmentationSample],
    *,
    threshold_grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Select global intensity threshold `t*` maximizing equal-group-weighted mean Dice on TRAIN only.

    Uses exact reverse-cumulative histogram counts so 2D and 3D cohorts evaluate all grid thresholds
    in a single linear pass, matching `(image >= t) & valid` + `group_segmentation_scores` exactly.
    Returns `(best_threshold, best_train_dice_pct)`.
    """
    require(len(train_samples) > 0, "Threshold fitting requires at least one training sample.")
    grid = np.asarray(
        DEFAULT_THRESHOLD_GRID if threshold_grid is None else threshold_grid,
        dtype=np.float64,
    )
    require(grid.ndim == 1 and grid.size > 0, "threshold_grid must be a non-empty 1D array.")
    grid = np.sort(np.unique(grid))
    m = int(grid.size)

    grouped_dice: dict[str, list[np.ndarray]] = defaultdict(list)
    for s in train_samples:
        valid = s.valid > 0
        truth = (s.target > 0) & valid
        n_ref = int(np.count_nonzero(truth))
        require(n_ref > 0, f"Sample {s.sample_id} has an empty reference mask.")

        val_vals = s.image[valid].astype(np.float64, copy=False)
        pos_vals = s.image[truth].astype(np.float64, copy=False)

        # For sorted grid, x >= grid[j] iff j < searchsorted(grid, x, side='right').
        val_bins = np.searchsorted(grid, val_vals, side="right")
        pos_bins = np.searchsorted(grid, pos_vals, side="right")
        val_counts = np.bincount(val_bins, minlength=m + 1)
        pos_counts = np.bincount(pos_bins, minlength=m + 1)

        n_pred = np.cumsum(val_counts[1:][::-1])[::-1].astype(np.float64)
        inter = np.cumsum(pos_counts[1:][::-1])[::-1].astype(np.float64)
        dice_curve = (2.0 * inter) / (n_pred + float(n_ref))
        grouped_dice[s.group_id].append(dice_curve)

    group_curves = np.array([np.mean(curves, axis=0) for curves in grouped_dice.values()], dtype=np.float64)
    mean_curve = group_curves.mean(axis=0)
    max_dice = float(mean_curve.max())
    best_indices = np.flatnonzero(mean_curve >= max_dice - 1e-12)
    best_idx = int(best_indices[len(best_indices) // 2])
    return float(grid[best_idx]), float(100.0 * mean_curve[best_idx])


def build_human_written_seg82_payload(
    architecture: str,
    threshold: float,
    *,
    readout_weight: float = 5.0,
    output_bias_eps: float = 1e-4,
) -> dict[str, Any]:
    """Build the human-written 82-parameter non-LLM prior for 2D or 3D segmentation.

    Sets filter 0 at scale 1 to a pure center-tap intensity detector:
    - center weight = 1.0, all surrounding taps = 0.0
    - `feature_bias[0] = -threshold`
    - `readout[0] = readout_weight` (> 0)
    - filters 1..5 and `spatial_bias` = 0.0
    - `output_bias = -output_bias_eps` (small negative tie-breaker because `dental-wsp` uses
      inclusive `logit >= 0.0`; setting `output_bias = 0.0` would classify every pixel as foreground
      when `ReLU(x - t) == 0`). With `readout_weight = 5.0` and `output_bias_eps = 1e-4`, the
      effective threshold is `threshold + 2e-5` (negligible relative to 8-bit or HU quantization).
    """
    require(readout_weight > 0.0, "readout_weight must be strictly positive.")
    require(output_bias_eps > 0.0, "output_bias_eps must be strictly positive for inclusive logit >= 0.")
    payload = zero_payload(architecture)
    if architecture in ("pano-wsp82-v1", "periapical-wsp82-v1"):
        payload["kernels"][0][1][1] = 1.0
    elif architecture == "cbct-wsp82-v1":
        payload["stencils"][0][0] = 1.0
    else:
        raise ValueError(f"Unsupported segmentation architecture: {architecture}")
    payload["feature_bias"][0] = float(-threshold)
    payload["readout"][0] = float(readout_weight)
    payload["output_bias"] = float(-output_bias_eps)
    validate_parameters(payload, expected_architecture=architecture)
    return payload


def evaluate_segmentation_baselines(
    *,
    task: str,
    architecture: str,
    train_samples: list[SegmentationSample],
    val_samples: list[SegmentationSample],
    design_lock_path: Path | None = None,
    responses_lock_path: Path | None = None,
    threshold_grid: np.ndarray | None = None,
    run_human_warmstart: bool = False,
    run_trained_position_only: bool = False,
    budgets: list[int] | None = None,
    seeds: list[int] | None = None,
    epochs: int = 150,
    lr: float = 3e-2,
    batch_size: int = 32,
    cosine_lr: bool = True,
    device: str = "cpu",
    out_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate trivial, non-LLM, human-prior, and reference WSP/position-only baselines on VAL.

    Returns `(summary_df, metadata_dict)`. Never accesses held-out test splits.
    """
    rows: list[dict[str, Any]] = []

    # 1. All-foreground baseline
    all_fg_masks = {s.sample_id: (s.valid > 0).copy() for s in val_samples}
    fg_metrics, _ = score_segmentation_masks(val_samples, all_fg_masks)
    rows.append(
        {
            "task": task,
            "baseline": "all_foreground",
            "k_budget": 0,
            "n_runs": 1,
            "threshold_param": np.nan,
            "dice_mean": fg_metrics["dice_pct"],
            "dice_std": np.nan,
            "bf1_mean": fg_metrics["bf1_pct"],
            "edge_recall_mean": fg_metrics["edge_recall_pct"],
            "box_iou_mean": fg_metrics["box_iou_pct"],
            "box_valid_vol_mean": fg_metrics["box_valid_vol_pct"],
            "excluded_tooth_mean": fg_metrics["excluded_tooth_pct"],
            "notes": "Predicts every valid pixel/voxel as foreground",
        }
    )

    # 2. Per-image Otsu threshold baseline
    otsu_thresholds = []
    otsu_masks: dict[str, np.ndarray] = {}
    for s in val_samples:
        t_o = compute_otsu_threshold(s.image, s.valid)
        otsu_thresholds.append(t_o)
        otsu_masks[s.sample_id] = (s.image >= t_o) & (s.valid > 0)
    otsu_metrics, _ = score_segmentation_masks(val_samples, otsu_masks)
    rows.append(
        {
            "task": task,
            "baseline": "per_image_otsu",
            "k_budget": 0,
            "n_runs": 1,
            "threshold_param": float(np.mean(otsu_thresholds)),
            "dice_mean": otsu_metrics["dice_pct"],
            "dice_std": np.nan,
            "bf1_mean": otsu_metrics["bf1_pct"],
            "edge_recall_mean": otsu_metrics["edge_recall_pct"],
            "box_iou_mean": otsu_metrics["box_iou_pct"],
            "box_valid_vol_mean": otsu_metrics["box_valid_vol_pct"],
            "excluded_tooth_mean": otsu_metrics["excluded_tooth_pct"],
            "notes": f"Per-sample Otsu on valid normalized intensities (mean t={np.mean(otsu_thresholds):.3f})",
        }
    )

    # 3. Fixed global intensity threshold chosen on TRAIN only
    t_star, train_dice_pct = fit_global_intensity_threshold(train_samples, threshold_grid=threshold_grid)
    global_masks = {s.sample_id: (s.image >= t_star) & (s.valid > 0) for s in val_samples}
    global_metrics, _ = score_segmentation_masks(val_samples, global_masks)
    hu_note = (
        f" (HU-window equiv. {-1000.0 + 4000.0 * t_star:.0f})"
        if architecture == "cbct-wsp82-v1"
        else ""
    )
    rows.append(
        {
            "task": task,
            "baseline": "global_train_threshold",
            "k_budget": len({s.group_id for s in train_samples}),
            "n_runs": 1,
            "threshold_param": t_star,
            "dice_mean": global_metrics["dice_pct"],
            "dice_std": np.nan,
            "bf1_mean": global_metrics["bf1_pct"],
            "edge_recall_mean": global_metrics["edge_recall_pct"],
            "box_iou_mean": global_metrics["box_iou_pct"],
            "box_valid_vol_mean": global_metrics["box_valid_vol_pct"],
            "excluded_tooth_mean": global_metrics["excluded_tooth_pct"],
            "notes": f"t*={t_star:.4f}{hu_note} chosen on train only (train Dice={train_dice_pct:.2f}%)",
        }
    )

    # 4a. Human-written 82-parameter init (frozen K=0)
    human_payload = build_human_written_seg82_payload(architecture, t_star)
    human_masks = predict_cohort(human_payload, val_samples)
    human_metrics, _ = score_segmentation_masks(val_samples, human_masks)
    rows.append(
        {
            "task": task,
            "baseline": "human_82p_frozen",
            "k_budget": 0,
            "n_runs": 1,
            "threshold_param": t_star,
            "dice_mean": human_metrics["dice_pct"],
            "dice_std": np.nan,
            "bf1_mean": human_metrics["bf1_pct"],
            "edge_recall_mean": human_metrics["edge_recall_pct"],
            "box_iou_mean": human_metrics["box_iou_pct"],
            "box_valid_vol_mean": human_metrics["box_valid_vol_pct"],
            "excluded_tooth_mean": human_metrics["excluded_tooth_pct"],
            "notes": f"1 center-tap filter (fb=-{t_star:.4f}, ro=5.0, ob=-1e-4); uses t* from train",
        }
    )

    # Reference: Frozen WSP mean +/- SD across all 6 locked replies
    if design_lock_path is not None and responses_lock_path is not None:
        d_bytes = design_lock_path.read_bytes()
        frozen_models = verify_response_lock(
            strict_json_loads(d_bytes),
            d_bytes,
            strict_json_loads(responses_lock_path.read_bytes()),
            expected_architecture=architecture,
        )
        per_reply = [
            score_segmentation_masks(val_samples, predict_cohort(payload, val_samples))[0]
            for payload in frozen_models.values()
        ]
        rows.append(
            {
                "task": task,
                "baseline": "frozen_wsp_mean6",
                "k_budget": 0,
                "n_runs": len(per_reply),
                "threshold_param": np.nan,
                "dice_mean": float(np.mean([r["dice_pct"] for r in per_reply])),
                "dice_std": float(np.std([r["dice_pct"] for r in per_reply], ddof=1)) if len(per_reply) > 1 else np.nan,
                "bf1_mean": float(np.mean([r["bf1_pct"] for r in per_reply])),
                "edge_recall_mean": float(np.mean([r["edge_recall_pct"] for r in per_reply])),
                "box_iou_mean": float(np.mean([r["box_iou_pct"] for r in per_reply])),
                "box_valid_vol_mean": float(np.mean([r["box_valid_vol_pct"] for r in per_reply])),
                "excluded_tooth_mean": float(np.mean([r["excluded_tooth_pct"] for r in per_reply])),
                "notes": "Mean +/- SD across all 6 locked frozen WSP replies (K=0)",
            }
        )

    # 4b. Optional human-written 82p warm-start and trained position-only reference under same schedule
    if (run_human_warmstart or run_trained_position_only) and budgets and seeds:
        by_group: dict[str, list[SegmentationSample]] = defaultdict(list)
        for s in train_samples:
            by_group[s.group_id].append(s)
        all_groups = sorted(by_group.keys())

        warm_by_k: dict[int, list[dict[str, float]]] = defaultdict(list)
        pos_by_k: dict[int, list[dict[str, float]]] = defaultdict(list)
        for seed in seeds:
            nested = sample_nested_group_budgets(all_groups, budgets, seed=seed)
            for k, selected_groups in nested.items():
                sub_train = [s for g in selected_groups for s in by_group[g]]
                cfg = TrainingConfig(
                    epochs=epochs,
                    lr=lr,
                    seed=seed,
                    device=device,
                    batch_size=batch_size,
                    cosine_lr=cosine_lr,
                )
                if run_human_warmstart:
                    trained_payload, _ = train_segmentation_budget(
                        architecture, sub_train, cfg, init_payload=human_payload
                    )
                    m, _ = score_segmentation_masks(val_samples, predict_cohort(trained_payload, val_samples))
                    warm_by_k[k].append(m)
                if run_trained_position_only:
                    pos_payload, _ = train_segmentation_budget(
                        architecture, sub_train, cfg, position_only=True
                    )
                    m, _ = score_segmentation_masks(val_samples, predict_cohort(pos_payload, val_samples))
                    pos_by_k[k].append(m)

        for label, bucket, note in (
            ("human_82p_warmstart", warm_by_k, "Human 82p prior warm-started with same Adam schedule"),
            ("trained_position_only", pos_by_k, "Trained spatial grid + output bias only"),
        ):
            for k in sorted(bucket.keys()):
                runs = bucket[k]
                rows.append(
                    {
                        "task": task,
                        "baseline": label,
                        "k_budget": int(k),
                        "n_runs": len(runs),
                        "threshold_param": t_star if label == "human_82p_warmstart" else np.nan,
                        "dice_mean": float(np.mean([r["dice_pct"] for r in runs])),
                        "dice_std": float(np.std([r["dice_pct"] for r in runs], ddof=1)) if len(runs) > 1 else np.nan,
                        "bf1_mean": float(np.mean([r["bf1_pct"] for r in runs])),
                        "edge_recall_mean": float(np.mean([r["edge_recall_pct"] for r in runs])),
                        "box_iou_mean": float(np.mean([r["box_iou_pct"] for r in runs])),
                        "box_valid_vol_mean": float(np.mean([r["box_valid_vol_pct"] for r in runs])),
                        "excluded_tooth_mean": float(np.mean([r["excluded_tooth_pct"] for r in runs])),
                        "notes": note,
                    }
                )

    df = pd.DataFrame(rows)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"{task}_val_baselines.csv", index=False)

    meta = {
        "task": task,
        "architecture": architecture,
        "global_threshold_t_star": t_star,
        "global_threshold_train_dice_pct": train_dice_pct,
        "human_payload": human_payload,
    }
    return df, meta


# ---------------------------------------------------------------------------
# Track 1: Classification Baselines
# ---------------------------------------------------------------------------


def extract_global_intensity_features(image: np.ndarray) -> np.ndarray:
    """Extract 3 global intensity summary features: `(mean, frac < 0.1, frac > 0.9)`."""
    arr = np.asarray(image, dtype=np.float64)
    return np.array(
        [
            float(arr.mean()),
            float(np.mean(arr < 0.1)),
            float(np.mean(arr > 0.9)),
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class GlobalIntensityLogReg:
    """Fitted L2-regularized logistic regression on standardized 3D intensity features."""

    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    bias: float

    def predict_logits(self, samples: list[ClassificationSample], *, margin_crop: bool = False) -> np.ndarray:
        feats = np.stack(
            [
                extract_global_intensity_features(
                    cls_lenet2d_margin_ablation(s.image) if margin_crop else s.image
                )
                for s in samples
            ],
            axis=0,
        )
        z = (feats - self.mean[None, :]) / self.std[None, :]
        return z @ self.weights + self.bias


def fit_global_intensity_logreg(
    train_samples: list[ClassificationSample],
    *,
    l2_reg: float = 1.0,
) -> GlobalIntensityLogReg:
    """Fit L2-regularized logistic regression on `(mean, frac<0.1, frac>0.9)` via SciPy L-BFGS-B."""
    require(len(train_samples) >= 2, "Logistic regression requires at least 2 training samples.")
    x_raw = np.stack([extract_global_intensity_features(s.image) for s in train_samples], axis=0)
    y = np.array([float(s.label) for s in train_samples], dtype=np.float64)

    mean = x_raw.mean(axis=0)
    std = np.maximum(x_raw.std(axis=0), 1e-6)
    x = (x_raw - mean[None, :]) / std[None, :]

    n_feat = x.shape[1]

    def _loss_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
        w = params[:n_feat]
        b = params[n_feat]
        logits = x @ w + b
        # Numerically stable softplus BCE: sum(log(1 + exp(logits)) - y * logits)
        loss_vec = np.logaddexp(0.0, logits) - y * logits
        loss = float(np.sum(loss_vec) + 0.5 * l2_reg * np.dot(w, w))
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        diff = probs - y
        grad_w = x.T @ diff + l2_reg * w
        grad_b = float(np.sum(diff))
        return loss, np.concatenate([grad_w, [grad_b]])

    res = minimize(_loss_and_grad, np.zeros(n_feat + 1, dtype=np.float64), method="L-BFGS-B", jac=True)
    return GlobalIntensityLogReg(
        mean=mean,
        std=std,
        weights=res.x[:n_feat].copy(),
        bias=float(res.x[n_feat]),
    )


def denpar_crown_edge_rule_score(image: np.ndarray) -> float:
    """Zero-parameter hand-coded rule from `benchmarks/denpar/cls_prompt.md` for `denpar_cls`.

    Applies the prompt's 5x5 vertical bright-above -> dark-below crown-edge filter (top 2 rows +0.1,
    middle row 0, bottom 2 rows -0.1) over the valid unpadded interior, then takes the mean signed
    squared response `mean(ReLU(r)^2 - ReLU(-r)^2) = mean(r * |r|)`.

    Nonlinear odd rectification avoids the telescoping-sum degeneracy of a linear mean of vertical
    differences (which reduces to top-border minus bottom-border intensity and vanishes identically
    under 10% border zeroing). Positive score predicts Upper (Maxillary) arch (`label = 1`).
    """
    img = np.asarray(image, dtype=np.float64)
    require(img.ndim == 2 and img.shape[0] >= 5 and img.shape[1] >= 5, "Image must be 2D and at least 5x5.")
    kernel = np.array(
        [
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.1, -0.1, -0.1, -0.1, -0.1],
            [-0.1, -0.1, -0.1, -0.1, -0.1],
        ],
        dtype=np.float64,
    )
    windows = np.lib.stride_tricks.sliding_window_view(img, (5, 5))
    resp = np.einsum("hwxy,xy->hw", windows, kernel)
    signed_energy = resp * np.abs(resp)
    return float(signed_energy.mean())


def evaluate_denpar_crown_edge_baseline(
    val_samples: list[ClassificationSample],
    *,
    border_fraction: float = 0.10,
    sign_flip_threshold: float = 0.35,
) -> dict[str, Any]:
    """Evaluate the zero-parameter vertical bright->dark crown-edge rule on `denpar_cls`."""
    y_true = np.array([s.label for s in val_samples], dtype=np.int64)
    scores = np.array([denpar_crown_edge_rule_score(s.image) for s in val_samples], dtype=np.float64)
    margin_scores = np.array(
        [denpar_crown_edge_rule_score(cls_lenet2d_margin_ablation(s.image, border_fraction)) for s in val_samples],
        dtype=np.float64,
    )
    roc_auc = compute_roc_auc(y_true, scores)
    pr_auc = compute_pr_auc(y_true, scores)
    margin_auc = compute_roc_auc(y_true, margin_scores)
    preds = (scores >= 0.0).astype(np.int64)
    acc = float(np.mean(preds == y_true))
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "accuracy": acc,
        "margin_ablated_roc_auc": float(margin_auc),
        "margin_auc_drop": float(roc_auc - margin_auc),
        "sign_flip_flagged": bool(roc_auc < sign_flip_threshold),
    }


def evaluate_classification_baselines(
    *,
    task: str,
    train_samples: list[ClassificationSample],
    val_samples: list[ClassificationSample],
    budgets: list[int],
    seeds: list[int],
    design_lock_path: Path | None = None,
    responses_lock_path: Path | None = None,
    l2_reg: float = 1.0,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Evaluate constant, global-intensity logistic regression, DenPAR crown-edge rule, and frozen WSP."""
    rows: list[dict[str, Any]] = []
    y_val = np.array([s.label for s in val_samples], dtype=np.int64)
    prevalence = float(y_val.mean())

    # 1. Constant-score baseline (exact AUC = 0.500, PR-AUC = prevalence)
    const_auc = compute_roc_auc(y_val, np.zeros(len(val_samples), dtype=np.float64))
    rows.append(
        {
            "task": task,
            "baseline": "constant_score",
            "k_budget": 0,
            "n_runs": 1,
            "roc_auc_mean": float(const_auc),
            "roc_auc_std": np.nan,
            "pr_auc_mean": prevalence,
            "margin_roc_auc_mean": float(const_auc),
            "margin_roc_auc_std": np.nan,
            "notes": f"Constant score 0.0 (val prevalence={prevalence:.3f})",
        }
    )

    # 2. DenPAR zero-parameter hand-coded vertical crown-edge rule
    if task == "denpar_cls":
        edge_res = evaluate_denpar_crown_edge_baseline(val_samples)
        rows.append(
            {
                "task": task,
                "baseline": "handcoded_vertical_crown_edge",
                "k_budget": 0,
                "n_runs": 1,
                "roc_auc_mean": edge_res["roc_auc"],
                "roc_auc_std": np.nan,
                "pr_auc_mean": edge_res["pr_auc"],
                "margin_roc_auc_mean": edge_res["margin_ablated_roc_auc"],
                "margin_roc_auc_std": np.nan,
                "notes": f"Zero-param 5x5 bright->dark edge rule (acc={edge_res['accuracy']:.3f})",
            }
        )

    # 3. Reference: Frozen WSP mean +/- SD across all 6 locked replies
    if design_lock_path is not None and responses_lock_path is not None:
        d_bytes = design_lock_path.read_bytes()
        frozen_models = verify_response_lock(
            strict_json_loads(d_bytes),
            d_bytes,
            strict_json_loads(responses_lock_path.read_bytes()),
            expected_architecture="cls-lenet2d-v1",
        )
        per_reply = [evaluate_classifier(p, val_samples) for p in frozen_models.values()]
        rows.append(
            {
                "task": task,
                "baseline": "frozen_wsp_mean6",
                "k_budget": 0,
                "n_runs": len(per_reply),
                "roc_auc_mean": float(np.mean([r["roc_auc"] for r in per_reply])),
                "roc_auc_std": float(np.std([r["roc_auc"] for r in per_reply], ddof=1)) if len(per_reply) > 1 else np.nan,
                "pr_auc_mean": float(np.mean([r["pr_auc"] for r in per_reply])),
                "margin_roc_auc_mean": float(np.mean([r["margin_ablated_roc_auc"] for r in per_reply])),
                "margin_roc_auc_std": (
                    float(np.std([r["margin_ablated_roc_auc"] for r in per_reply], ddof=1))
                    if len(per_reply) > 1
                    else np.nan
                ),
                "notes": "Mean +/- SD across all 6 locked frozen WSP replies (K=0)",
            }
        )

    # 4. Global intensity logistic regression fit on exact same stratified K-budget subsets
    by_group: dict[str, list[ClassificationSample]] = defaultdict(list)
    for s in train_samples:
        by_group[s.group_id].append(s)

    by_k: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for seed in seeds:
        nested = sample_stratified_cls_budgets(train_samples, budgets, seed=seed)
        for k, selected_groups in nested.items():
            sub_train = [s for g in selected_groups for s in by_group[g]]
            clf = fit_global_intensity_logreg(sub_train, l2_reg=l2_reg)
            full_logits = clf.predict_logits(val_samples, margin_crop=False)
            margin_logits = clf.predict_logits(val_samples, margin_crop=True)
            by_k[k].append(
                (
                    compute_roc_auc(y_val, full_logits),
                    compute_pr_auc(y_val, full_logits),
                    compute_roc_auc(y_val, margin_logits),
                )
            )

    for k in sorted(by_k.keys()):
        arr = np.array(by_k[k], dtype=np.float64)
        rows.append(
            {
                "task": task,
                "baseline": "global_intensity_logreg",
                "k_budget": int(k),
                "n_runs": len(arr),
                "roc_auc_mean": float(arr[:, 0].mean()),
                "roc_auc_std": float(arr[:, 0].std(ddof=1)) if len(arr) > 1 else np.nan,
                "pr_auc_mean": float(arr[:, 1].mean()),
                "margin_roc_auc_mean": float(arr[:, 2].mean()),
                "margin_roc_auc_std": float(arr[:, 2].std(ddof=1)) if len(arr) > 1 else np.nan,
                "notes": "L2 logreg on (mean, frac<0.1, frac>0.9) fit on K-budget train subset",
            }
        )

    df = pd.DataFrame(rows)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"{task}_val_baselines.csv", index=False)
    return df
