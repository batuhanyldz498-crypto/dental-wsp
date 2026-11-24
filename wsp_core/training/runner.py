"""Three-arm benchmark runner (Frozen WSP, Random-Init Adam, WSP Warm-Start Adam) & learning curves."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wsp_core.evaluation.classification import ClassificationSample, evaluate_classifier
from wsp_core.evaluation.segmentation import (
    SegmentationSample,
    bounding_slices,
    contour_and_band,
    group_segmentation_scores,
)
from wsp_core.guards import require
from wsp_core.schemas import strict_json_loads, verify_response_lock
from wsp_core.training.trainer import (
    TrainingConfig,
    sample_nested_group_budgets,
    sample_stratified_cls_budgets,
    train_classification_budget,
    train_segmentation_budget,
)


def run_classification_three_arm_validation(
    *,
    task: str,
    train_samples: list[ClassificationSample],
    val_samples: list[ClassificationSample],
    design_lock_path: Path,
    responses_lock_path: Path,
    budgets: list[int],
    seeds: list[int],
    epochs: int = 200,
    lr: float = 5e-3,
    device: str = "cpu",
    out_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Arm 1 (Frozen WSP), Arm 2 (Random-Init Adam), and Arm 3 (WSP Warm-Start Adam) on validation."""
    design_bytes = design_lock_path.read_bytes()
    design = strict_json_loads(design_bytes)
    resp = strict_json_loads(responses_lock_path.read_bytes())
    frozen_models = verify_response_lock(design, design_bytes, resp, expected_architecture="cls-lenet2d-v1")
    require(len(frozen_models) > 0, f"No valid frozen replies found in {responses_lock_path}")

    rows: list[dict[str, Any]] = []

    # --- Arm 1: Frozen WSP (K = 0, zero optimizer steps) ---
    for slot, payload in frozen_models.items():
        metrics = evaluate_classifier(payload, val_samples)
        rows.append(
            {
                "task": task,
                "arm": "frozen_wsp",
                "k_budget": 0,
                "seed": -1,
                "init_slot": slot,
                **metrics,
            }
        )

    # Pre-index training samples by group_id for fast budget subsetting
    by_group: dict[str, list[ClassificationSample]] = {}
    for s in train_samples:
        by_group.setdefault(s.group_id, []).append(s)

    for seed in seeds:
        nested = sample_stratified_cls_budgets(train_samples, budgets, seed=seed)
        for k, selected_groups in nested.items():
            sub_train = [s for g in selected_groups for s in by_group[g]]
            cfg = TrainingConfig(epochs=epochs, lr=lr, seed=seed, device=device)

            # --- Arm 2: Random-Init + Adam ---
            rand_payload, _ = train_classification_budget(sub_train, cfg, init_payload=None)
            rand_metrics = evaluate_classifier(rand_payload, val_samples)
            rows.append(
                {
                    "task": task,
                    "arm": "random_init_adam",
                    "k_budget": k,
                    "seed": seed,
                    "init_slot": "random",
                    **rand_metrics,
                }
            )

            # --- Arm 3: WSP Warm-Start + Adam (every valid frozen reply, no cherry-picking) ---
            for slot, frozen_payload in frozen_models.items():
                warm_payload, _ = train_classification_budget(sub_train, cfg, init_payload=frozen_payload)
                warm_metrics = evaluate_classifier(warm_payload, val_samples)
                rows.append(
                    {
                        "task": task,
                        "arm": "wsp_warmstart_adam",
                        "k_budget": k,
                        "seed": seed,
                        "init_slot": slot,
                        **warm_metrics,
                    }
                )

    runs_df = pd.DataFrame(rows)
    summary_df = (
        runs_df.groupby(["arm", "k_budget"], as_index=False)
        .agg(
            n_runs=("roc_auc", "count"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            margin_roc_auc_mean=("margin_ablated_roc_auc", "mean"),
            margin_roc_auc_std=("margin_ablated_roc_auc", "std"),
            sign_flip_count=("sign_flip_flagged", "sum"),
        )
        .sort_values(["arm", "k_budget"])
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        runs_df.to_csv(out_dir / f"{task}_val_runs.csv", index=False)
        summary_df.to_csv(out_dir / f"{task}_val_summary.csv", index=False)
        plot_classification_learning_curve(
            summary_df,
            task=task,
            out_path=out_dir / f"{task}_learning_curve.png",
        )

    return runs_df, summary_df


def plot_classification_learning_curve(
    summary_df: pd.DataFrame,
    *,
    task: str,
    out_path: Path | None = None,
) -> Any:
    """Plot 2-panel learning curves: Full-Frame ROC-AUC and Margin-Crop ROC-AUC vs. K."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True, layout="constrained")
    panels = [
        ("roc_auc_mean", "roc_auc_std", "Full-Frame Validation ROC-AUC"),
        ("margin_roc_auc_mean", "margin_roc_auc_std", "Margin-Crop (10% Border Zeroed) ROC-AUC"),
    ]
    styles = {
        "random_init_adam": ("o-", "#263d57", "Random-Init + Adam"),
        "wsp_warmstart_adam": ("s-", "#147b72", "WSP Warm-Start + Adam"),
    }

    frozen = summary_df[summary_df["arm"] == "frozen_wsp"]
    for ax, (mean_col, std_col, title) in zip(axes, panels):
        if not frozen.empty:
            val = float(frozen[mean_col].iloc[0])
            ax.axhline(val, color="#a44624", linestyle="--", linewidth=1.8, label=f"Frozen WSP K=0 ({val:.3f})")
        for arm, (fmt, color, label) in styles.items():
            sub = summary_df[summary_df["arm"] == arm].sort_values("k_budget")
            if sub.empty:
                continue
            ks = sub["k_budget"].to_numpy()
            means = sub[mean_col].to_numpy()
            stds = np.nan_to_num(sub[std_col].to_numpy(), nan=0.0)
            ax.plot(ks, means, fmt, color=color, linewidth=2.0, markersize=6, label=label)
            ax.fill_between(ks, means - stds, means + stds, color=color, alpha=0.14)
        ax.axhline(0.5, color="#999999", linestyle=":", linewidth=1.0)
        ax.set_xscale("log", base=2)
        ax.set_ylim(0.30, 1.02)
        ax.set_xlabel("Labeled training groups (K)")
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Validation ROC-AUC")
    axes[1].legend(loc="lower right", frameon=True)
    fig.suptitle(f"{task} · Frozen WSP vs. Random-Init Adam vs. WSP Warm-Start Adam (Validation Only)", fontsize=12)

    if out_path is not None:
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    return fig


def _precompute_seg_val_cache(val_samples: list[SegmentationSample]) -> list[dict[str, Any]]:
    """Precompute reference boxes, contours, and bands once across all validation samples."""
    cache = []
    for s in val_samples:
        valid = s.valid > 0
        truth = (s.target > 0) & valid
        ref_box = bounding_slices(truth)
        ref_edge, ref_band = contour_and_band(truth, valid)
        ref_area = int(np.count_nonzero(truth))
        ref_box_vol = int(np.prod([sl.stop - sl.start for sl in ref_box])) if ref_box is not None else 0
        cache.append(
            {
                "sample": s,
                "valid": valid,
                "truth": truth,
                "ref_box": ref_box,
                "ref_edge": ref_edge,
                "ref_band": ref_band,
                "ref_area": ref_area,
                "ref_box_vol": ref_box_vol,
            }
        )
    return cache


def _predict_masks_torch(
    architecture: str,
    payload: dict[str, Any],
    val_x: Any,
    val_samples: list[SegmentationSample],
    device: str,
) -> dict[str, np.ndarray]:
    """Fast GPU/CPU batched PyTorch inference returning valid boolean masks."""
    import torch
    from wsp_core.architectures.seg_wsp82_2d import build_torch_seg_wsp82_2d
    from wsp_core.architectures.seg_wsp82_3d import build_torch_seg_wsp82_3d

    if architecture in ("pano-wsp82-v1", "periapical-wsp82-v1"):
        model = build_torch_seg_wsp82_2d(architecture=architecture)
    else:
        model = build_torch_seg_wsp82_3d()
    model.load_from_wsp_dict(payload)
    model = model.to(device).eval()
    masks: dict[str, np.ndarray] = {}
    batch_size = 16 if architecture != "cbct-wsp82-v1" else 4
    with torch.no_grad():
        for start in range(0, len(val_samples), batch_size):
            xb = val_x[start : start + batch_size].to(device)
            preds = (model(xb).squeeze(1) >= 0.0).cpu().numpy()
            for offset, s in enumerate(val_samples[start : start + batch_size]):
                masks[s.sample_id] = preds[offset] & (s.valid > 0)
    return masks


def _score_seg_payload_fast(
    architecture: str,
    payload: dict[str, Any],
    val_x: Any,
    val_samples: list[SegmentationSample],
    val_cache: list[dict[str, Any]],
    device: str,
) -> dict[str, float]:
    """Score Arm-A region Dice/IoU, boundary F1, edge recall, box IoU, and excluded tooth rate cleanly."""
    masks = _predict_masks_torch(architecture, payload, val_x, val_samples, device=device)
    reg, _ = group_segmentation_scores(val_samples, masks)

    by_group: dict[str, list[tuple[float, float, float, float]]] = {}
    for item in val_cache:
        s = item["sample"]
        valid = item["valid"]
        truth = item["truth"]
        ref_box = item["ref_box"]
        ref_edge = item["ref_edge"]
        ref_band = item["ref_band"]
        ref_area = item["ref_area"]
        ref_box_vol = item["ref_box_vol"]

        pred = masks[s.sample_id]
        pred_edge, pred_band = contour_and_band(pred, valid)
        prec = float(np.count_nonzero(ref_band[pred_edge]) / pred_edge.size) if pred_edge.size else 0.0
        recall = float(pred_band[ref_edge].mean()) if ref_edge.size else 0.0
        bf1 = 2.0 * prec * recall / (prec + recall) if (prec + recall) > 0.0 else 0.0

        pred_box = bounding_slices(pred)
        proposal = np.zeros_like(valid, dtype=bool)
        if pred_box is not None:
            proposal[pred_box] = True
        prop_vol = int(np.count_nonzero(proposal))
        overlap = int(np.count_nonzero(proposal[ref_box])) if ref_box is not None else 0
        denom = prop_vol + ref_box_vol - overlap
        box_iou = overlap / denom if denom > 0 else 0.0
        excl_tooth = 1.0 - (int(np.count_nonzero(truth & proposal)) / ref_area) if ref_area > 0 else 0.0
        by_group.setdefault(s.group_id, []).append((100.0 * bf1, 100.0 * recall, 100.0 * box_iou, 100.0 * excl_tooth))

    group_means = np.array([np.mean(v, axis=0) for v in by_group.values()], dtype=np.float64)
    return {
        "dice_pct": 100.0 * reg["dice"],
        "iou_pct": 100.0 * reg["iou"],
        "precision_pct": 100.0 * reg["precision"],
        "recall_pct": 100.0 * reg["recall"],
        "bf1_a_pct": float(group_means[:, 0].mean()),
        "edge_recall_a_pct": float(group_means[:, 1].mean()),
        "box_iou_a_pct": float(group_means[:, 2].mean()),
        "excluded_tooth_a_pct": float(group_means[:, 3].mean()),
    }


def run_segmentation_three_arm_validation(
    *,
    task: str,
    architecture: str,
    train_samples: list[SegmentationSample],
    val_samples: list[SegmentationSample],
    design_lock_path: Path,
    responses_lock_path: Path,
    budgets: list[int],
    seeds: list[int],
    epochs: int = 200,
    lr: float = 1e-2,
    device: str = "cpu",
    out_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Frozen WSP (K=0), Random-Init Adam, WSP Warm-Start Adam, and Trained Position-Only on validation."""
    import torch

    design_bytes = design_lock_path.read_bytes()
    design = strict_json_loads(design_bytes)
    resp = strict_json_loads(responses_lock_path.read_bytes())
    frozen_models = verify_response_lock(design, design_bytes, resp, expected_architecture=architecture)
    require(len(frozen_models) > 0, f"No valid frozen replies found in {responses_lock_path}")

    val_cache = _precompute_seg_val_cache(val_samples)
    val_x = torch.stack([torch.from_numpy(s.image.astype(np.float32)).unsqueeze(0) for s in val_samples])
    rows: list[dict[str, Any]] = []

    # Arm 1: Frozen WSP (K = 0)
    for slot, payload in frozen_models.items():
        metrics = _score_seg_payload_fast(architecture, payload, val_x, val_samples, val_cache, device)
        rows.append({"task": task, "arm": "frozen_wsp", "k_budget": 0, "seed": -1, "init_slot": slot, **metrics})
    print(f"[{task}] Scored Arm 1 Frozen WSP ({len(frozen_models)} replies)")

    by_group: dict[str, list[SegmentationSample]] = {}
    for s in train_samples:
        by_group.setdefault(s.group_id, []).append(s)
    all_groups = sorted(by_group.keys())

    for seed in seeds:
        nested = sample_nested_group_budgets(all_groups, budgets, seed=seed)
        for k, selected_groups in nested.items():
            sub_train = [s for g in selected_groups for s in by_group[g]]
            cfg = TrainingConfig(epochs=epochs, lr=lr, seed=seed, device=device)

            # Control Arm: Trained Position-Only (spatial grid + output bias only)
            pos_payload, _ = train_segmentation_budget(architecture, sub_train, cfg, position_only=True)
            rows.append(
                {
                    "task": task,
                    "arm": "trained_position_only",
                    "k_budget": k,
                    "seed": seed,
                    "init_slot": "position_only",
                    **_score_seg_payload_fast(architecture, pos_payload, val_x, val_samples, val_cache, device),
                }
            )

            # Arm 2: Random-Init + Adam
            rand_payload, _ = train_segmentation_budget(architecture, sub_train, cfg, init_payload=None)
            rows.append(
                {
                    "task": task,
                    "arm": "random_init_adam",
                    "k_budget": k,
                    "seed": seed,
                    "init_slot": "random",
                    **_score_seg_payload_fast(architecture, rand_payload, val_x, val_samples, val_cache, device),
                }
            )

            # Arm 3: WSP Warm-Start + Adam (every valid frozen reply, no cherry-picking)
            for slot, frozen_payload in frozen_models.items():
                warm_payload, _ = train_segmentation_budget(architecture, sub_train, cfg, init_payload=frozen_payload)
                rows.append(
                    {
                        "task": task,
                        "arm": "wsp_warmstart_adam",
                        "k_budget": k,
                        "seed": seed,
                        "init_slot": slot,
                        **_score_seg_payload_fast(architecture, warm_payload, val_x, val_samples, val_cache, device),
                    }
                )
            print(f"[{task}] Completed seed={seed}, K={k}")

    runs_df = pd.DataFrame(rows)
    summary_df = (
        runs_df.groupby(["arm", "k_budget"], as_index=False)
        .agg(
            n_runs=("dice_pct", "count"),
            dice_mean=("dice_pct", "mean"),
            dice_std=("dice_pct", "std"),
            bf1_mean=("bf1_a_pct", "mean"),
            edge_recall_mean=("edge_recall_a_pct", "mean"),
            edge_recall_std=("edge_recall_a_pct", "std"),
            excluded_tooth_mean=("excluded_tooth_a_pct", "mean"),
        )
        .sort_values(["arm", "k_budget"])
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        runs_df.to_csv(out_dir / f"{task}_val_runs.csv", index=False)
        summary_df.to_csv(out_dir / f"{task}_val_summary.csv", index=False)
        plot_segmentation_learning_curve(summary_df, task=task, out_path=out_dir / f"{task}_learning_curve.png")

    return runs_df, summary_df


def plot_segmentation_learning_curve(
    summary_df: pd.DataFrame,
    *,
    task: str,
    out_path: Path | None = None,
) -> Any:
    """Plot 2-panel segmentation learning curves: Region Dice (%) and Reference-Edge Recall (%) vs. K."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")
    panels = [
        ("dice_mean", "dice_std", "Validation Region Dice (%)", "Dice (%)"),
        ("edge_recall_mean", "edge_recall_std", "Validation Reference-Edge Recall (%)", "Edge Recall (%)"),
    ]
    styles = {
        "trained_position_only": ("^--", "#7c6f64", "Trained Position-Only"),
        "random_init_adam": ("o-", "#263d57", "Random-Init + Adam"),
        "wsp_warmstart_adam": ("s-", "#147b72", "WSP Warm-Start + Adam"),
    }

    frozen = summary_df[summary_df["arm"] == "frozen_wsp"]
    for ax, (mean_col, std_col, title, ylabel) in zip(axes, panels):
        if not frozen.empty:
            val = float(frozen[mean_col].iloc[0])
            ax.axhline(val, color="#a44624", linestyle="--", linewidth=1.8, label=f"Frozen WSP K=0 ({val:.1f}%)")
        for arm, (fmt, color, label) in styles.items():
            sub = summary_df[summary_df["arm"] == arm].sort_values("k_budget")
            if sub.empty:
                continue
            ks = sub["k_budget"].to_numpy()
            means = sub[mean_col].to_numpy()
            stds = np.nan_to_num(sub[std_col].to_numpy(), nan=0.0)
            ax.plot(ks, means, fmt, color=color, linewidth=2.0, markersize=6, label=label)
            ax.fill_between(ks, means - stds, means + stds, color=color, alpha=0.14)
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Labeled training groups (K)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)

    axes[1].legend(loc="lower right", frameon=True)
    fig.suptitle(f"{task} · 82-Param Frozen WSP vs. Position-Only vs. Random-Init vs. WSP Warm-Start (Validation)", fontsize=11.5)

    if out_path is not None:
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    return fig
