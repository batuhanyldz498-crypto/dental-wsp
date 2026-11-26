"""Publication-grade deterministic X-ray and CBCT slice overlay comparison figures."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from wsp_core.architectures.seg_wsp82_2d import seg_wsp82_2d_logits, seg_wsp82_2d_without_grid
from wsp_core.architectures.seg_wsp82_3d import seg_wsp82_3d_logits, seg_wsp82_3d_without_grid
from wsp_core.evaluation.segmentation import SegmentationSample, contour_and_band


def select_deterministic_examples(
    samples: list[SegmentationSample],
    n: int = 2,
    salt: str = "dental-wsp-viz-v1",
) -> list[SegmentationSample]:
    """Deterministically pick `n` validation samples via salted SHA-256 (zero cherry-picking)."""

    def _key(sample: SegmentationSample) -> bytes:
        return hashlib.sha256(f"{salt}\0{sample.sample_id}".encode("utf-8")).digest()

    return sorted(samples, key=_key)[:n]


def _predict_single(payload: dict[str, Any], sample: SegmentationSample, threshold: float = 0.0) -> np.ndarray:
    arch = payload["architecture"]
    if arch in ("pano-wsp82-v1", "periapical-wsp82-v1"):
        logits = seg_wsp82_2d_logits(sample.image, payload)
    elif arch == "cbct-wsp82-v1":
        logits = seg_wsp82_3d_logits(sample.image, payload)
    else:
        raise ValueError(f"Unsupported architecture for visualization: {arch}")
    return (logits >= threshold) & (sample.valid > 0)


def _without_grid(payload: dict[str, Any]) -> dict[str, Any]:
    if payload["architecture"] == "cbct-wsp82-v1":
        return seg_wsp82_3d_without_grid(payload)
    return seg_wsp82_2d_without_grid(payload)


def _dice_and_recall(truth: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    t = (truth > 0) & (valid > 0)
    p = (pred > 0) & (valid > 0)
    inter = int(np.count_nonzero(t & p))
    denom = int(np.count_nonzero(t)) + int(np.count_nonzero(p))
    dice = 100.0 * (2.0 * inter / denom) if denom > 0 else 0.0
    ref_edge, _ = contour_and_band(t, valid > 0)
    _, pred_band = contour_and_band(p, valid > 0)
    recall = 100.0 * float(pred_band[ref_edge].mean()) if ref_edge.size else 0.0
    return dice, recall


def _draw_contour(ax: Any, mask_2d: np.ndarray, color: str, linestyle: str, lw: float = 1.0) -> None:
    if np.any(mask_2d):
        h, w = mask_2d.shape
        ax.contour(
            np.arange(-1, w + 1),
            np.arange(-1, h + 1),
            np.pad(mask_2d.astype(np.uint8), 1),
            levels=[0.5],
            colors=[color],
            linestyles=[linestyle],
            linewidths=[lw],
        )


def _to_2d_view(
    sample: SegmentationSample,
    pred: np.ndarray,
    slice_axis: str = "axial",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 2D (image, truth, pred) arrays; for RAS+ 3D CBCT (Z/I->S, Y/P->A, X/L->R) displays upright."""
    if sample.image.ndim == 2:
        return sample.image, sample.target > 0, pred > 0
    coords = np.nonzero(sample.target > 0)
    cz = int(np.round(coords[0].mean())) if coords[0].size else sample.image.shape[0] // 2
    cy = int(np.round(coords[1].mean())) if coords[1].size else sample.image.shape[1] // 2
    if slice_axis == "coronal":
        # Flip Z vertically for display only so superior maxilla sits at top and inferior mandible at bottom
        return sample.image[::-1, cy, :], sample.target[::-1, cy, :] > 0, pred[::-1, cy, :] > 0
    # Flip Y vertically for display only so anterior incisors sit at top and posterior spine at bottom
    return sample.image[cz, ::-1, :], sample.target[cz, ::-1, :] > 0, pred[cz, ::-1, :] > 0


def plot_frozen_replies_grid(
    sample: SegmentationSample,
    models: dict[str, dict[str, Any]],
    *,
    title: str = "Frozen WSP Replies · Intact (A) vs. Image-Only (B)",
    attribution: str = "",
    slice_axis: str = "axial",
    out_path: str | Path | None = None,
) -> Any:
    """Plot all frozen replies side-by-side (Intact A vs. Image-Only B) on one validation radiograph/slice."""
    slots = list(models.keys())
    n_rows = len(slots)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11.5, 2.2 * n_rows + 1.0), squeeze=False)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.92, bottom=0.04, wspace=0.04, hspace=0.18)

    gold, cyan = "#ffd166", "#38e8f5"
    handles = [
        Line2D([0], [0], color="#c99700", lw=2.0, label="Ground Truth (solid gold)"),
        Line2D([0], [0], color="#0891b2", lw=2.0, ls="--", label="Prediction (dashed cyan)"),
    ]
    fig.suptitle(title, y=0.985, fontsize=13, fontweight="bold")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.958), ncol=2, frameon=False)

    for row_idx, slot in enumerate(slots):
        payload_a = models[slot]
        payload_b = _without_grid(payload_a)
        for col_idx, (arm_label, payload) in enumerate((("A · Intact", payload_a), ("B · Image-Only", payload_b))):
            ax = axes[row_idx, col_idx]
            pred = _predict_single(payload, sample)
            dice, recall = _dice_and_recall(sample.target, pred, sample.valid)
            img2d, truth2d, pred2d = _to_2d_view(sample, pred, slice_axis=slice_axis)
            ax.imshow(img2d, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            _draw_contour(ax, truth2d, gold, "solid", lw=0.95)
            _draw_contour(ax, pred2d, cyan, "dashed", lw=0.95)
            ax.set_title(f"{arm_label}  |  Dice {dice:.1f}% · Edge Rec {recall:.1f}%", fontsize=9.5, pad=3)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(slot, fontsize=10, fontweight="bold")

    if attribution:
        fig.text(0.5, 0.008, attribution, ha="center", va="bottom", fontsize=8.5, color="#555555")
    if out_path is not None:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=180, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_four_arm_comparison(
    sample: SegmentationSample,
    arm_payloads: dict[str, dict[str, Any]],
    *,
    title: str = "4-Arm Comparison on Same Validation Radiograph",
    attribution: str = "",
    slice_axis: str = "axial",
    out_path: str | Path | None = None,
) -> Any:
    """Plot a 2x2 side-by-side X-ray comparison across Frozen WSP, Position-Only, Random-Init, and Warm-Start."""
    items = list(arm_payloads.items())
    n_cols = 2
    n_rows = (len(items) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12.2, 3.4 * n_rows + 0.9), squeeze=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.06, wspace=0.04, hspace=0.18)

    gold, cyan = "#ffd166", "#38e8f5"
    handles = [
        Line2D([0], [0], color="#c99700", lw=2.2, label="Ground Truth (solid gold)"),
        Line2D([0], [0], color="#0891b2", lw=2.2, ls="--", label="Prediction (dashed cyan)"),
    ]
    fig.suptitle(title, y=0.97, fontsize=13, fontweight="bold")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=2, frameon=False)

    for idx, (label, payload) in enumerate(items):
        ax = axes[idx // n_cols, idx % n_cols]
        pred = _predict_single(payload, sample)
        dice, recall = _dice_and_recall(sample.target, pred, sample.valid)
        img2d, truth2d, pred2d = _to_2d_view(sample, pred, slice_axis=slice_axis)
        ax.imshow(img2d, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        _draw_contour(ax, truth2d, gold, "solid", lw=1.05)
        _draw_contour(ax, pred2d, cyan, "dashed", lw=1.05)
        ax.set_title(f"{label}  |  Dice {dice:.1f}% · Edge Rec {recall:.1f}%", fontsize=10, fontweight="bold", pad=4)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(len(items), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    if attribution:
        fig.text(0.5, 0.01, attribution, ha="center", va="bottom", fontsize=8.5, color="#555555")
    if out_path is not None:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=180, bbox_inches="tight")
        plt.close(fig)
    return fig
