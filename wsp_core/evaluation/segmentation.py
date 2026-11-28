"""Unified 2D/3D segmentation evaluation suite for `dental-wsp`.

Includes:
- Group-weighted region Dice, IoU, Precision, Recall
- Pathway ablations (A intact, B image-only, C position-only) with invariance check
- Paired cross-group swaps on identical valid-pixel / valid-voxel patterns
- Inside/outside reference-ROI false positives per 1,000 total valid pixels/voxels
  and outside-ROI share of net grid-removed FPs (A vs B)
- Full-canvas/volume boundary/surface F1 and reference-edge recall (2-px / 2-voxel tolerance)
- In-box predicted-edge precision and paired reference-edge fate (preserved, gained, lost, missed)
- Coarse enclosing-box localization: IoU, reference-box coverage, predicted box valid-area/volume
  retention fraction, and excluded annotated tooth-pixel/voxel rate
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from wsp_core.architectures.seg_wsp82_2d import (
    seg_wsp82_2d_logits,
    seg_wsp82_2d_position_logits,
    seg_wsp82_2d_without_grid,
)
from wsp_core.architectures.seg_wsp82_3d import (
    seg_wsp82_3d_logits,
    seg_wsp82_3d_position_logits,
    seg_wsp82_3d_without_grid,
)
from wsp_core.guards import require

# Clear column headers for newly generated CSV outputs (legacy sealed CSVs retain historical names).
OUTSIDE_FP_SHARE_COLUMN = "Outside-ROI share of grid-removed FPs (A vs B) (%)"
BOX_VALID_FRAC_COLUMN = "Predicted box valid fraction A (%)"


@dataclass(frozen=True)
class SegmentationSample:
    """Single 2D or 3D segmentation item on its standardized canvas."""

    sample_id: str
    group_id: str
    image: np.ndarray  # (H, W) or (D, H, W), float32 in [0, 1]
    target: np.ndarray  # bool mask matching image.shape
    valid: np.ndarray  # bool valid-region mask matching image.shape


def _expand_cardinal(mask: np.ndarray) -> np.ndarray:
    """One inclusive cardinal-neighbor dilation (4-neighbor in 2D, 6-neighbor in 3D)."""
    out = mask.copy()
    for axis in range(mask.ndim):
        s_dst_fwd = [slice(None)] * mask.ndim
        s_src_fwd = [slice(None)] * mask.ndim
        s_dst_fwd[axis] = slice(1, None)
        s_src_fwd[axis] = slice(None, -1)
        out[tuple(s_dst_fwd)] |= mask[tuple(s_src_fwd)]

        s_dst_bwd = [slice(None)] * mask.ndim
        s_src_bwd = [slice(None)] * mask.ndim
        s_dst_bwd[axis] = slice(None, -1)
        s_src_bwd[axis] = slice(1, None)
        out[tuple(s_dst_bwd)] |= mask[tuple(s_src_bwd)]
    return out


def contour_and_band(mask: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inner foreground contour and inclusive 2-step cardinal tolerance band."""
    fg = mask & valid
    bg = valid & ~mask
    edge = fg & _expand_cardinal(bg)
    band = valid & _expand_cardinal(_expand_cardinal(edge))
    return np.flatnonzero(edge), band.ravel()


def bounding_slices(mask: np.ndarray) -> tuple[slice, ...] | None:
    """Tight axis-aligned enclosing box as exclusive-stop slices, or None if empty."""
    coords = np.nonzero(mask)
    if len(coords) == 0 or coords[0].size == 0:
        return None
    return tuple(slice(int(c.min()), int(c.max()) + 1) for c in coords)


def _forward_bundle(architecture: str) -> tuple[Callable, Callable, Callable]:
    if architecture in ("pano-wsp82-v1", "periapical-wsp82-v1"):
        return seg_wsp82_2d_logits, seg_wsp82_2d_position_logits, seg_wsp82_2d_without_grid
    if architecture == "cbct-wsp82-v1":
        return seg_wsp82_3d_logits, seg_wsp82_3d_position_logits, seg_wsp82_3d_without_grid
    raise ValueError(f"Unsupported segmentation architecture: {architecture}")


def group_segmentation_scores(
    samples: list[SegmentationSample],
    predictions: dict[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute equal-group-weighted Dice, IoU, Recall, Precision plus per-group Dice."""
    grouped: dict[str, list[list[float]]] = defaultdict(list)
    for sample in samples:
        valid = sample.valid > 0
        truth = (sample.target > 0) & valid
        pred = (predictions[sample.sample_id] > 0) & valid
        n_ref = int(truth.sum())
        n_pred = int(pred.sum())
        inter = int((pred & truth).sum())
        require(n_ref > 0, f"Sample {sample.sample_id} has an empty reference mask.")
        dice = 2.0 * inter / (n_pred + n_ref)
        iou = inter / (n_pred + n_ref - inter)
        recall = inter / n_ref
        precision = inter / n_pred if n_pred > 0 else 0.0
        grouped[sample.group_id].append([dice, iou, recall, precision])

    group_means = {g: np.mean(vals, axis=0) for g, vals in grouped.items()}
    overall = np.mean(list(group_means.values()), axis=0)
    metrics = {
        "dice": float(overall[0]),
        "iou": float(overall[1]),
        "recall": float(overall[2]),
        "precision": float(overall[3]),
        "groups": len(group_means),
    }
    group_dice = {g: float(vals[0]) for g, vals in group_means.items()}
    return metrics, group_dice


def predict_cohort(
    payload: dict[str, Any],
    samples: list[SegmentationSample],
    threshold: float = 0.0,
    forward_fn: Callable[[np.ndarray, dict[str, Any]], np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Run frozen forward pass on all samples and return thresholded valid boolean masks."""
    if forward_fn is None:
        forward_fn, _, _ = _forward_bundle(payload["architecture"])
    masks: dict[str, np.ndarray] = {}
    for sample in samples:
        logits = forward_fn(sample.image, payload)
        require(
            logits.shape == sample.image.shape and bool(np.isfinite(logits).all()),
            "Invalid segmentation logits.",
        )
        masks[sample.sample_id] = (logits >= threshold) & (sample.valid > 0)
    return masks


def check_position_invariance(
    payload: dict[str, Any],
    samples: list[SegmentationSample],
    threshold: float = 0.0,
) -> None:
    """Verify Condition C produces byte-identical logits across distinct inputs on the same canvas."""
    if len(samples) < 2:
        return
    forward_full, forward_pos, _ = _forward_bundle(payload["architecture"])
    first = samples[0]
    second = next(
        (
            s
            for s in samples[1:]
            if s.image.shape == first.image.shape and not np.array_equal(s.image, first.image)
        ),
        None,
    )
    require(second is not None, "Position invariance check requires two distinct images of equal shape.")
    logits1 = forward_pos(first.image, payload)
    logits2 = forward_pos(second.image, payload)
    readout_zero = {**payload, "readout": [0.0] * len(payload["readout"])}
    oracle = forward_full(first.image, readout_zero)
    common_valid = first.valid > 0
    require(
        np.array_equal(logits1, logits2)
        and np.allclose(logits1, oracle, atol=1e-6, rtol=0.0)
        and np.array_equal((logits1 >= threshold) & common_valid, (logits2 >= threshold) & common_valid),
        "Position-only invariance check failed.",
    )


def evaluate_segmentation_ablation(
    payload: dict[str, Any],
    samples: list[SegmentationSample],
    threshold: float = 0.0,
) -> dict[str, Any]:
    """Run Conditions A (intact), B (zero grid), and C (position-only) with invariance check."""
    forward_full, forward_pos, without_grid = _forward_bundle(payload["architecture"])
    check_position_invariance(payload, samples, threshold=threshold)

    masks_a = predict_cohort(payload, samples, threshold=threshold, forward_fn=forward_full)
    masks_b = predict_cohort(without_grid(payload), samples, threshold=threshold, forward_fn=forward_full)
    masks_c = predict_cohort(payload, samples, threshold=threshold, forward_fn=forward_pos)

    out: dict[str, Any] = {}
    for arm, masks in (("A", masks_a), ("B", masks_b), ("C", masks_c)):
        metrics, g_dice = group_segmentation_scores(samples, masks)
        out[arm] = {"metrics": metrics, "group_dice": g_dice, "masks": masks}
    return out


def _eligible_swap_donors(samples: list[SegmentationSample]) -> dict[str, list[str]]:
    by_canvas: dict[tuple[tuple[int, ...], bytes], list[SegmentationSample]] = defaultdict(list)
    for s in samples:
        valid = s.valid > 0
        by_canvas[(valid.shape, np.packbits(valid).tobytes())].append(s)
    donors: dict[str, list[str]] = {}
    for members in by_canvas.values():
        for rec in members:
            donors[rec.sample_id] = [d.sample_id for d in members if d.group_id != rec.group_id]
    return donors


def evaluate_segmentation_swaps(
    payload: dict[str, Any],
    samples: list[SegmentationSample],
    threshold: float = 0.0,
) -> dict[str, Any]:
    """Score intact (A), image-only (B), and position-only (C) correct-vs-swapped Dice gaps."""
    ablation = evaluate_segmentation_ablation(payload, samples, threshold=threshold)
    donors = _eligible_swap_donors(samples)
    eligible = [s for s in samples if donors[s.sample_id]]
    require(len(eligible) >= 2, "Need at least two samples sharing a valid-pixel mask for paired swaps.")

    truth = {s.sample_id: ((s.target > 0) & (s.valid > 0)) for s in samples}
    ref_area = {sid: int(mask.sum()) for sid, mask in truth.items()}

    def _swap_summary(masks: dict[str, np.ndarray]) -> dict[str, Any]:
        pred_area = {sid: int(m.sum()) for sid, m in masks.items()}

        def _dice(rec_id: str, donor_id: str) -> float:
            inter = int(np.count_nonzero(truth[rec_id] & masks[donor_id]))
            return 2.0 * inter / (ref_area[rec_id] + pred_area[donor_id])

        by_group_own: dict[str, list[float]] = defaultdict(list)
        by_group_other: dict[str, list[float]] = defaultdict(list)
        for s in eligible:
            own_val = _dice(s.sample_id, s.sample_id)
            other_val = float(np.mean([_dice(s.sample_id, d) for d in donors[s.sample_id]]))
            by_group_own[s.group_id].append(own_val)
            by_group_other[s.group_id].append(other_val)

        groups = list(by_group_own.keys())
        own_arr = np.array([np.mean(by_group_own[g]) for g in groups], dtype=np.float64)
        other_arr = np.array([np.mean(by_group_other[g]) for g in groups], dtype=np.float64)
        loss_pp = 100.0 * (own_arr - other_arr)
        return {
            "correct_dice_pct": float(100.0 * own_arr.mean()),
            "swapped_dice_pct": float(100.0 * other_arr.mean()),
            "swap_loss_pp": float(loss_pp.mean()),
            "swap_loss_median_pp": float(np.median(loss_pp)),
            "swap_loss_positive_groups": int(np.count_nonzero(loss_pp > 1e-10)),
            "group_losses_pp": loss_pp,
        }

    sum_a = _swap_summary(ablation["A"]["masks"])
    sum_b = _swap_summary(ablation["B"]["masks"])
    sum_c = _swap_summary(ablation["C"]["masks"])
    require(
        np.allclose(sum_c["group_losses_pp"], 0.0, atol=1e-10),
        "Position-only control failed swap invariance check.",
    )
    grid_effect_pp = sum_a["group_losses_pp"] - sum_b["group_losses_pp"]
    return {
        "pair_count": sum(len(donors[s.sample_id]) for s in eligible),
        "groups": len(sum_a["group_losses_pp"]),
        "total_groups": len({s.group_id for s in samples}),
        "full_cohort_dice_a_pct": float(100.0 * ablation["A"]["metrics"]["dice"]),
        "full_cohort_dice_b_pct": float(100.0 * ablation["B"]["metrics"]["dice"]),
        "full_cohort_dice_c_pct": float(100.0 * ablation["C"]["metrics"]["dice"]),
        "intact": {k: v for k, v in sum_a.items() if k != "group_losses_pp"},
        "image_only": {k: v for k, v in sum_b.items() if k != "group_losses_pp"},
        "position_only": {k: v for k, v in sum_c.items() if k != "group_losses_pp"},
        "grid_effect_dice_gap_pp": float(grid_effect_pp.mean()),
        "grid_effect_median_pp": float(np.median(grid_effect_pp)),
        "grid_effect_positive_groups": int(np.count_nonzero(grid_effect_pp > 1e-10)),
    }


def score_segmentation_masks(
    samples: list[SegmentationSample],
    predictions: dict[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, float]]:
    """Score arbitrary binary prediction masks using the exact benchmark region, boundary, and box metrics.

    Returns `(metrics_dict, per_group_dice_pct)` where every metric is equal-group-weighted in `%`.
    Shared by trivial/non-LLM baselines and learned arms for strict metric parity.
    """
    reg, group_dice = group_segmentation_scores(samples, predictions)
    by_group: dict[str, list[tuple[float, float, float, float, float]]] = defaultdict(list)

    for s in samples:
        valid = s.valid > 0
        valid_area = int(np.count_nonzero(valid))
        truth = (s.target > 0) & valid
        pred = (predictions[s.sample_id] > 0) & valid
        ref_area = int(np.count_nonzero(truth))
        ref_box = bounding_slices(truth)
        require(ref_box is not None and ref_area > 0, f"Empty reference for {s.sample_id}.")
        ref_edge, ref_band = contour_and_band(truth, valid)
        ref_box_vol = int(np.prod([sl.stop - sl.start for sl in ref_box]))

        pred_edge, pred_band = contour_and_band(pred, valid)
        prec = float(np.count_nonzero(ref_band[pred_edge]) / pred_edge.size) if pred_edge.size else 0.0
        recall = float(pred_band[ref_edge].mean()) if ref_edge.size else 0.0
        bf1 = 2.0 * prec * recall / (prec + recall) if (prec + recall) > 0.0 else 0.0

        pred_box = bounding_slices(pred)
        proposal = np.zeros_like(valid, dtype=bool)
        if pred_box is not None:
            proposal[pred_box] = True
        prop_vol = int(np.count_nonzero(proposal))
        prop_valid_vol = int(np.count_nonzero(proposal & valid))
        box_valid_frac = prop_valid_vol / valid_area if valid_area > 0 else 0.0
        overlap = int(np.count_nonzero(proposal[ref_box]))
        denom = prop_vol + ref_box_vol - overlap
        box_iou = overlap / denom if denom > 0 else 0.0
        excl_tooth = 1.0 - (int(np.count_nonzero(truth & proposal)) / ref_area)
        by_group[s.group_id].append(
            (100.0 * bf1, 100.0 * recall, 100.0 * box_iou, 100.0 * box_valid_frac, 100.0 * excl_tooth)
        )

    group_means = np.array([np.mean(v, axis=0) for v in by_group.values()], dtype=np.float64)
    metrics = {
        "dice_pct": 100.0 * reg["dice"],
        "iou_pct": 100.0 * reg["iou"],
        "precision_pct": 100.0 * reg["precision"],
        "recall_pct": 100.0 * reg["recall"],
        "bf1_pct": float(group_means[:, 0].mean()),
        "edge_recall_pct": float(group_means[:, 1].mean()),
        "box_iou_pct": float(group_means[:, 2].mean()),
        "box_valid_vol_pct": float(group_means[:, 3].mean()),
        "excluded_tooth_pct": float(group_means[:, 4].mean()),
        "groups": int(reg["groups"]),
    }
    group_dice_pct = {g: 100.0 * d for g, d in group_dice.items()}
    return metrics, group_dice_pct


def evaluate_segmentation_paired_audit(
    payload: dict[str, Any],
    samples: list[SegmentationSample],
    threshold: float = 0.0,
) -> dict[str, Any]:
    """Paired intact (A) vs. image-only (B) false-positive, boundary, and enclosing-box audit.

    Notes on key metrics
    --------------------
    - `outside_fp_reduction_share_pct` (dict key retained for backward compatibility; labeled
      `"Outside-ROI share of grid-removed FPs (A vs B) (%)"` in new CSVs and tables):
      Defined as `100 * fp_saved_out / (fp_saved_out + fp_saved_in)` when total net FPs removed
      by the spatial grid (`fp_saved_out + fp_saved_in`, comparing Arm A vs. Arm B) is positive,
      else `0.0`. This is a signed share of *net* grid-removed FPs, NOT a percentage reduction of
      outside-ROI FPs: it can exceed `100%` when the grid removes outside-ROI FPs while increasing
      inside-ROI FPs (`fp_saved_in < 0`, shrinking the denominator), or become negative when
      outside-ROI FPs increase (`fp_saved_out < 0`) while inside-ROI FPs decrease more strongly.
    - `box_valid_vol_a_pct`, `box_valid_vol_b_pct`:
      Percentage of valid pixels/voxels retained inside the predicted enclosing bounding box
      (`100 * |pred_box ∩ valid| / |valid|`). Reported alongside `excluded_tooth_a_pct` and
      `box_iou_a_pct` because near-0% excluded tooth voxels can be trivial when the predicted
      box spans a large fraction of the valid canvas/volume.
    """
    ablation = evaluate_segmentation_ablation(payload, samples, threshold=threshold)
    masks_a = ablation["A"]["masks"]
    masks_b = ablation["B"]["masks"]

    per_group: dict[str, list[dict[str, float]]] = defaultdict(list)

    for s in samples:
        valid = s.valid > 0
        valid_area = int(np.count_nonzero(valid))
        truth = (s.target > 0) & valid
        ref_area = int(np.count_nonzero(truth))
        ref_box = bounding_slices(truth)
        require(ref_box is not None and ref_area > 0, f"Empty reference for {s.sample_id}.")
        ref_edge, ref_band = contour_and_band(truth, valid)
        require(ref_edge.size > 0, f"Reference {s.sample_id} has no evaluable boundary.")

        box_dims = [sl.stop - sl.start for sl in ref_box]
        ref_box_vol = int(np.prod(box_dims))

        arm_stats: dict[str, dict[str, Any]] = {}
        for arm, mask in (("A", masks_a[s.sample_id]), ("B", masks_b[s.sample_id])):
            pred = mask & valid
            pred_area = int(np.count_nonzero(pred))
            in_box = pred[ref_box]
            inside = int(np.count_nonzero(in_box))
            tp = int(np.count_nonzero(in_box & truth[ref_box]))
            fp_in = 1000.0 * (inside - tp) / valid_area
            fp_out = 1000.0 * (pred_area - inside) / valid_area

            pred_edge, pred_band = contour_and_band(pred, valid)
            prec_full = float(np.count_nonzero(ref_band[pred_edge]) / pred_edge.size) if pred_edge.size else 0.0
            hit_ref = pred_band[ref_edge]
            recall_full = float(hit_ref.mean())
            bf1 = (
                2.0 * prec_full * recall_full / (prec_full + recall_full)
                if (prec_full + recall_full) > 0.0
                else 0.0
            )

            # In-box predicted edge precision (extract edges full-canvas first, then filter by ref_box)
            if pred_edge.size > 0:
                coords = np.unravel_index(pred_edge, valid.shape)
                in_bounds = np.ones(pred_edge.size, dtype=bool)
                for coord, sl in zip(coords, ref_box):
                    in_bounds &= (coord >= sl.start) & (coord < sl.stop)
                edge_in_box = pred_edge[in_bounds]
                in_box_prec = float(ref_band[edge_in_box].mean()) if edge_in_box.size else 0.0
            else:
                in_box_prec = 0.0

            # Enclosing predicted box localization, valid-volume retention, & excluded tooth rate
            pred_box = bounding_slices(pred)
            proposal = np.zeros_like(valid, dtype=bool)
            if pred_box is not None:
                proposal[pred_box] = True
            prop_vol = int(np.count_nonzero(proposal))
            prop_valid_vol = int(np.count_nonzero(proposal & valid))
            box_valid_frac = prop_valid_vol / valid_area if valid_area > 0 else 0.0
            box_overlap = int(np.count_nonzero(proposal[ref_box]))
            box_iou = box_overlap / (prop_vol + ref_box_vol - box_overlap) if (prop_vol + ref_box_vol - box_overlap) else 0.0
            box_cov = box_overlap / ref_box_vol
            retained_tooth = int(np.count_nonzero(truth & proposal))
            excluded_tooth_rate = 1.0 - (retained_tooth / ref_area)

            arm_stats[arm] = {
                "fp_in": fp_in,
                "fp_out": fp_out,
                "bf1_pct": 100.0 * bf1,
                "edge_recall_pct": 100.0 * recall_full,
                "in_box_prec_pct": 100.0 * in_box_prec,
                "box_iou_pct": 100.0 * box_iou,
                "box_cov_pct": 100.0 * box_cov,
                "box_valid_vol_pct": 100.0 * box_valid_frac,
                "excluded_tooth_pct": 100.0 * excluded_tooth_rate,
                "hit_ref": hit_ref,
            }

        hit_a = arm_stats["A"]["hit_ref"]
        hit_b = arm_stats["B"]["hit_ref"]
        status = 2 * hit_a.astype(np.uint8) + hit_b.astype(np.uint8)
        counts = np.bincount(status, minlength=4) / float(ref_edge.size)

        per_group[s.group_id].append(
            {
                "fp_in_a": arm_stats["A"]["fp_in"],
                "fp_in_b": arm_stats["B"]["fp_in"],
                "fp_out_a": arm_stats["A"]["fp_out"],
                "fp_out_b": arm_stats["B"]["fp_out"],
                "fp_saved_in": arm_stats["B"]["fp_in"] - arm_stats["A"]["fp_in"],
                "fp_saved_out": arm_stats["B"]["fp_out"] - arm_stats["A"]["fp_out"],
                "bf1_a_pct": arm_stats["A"]["bf1_pct"],
                "bf1_b_pct": arm_stats["B"]["bf1_pct"],
                "bf1_gain_pp": arm_stats["A"]["bf1_pct"] - arm_stats["B"]["bf1_pct"],
                "edge_recall_a_pct": arm_stats["A"]["edge_recall_pct"],
                "edge_recall_b_pct": arm_stats["B"]["edge_recall_pct"],
                "edge_recall_gain_pp": arm_stats["A"]["edge_recall_pct"] - arm_stats["B"]["edge_recall_pct"],
                "in_box_prec_a_pct": arm_stats["A"]["in_box_prec_pct"],
                "in_box_prec_b_pct": arm_stats["B"]["in_box_prec_pct"],
                "in_box_prec_gain_pp": arm_stats["A"]["in_box_prec_pct"] - arm_stats["B"]["in_box_prec_pct"],
                "box_iou_a_pct": arm_stats["A"]["box_iou_pct"],
                "box_iou_b_pct": arm_stats["B"]["box_iou_pct"],
                "box_cov_a_pct": arm_stats["A"]["box_cov_pct"],
                "box_cov_b_pct": arm_stats["B"]["box_cov_pct"],
                "box_valid_vol_a_pct": arm_stats["A"]["box_valid_vol_pct"],
                "box_valid_vol_b_pct": arm_stats["B"]["box_valid_vol_pct"],
                "excluded_tooth_a_pct": arm_stats["A"]["excluded_tooth_pct"],
                "excluded_tooth_b_pct": arm_stats["B"]["excluded_tooth_pct"],
                "edges_missed_pct": 100.0 * float(counts[0]),
                "edges_lost_pct": 100.0 * float(counts[1]),
                "edges_gained_pct": 100.0 * float(counts[2]),
                "edges_preserved_pct": 100.0 * float(counts[3]),
            }
        )

    # Weight each provisional/patient group equally
    group_rows = [
        {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        for rows in per_group.values()
    ]
    summary = {k: float(np.mean([g[k] for g in group_rows])) for k in group_rows[0]}
    total_fp_saved = summary["fp_saved_out"] + summary["fp_saved_in"]
    summary["outside_fp_reduction_share_pct"] = (
        100.0 * summary["fp_saved_out"] / total_fp_saved if total_fp_saved > 0 else 0.0
    )
    summary["in_box_prec_positive_groups"] = int(
        sum(g["in_box_prec_gain_pp"] > 1e-10 for g in group_rows)
    )
    summary["groups"] = len(group_rows)
    return summary
