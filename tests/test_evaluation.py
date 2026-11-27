"""Synthetic 2D and 3D end-to-end tests for segmentation and classification evaluation suites."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.dentvoxel.adapter import (
    audit_intensity_calibration,
    extract_axial_sinus_slices,
    prepare_cbct_3d_canvas,
)
from wsp_core.evaluation.classification import (
    ClassificationSample,
    compute_pr_auc,
    compute_roc_auc,
    evaluate_classifier,
)
from wsp_core.evaluation.segmentation import (
    BOX_VALID_FRAC_COLUMN,
    OUTSIDE_FP_SHARE_COLUMN,
    SegmentationSample,
    evaluate_segmentation_paired_audit,
    evaluate_segmentation_swaps,
    predict_cohort,
    score_segmentation_masks,
)
from wsp_core.schemas import zero_payload


def _synthetic_2d_cohort() -> tuple[list[SegmentationSample], dict]:
    """Create 4 synthetic radiographs with bright tooth rectangles at shifted horizontal positions."""
    samples: list[SegmentationSample] = []
    shifts = [8, 18, 28, 38]
    for idx, x0 in enumerate(shifts):
        img = np.full((32, 64), 0.15, dtype=np.float32)
        target = np.zeros((32, 64), dtype=bool)
        valid = np.ones((32, 64), dtype=bool)
        # Tooth phantom in middle vertical band, shifted horizontally per group
        img[10:22, x0 : x0 + 12] = 0.90
        target[10:22, x0 : x0 + 12] = True
        # Bright off-arch distractor at top-left border to verify spatial-grid suppression
        img[1:5, 1:8] = 0.90
        samples.append(
            SegmentationSample(sample_id=f"s{idx}", group_id=f"g{idx}", image=img, target=target, valid=valid)
        )

    payload = zero_payload("pano-wsp82-v1")
    # Filter 0 activates on bright pixels (>0.5)
    payload["kernels"][0][1][1] = 4.0
    payload["feature_bias"][0] = -2.0
    payload["readout"][0] = 2.0
    # Spatial grid strongly penalizes top row (suppressing the top-left distractor)
    payload["spatial_bias"] = [
        [-6.0, -6.0, -6.0, -6.0, -6.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [-6.0, -6.0, -6.0, -6.0, -6.0],
    ]
    payload["output_bias"] = -0.5
    return samples, payload


def test_2d_swaps_and_paired_audit_end_to_end():
    samples, payload = _synthetic_2d_cohort()
    swaps = evaluate_segmentation_swaps(payload, samples)
    assert swaps["pair_count"] == 12  # 4 * 3 directed cross-group pairs
    assert swaps["intact"]["correct_dice_pct"] > 90.0
    assert swaps["intact"]["swap_loss_pp"] > 20.0
    assert swaps["position_only"]["swap_loss_pp"] == pytest.approx(0.0, abs=1e-10)

    audit = evaluate_segmentation_paired_audit(payload, samples)
    # Grid suppresses top-left distractor -> almost all FP reduction is outside the tooth box
    assert audit["fp_saved_out"] > 0.0
    assert audit["outside_fp_reduction_share_pct"] > 95.0
    # Paired reference-edge transition shares sum to 100%
    edge_sum = (
        audit["edges_preserved_pct"]
        + audit["edges_gained_pct"]
        + audit["edges_lost_pct"]
        + audit["edges_missed_pct"]
    )
    assert edge_sum == pytest.approx(100.0, abs=1e-5)
    # Tight intact box covers the tooth completely -> 0% excluded tooth pixels
    assert audit["excluded_tooth_a_pct"] == pytest.approx(0.0, abs=1e-5)


def test_3d_cbct_swaps_audit_and_hu_check():
    # Build 3 synthetic HU volumes with shifted bright enamel cubes (+2200 HU) in soft tissue (+100 HU)
    raw_vols = []
    samples: list[SegmentationSample] = []
    for idx, x0 in enumerate((3, 6, 9)):
        vol_hu = np.full((12, 16, 16), -900.0, dtype=np.float32)  # air background
        vol_hu[2:10, 2:14, 2:14] = 100.0  # soft tissue
        tooth = np.zeros((12, 16, 16), dtype=bool)
        vol_hu[4:8, 6:10, x0 : x0 + 4] = 2400.0  # bright enamel/dentin
        tooth[4:8, 6:10, x0 : x0 + 4] = True
        raw_vols.append(vol_hu)
        img_c, msk_c, val_c = prepare_cbct_3d_canvas(vol_hu, tooth, canvas=(12, 16, 16), norm_mode="hu_window")
        samples.append(
            SegmentationSample(sample_id=f"scan{idx}", group_id=f"scan{idx}", image=img_c, target=msk_c, valid=val_c)
        )

    hu_report = audit_intensity_calibration(raw_vols)
    assert hu_report["appears_hu_calibrated"] is True
    assert hu_report["recommended_mode"] == "hu_window"

    payload = zero_payload("cbct-wsp82-v1")
    payload["stencils"][0][0] = 6.0  # center tap
    payload["feature_bias"][0] = -3.6  # triggers above ~0.6 scaled (~+1400 HU)
    payload["readout"][0] = 2.0
    payload["output_bias"] = -0.5

    swaps = evaluate_segmentation_swaps(payload, samples)
    assert swaps["intact"]["correct_dice_pct"] > 95.0
    assert swaps["intact"]["swap_loss_pp"] > 20.0
    audit = evaluate_segmentation_paired_audit(payload, samples)
    assert audit["excluded_tooth_a_pct"] == pytest.approx(0.0, abs=1e-5)

    # Verify scan-split axial sinus slice extraction inherits scan_id as group_id
    sinus = np.zeros((12, 16, 16), dtype=bool)
    sinus[8:11, 4:12, 4:12] = True
    slices = extract_axial_sinus_slices("scan0", raw_vols[0], sinus, stride=1, min_sinus_voxels=20)
    assert all(s.group_id == "scan0" for s in slices)
    assert sum(s.label for s in slices) == 3


def test_classification_metrics_and_sign_flip_detection():
    y_true = np.array([0, 0, 1, 1])
    assert compute_roc_auc(y_true, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert compute_roc_auc(y_true, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert compute_pr_auc(y_true, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)

    # Construct a tiny classifier that detects bright central spots (and test its sign-flipped twin)
    samples = []
    for i in range(6):
        img = np.full((64, 64), 0.1, dtype=np.float32)
        label = 1 if i >= 3 else 0
        if label == 1:
            img[24:40, 24:40] = 0.95
        samples.append(ClassificationSample(sample_id=f"c{i}", group_id=f"g{i}", image=img, label=label))

    good = zero_payload("cls-lenet2d-v1")
    good["conv1_weights"][0][0][2][2] = 2.0
    good["conv1_bias"][0] = -0.8
    good["conv2_weights"][0][0][1][1] = 2.0
    good["head_weights"][4] = 1.0  # channel-0 global max
    good["head_bias"] = -0.5

    res_good = evaluate_classifier(good, samples)
    assert res_good["roc_auc"] == pytest.approx(1.0)
    assert res_good["sign_flip_flagged"] is False
    assert res_good["margin_ablated_roc_auc"] == pytest.approx(1.0)

    flipped = {**good, "head_weights": [-w for w in good["head_weights"]]}
    res_flip = evaluate_classifier(flipped, samples)
    assert res_flip["roc_auc"] == pytest.approx(0.0)
    assert res_flip["sign_flip_flagged"] is True


def test_prepare_panorama_handles_rgba_mask_identically_to_grayscale(tmp_path):
    from PIL import Image
    from benchmarks.akudental.adapter import prepare_panorama

    img = np.full((64, 64), 40, dtype=np.uint8)
    img[16:48, 20:44] = 220
    msk_l = np.zeros((64, 64), dtype=np.uint8)
    msk_l[16:48, 20:44] = 255
    msk_rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    msk_rgba[..., :3] = msk_l[..., None]
    msk_rgba[..., 3] = 255  # Constant opaque alpha (matches DenPAR Testing/431.png)

    img_p = tmp_path / "img.jpg"
    msk_l_p = tmp_path / "msk_l.png"
    msk_rgba_p = tmp_path / "msk_rgba.png"
    Image.fromarray(img, mode="L").save(img_p)
    Image.fromarray(msk_l, mode="L").save(msk_l_p)
    Image.fromarray(msk_rgba, mode="RGBA").save(msk_rgba_p)

    i1, t1, v1 = prepare_panorama(img_p, msk_l_p, canvas=(64, 64))
    i2, t2, v2 = prepare_panorama(img_p, msk_rgba_p, canvas=(64, 64))
    assert np.array_equal(i1, i2)
    assert np.array_equal(t1, t2)
    assert np.array_equal(v1, v2)
    assert float(t2.mean()) == pytest.approx(768.0 / 4096.0)


def test_box_valid_volume_fraction_and_outside_fp_share_over_100():
    """Verify box_valid_vol_a_pct / b_pct and that outside_fp_reduction_share_pct can exceed 100%."""
    samples, payload = _synthetic_2d_cohort()
    audit = evaluate_segmentation_paired_audit(payload, samples)

    # Intact box tightly matches the 12x12 tooth rectangle inside a 32x64 valid canvas (144 / 2048 = 7.03125%)
    assert audit["box_valid_vol_a_pct"] == pytest.approx(100.0 * (12 * 12) / (32 * 64), abs=1e-4)
    # Image-only includes the top-left distractor [1:5, 1:8], so its enclosing box is much larger
    assert audit["box_valid_vol_b_pct"] > audit["box_valid_vol_a_pct"]
    assert "Outside-ROI share of grid-removed FPs (A vs B)" in OUTSIDE_FP_SHARE_COLUMN
    assert "valid fraction" in BOX_VALID_FRAC_COLUMN

    # score_segmentation_masks matches evaluate_segmentation_paired_audit on arm A
    masks_a = predict_cohort(payload, samples)
    scored, per_group = score_segmentation_masks(samples, masks_a)
    assert scored["box_valid_vol_pct"] == pytest.approx(audit["box_valid_vol_a_pct"], abs=1e-6)
    assert scored["box_iou_pct"] == pytest.approx(audit["box_iou_a_pct"], abs=1e-6)
    assert scored["excluded_tooth_pct"] == pytest.approx(audit["excluded_tooth_a_pct"], abs=1e-6)
    assert scored["bf1_pct"] == pytest.approx(audit["bf1_a_pct"], abs=1e-6)
    assert len(per_group) == 4

    # Construct a case where spatial grid suppresses outside FPs AND increases inside-ROI FPs
    # -> fp_saved_out > 0, fp_saved_in < 0, total_fp_saved > 0 -> outside_fp_reduction_share_pct > 100%
    over_samples: list[SegmentationSample] = []
    for idx in range(2):
        img = np.full((32, 64), 0.10, dtype=np.float32)
        target = np.zeros((32, 64), dtype=bool)
        valid = np.ones((32, 64), dtype=bool)
        # Disconnected tooth pixels inside ref_box [14:18, 24:40] leaving interior background at 0.45
        target[14:16, 24:28] = True
        target[16:18, 36:40] = True
        img[14:18, 24:40] = 0.45  # in B: 4*0.45 - 1 - 1 = -0.2 < 0; in A: grid middle adds ~+0.5 -> +0.3 > 0
        img[target] = 0.90
        img[1:7, 1:50] = 0.90  # 294 outside-ROI distractor pixels at top (suppressed by grid in A)
        img[0, idx] = 0.20  # make samples distinct for position-invariance check
        over_samples.append(SegmentationSample(f"ov{idx}", f"og{idx}", img, target, valid))

    p2 = zero_payload("pano-wsp82-v1")
    p2["kernels"][0][1][1] = 4.0
    p2["feature_bias"][0] = -1.0  # activates above 0.25 -> 0.10 gives 0.0, 0.45 gives 0.8, 0.90 gives 2.6
    p2["readout"][0] = 1.0
    p2["output_bias"] = -1.0
    p2["spatial_bias"] = [
        [-4.0, -4.0, -4.0, -4.0, -4.0],  # suppresses top distractor (large fp_saved_out > 0)
        [+0.8, +0.8, +0.8, +0.8, +0.8],  # pushes interior 0.45 above 0 (fp_saved_in < 0) while 0.10 stays < 0
        [-4.0, -4.0, -4.0, -4.0, -4.0],
    ]
    over_audit = evaluate_segmentation_paired_audit(p2, over_samples)
    assert over_audit["fp_saved_out"] > 0.0
    assert over_audit["fp_saved_in"] < 0.0
    assert over_audit["outside_fp_reduction_share_pct"] > 100.0

    # All-foreground mask retains 100% of valid canvas and excludes 0% of tooth pixels
    all_fg = {s.sample_id: s.valid.copy() for s in samples}
    fg_metrics, _ = score_segmentation_masks(samples, all_fg)
    assert fg_metrics["box_valid_vol_pct"] == pytest.approx(100.0)
    assert fg_metrics["excluded_tooth_pct"] == pytest.approx(0.0)
    assert fg_metrics["bf1_pct"] == pytest.approx(0.0)
    assert fg_metrics["edge_recall_pct"] == pytest.approx(0.0)
