"""Unit tests for trivial/non-LLM baselines, human-written 82p prior, and Hanley-McNeil CIs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from wsp_core.evaluation.baselines import (
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
from wsp_core.evaluation.classification import ClassificationSample, hanley_mcneil_auc_ci
from wsp_core.evaluation.segmentation import (
    SegmentationSample,
    group_segmentation_scores,
    predict_cohort,
    score_segmentation_masks,
)
from wsp_core.registry import REPO_ROOT

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _make_synthetic_2d_seg(n: int = 6, *, seed: int = 0) -> list[SegmentationSample]:
    rng = np.random.default_rng(seed)
    samples = []
    for idx in range(n):
        img = rng.uniform(0.05, 0.25, size=(32, 64)).astype(np.float32)
        target = np.zeros((32, 64), dtype=bool)
        valid = np.zeros((32, 64), dtype=bool)
        valid[2:30, 4:60] = True
        x0 = 12 + (idx % 3) * 12
        img[10:22, x0 : x0 + 10] = rng.uniform(0.75, 0.95, size=(12, 10)).astype(np.float32)
        target[10:22, x0 : x0 + 10] = True
        img[~valid] = 0.0
        samples.append(SegmentationSample(f"s{idx}", f"g{idx}", img, target, valid))
    return samples


def _make_synthetic_3d_seg(n: int = 4, *, seed: int = 1) -> list[SegmentationSample]:
    rng = np.random.default_rng(seed)
    samples = []
    for idx in range(n):
        vol = rng.uniform(0.05, 0.25, size=(12, 16, 16)).astype(np.float32)
        target = np.zeros((12, 16, 16), dtype=bool)
        valid = np.ones((12, 16, 16), dtype=bool)
        x0 = 3 + idx * 2
        vol[4:8, 5:11, x0 : x0 + 4] = rng.uniform(0.75, 0.95, size=(4, 6, 4)).astype(np.float32)
        target[4:8, 5:11, x0 : x0 + 4] = True
        samples.append(SegmentationSample(f"v{idx}", f"vg{idx}", vol, target, valid))
    return samples


def test_otsu_threshold_separates_bimodal_valid_region():
    img = np.full((32, 32), 0.15, dtype=np.float32)
    valid = np.zeros((32, 32), dtype=bool)
    valid[4:28, 4:28] = True
    img[10:22, 10:22] = 0.85
    # Padding outside valid should not bias Otsu
    img[~valid] = 1.0
    t = compute_otsu_threshold(img, valid)
    assert 0.20 < t < 0.80
    pred = (img >= t) & valid
    expected = np.zeros((32, 32), dtype=bool)
    expected[10:22, 10:22] = True
    assert np.array_equal(pred, expected)


def test_fast_global_threshold_matches_naive_group_dice_exactly():
    train = _make_synthetic_2d_seg(6, seed=42)
    grid = np.round(np.linspace(0.10, 0.90, 17), 4)
    t_fast, dice_fast = fit_global_intensity_threshold(train, threshold_grid=grid)

    # Naive reference using group_segmentation_scores
    naive_dices = []
    for t in grid:
        preds = {s.sample_id: (s.image >= t) & (s.valid > 0) for s in train}
        metrics, _ = group_segmentation_scores(train, preds)
        naive_dices.append(100.0 * metrics["dice"])
    naive_arr = np.asarray(naive_dices, dtype=np.float64)
    best_dice = float(naive_arr.max())
    best_indices = np.flatnonzero(naive_arr >= best_dice - 1e-12)
    best_t = float(grid[int(best_indices[len(best_indices) // 2])])

    assert t_fast == pytest.approx(best_t, abs=1e-12)
    assert dice_fast == pytest.approx(best_dice, abs=1e-10)


@pytest.mark.parametrize("arch", ["pano-wsp82-v1", "periapical-wsp82-v1", "cbct-wsp82-v1"])
def test_human_written_82p_prior_matches_global_threshold_frozen(arch: str):
    samples = _make_synthetic_3d_seg(3) if arch == "cbct-wsp82-v1" else _make_synthetic_2d_seg(3)
    t = 0.50
    payload = build_human_written_seg82_payload(arch, t)
    assert payload["architecture"] == arch
    assert payload["feature_bias"][0] == pytest.approx(-t)
    assert payload["readout"][0] > 0.0
    assert all(v == 0.0 for v in payload["readout"][1:])
    assert np.all(np.asarray(payload["spatial_bias"]) == 0.0)

    human_masks = predict_cohort(payload, samples)
    thresh_masks = {s.sample_id: (s.image >= t) & (s.valid > 0) for s in samples}
    m_human, _ = score_segmentation_masks(samples, human_masks)
    m_thresh, _ = score_segmentation_masks(samples, thresh_masks)
    assert m_human["dice_pct"] == pytest.approx(m_thresh["dice_pct"], abs=1e-4)
    assert m_human["dice_pct"] > 95.0


def test_evaluate_segmentation_baselines_2d_and_3d(tmp_path: Path):
    train2d = _make_synthetic_2d_seg(4, seed=0)
    val2d = _make_synthetic_2d_seg(3, seed=1)
    df2d, meta2d = evaluate_segmentation_baselines(
        task="akudental_seg",
        architecture="pano-wsp82-v1",
        train_samples=train2d,
        val_samples=val2d,
        design_lock_path=REPO_ROOT / "benchmarks/akudental/locks/akudental_seg.design.lock.json",
        responses_lock_path=REPO_ROOT / "benchmarks/akudental/locks/akudental_seg.responses.lock.json",
        out_dir=tmp_path,
    )
    assert (tmp_path / "akudental_seg_val_baselines.csv").is_file()
    names = set(df2d["baseline"])
    assert {"all_foreground", "per_image_otsu", "global_train_threshold", "human_82p_frozen", "frozen_wsp_mean6"} <= names
    assert 0.25 < meta2d["global_threshold_t_star"] < 0.75

    # Verify global threshold and human 82p frozen both reach >95% Dice on clean synthetic data
    row_thresh = df2d[df2d["baseline"] == "global_train_threshold"].iloc[0]
    row_human = df2d[df2d["baseline"] == "human_82p_frozen"].iloc[0]
    assert row_thresh["dice_mean"] > 95.0
    assert row_human["dice_mean"] == pytest.approx(row_thresh["dice_mean"], abs=1e-4)

    # 3D smoke check
    train3d = _make_synthetic_3d_seg(3, seed=2)
    val3d = _make_synthetic_3d_seg(2, seed=3)
    df3d, _ = evaluate_segmentation_baselines(
        task="dentvoxel_seg",
        architecture="cbct-wsp82-v1",
        train_samples=train3d,
        val_samples=val3d,
    )
    assert float(df3d[df3d["baseline"] == "human_82p_frozen"]["dice_mean"].iloc[0]) > 95.0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch optional extra not installed locally")
def test_human_82p_warmstart_baseline_runs_cleanly():
    train2d = _make_synthetic_2d_seg(4, seed=5)
    val2d = _make_synthetic_2d_seg(2, seed=6)
    df, _ = evaluate_segmentation_baselines(
        task="denpar_seg",
        architecture="periapical-wsp82-v1",
        train_samples=train2d,
        val_samples=val2d,
        run_human_warmstart=True,
        run_trained_position_only=True,
        budgets=[2],
        seeds=[0],
        epochs=3,
        lr=1e-2,
        batch_size=2,
        device="cpu",
    )
    assert "human_82p_warmstart" in set(df["baseline"])
    assert "trained_position_only" in set(df["baseline"])


def test_classification_baselines_logreg_and_denpar_crown_edge_rule(tmp_path: Path):
    # Synthetic Upper (bright crown above dark occlusal air) vs Lower (dark air above bright crown)
    train_cls: list[ClassificationSample] = []
    val_cls: list[ClassificationSample] = []
    for split_list, n_samples, offset in ((train_cls, 12, 0), (val_cls, 10, 100)):
        for i in range(n_samples):
            label = 1 if i % 2 == 0 else 0
            img = np.full((64, 64), 0.10, dtype=np.float32)
            if label == 1:
                # Upper arch: bone at top (0.5), bright crown middle (0.95), dark occlusal air below (0.05)
                img[6:28, 10:54] = 0.50
                img[28:40, 10:54] = 0.95
                img[40:58, 10:54] = 0.05
            else:
                # Lower arch: dark occlusal air above (0.05), bright crown middle (0.95), bone below (0.5)
                img[6:24, 10:54] = 0.05
                img[24:36, 10:54] = 0.95
                img[36:58, 10:54] = 0.50
            split_list.append(ClassificationSample(f"c{offset+i}", f"g{offset+i}", img, label))

    # Hand-coded rule separates Upper > 0 vs Lower < 0 on BOTH full and margin-cropped inputs
    assert denpar_crown_edge_rule_score(val_cls[0].image) > 0.0  # label=1 (Upper)
    assert denpar_crown_edge_rule_score(val_cls[1].image) < 0.0  # label=0 (Lower)
    edge_res = evaluate_denpar_crown_edge_baseline(val_cls)
    assert edge_res["roc_auc"] == pytest.approx(1.0)
    assert edge_res["margin_ablated_roc_auc"] == pytest.approx(1.0)
    assert edge_res["accuracy"] == pytest.approx(1.0)

    # Global intensity logreg on synthetic sinus-like dark-air vs solid-tissue task
    sinus_train = []
    sinus_val = []
    for bucket, count, off in ((sinus_train, 10, 0), (sinus_val, 8, 50)):
        for i in range(count):
            label = 1 if i % 2 == 0 else 0
            img = np.full((64, 64), 0.50, dtype=np.float32)
            if label == 1:
                img[16:48, 16:48] = 0.02  # large dark air cavity -> high frac < 0.1
            else:
                img[24:40, 24:40] = 0.95  # bright teeth -> high frac > 0.9
            bucket.append(ClassificationSample(f"x{off+i}", f"xg{off+i}", img, label))

    f_pos = extract_global_intensity_features(sinus_train[0].image)
    assert f_pos.shape == (3,)
    assert f_pos[1] > 0.20  # frac < 0.1
    clf = fit_global_intensity_logreg(sinus_train)
    assert clf.predict_logits(sinus_val)[0] > clf.predict_logits(sinus_val)[1]

    df_cls = evaluate_classification_baselines(
        task="denpar_cls",
        train_samples=train_cls,
        val_samples=val_cls,
        budgets=[2, 6],
        seeds=[0, 1],
        design_lock_path=REPO_ROOT / "benchmarks/denpar/locks/denpar_cls.design.lock.json",
        responses_lock_path=REPO_ROOT / "benchmarks/denpar/locks/denpar_cls.responses.lock.json",
        out_dir=tmp_path,
    )
    assert (tmp_path / "denpar_cls_val_baselines.csv").is_file()
    assert {"constant_score", "handcoded_vertical_crown_edge", "frozen_wsp_mean6", "global_intensity_logreg"} <= set(
        df_cls["baseline"]
    )
    const_row = df_cls[df_cls["baseline"] == "constant_score"].iloc[0]
    assert const_row["roc_auc_mean"] == pytest.approx(0.5)
    assert const_row["pr_auc_mean"] == pytest.approx(0.5)


def test_hanley_mcneil_auc_ci_matches_known_values():
    # For akudental_cls test (n_pos=50, n_neg=16, AUC=0.5429), 95% CI half-width is ~0.161
    lo, hi, se = hanley_mcneil_auc_ci(0.542917, n_pos=50, n_neg=16)
    assert 0.075 < se < 0.090
    assert 0.37 < lo < 0.40
    assert 0.69 < hi < 0.72
    # Bounds stay clipped inside [0, 1]
    lo_hi, hi_hi, _ = hanley_mcneil_auc_ci(0.995, n_pos=50, n_neg=16)
    assert 0.0 <= lo_hi <= 0.995 <= hi_hi <= 1.0


def test_cli_parsers_for_run_baselines_and_factorial_controls():
    from wsp_core.cli import build_parser

    parser = build_parser()
    args_base = parser.parse_args(
        [
            "run-baselines",
            "dentvoxel_seg",
            "--data-dir",
            "data/DentVoxel.zip",
            "--split-file",
            "runs/splits.json",
            "--out-dir",
            "results/validation_controls",
            "--run-human-warmstart",
            "--budgets",
            "1,2,4",
            "--seeds",
            "0,1",
        ]
    )
    assert args_base.command == "run-baselines"
    assert args_base.task == "dentvoxel_seg"
    assert args_base.run_human_warmstart is True

    args_fact = parser.parse_args(
        [
            "run-factorial-controls",
            "dentvoxel_seg",
            "--data-dir",
            "data/DentVoxel.zip",
            "--split-file",
            "runs/splits.json",
            "--out-dir",
            "results/validation_controls",
            "--budgets",
            "1,2,4,8",
            "--seeds",
            "0,1,2,3,4",
            "--steps",
            "480,1800",
        ]
    )
    assert args_fact.command == "run-factorial-controls"
    assert args_fact.steps == "480,1800"
    assert args_fact.n_boot == 2000
