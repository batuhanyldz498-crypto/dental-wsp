"""Tests for sealed test-set guards, nested label-budget subsampling, and optional PyTorch trainer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from wsp_core.evaluation.classification import ClassificationSample
from wsp_core.evaluation.segmentation import SegmentationSample
from wsp_core.guards import StudyStop, exposure_guard, require_split_allowed
from wsp_core.schemas import sha256_bytes, stable_json_bytes, zero_payload
from wsp_core.training.controls import (
    ScheduleSpec,
    bootstrap_group_mean_ci,
    plot_factorial_controls_svg,
    run_segmentation_factorial_controls,
)
from wsp_core.training.runner import (
    _flush_runs_csv,
    _load_existing_runs,
    plot_classification_learning_curve,
    plot_segmentation_learning_curve,
)
from wsp_core.training.trainer import (
    TrainingConfig,
    compute_prior_output_bias,
    sample_nested_group_budgets,
    sample_stratified_cls_budgets,
    train_segmentation_budget,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_sealed_test_guard_blocks_without_valid_authorization(tmp_path: Path):
    # Validation/train allowed
    require_split_allowed("val", task="akudental_seg")
    require_split_allowed("train", task="dentvoxel_seg")

    # Test blocked by default
    with pytest.raises(StudyStop, match="Test split is sealed"):
        require_split_allowed("test", task="akudental_seg")

    # Wrong task in authorization receipt blocked
    auth_file = tmp_path / "test_authorization.json"
    auth_file.write_text(
        json.dumps({"approved": True, "commit_sha": "abc123", "authorized_tasks": ["denpar_seg"]}),
        encoding="utf-8",
    )
    with pytest.raises(StudyStop, match="does not authorize"):
        require_split_allowed("test", task="akudental_seg", allow_test=True, authorization_file=auth_file)

    # Matching authorized task passes
    require_split_allowed("test", task="denpar_seg", allow_test=True, authorization_file=auth_file)


def test_exposure_guard_stops_on_marker(tmp_path: Path):
    exposure_guard(tmp_path)  # clean dir passes
    (tmp_path / "final_test_exposure.private.json").write_text("{}", encoding="utf-8")
    with pytest.raises(StudyStop, match="exposure marker"):
        exposure_guard(tmp_path)


def test_nested_group_budgets_strict_nesting_and_determinism():
    groups = [f"patient_{i:02d}" for i in range(20)]
    b1 = sample_nested_group_budgets(groups, [1, 2, 4, 8, 16], seed=7)
    b2 = sample_nested_group_budgets(groups, [1, 2, 4, 8, 16], seed=7)
    assert b1 == b2
    assert set(b1[1]) < set(b1[2]) < set(b1[4]) < set(b1[8]) < set(b1[16])
    assert len(b1[8]) == 8


def test_incremental_checkpoint_resume_helper(tmp_path: Path):
    csv_path = tmp_path / "runs.csv"
    rows, done = _load_existing_runs(csv_path)
    assert rows == [] and done == set()

    sample_rows = [
        {"task": "akudental_seg", "arm": "frozen_wsp", "k_budget": 0, "seed": -1, "init_slot": "astra_1", "dice_pct": 49.7},
        {"task": "akudental_seg", "arm": "random_init_adam", "k_budget": 4, "seed": 0, "init_slot": "random", "dice_pct": 55.2},
    ]
    _flush_runs_csv(sample_rows, csv_path)
    reloaded, done2 = _load_existing_runs(csv_path)
    assert len(reloaded) == 2
    assert ("frozen_wsp", 0, -1, "astra_1") in done2
    assert ("random_init_adam", 4, 0, "random") in done2


def test_stratified_cls_budgets_guarantee_both_classes_at_k2():
    dummy = np.zeros((64, 64), dtype=np.float32)
    samples = [ClassificationSample(f"s{i}", f"g{i}", dummy, 1 if i < 15 else 0) for i in range(20)]
    by_g = {s.group_id: s.label for s in samples}
    for seed in range(10):
        nested = sample_stratified_cls_budgets(samples, [2, 4, 8, 16], seed=seed)
        labels_k2 = {by_g[g] for g in nested[2]}
        assert labels_k2 == {0, 1}
        assert set(nested[2]) < set(nested[4]) < set(nested[8]) < set(nested[16])


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch optional extra not installed locally")
def test_random_init_and_wsp_warmstart_training_loop():
    img = np.full((32, 64), 0.1, dtype=np.float32)
    target = np.zeros((32, 64), dtype=bool)
    img[10:22, 20:44] = 0.9
    target[10:22, 20:44] = True
    sample = SegmentationSample("s0", "g0", img, target, np.ones((32, 64), dtype=bool))

    cfg = TrainingConfig(epochs=5, lr=1e-2, seed=0)
    rand_payload, rand_losses = train_segmentation_budget("pano-wsp82-v1", [sample], cfg)
    assert rand_payload["architecture"] == "pano-wsp82-v1"
    assert len(rand_losses) == 5

    warm_init = zero_payload("pano-wsp82-v1")
    warm_payload, warm_losses = train_segmentation_budget("pano-wsp82-v1", [sample], cfg, init_payload=warm_init)
    assert warm_payload["architecture"] == "pano-wsp82-v1"
    assert len(warm_losses) == 5


def test_learning_curve_plots_support_held_out_test_split_label(tmp_path: Path):
    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")
    cls_df = pd.DataFrame(
        [
            {"arm": "frozen_wsp", "k_budget": 0, "roc_auc_mean": 0.87, "roc_auc_std": 0.05, "margin_roc_auc_mean": 0.85, "margin_roc_auc_std": 0.07},
            {"arm": "random_init_adam", "k_budget": 2, "roc_auc_mean": 0.88, "roc_auc_std": 0.03, "margin_roc_auc_mean": 0.82, "margin_roc_auc_std": 0.04},
            {"arm": "wsp_warmstart_adam", "k_budget": 2, "roc_auc_mean": 0.98, "roc_auc_std": 0.01, "margin_roc_auc_mean": 0.97, "margin_roc_auc_std": 0.02},
        ]
    )
    seg_df = pd.DataFrame(
        [
            {"arm": "frozen_wsp", "k_budget": 0, "dice_mean": 47.4, "dice_std": 3.2, "edge_recall_mean": 96.1, "edge_recall_std": 2.1},
            {"arm": "random_init_adam", "k_budget": 1, "dice_mean": 0.0, "dice_std": 0.0, "edge_recall_mean": 0.0, "edge_recall_std": 0.0},
            {"arm": "wsp_warmstart_adam", "k_budget": 1, "dice_mean": 60.4, "dice_std": 4.7, "edge_recall_mean": 91.7, "edge_recall_std": 3.6},
        ]
    )
    cls_png = tmp_path / "dentvoxel_cls_test_learning_curve.png"
    seg_png = tmp_path / "dentvoxel_seg_test_learning_curve.png"
    plot_classification_learning_curve(cls_df, task="dentvoxel_cls", out_path=cls_png, split_label="test")
    plot_segmentation_learning_curve(seg_df, task="dentvoxel_seg", out_path=seg_png, split_label="test")
    assert cls_png.is_file() and cls_png.stat().st_size > 1000
    assert seg_png.is_file() and seg_png.stat().st_size > 1000


def test_compute_prior_output_bias_uses_valid_region_only():
    img = np.zeros((16, 16), dtype=np.float32)
    valid = np.zeros((16, 16), dtype=bool)
    valid[0:10, 0:10] = True  # 100 valid pixels
    target = np.zeros((16, 16), dtype=bool)
    target[0:2, 0:10] = True  # 20 valid foreground pixels -> p = 0.20
    target[12:16, 12:16] = True  # invalid region foreground must be ignored
    s = SegmentationSample("s0", "g0", img, target, valid)
    bias = compute_prior_output_bias([s])
    assert bias == pytest.approx(np.log(0.20 / 0.80), abs=1e-6)


def test_bootstrap_group_mean_ci_properties():
    # Identical groups -> zero-width CI equal to the constant mean
    const_mat = np.full((3, 10), 55.0, dtype=np.float64)
    lo, hi = bootstrap_group_mean_ci(const_mat, n_boot=500, seed=0)
    assert lo == pytest.approx(55.0)
    assert hi == pytest.approx(55.0)

    # Heterogeneous groups -> CI brackets the mean strictly and is deterministic
    rng = np.random.default_rng(123)
    mat = rng.uniform(30.0, 80.0, size=(5, 20))
    lo1, hi1 = bootstrap_group_mean_ci(mat, n_boot=1000, seed=2026)
    lo2, hi2 = bootstrap_group_mean_ci(mat, n_boot=1000, seed=2026)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 < float(mat.mean()) < hi1


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch optional extra not installed locally")
def test_trainer_defaults_reproduce_v010_bit_for_bit_on_cpu():
    """Regression test: default TrainingConfig reproduces the original v0.1.0 epoch loop bit-for-bit."""
    import torch
    import torch.nn.functional as F
    from wsp_core.architectures.seg_wsp82_2d import build_torch_seg_wsp82_2d
    from wsp_core.architectures.seg_wsp82_3d import build_torch_seg_wsp82_3d
    from wsp_core.training.trainer import _soft_dice_loss

    def _v010_reference(arch: str, samples: list[SegmentationSample], epochs: int, lr: float, seed: int, bs_cfg: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_torch_seg_wsp82_3d() if arch == "cbct-wsp82-v1" else build_torch_seg_wsp82_2d(arch)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=lr * 0.05)
        x_all = torch.stack([torch.from_numpy(s.image.astype(np.float32)).unsqueeze(0) for s in samples])
        y_all = torch.stack([torch.from_numpy((s.target > 0).astype(np.float32)).unsqueeze(0) for s in samples])
        v_all = torch.stack([torch.from_numpy((s.valid > 0).astype(np.float32)).unsqueeze(0) for s in samples])
        n = len(samples)
        bs = max(1, min(4 if arch == "cbct-wsp82-v1" else bs_cfg, n))
        gen = torch.Generator().manual_seed(seed)
        losses = []
        model.train()
        for _ in range(epochs):
            perm = torch.randperm(n, generator=gen) if n > bs else torch.arange(n)
            ep = torch.zeros(())
            steps = 0
            for start in range(0, n, bs):
                idx = perm[start : start + bs]
                optimizer.zero_grad()
                logits = model(x_all[idx])
                bce = (F.binary_cross_entropy_with_logits(logits, y_all[idx], reduction="none") * v_all[idx]).sum() / v_all[idx].sum().clamp_min(1.0)
                loss = 0.5 * bce + 0.5 * _soft_dice_loss(logits, y_all[idx], v_all[idx])
                loss.backward()
                optimizer.step()
                ep = ep + loss.detach()
                steps += 1
            scheduler.step()
            losses.append(float((ep / max(1, steps)).item()))
        return model.export_to_wsp_dict(), losses

    img2 = np.full((32, 32), 0.1, dtype=np.float32)
    t2 = np.zeros((32, 32), dtype=bool)
    img2[10:22, 10:22] = 0.9
    t2[10:22, 10:22] = True
    s2 = [SegmentationSample("s0", "g0", img2, t2, np.ones((32, 32), dtype=bool))]

    cfg = TrainingConfig(epochs=4, lr=0.03, seed=7, batch_size=4, cosine_lr=True)
    p2_new, l2_new = train_segmentation_budget("pano-wsp82-v1", s2, cfg)
    p2_ref, l2_ref = _v010_reference("pano-wsp82-v1", s2, 4, 0.03, 7, 4)
    assert sha256_bytes(stable_json_bytes(p2_new)) == sha256_bytes(stable_json_bytes(p2_ref))
    assert l2_new == l2_ref

    img3 = np.full((8, 8, 8), 0.1, dtype=np.float32)
    t3 = np.zeros((8, 8, 8), dtype=bool)
    img3[2:5, 2:5, 2:5] = 0.9
    t3[2:5, 2:5, 2:5] = True
    s3 = [SegmentationSample("v0", "vg0", img3, t3, np.ones((8, 8, 8), dtype=bool))]

    p3_new, l3_new = train_segmentation_budget("cbct-wsp82-v1", s3, cfg)
    p3_ref, l3_ref = _v010_reference("cbct-wsp82-v1", s3, 4, 0.03, 7, 4)
    assert sha256_bytes(stable_json_bytes(p3_new)) == sha256_bytes(stable_json_bytes(p3_ref))
    assert l3_new == l3_ref


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch optional extra not installed locally")
def test_trainer_max_steps_and_prior_output_bias_controls():
    samples = []
    for i in range(6):
        img = np.full((16, 16), 0.1, dtype=np.float32)
        target = np.zeros((16, 16), dtype=bool)
        img[6:8, 6:8] = 0.9  # 4 / 256 = 1.5625% sparse foreground
        target[6:8, 6:8] = True
        samples.append(SegmentationSample(f"s{i}", f"g{i}", img, target, np.ones((16, 16), dtype=bool)))

    # Fixed max_steps executes exact step count independent of K (K=1 vs K=6)
    cfg_steps = TrainingConfig(epochs=2, lr=0.03, seed=0, batch_size=2, cosine_lr=True, max_steps=11, init_output_bias="prior")
    p_k1, l_k1 = train_segmentation_budget("pano-wsp82-v1", samples[:1], cfg_steps)
    p_k6, l_k6 = train_segmentation_budget("pano-wsp82-v1", samples, cfg_steps)
    assert len(l_k1) == 11
    assert len(l_k6) == 11
    # Prior-bias init starts around log(0.015625 / 0.984375) ≈ -4.14 instead of 0.0
    assert p_k1["output_bias"] < -2.0
    assert p_k6["output_bias"] < -2.0

    # Invalid init_output_bias or combining prior with explicit init_payload raises ValueError
    with pytest.raises(ValueError, match="Unsupported init_output_bias"):
        train_segmentation_budget("pano-wsp82-v1", samples[:1], TrainingConfig(epochs=1, init_output_bias="bogus"))
    with pytest.raises(ValueError, match="cannot be combined"):
        train_segmentation_budget(
            "pano-wsp82-v1",
            samples[:1],
            TrainingConfig(epochs=1, init_output_bias="prior"),
            init_payload=zero_payload("pano-wsp82-v1"),
        )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch optional extra not installed locally")
def test_factorial_controls_runner_and_svg_output(tmp_path: Path):
    import matplotlib
    from wsp_core.registry import REPO_ROOT

    matplotlib.use("Agg")
    rng = np.random.default_rng(11)
    train_s, val_s = [], []
    for bucket, n, off in ((train_s, 4, 0), (val_s, 3, 10)):
        for i in range(n):
            vol = rng.uniform(0.05, 0.25, size=(8, 12, 12)).astype(np.float32)
            target = np.zeros((8, 12, 12), dtype=bool)
            vol[2:6, 4:8, 4:8] = rng.uniform(0.75, 0.95, size=(4, 4, 4)).astype(np.float32)
            target[2:6, 4:8, 4:8] = True
            bucket.append(SegmentationSample(f"v{off+i}", f"vg{off+i}", vol, target, np.ones((8, 12, 12), dtype=bool)))

    out = run_segmentation_factorial_controls(
        task="dentvoxel_seg",
        architecture="cbct-wsp82-v1",
        train_samples=train_s,
        val_samples=val_s,
        design_lock_path=REPO_ROOT / "benchmarks/dentvoxel/locks/dentvoxel_seg.design.lock.json",
        responses_lock_path=REPO_ROOT / "benchmarks/dentvoxel/locks/dentvoxel_seg.responses.lock.json",
        budgets=[1, 2],
        seeds=[0, 1],
        schedules=[
            ScheduleSpec("original_epochs_2", epochs=2, max_steps=None),
            ScheduleSpec("fixed_steps_4", epochs=2, max_steps=4),
        ],
        lr=0.03,
        batch_size=2,
        device="cpu",
        out_dir=tmp_path,
        resume=True,
        n_boot=200,
    )
    summary_df = out["summary_df"]
    paired_df = out["paired_summary_df"]
    svg_path = out["svg_path"]
    assert set(summary_df["arm"]) == {"random_zero_bias", "random_prior_bias", "human_init", "wsp_warmstart"}
    assert set(summary_df["schedule"]) == {"original_epochs_2", "fixed_steps_4"}
    assert {"dice_mean", "dice_boot_ci95_low", "dice_boot_ci95_high", "n_collapsed"} <= set(summary_df.columns)
    assert "wsp_warmstart_vs_random_prior_bias" in set(paired_df["comparison"])
    assert svg_path is not None and svg_path.is_file() and svg_path.suffix == ".svg"
    assert "<svg" in svg_path.read_text(encoding="utf-8")

    # Verify resume skips completed runs cleanly
    out2 = run_segmentation_factorial_controls(
        task="dentvoxel_seg",
        architecture="cbct-wsp82-v1",
        train_samples=train_s,
        val_samples=val_s,
        design_lock_path=REPO_ROOT / "benchmarks/dentvoxel/locks/dentvoxel_seg.design.lock.json",
        responses_lock_path=REPO_ROOT / "benchmarks/dentvoxel/locks/dentvoxel_seg.responses.lock.json",
        budgets=[1, 2],
        seeds=[0, 1],
        schedules=[
            ScheduleSpec("original_epochs_2", epochs=2, max_steps=None),
            ScheduleSpec("fixed_steps_4", epochs=2, max_steps=4),
        ],
        lr=0.03,
        batch_size=2,
        device="cpu",
        out_dir=tmp_path,
        resume=True,
        n_boot=200,
    )
    assert len(out2["runs_df"]) == len(out["runs_df"])

    # Confirm SVG plotting helper works standalone too
    svg2 = plot_factorial_controls_svg(summary_df, task="dentvoxel_seg", out_path=tmp_path / "standalone.svg")
    assert svg2.is_file()
