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
from wsp_core.schemas import zero_payload
from wsp_core.training.runner import (
    _flush_runs_csv,
    _load_existing_runs,
    plot_classification_learning_curve,
    plot_segmentation_learning_curve,
)
from wsp_core.training.trainer import (
    TrainingConfig,
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
