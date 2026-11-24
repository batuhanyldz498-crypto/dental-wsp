"""Tests for sealed test-set guards, nested label-budget subsampling, and optional PyTorch trainer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from wsp_core.evaluation.segmentation import SegmentationSample
from wsp_core.guards import StudyStop, exposure_guard, require_split_allowed
from wsp_core.schemas import zero_payload
from wsp_core.training.trainer import (
    TrainingConfig,
    sample_nested_group_budgets,
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
