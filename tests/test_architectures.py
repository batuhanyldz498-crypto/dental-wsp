"""Tests for 2D/3D segmentation and 2D classification forward oracles and PyTorch parity."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from wsp_core.architectures.cls_lenet2d import (
    build_torch_cls_lenet2d,
    cls_lenet2d_logits,
    cls_lenet2d_margin_ablation,
)
from wsp_core.architectures.seg_wsp82_2d import (
    build_torch_seg_wsp82_2d,
    seg_wsp82_2d_logits,
    seg_wsp82_2d_position_logits,
    seg_wsp82_2d_without_grid,
)
from wsp_core.architectures.seg_wsp82_3d import (
    build_torch_seg_wsp82_3d,
    seg_wsp82_3d_logits,
    seg_wsp82_3d_position_logits,
    seg_wsp82_3d_without_grid,
)
from wsp_core.schemas import ARCHITECTURE_SHAPES

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _random_payload(arch: str, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    payload = {"architecture": arch}
    for key, shape in ARCHITECTURE_SHAPES[arch].items():
        if shape == ():
            payload[key] = float(rng.normal(0.0, 0.5))
        else:
            payload[key] = rng.normal(0.0, 0.5, size=shape).tolist()
    return payload


def test_seg_wsp82_2d_oracle_and_position_invariance():
    rng = np.random.default_rng(1)
    img1 = rng.uniform(0, 1, size=(32, 64)).astype(np.float32)
    img2 = rng.uniform(0, 1, size=(32, 64)).astype(np.float32)
    payload = _random_payload("pano-wsp82-v1", seed=10)

    logits_a = seg_wsp82_2d_logits(img1, payload)
    logits_b = seg_wsp82_2d_logits(img1, seg_wsp82_2d_without_grid(payload))
    pos1 = seg_wsp82_2d_position_logits(img1, payload)
    pos2 = seg_wsp82_2d_position_logits(img2, payload)

    assert logits_a.shape == (32, 64)
    assert np.array_equal(pos1, pos2)
    # Exact additive logit relation: A = B + (pos - output_bias)
    assert np.allclose(logits_a, logits_b + (pos1 - payload["output_bias"]), atol=1e-5)


def test_seg_wsp82_3d_oracle_and_position_invariance():
    rng = np.random.default_rng(2)
    vol1 = rng.uniform(0, 1, size=(12, 16, 16)).astype(np.float32)
    vol2 = rng.uniform(0, 1, size=(12, 16, 16)).astype(np.float32)
    payload = _random_payload("cbct-wsp82-v1", seed=20)

    logits_a = seg_wsp82_3d_logits(vol1, payload)
    logits_b = seg_wsp82_3d_logits(vol1, seg_wsp82_3d_without_grid(payload))
    pos1 = seg_wsp82_3d_position_logits(vol1, payload)
    pos2 = seg_wsp82_3d_position_logits(vol2, payload)

    assert logits_a.shape == (12, 16, 16)
    assert np.array_equal(pos1, pos2)
    assert np.allclose(logits_a, logits_b + (pos1 - payload["output_bias"]), atol=1e-5)


def test_cls_lenet2d_oracle_and_margin_ablation():
    rng = np.random.default_rng(3)
    img = rng.uniform(0.2, 0.8, size=(64, 64)).astype(np.float32)
    payload = _random_payload("cls-lenet2d-v1", seed=30)
    logit = cls_lenet2d_logits(img, payload)
    assert np.isfinite(logit)

    cropped = cls_lenet2d_margin_ablation(img, border_fraction=0.10)
    assert np.all(cropped[:6, :] == 0.0)
    assert np.all(cropped[:, :6] == 0.0)
    assert np.any(cropped[10:-10, 10:-10] > 0.0)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch optional extra not installed locally")
def test_torch_numpy_equivalence():
    import torch

    rng = np.random.default_rng(7)

    # 2D 82-param parity
    p2d = _random_payload("pano-wsp82-v1", seed=11)
    img = rng.uniform(0, 1, size=(32, 64)).astype(np.float32)
    m2d = build_torch_seg_wsp82_2d("pano-wsp82-v1")
    m2d.load_from_wsp_dict(p2d)
    with torch.no_grad():
        t2d = m2d(torch.from_numpy(img)[None, None]).squeeze().numpy()
    assert np.allclose(seg_wsp82_2d_logits(img, p2d), t2d, atol=1e-5)

    # 3D 82-param parity
    p3d = _random_payload("cbct-wsp82-v1", seed=12)
    vol = rng.uniform(0, 1, size=(12, 16, 16)).astype(np.float32)
    m3d = build_torch_seg_wsp82_3d()
    m3d.load_from_wsp_dict(p3d)
    with torch.no_grad():
        t3d = m3d(torch.from_numpy(vol)[None, None]).squeeze().numpy()
    assert np.allclose(seg_wsp82_3d_logits(vol, p3d), t3d, atol=1e-5)

    # 2D LeNet classifier parity
    pcls = _random_payload("cls-lenet2d-v1", seed=13)
    cimg = rng.uniform(0, 1, size=(64, 64)).astype(np.float32)
    mcls = build_torch_cls_lenet2d()
    mcls.load_from_wsp_dict(pcls)
    with torch.no_grad():
        tcls = float(mcls(torch.from_numpy(cimg)[None, None]).item())
    assert np.isclose(cls_lenet2d_logits(cimg, pcls), tcls, atol=1e-5)
