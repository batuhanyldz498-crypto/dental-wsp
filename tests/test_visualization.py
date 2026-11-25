"""Tests for deterministic X-ray and 3D CBCT slice overlay visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from wsp_core.evaluation.segmentation import SegmentationSample
from wsp_core.schemas import zero_payload
from wsp_core.visualization import (
    plot_four_arm_comparison,
    plot_frozen_replies_grid,
    select_deterministic_examples,
)


def test_deterministic_selection_and_2d_3d_overlay_exports(tmp_path: Path):
    samples_2d = []
    for i in range(4):
        img = np.full((32, 64), 0.2, dtype=np.float32)
        target = np.zeros((32, 64), dtype=bool)
        img[10:22, 16:48] = 0.85
        target[10:22, 16:48] = True
        samples_2d.append(SegmentationSample(f"img_{i}", f"g_{i}", img, target, np.ones((32, 64), dtype=bool)))

    sel1 = select_deterministic_examples(samples_2d, n=2)
    sel2 = select_deterministic_examples(samples_2d, n=2)
    assert [s.sample_id for s in sel1] == [s.sample_id for s in sel2]
    assert len(sel1) == 2

    p2d = zero_payload("pano-wsp82-v1")
    p2d["kernels"][0][1][1] = 2.0
    p2d["readout"][0] = 1.0
    p2d["output_bias"] = -0.8

    out_grid = tmp_path / "grid_2d.png"
    plot_frozen_replies_grid(sel1[0], {"astra_1": p2d, "opus_1": p2d}, out_path=out_grid)
    assert out_grid.is_file() and out_grid.stat().st_size > 1000

    out_4arm = tmp_path / "four_arm_2d.png"
    plot_four_arm_comparison(sel1[0], {"Frozen": p2d, "Position-Only": p2d, "Random-Init": p2d, "Warm-Start": p2d}, out_path=out_4arm)
    assert out_4arm.is_file() and out_4arm.stat().st_size > 1000

    # 3D CBCT axial slice visualization smoke test
    vol = np.full((12, 16, 16), 0.1, dtype=np.float32)
    msk = np.zeros((12, 16, 16), dtype=bool)
    vol[4:8, 5:11, 5:11] = 0.9
    msk[4:8, 5:11, 5:11] = True
    s3d = SegmentationSample("scan_0001", "scan_0001", vol, msk, np.ones((12, 16, 16), dtype=bool))
    p3d = zero_payload("cbct-wsp82-v1")
    out_3d = tmp_path / "grid_3d.png"
    plot_frozen_replies_grid(s3d, {"astra_1": p3d}, slice_axis="axial", out_path=out_3d)
    assert out_3d.is_file() and out_3d.stat().st_size > 1000
