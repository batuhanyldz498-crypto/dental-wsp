"""82-parameter 3D multiscale 7-tap cross-stencil WSP segmenter (`cbct-wsp82-v1`).

Parameter budget (exact 82-parameter parity with 2D `pano-wsp82-v1` and `periapical-wsp82-v1`):
- 6 filters x 7-tap 3D 6-connected stencil [center, -z, +z, -y, +y, -x, +x] = 42
- 6 feature biases = 6
- 6 readout weights = 6
- 3 x 3 x 3 aligned-corners trilinear spatial grid = 27
- 1 scalar output bias = 1
Total = 82 parameters.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wsp_core.architectures.seg_wsp82_2d import basis_1d
from wsp_core.schemas import ARCHITECTURE_SHAPES, validate_parameters

SCALES_3D = (1, 2, 4)


def seg_wsp82_3d_position_logits(volume: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
    """Position-only 3D forward pass (Condition C): trilinear 3x3x3 grid + output bias."""
    arrays = validate_parameters(payload, expected_architecture="cbct-wsp82-v1")
    depth, height, width = volume.shape
    bd = basis_1d(depth, 3)
    bh = basis_1d(height, 3)
    bw = basis_1d(width, 3)
    spatial = np.einsum("di,hj,wk,ijk->dhw", bd, bh, bw, arrays["spatial_bias"], optimize=True)
    return (spatial + arrays["output_bias"]).astype(np.float32, copy=False)


def seg_wsp82_3d_logits(volume: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
    """Exact NumPy forward oracle for `cbct-wsp82-v1` on a 3D (D, H, W) volume."""
    arrays = validate_parameters(payload, expected_architecture="cbct-wsp82-v1")
    depth, height, width = volume.shape
    if depth % 4 or height % 4 or width % 4:
        raise ValueError("3D canvas dimensions must be divisible by 4.")

    out = seg_wsp82_3d_position_logits(volume, payload)
    vol = np.asarray(volume, dtype=np.float32)

    for scale_idx, scale in enumerate(SCALES_3D):
        if scale == 1:
            low = vol
        else:
            low = vol.reshape(
                depth // scale, scale, height // scale, scale, width // scale, scale
            ).mean(axis=(1, 3, 5))
        pad = np.pad(low, 1, mode="constant", constant_values=0.0)
        c = pad[1:-1, 1:-1, 1:-1]
        zm = pad[:-2, 1:-1, 1:-1]
        zp = pad[2:, 1:-1, 1:-1]
        ym = pad[1:-1, :-2, 1:-1]
        yp = pad[1:-1, 2:, 1:-1]
        xm = pad[1:-1, 1:-1, :-2]
        xp = pad[1:-1, 1:-1, 2:]

        for filt_idx in (2 * scale_idx, 2 * scale_idx + 1):
            w = arrays["stencils"][filt_idx]
            feat = (
                w[0] * c
                + w[1] * zm
                + w[2] * zp
                + w[3] * ym
                + w[4] * yp
                + w[5] * xm
                + w[6] * xp
                + arrays["feature_bias"][filt_idx]
            )
            activated = np.maximum(feat, np.float32(0.0))
            if scale > 1:
                activated = activated.repeat(scale, axis=0).repeat(scale, axis=1).repeat(scale, axis=2)
            out = out + arrays["readout"][filt_idx] * activated
    return out.astype(np.float32, copy=False)


def seg_wsp82_3d_without_grid(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `payload` with the 3x3x3 spatial grid zeroed (Condition B)."""
    return {
        **payload,
        "spatial_bias": np.zeros(ARCHITECTURE_SHAPES["cbct-wsp82-v1"]["spatial_bias"]).tolist(),
    }


def build_torch_seg_wsp82_3d(position_only: bool = False) -> Any:
    """Create a differentiable PyTorch `SegWSP82_3D` module matching `seg_wsp82_3d_logits`."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SegWSP82_3D(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.architecture = "cbct-wsp82-v1"
            self.position_only = position_only
            self.stencils = nn.Parameter(torch.empty(6, 7))
            self.feature_bias = nn.Parameter(torch.zeros(6))
            self.readout = nn.Parameter(torch.empty(6))
            self.spatial_bias = nn.Parameter(torch.zeros(3, 3, 3))
            self.output_bias = nn.Parameter(torch.tensor(0.0))
            self.reset_parameters()
            if self.position_only:
                self.stencils.requires_grad_(False)
                self.feature_bias.requires_grad_(False)
                self.readout.requires_grad_(False)

        def reset_parameters(self) -> None:
            nn.init.uniform_(self.stencils, -0.5, 0.5)
            nn.init.constant_(self.feature_bias, 0.1)  # keep tiny 6-filter ReLUs alive at init
            nn.init.uniform_(self.readout, -0.5, 0.5)
            nn.init.zeros_(self.spatial_bias)
            nn.init.zeros_(self.output_bias)

        def load_from_wsp_dict(self, payload: dict[str, Any]) -> None:
            arrays = validate_parameters(payload, expected_architecture=self.architecture)
            with torch.no_grad():
                self.stencils.copy_(torch.from_numpy(arrays["stencils"]))
                self.feature_bias.copy_(torch.from_numpy(arrays["feature_bias"]))
                self.readout.copy_(torch.from_numpy(arrays["readout"]))
                self.spatial_bias.copy_(torch.from_numpy(arrays["spatial_bias"]))
                self.output_bias.copy_(torch.tensor(float(arrays["output_bias"])))

        def export_to_wsp_dict(self) -> dict[str, Any]:
            with torch.no_grad():
                return {
                    "architecture": self.architecture,
                    "stencils": self.stencils.detach().cpu().float().tolist(),
                    "feature_bias": self.feature_bias.detach().cpu().float().tolist(),
                    "readout": self.readout.detach().cpu().float().tolist(),
                    "spatial_bias": self.spatial_bias.detach().cpu().float().tolist(),
                    "output_bias": float(self.output_bias.detach().cpu().item()),
                }

        @staticmethod
        def _basis_torch(length: int, nodes: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
            coord = torch.linspace(0, nodes - 1, length, device=device, dtype=dtype)
            grid = torch.arange(nodes, device=device, dtype=dtype)
            return torch.clamp(1.0 - torch.abs(coord[:, None] - grid[None, :]), min=0.0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, 1, D, H, W)
            _, _, depth, height, width = x.shape
            if depth % 4 or height % 4 or width % 4:
                raise ValueError("3D canvas dimensions must be divisible by 4.")
            bd = self._basis_torch(depth, 3, x.device, x.dtype)
            bh = self._basis_torch(height, 3, x.device, x.dtype)
            bw = self._basis_torch(width, 3, x.device, x.dtype)
            spatial = torch.einsum("di,hj,wk,ijk->dhw", bd, bh, bw, self.spatial_bias) + self.output_bias
            out = spatial.unsqueeze(0).unsqueeze(0).expand(x.shape[0], 1, depth, height, width).clone()
            if self.position_only:
                return out

            for scale_idx, scale in enumerate(SCALES_3D):
                low = x if scale == 1 else F.avg_pool3d(x, kernel_size=scale, stride=scale)
                pad = F.pad(low, (1, 1, 1, 1, 1, 1), mode="constant", value=0.0)
                c = pad[:, :, 1:-1, 1:-1, 1:-1]
                zm = pad[:, :, :-2, 1:-1, 1:-1]
                zp = pad[:, :, 2:, 1:-1, 1:-1]
                ym = pad[:, :, 1:-1, :-2, 1:-1]
                yp = pad[:, :, 1:-1, 2:, 1:-1]
                xm = pad[:, :, 1:-1, 1:-1, :-2]
                xp = pad[:, :, 1:-1, 1:-1, 2:]

                for filt_idx in (2 * scale_idx, 2 * scale_idx + 1):
                    w = self.stencils[filt_idx]
                    feat = (
                        w[0] * c
                        + w[1] * zm
                        + w[2] * zp
                        + w[3] * ym
                        + w[4] * yp
                        + w[5] * xm
                        + w[6] * xp
                        + self.feature_bias[filt_idx]
                    )
                    activated = F.relu(feat)
                    if scale > 1:
                        activated = (
                            activated.repeat_interleave(scale, dim=-3)
                            .repeat_interleave(scale, dim=-2)
                            .repeat_interleave(scale, dim=-1)
                        )
                    out = out + self.readout[filt_idx] * activated
            return out

    return SegWSP82_3D()
