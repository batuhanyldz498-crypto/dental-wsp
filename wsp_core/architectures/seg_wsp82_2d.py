"""82-parameter 2D multiscale WSP segmenter (`pano-wsp82-v1` and `periapical-wsp82-v1`).

Matches the exact forward oracle of AKUDENTAL Study 1 (`AKUDENTAL_WSP_Spatial_Ablation_Colab.ipynb`)
and exposes an equivalent differentiable PyTorch module when `torch` is installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wsp_core.schemas import ARCHITECTURE_SHAPES, validate_parameters

SCALES_2D = (1, 4, 16)


def basis_1d(length: int, nodes: int) -> np.ndarray:
    """Aligned-corners piecewise-linear interpolation basis of shape (length, nodes)."""
    coord = np.linspace(0, nodes - 1, length, dtype=np.float32)
    return np.maximum(1.0 - np.abs(coord[:, None] - np.arange(nodes, dtype=np.float32)), 0.0)


def seg_wsp82_2d_position_logits(image: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
    """Position-only forward pass (Condition C): bilinear 3x5 spatial grid + output bias."""
    arrays = validate_parameters(payload)
    height, width = image.shape
    return (basis_1d(height, 3) @ arrays["spatial_bias"] @ basis_1d(width, 5).T) + arrays["output_bias"]


def seg_wsp82_2d_logits(image: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
    """Exact NumPy forward oracle for `pano-wsp82-v1` and `periapical-wsp82-v1`."""
    arrays = validate_parameters(payload)
    height, width = image.shape
    if height % 16 or width % 16:
        raise ValueError("2D canvas dimensions must be divisible by 16.")

    out = (basis_1d(height, 3) @ arrays["spatial_bias"] @ basis_1d(width, 5).T) + arrays["output_bias"]
    img = np.asarray(image, dtype=np.float32)
    for scale_idx, scale in enumerate(SCALES_2D):
        if scale == 1:
            low = img
        else:
            low = img.reshape(height // scale, scale, width // scale, scale).mean(axis=(1, 3))
        windows = np.lib.stride_tricks.sliding_window_view(np.pad(low, 1), (3, 3))
        for filt_idx in (2 * scale_idx, 2 * scale_idx + 1):
            feature = np.einsum("hwxy,xy->hw", windows, arrays["kernels"][filt_idx]) + arrays["feature_bias"][filt_idx]
            activated = np.maximum(feature, np.float32(0.0))
            if scale > 1:
                activated = activated.repeat(scale, axis=0).repeat(scale, axis=1)
            out = out + arrays["readout"][filt_idx] * activated
    return out.astype(np.float32, copy=False)


def seg_wsp82_2d_without_grid(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `payload` with the 3x5 spatial grid zeroed (Condition B)."""
    arch = payload["architecture"]
    return {**payload, "spatial_bias": np.zeros(ARCHITECTURE_SHAPES[arch]["spatial_bias"]).tolist()}


def build_torch_seg_wsp82_2d(architecture: str = "pano-wsp82-v1", position_only: bool = False) -> Any:
    """Create a differentiable PyTorch `SegWSP82_2D` module matching `seg_wsp82_2d_logits`."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if architecture not in ("pano-wsp82-v1", "periapical-wsp82-v1"):
        raise ValueError(f"Unsupported 2D segmentation architecture: {architecture}")

    class SegWSP82_2D(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.architecture = architecture
            self.position_only = position_only
            self.kernels = nn.Parameter(torch.empty(6, 3, 3))
            self.feature_bias = nn.Parameter(torch.zeros(6))
            self.readout = nn.Parameter(torch.empty(6))
            self.spatial_bias = nn.Parameter(torch.zeros(3, 5))
            self.output_bias = nn.Parameter(torch.tensor(0.0))
            self.reset_parameters()
            if self.position_only:
                self.kernels.requires_grad_(False)
                self.feature_bias.requires_grad_(False)
                self.readout.requires_grad_(False)

        def reset_parameters(self) -> None:
            nn.init.kaiming_uniform_(self.kernels, a=0.0)
            nn.init.zeros_(self.feature_bias)
            nn.init.uniform_(self.readout, -0.5, 0.5)
            nn.init.zeros_(self.spatial_bias)
            nn.init.zeros_(self.output_bias)

        def load_from_wsp_dict(self, payload: dict[str, Any]) -> None:
            arrays = validate_parameters(payload, expected_architecture=self.architecture)
            with torch.no_grad():
                self.kernels.copy_(torch.from_numpy(arrays["kernels"]))
                self.feature_bias.copy_(torch.from_numpy(arrays["feature_bias"]))
                self.readout.copy_(torch.from_numpy(arrays["readout"]))
                self.spatial_bias.copy_(torch.from_numpy(arrays["spatial_bias"]))
                self.output_bias.copy_(torch.tensor(float(arrays["output_bias"])))

        def export_to_wsp_dict(self) -> dict[str, Any]:
            with torch.no_grad():
                return {
                    "architecture": self.architecture,
                    "kernels": self.kernels.detach().cpu().float().tolist(),
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
            # x: (B, 1, H, W)
            _, _, height, width = x.shape
            if height % 16 or width % 16:
                raise ValueError("2D canvas dimensions must be divisible by 16.")
            bh = self._basis_torch(height, 3, x.device, x.dtype)
            bw = self._basis_torch(width, 5, x.device, x.dtype)
            spatial = (bh @ self.spatial_bias @ bw.T) + self.output_bias
            out = spatial.unsqueeze(0).unsqueeze(0).expand(x.shape[0], 1, height, width).clone()
            if self.position_only:
                return out
            for scale_idx, scale in enumerate(SCALES_2D):
                low = x if scale == 1 else F.avg_pool2d(x, kernel_size=scale, stride=scale)
                for filt_idx in (2 * scale_idx, 2 * scale_idx + 1):
                    w = self.kernels[filt_idx].view(1, 1, 3, 3)
                    feat = F.conv2d(low, w, bias=self.feature_bias[filt_idx : filt_idx + 1], padding=1)
                    activated = F.relu(feat)
                    if scale > 1:
                        activated = activated.repeat_interleave(scale, dim=-2).repeat_interleave(scale, dim=-1)
                    out = out + self.readout[filt_idx] * activated
            return out

    return SegWSP82_2D()
