"""261-parameter compact 2D LeNet-style binary classifier (`cls-lenet2d-v1`).

Direct Lindholz et al. classification mirror across AKUDENTAL, DenPAR, and DentVoxel axial slices:
- Input: (64, 64) grayscale in [0, 1]
- Conv1: 4 x 1 x 5 x 5 (pad 2) + bias (4) + ReLU + 4x4 avg pool -> (4, 16, 16)
- Conv2: 4 x 4 x 3 x 3 (pad 1) + bias (4) + ReLU -> (4, 16, 16)
- Global pool: [channel_mean(4), channel_max(4)] -> (8,)
- Linear head: weights (8,) + bias () -> scalar logit
Total = 100 + 4 + 144 + 4 + 8 + 1 = 261 parameters.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wsp_core.schemas import validate_parameters


def cls_lenet2d_logits(image: np.ndarray, payload: dict[str, Any]) -> float:
    """Exact NumPy forward oracle for `cls-lenet2d-v1` on a (64, 64) image."""
    arrays = validate_parameters(payload, expected_architecture="cls-lenet2d-v1")
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2 or img.shape[0] % 4 != 0 or img.shape[1] % 4 != 0:
        raise ValueError("Classifier input must be a 2D image with dimensions divisible by 4.")
    height, width = img.shape

    # Conv1: 4 filters of 5x5, zero-padding 2
    windows1 = np.lib.stride_tricks.sliding_window_view(np.pad(img, 2), (5, 5))
    c1 = np.einsum("hwxy,oxy->ohw", windows1, arrays["conv1_weights"][:, 0], optimize=True)
    c1 = np.maximum(c1 + arrays["conv1_bias"][:, None, None], np.float32(0.0))
    # 4x4 average pooling
    p1 = c1.reshape(4, height // 4, 4, width // 4, 4).mean(axis=(2, 4))

    # Conv2: 4 filters of 3x3 over 4 channels, zero-padding 1
    pad_p1 = np.pad(p1, ((0, 0), (1, 1), (1, 1)))
    windows2 = np.lib.stride_tricks.sliding_window_view(pad_p1, (3, 3), axis=(1, 2))
    c2 = np.einsum("ihwxy,oixy->ohw", windows2, arrays["conv2_weights"], optimize=True)
    c2 = np.maximum(c2 + arrays["conv2_bias"][:, None, None], np.float32(0.0))

    pooled = np.concatenate([c2.mean(axis=(1, 2)), c2.max(axis=(1, 2))], axis=0)
    logit = float(np.dot(pooled, arrays["head_weights"]) + arrays["head_bias"])
    return logit


def cls_lenet2d_margin_ablation(image: np.ndarray, border_fraction: float = 0.10) -> np.ndarray:
    """Zero out the outer border strip to audit film-margin shortcut reliance."""
    out = np.asarray(image, dtype=np.float32).copy()
    h, w = out.shape
    dy = max(1, int(round(h * border_fraction)))
    dx = max(1, int(round(w * border_fraction)))
    out[:dy, :] = 0.0
    out[-dy:, :] = 0.0
    out[:, :dx] = 0.0
    out[:, -dx:] = 0.0
    return out


def build_torch_cls_lenet2d() -> Any:
    """Create a differentiable PyTorch `ClsLeNet2D` module matching `cls_lenet2d_logits`."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ClsLeNet2D(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.architecture = "cls-lenet2d-v1"
            self.conv1 = nn.Conv2d(1, 4, kernel_size=5, padding=2, bias=True)
            self.conv2 = nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=True)
            self.head = nn.Linear(8, 1, bias=True)

        def load_from_wsp_dict(self, payload: dict[str, Any]) -> None:
            arrays = validate_parameters(payload, expected_architecture=self.architecture)
            with torch.no_grad():
                self.conv1.weight.copy_(torch.from_numpy(arrays["conv1_weights"]))
                self.conv1.bias.copy_(torch.from_numpy(arrays["conv1_bias"]))
                self.conv2.weight.copy_(torch.from_numpy(arrays["conv2_weights"]))
                self.conv2.bias.copy_(torch.from_numpy(arrays["conv2_bias"]))
                self.head.weight.copy_(torch.from_numpy(arrays["head_weights"]).view(1, 8))
                self.head.bias.copy_(torch.tensor([float(arrays["head_bias"])]))

        def export_to_wsp_dict(self) -> dict[str, Any]:
            with torch.no_grad():
                return {
                    "architecture": self.architecture,
                    "conv1_weights": self.conv1.weight.detach().cpu().float().tolist(),
                    "conv1_bias": self.conv1.bias.detach().cpu().float().tolist(),
                    "conv2_weights": self.conv2.weight.detach().cpu().float().tolist(),
                    "conv2_bias": self.conv2.bias.detach().cpu().float().tolist(),
                    "head_weights": self.head.weight.detach().cpu().float().view(8).tolist(),
                    "head_bias": float(self.head.bias.detach().cpu().item()),
                }

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, 1, H, W) -> returns (B,) logits
            h = F.relu(self.conv1(x))
            h = F.avg_pool2d(h, kernel_size=4, stride=4)
            h = F.relu(self.conv2(h))
            mean_pool = h.mean(dim=(-2, -1))
            max_pool = h.amax(dim=(-2, -1))
            features = torch.cat([mean_pool, max_pool], dim=1)
            return self.head(features).squeeze(-1)

    return ClsLeNet2D()
