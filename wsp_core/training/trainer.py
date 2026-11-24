"""Same-architecture Adam trainer supporting Random-Init, WSP Warm-Start, and Position-Only arms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from wsp_core.architectures.cls_lenet2d import build_torch_cls_lenet2d
from wsp_core.architectures.seg_wsp82_2d import build_torch_seg_wsp82_2d
from wsp_core.architectures.seg_wsp82_3d import build_torch_seg_wsp82_3d
from wsp_core.evaluation.classification import ClassificationSample
from wsp_core.evaluation.segmentation import SegmentationSample
from wsp_core.guards import require


@dataclass(frozen=True)
class TrainingConfig:
    """Prespecified optimization hyperparameters shared across Random-Init and WSP Warm-Start."""

    epochs: int = 200
    lr: float = 1e-2
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "cpu"


def sample_nested_group_budgets(
    group_ids: list[str],
    budgets: list[int],
    seed: int,
) -> dict[int, list[str]]:
    """Deterministically draw nested group subsets so K_1 ⊂ K_2 ⊂ ... for low-variance curves."""
    unique_groups = sorted(set(group_ids))
    require(len(unique_groups) > 0, "Cannot sample label budgets from an empty group list.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique_groups).tolist()
    result: dict[int, list[str]] = {}
    for k in sorted(set(budgets)):
        require(k >= 1, f"Label budget K must be >= 1, got {k}.")
        effective_k = min(k, len(perm))
        result[k] = perm[:effective_k]
    return result


def _soft_dice_loss(logits: Any, targets: Any, valid: Any, eps: float = 1e-6) -> Any:
    import torch

    probs = torch.sigmoid(logits) * valid
    t = targets * valid
    dims = tuple(range(1, probs.ndim))
    inter = (probs * t).sum(dim=dims)
    denom = probs.sum(dim=dims) + t.sum(dim=dims)
    return 1.0 - ((2.0 * inter + eps) / (denom + eps)).mean()


def train_segmentation_budget(
    architecture: str,
    train_samples: list[SegmentationSample],
    config: TrainingConfig,
    *,
    init_payload: dict[str, Any] | None = None,
    position_only: bool = False,
) -> tuple[dict[str, Any], list[float]]:
    """Train a compact 82-parameter 2D or 3D segmenter with Adam and return exported WSP payload."""
    import torch
    import torch.nn.functional as F

    require(len(train_samples) > 0, "Training requires at least one segmentation sample.")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if architecture in ("pano-wsp82-v1", "periapical-wsp82-v1"):
        model = build_torch_seg_wsp82_2d(architecture=architecture, position_only=position_only)
    elif architecture == "cbct-wsp82-v1":
        model = build_torch_seg_wsp82_3d(position_only=position_only)
    else:
        raise ValueError(f"Unsupported segmentation architecture: {architecture}")

    if init_payload is not None:
        model.load_from_wsp_dict(init_payload)

    device = torch.device(config.device)
    model = model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    require(len(trainable) > 0, "No trainable parameters found.")
    optimizer = torch.optim.Adam(trainable, lr=config.lr, weight_decay=config.weight_decay)

    x_batch = torch.stack([torch.from_numpy(s.image.astype(np.float32)).unsqueeze(0) for s in train_samples]).to(device)
    y_batch = torch.stack([torch.from_numpy((s.target > 0).astype(np.float32)).unsqueeze(0) for s in train_samples]).to(device)
    v_batch = torch.stack([torch.from_numpy((s.valid > 0).astype(np.float32)).unsqueeze(0) for s in train_samples]).to(device)

    losses: list[float] = []
    model.train()
    for _ in range(config.epochs):
        optimizer.zero_grad()
        logits = model(x_batch)
        bce = (F.binary_cross_entropy_with_logits(logits, y_batch, reduction="none") * v_batch).sum() / v_batch.sum().clamp_min(1.0)
        dice = _soft_dice_loss(logits, y_batch, v_batch)
        loss = 0.5 * bce + 0.5 * dice
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    return model.export_to_wsp_dict(), losses


def train_classification_budget(
    train_samples: list[ClassificationSample],
    config: TrainingConfig,
    *,
    init_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[float]]:
    """Train the 261-parameter `cls-lenet2d-v1` classifier with Adam and return exported WSP payload."""
    import torch
    import torch.nn.functional as F

    require(len(train_samples) > 0, "Training requires at least one classification sample.")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = build_torch_cls_lenet2d()
    if init_payload is not None:
        model.load_from_wsp_dict(init_payload)

    device = torch.device(config.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    x_batch = torch.stack([torch.from_numpy(s.image.astype(np.float32)).unsqueeze(0) for s in train_samples]).to(device)
    y_batch = torch.tensor([float(s.label) for s in train_samples], dtype=torch.float32, device=device)

    n_pos = float((y_batch == 1).sum().item())
    n_neg = float((y_batch == 0).sum().item())
    pos_weight = torch.tensor([n_neg / n_pos], device=device) if (n_pos > 0 and n_neg > 0) else None

    losses: list[float] = []
    model.train()
    for _ in range(config.epochs):
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = F.binary_cross_entropy_with_logits(logits, y_batch, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    return model.export_to_wsp_dict(), losses
