"""Same-architecture Adam trainer supporting Random-Init, WSP Warm-Start, and Position-Only arms."""

from __future__ import annotations

from dataclasses import dataclass
import math
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
    """Prespecified optimization hyperparameters shared across Random-Init and WSP Warm-Start.

    Optional controls (defaults reproduce v0.1.0 epoch-based training bit-for-bit on CPU):
    - `max_steps`: if set, runs exactly `max_steps` optimizer steps independent of `K` with cosine LR
      decay stepped once per optimizer step (`T_max = max_steps`).
    - `init_output_bias`: `"zero"` (default) or `"prior"` (`log(p / (1 - p))`, where `p` is the mean
      foreground fraction over valid voxels/pixels of the `K` training samples; only valid when
      `init_payload is None`).
    """

    epochs: int = 200
    lr: float = 1e-2
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "cpu"
    batch_size: int = 16
    cosine_lr: bool = True
    max_steps: int | None = None
    init_output_bias: str = "zero"


def compute_prior_output_bias(train_samples: list[SegmentationSample]) -> float:
    """Compute log-odds prior bias `log(p / (1 - p))` from mean valid-region foreground fraction `p`."""
    require(len(train_samples) > 0, "Prior output bias requires at least one training sample.")
    fracs: list[float] = []
    for s in train_samples:
        valid = s.valid > 0
        n_valid = int(np.count_nonzero(valid))
        require(n_valid > 0, f"Sample {s.sample_id} has an empty valid mask.")
        n_fg = int(np.count_nonzero((s.target > 0) & valid))
        fracs.append(n_fg / n_valid)
    p = float(np.clip(np.mean(fracs), 1e-6, 1.0 - 1e-6))
    return float(math.log(p / (1.0 - p)))


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


def sample_stratified_cls_budgets(
    samples: list[ClassificationSample],
    budgets: list[int],
    seed: int,
) -> dict[int, list[str]]:
    """Draw nested group budgets for binary classification guaranteeing >=1 pos and >=1 neg group when K >= 2."""
    group_labels: dict[str, set[int]] = {}
    for s in samples:
        group_labels.setdefault(s.group_id, set()).add(int(s.label))

    pos_groups = sorted(g for g, ls in group_labels.items() if 1 in ls and 0 not in ls)
    neg_groups = sorted(g for g, ls in group_labels.items() if 0 in ls and 1 not in ls)
    mixed_groups = sorted(g for g, ls in group_labels.items() if 0 in ls and 1 in ls)

    rng = np.random.default_rng(seed)
    if pos_groups and neg_groups:
        pos_perm = rng.permutation(pos_groups).tolist()
        neg_perm = rng.permutation(neg_groups).tolist()
        # Anchor the first two positions with 1 positive and 1 negative group, then shuffle remaining groups.
        rest = pos_perm[1:] + neg_perm[1:] + mixed_groups
        rest_perm = rng.permutation(rest).tolist() if rest else []
        ordered = [pos_perm[0], neg_perm[0]] + rest_perm
    else:
        ordered = rng.permutation(sorted(group_labels.keys())).tolist()

    result: dict[int, list[str]] = {}
    for k in sorted(set(budgets)):
        require(k >= 1, f"Label budget K must be >= 1, got {k}.")
        result[k] = ordered[: min(k, len(ordered))]
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
    if config.init_output_bias not in ("zero", "prior"):
        raise ValueError(f"Unsupported init_output_bias: {config.init_output_bias!r}. Expected 'zero' or 'prior'.")
    if config.max_steps is not None:
        require(config.max_steps >= 1, f"max_steps must be >= 1, got {config.max_steps}.")
    if init_payload is not None and config.init_output_bias != "zero":
        raise ValueError("init_output_bias='prior' cannot be combined with an explicit init_payload.")

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
    elif config.init_output_bias == "prior":
        prior_bias = compute_prior_output_bias(train_samples)
        with torch.no_grad():
            model.output_bias.fill_(prior_bias)

    device = torch.device(config.device)
    model = model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    require(len(trainable) > 0, "No trainable parameters found.")
    optimizer = torch.optim.Adam(trainable, lr=config.lr, weight_decay=config.weight_decay)
    t_max = max(1, config.max_steps) if config.max_steps is not None else max(1, config.epochs)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=config.lr * 0.05)
        if config.cosine_lr
        else None
    )

    x_all = torch.stack([torch.from_numpy(s.image.astype(np.float32)).unsqueeze(0) for s in train_samples])
    y_all = torch.stack([torch.from_numpy((s.target > 0).astype(np.float32)).unsqueeze(0) for s in train_samples])
    v_all = torch.stack([torch.from_numpy((s.valid > 0).astype(np.float32)).unsqueeze(0) for s in train_samples])
    total_bytes = (x_all.numel() + y_all.numel() + v_all.numel()) * 4
    pre_staged = device.type == "cuda" and total_bytes <= 2 * 1024**3
    if pre_staged:
        x_all = x_all.to(device)
        y_all = y_all.to(device)
        v_all = v_all.to(device)

    n_samples = len(train_samples)
    default_bs = 4 if architecture == "cbct-wsp82-v1" else config.batch_size
    bs = max(1, min(default_bs, n_samples))
    gen = torch.Generator().manual_seed(config.seed)

    losses: list[float] = []
    model.train()
    if config.max_steps is None:
        for _ in range(config.epochs):
            perm = torch.randperm(n_samples, generator=gen) if n_samples > bs else torch.arange(n_samples)
            epoch_loss = torch.zeros((), device=device)
            n_steps = 0
            for start in range(0, n_samples, bs):
                idx = perm[start : start + bs]
                if pre_staged:
                    idx_dev = idx.to(device)
                    xb, yb, vb = x_all[idx_dev], y_all[idx_dev], v_all[idx_dev]
                else:
                    xb, yb, vb = x_all[idx].to(device), y_all[idx].to(device), v_all[idx].to(device)
                optimizer.zero_grad()
                logits = model(xb)
                bce = (F.binary_cross_entropy_with_logits(logits, yb, reduction="none") * vb).sum() / vb.sum().clamp_min(1.0)
                dice = _soft_dice_loss(logits, yb, vb)
                loss = 0.5 * bce + 0.5 * dice
                loss.backward()
                optimizer.step()
                epoch_loss = epoch_loss + loss.detach()
                n_steps += 1
            if scheduler is not None:
                scheduler.step()
            losses.append(float((epoch_loss / max(1, n_steps)).cpu().item()))
    else:
        steps_done = 0
        while steps_done < config.max_steps:
            perm = torch.randperm(n_samples, generator=gen) if n_samples > bs else torch.arange(n_samples)
            for start in range(0, n_samples, bs):
                if steps_done >= config.max_steps:
                    break
                idx = perm[start : start + bs]
                if pre_staged:
                    idx_dev = idx.to(device)
                    xb, yb, vb = x_all[idx_dev], y_all[idx_dev], v_all[idx_dev]
                else:
                    xb, yb, vb = x_all[idx].to(device), y_all[idx].to(device), v_all[idx].to(device)
                optimizer.zero_grad()
                logits = model(xb)
                bce = (F.binary_cross_entropy_with_logits(logits, yb, reduction="none") * vb).sum() / vb.sum().clamp_min(1.0)
                dice = _soft_dice_loss(logits, yb, vb)
                loss = 0.5 * bce + 0.5 * dice
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                losses.append(float(loss.detach().cpu().item()))
                steps_done += 1

    if pre_staged:
        del x_all, y_all, v_all
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
    n_iters = config.max_steps if config.max_steps is not None else config.epochs
    for _ in range(n_iters):
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = F.binary_cross_entropy_with_logits(logits, y_batch, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    return model.export_to_wsp_dict(), losses
