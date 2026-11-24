"""BYOD dataset adapter for DenPAR intra-oral periapical radiographs (`denpar_seg`, `denpar_cls`)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.akudental.adapter import prepare_cls_canvas, prepare_panorama
from wsp_core.evaluation.classification import ClassificationSample
from wsp_core.evaluation.segmentation import SegmentationSample
from wsp_core.guards import require, require_split_allowed

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _find_by_stem(folder: Path, stem: str) -> Path:
    for ext in IMAGE_EXTS:
        cand = folder / f"{stem}{ext}"
        if cand.is_file():
            return cand
    direct = folder / stem
    if direct.is_file():
        return direct
    raise FileNotFoundError(f"Could not find file for stem '{stem}' in {folder}")


def make_group_splits(
    group_ids: list[str],
    *,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    seed: int = 2026,
) -> dict[str, str]:
    """Deterministically assign unique group/patient IDs to train/val/test splits."""
    unique = sorted(set(group_ids))
    require(len(unique) >= 3, "Need at least 3 distinct groups to form train/val/test splits.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique).tolist()
    n_train = max(1, int(round(len(perm) * train_frac)))
    n_val = max(1, int(round(len(perm) * val_frac)))
    if n_train + n_val >= len(perm):
        n_val = max(1, len(perm) - n_train - 1)
    splits: dict[str, str] = {}
    for idx, gid in enumerate(perm):
        if idx < n_train:
            splits[gid] = "train"
        elif idx < n_train + n_val:
            splits[gid] = "val"
        else:
            splits[gid] = "test"
    return splits


def load_denpar_seg(
    data_dir: str | Path,
    *,
    split: str = "val",
    canvas: tuple[int, int] = (384, 384),
    allow_test: bool = False,
    authorization_file: str | Path | None = None,
) -> list[SegmentationSample]:
    """Load DenPAR periapical tooth-union segmentation samples for the requested split."""
    require_split_allowed(split, task="denpar_seg", allow_test=allow_test, authorization_file=authorization_file)
    root = Path(data_dir)
    meta_path = root / "metadata.csv"
    require(meta_path.is_file(), f"Missing DenPAR metadata CSV: {meta_path}")

    samples: list[SegmentationSample] = []
    with open(meta_path, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            sid = row["image_id"]
            gid = row.get("group_id") or sid
            img_path = _find_by_stem(root / "images", sid)
            msk_path = _find_by_stem(root / "masks", sid)
            img, target, valid = prepare_panorama(img_path, msk_path, canvas=canvas)
            samples.append(
                SegmentationSample(sample_id=sid, group_id=gid, image=img, target=target, valid=valid)
            )
    require(len(samples) > 0, f"No DenPAR segmentation samples found for split '{split}'.")
    return samples


def load_denpar_cls(
    data_dir: str | Path,
    *,
    split: str = "val",
    size: tuple[int, int] = (64, 64),
    allow_test: bool = False,
    authorization_file: str | Path | None = None,
) -> list[ClassificationSample]:
    """Load DenPAR alveolar bone-loss binary classification samples for the requested split."""
    require_split_allowed(split, task="denpar_cls", allow_test=allow_test, authorization_file=authorization_file)
    root = Path(data_dir)
    meta_path = root / "metadata.csv"
    require(meta_path.is_file(), f"Missing DenPAR metadata CSV: {meta_path}")

    samples: list[ClassificationSample] = []
    with open(meta_path, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            sid = row["image_id"]
            gid = row.get("group_id") or sid
            label = int(row["bone_loss_label"])
            require(label in (0, 1), f"Invalid bone_loss_label for {sid}: {label}")
            img_path = _find_by_stem(root / "images", sid)
            img = prepare_cls_canvas(img_path, size=size)
            samples.append(ClassificationSample(sample_id=sid, group_id=gid, image=img, label=label))
    require(len(samples) > 0, f"No DenPAR classification samples found for split '{split}'.")
    return samples


def verify_denpar_directory(data_dir: Path) -> dict[str, Any]:
    """Read-only structural check of a local DenPAR BYOD folder."""
    require(data_dir.is_dir(), f"DenPAR directory not found: {data_dir}")
    images = [p for p in (data_dir / "images").glob("*") if p.suffix.lower() in IMAGE_EXTS] if (data_dir / "images").is_dir() else []
    masks = [p for p in (data_dir / "masks").glob("*") if p.suffix.lower() in IMAGE_EXTS] if (data_dir / "masks").is_dir() else []
    meta_exists = (data_dir / "metadata.csv").is_file()
    return {
        "dataset": "denpar",
        "images_found": len(images),
        "masks_found": len(masks),
        "metadata_csv_present": meta_exists,
    }
