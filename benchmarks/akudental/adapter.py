"""BYOD dataset adapter for AKUDENTAL panoramic radiographs (`akudental_seg`, `akudental_cls`)."""

from __future__ import annotations

import csv
import zipfile
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from wsp_core.evaluation.classification import ClassificationSample
from wsp_core.evaluation.segmentation import SegmentationSample
from wsp_core.guards import require, require_split_allowed
from wsp_core.schemas import sha256_file, strict_json_loads

RESTORATION_KEYWORDS = ("implant", "crown", "filling", "bridge")


def prepare_panorama(
    image_source: Any,
    mask_source: Any,
    canvas: tuple[int, int] = (384, 768),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact Study 1 preprocessing: 0.5–99.5 percentile scale, centered aspect-preserving resize."""
    with Image.open(image_source) as img:
        gray = np.asarray(img.convert("L"), dtype=np.float32)
    with Image.open(mask_source) as msk:
        target = np.asarray(msk, dtype=np.uint8) > 0
    if gray.shape != target.shape:
        raise ValueError("Image and mask geometry differ.")
    height, width = map(int, canvas)
    if height % 16 or width % 16 or min(height, width) < 32:
        raise ValueError("Canvas sides must be >= 32 and divisible by 16.")

    lower, upper = np.percentile(gray, (0.5, 99.5))
    if upper <= lower:
        raise ValueError("Image has no usable grayscale range.")
    scaled = np.clip((gray - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)

    ratio = min(height / gray.shape[0], width / gray.shape[1])
    resized_h = max(1, min(height, round(gray.shape[0] * ratio)))
    resized_w = max(1, min(width, round(gray.shape[1] * ratio)))
    resized = np.asarray(
        Image.fromarray(scaled).resize((resized_w, resized_h), resample=Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    mask = np.asarray(
        Image.fromarray(target.astype(np.uint8)).resize((resized_w, resized_h), resample=Image.Resampling.NEAREST),
        dtype=np.uint8,
    ) > 0

    yy = (height - resized_h) // 2
    xx = (width - resized_w) // 2
    image_canvas = np.zeros((height, width), dtype=np.float32)
    target_canvas = np.zeros((height, width), dtype=bool)
    valid_canvas = np.zeros((height, width), dtype=bool)
    image_canvas[yy : yy + resized_h, xx : xx + resized_w] = resized
    target_canvas[yy : yy + resized_h, xx : xx + resized_w] = mask
    valid_canvas[yy : yy + resized_h, xx : xx + resized_w] = True
    return image_canvas, target_canvas, valid_canvas


def prepare_cls_canvas(image_source: Any, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Standardized 64x64 grayscale canvas for `akudental_cls` (`cls-lenet2d-v1`)."""
    with Image.open(image_source) as img:
        gray = np.asarray(img.convert("L"), dtype=np.float32)
    lower, upper = np.percentile(gray, (0.5, 99.5))
    if upper <= lower:
        raise ValueError("Image has no usable grayscale range.")
    scaled = np.clip((gray - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)
    height, width = size
    ratio = min(height / gray.shape[0], width / gray.shape[1])
    rh = max(1, min(height, round(gray.shape[0] * ratio)))
    rw = max(1, min(width, round(gray.shape[1] * ratio)))
    resized = np.asarray(
        Image.fromarray(scaled).resize((rw, rh), resample=Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    canvas = np.zeros((height, width), dtype=np.float32)
    yy, xx = (height - rh) // 2, (width - rw) // 2
    canvas[yy : yy + rh, xx : xx + rw] = resized
    return canvas


def load_akudental_seg_zip(
    archive_path: str | Path,
    split_plan_path: str | Path,
    *,
    split: str = "val",
    canvas: tuple[int, int] = (384, 768),
    allow_test: bool = False,
    authorization_file: str | Path | None = None,
) -> list[SegmentationSample]:
    """Load AKUDENTAL segmentation cohort from the verified ZIP + split_plan.private.json."""
    require_split_allowed(split, task="akudental_seg", allow_test=allow_test, authorization_file=authorization_file)
    archive_path = Path(archive_path)
    split_plan_path = Path(split_plan_path)
    plan = strict_json_loads(split_plan_path.read_bytes())
    assignments = {r["image_id"]: r["group"] for r in plan["rows"] if r["split"] == split}
    require(len(assignments) > 0, f"No AKUDENTAL rows found for split '{split}'.")

    samples: list[SegmentationSample] = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        manifest = strict_json_loads(zf.read("manifest.json"))
        require(
            manifest.get("source_coco_foreground") == list(range(1, 33)),
            "Archive tooth-union foreground classes differ from classes 1–32.",
        )
        reader = csv.DictReader(StringIO(zf.read("images.csv").decode("utf-8-sig")))
        for row in reader:
            cid = row["canonical_image_id"]
            if cid not in assignments:
                continue
            with zf.open(row["image_path"]) as img_f, zf.open(row["mask_path"]) as msk_f:
                img, target, valid = prepare_panorama(img_f, msk_f, canvas=canvas)
            samples.append(
                SegmentationSample(
                    sample_id=cid,
                    group_id=assignments[cid],
                    image=img,
                    target=target,
                    valid=valid,
                )
            )
    require(len(samples) == len(assignments), "Loaded AKUDENTAL sample count disagrees with split plan.")
    return samples


def load_akudental_cls_coco(
    coco_json_path: str | Path,
    images_dir: str | Path,
    split_assignments: dict[str, tuple[str, str]],
    *,
    split: str = "val",
    size: tuple[int, int] = (64, 64),
    allow_test: bool = False,
    authorization_file: str | Path | None = None,
) -> list[ClassificationSample]:
    """Build `akudental_cls` binary restoration-presence samples from an AKUDENTAL COCO annotation file.

    `split_assignments` maps filename/image_id -> (split_name, group_id).
    """
    require_split_allowed(split, task="akudental_cls", allow_test=allow_test, authorization_file=authorization_file)
    coco = strict_json_loads(Path(coco_json_path).read_bytes())
    restoration_cat_ids = {
        cat["id"]
        for cat in coco.get("categories", [])
        if any(kw in str(cat.get("name", "")).lower() for kw in RESTORATION_KEYWORDS)
    }
    require(len(restoration_cat_ids) > 0, "No restoration categories (implant/crown/filling/bridge) found in COCO JSON.")

    has_restoration: set[Any] = {
        ann["image_id"] for ann in coco.get("annotations", []) if ann.get("category_id") in restoration_cat_ids
    }
    images_dir = Path(images_dir)
    samples: list[ClassificationSample] = []
    for img_meta in coco.get("images", []):
        key = str(img_meta["file_name"])
        if key not in split_assignments:
            continue
        assigned_split, group_id = split_assignments[key]
        if assigned_split != split:
            continue
        img_arr = prepare_cls_canvas(images_dir / key, size=size)
        samples.append(
            ClassificationSample(
                sample_id=key,
                group_id=group_id,
                image=img_arr,
                label=1 if img_meta["id"] in has_restoration else 0,
            )
        )
    require(len(samples) > 0, f"No AKUDENTAL classification samples found for split '{split}'.")
    return samples


def verify_akudental_source(path: Path) -> dict[str, Any]:
    """Read-only verification of an AKUDENTAL ZIP archive or directory."""
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            require("manifest.json" in names and "images.csv" in names, "ZIP missing manifest.json or images.csv.")
            manifest = strict_json_loads(zf.read("manifest.json"))
            rows = list(csv.DictReader(StringIO(zf.read("images.csv").decode("utf-8-sig"))))
        return {
            "dataset": "akudental",
            "format": "init_zip",
            "sha256": sha256_file(path),
            "image_count": len(rows),
            "foreground_classes": manifest.get("source_coco_foreground"),
        }
    require(path.is_dir(), f"AKUDENTAL path not found: {path}")
    jsons = sorted(path.glob("*.json"))
    return {
        "dataset": "akudental",
        "format": "directory",
        "json_files_count": len(jsons),
    }
