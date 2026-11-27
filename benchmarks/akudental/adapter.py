"""BYOD dataset adapter for official public AKUDENTAL panoramic radiographs (`akudental_seg`, `akudental_cls`).

Loads directly from the official public AKUDENTAL directory (`akudental_instances.json` + `images/`),
rasterizing FDI permanent tooth classes 1–32 for `akudental_seg` and using restorative classes 33–35
(`Bridge`, `Filling - Crown`, `Implant`) for `akudental_cls`.

Also retains `load_akudental_seg_zip` for legacy Study 1 archive compatibility.
"""

from __future__ import annotations

import csv
import zipfile
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from wsp_core.evaluation.classification import ClassificationSample
from wsp_core.evaluation.segmentation import SegmentationSample
from wsp_core.guards import require, require_split_allowed
from wsp_core.schemas import sha256_file, strict_json_loads

TOOTH_CATEGORY_IDS = frozenset(range(1, 33))
RESTORATION_KEYWORDS = ("implant", "crown", "filling", "bridge")


def prepare_panorama(
    image_source: Any,
    mask_source: Any,
    canvas: tuple[int, int] = (384, 768),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-image 0.5–99.5 percentile scaling, centered aspect-preserving resize, valid mask."""
    if isinstance(image_source, np.ndarray):
        gray = image_source.astype(np.float32, copy=False)
    else:
        with Image.open(image_source) as img:
            gray = np.asarray(img.convert("L"), dtype=np.float32)

    if isinstance(mask_source, np.ndarray):
        target = (mask_source[..., :3].max(axis=-1) > 0) if mask_source.ndim == 3 else (mask_source > 0)
    else:
        with Image.open(mask_source) as msk:
            target = np.asarray(msk.convert("L"), dtype=np.uint8) > 0

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
    mask = (
        np.asarray(
            Image.fromarray(target.astype(np.uint8)).resize((resized_w, resized_h), resample=Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
        > 0
    )

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


def make_akudental_public_splits(
    coco: dict[str, Any],
    *,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    seed: int = 2026,
) -> dict[str, str]:
    """Deterministic stratified 60/20/20 split by restoration presence across the 333 public images."""
    rest_ids = {
        c["id"]
        for c in coco.get("categories", [])
        if any(kw in str(c.get("name", "")).lower() for kw in RESTORATION_KEYWORDS)
    }
    pos_img_ids = {a["image_id"] for a in coco.get("annotations", []) if a.get("category_id") in rest_ids}

    pos_names = sorted(str(im["file_name"]) for im in coco["images"] if im["id"] in pos_img_ids)
    neg_names = sorted(str(im["file_name"]) for im in coco["images"] if im["id"] not in pos_img_ids)

    rng = np.random.default_rng(seed)
    assignments: dict[str, str] = {}
    for bucket in (pos_names, neg_names):
        perm = rng.permutation(bucket).tolist()
        n_train = max(1, int(round(len(perm) * train_frac)))
        n_val = max(1, int(round(len(perm) * val_frac)))
        if n_train + n_val >= len(perm):
            n_val = max(1, len(perm) - n_train - 1)
        for idx, fname in enumerate(perm):
            assignments[fname] = "train" if idx < n_train else ("val" if idx < n_train + n_val else "test")
    return assignments


def _rasterize_tooth_union(height: int, width: int, anns: list[dict[str, Any]]) -> np.ndarray:
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for ann in anns:
        if ann.get("category_id") not in TOOTH_CATEGORY_IDS:
            continue
        seg = ann.get("segmentation", [])
        if isinstance(seg, list):
            for poly in seg:
                if len(poly) >= 6:
                    pts = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly), 2)]
                    draw.polygon(pts, fill=1)
    return np.asarray(canvas, dtype=np.uint8) > 0


def load_akudental_seg_public(
    data_dir: str | Path,
    *,
    split: str = "val",
    canvas: tuple[int, int] = (384, 768),
    seed: int = 2026,
    allow_test: bool = False,
    authorization_file: str | Path | None = None,
) -> list[SegmentationSample]:
    """Load AKUDENTAL segmentation samples directly from the official public folder."""
    require_split_allowed(split, task="akudental_seg", allow_test=allow_test, authorization_file=authorization_file)
    root = Path(data_dir)
    coco = strict_json_loads((root / "akudental_instances.json").read_bytes())
    splits = make_akudental_public_splits(coco, seed=seed)

    anns_by_img: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_img[ann["image_id"]].append(ann)

    images_dir = root / "images"
    samples: list[SegmentationSample] = []
    for im in sorted(coco["images"], key=lambda x: str(x["file_name"])):
        fname = str(im["file_name"])
        if splits.get(fname) != split:
            continue
        img_path = images_dir / fname
        with Image.open(img_path) as opened:
            gray = np.asarray(opened.convert("L"), dtype=np.float32)
        mask = _rasterize_tooth_union(gray.shape[0], gray.shape[1], anns_by_img[im["id"]])
        img_c, msk_c, val_c = prepare_panorama(gray, mask, canvas=canvas)
        samples.append(
            SegmentationSample(
                sample_id=Path(fname).stem,
                group_id=Path(fname).stem,
                image=img_c,
                target=msk_c,
                valid=val_c,
            )
        )
    require(len(samples) > 0, f"No AKUDENTAL segmentation samples found for split '{split}'.")
    return samples


def load_akudental_cls_public(
    data_dir: str | Path,
    *,
    split: str = "val",
    size: tuple[int, int] = (64, 64),
    seed: int = 2026,
    allow_test: bool = False,
    authorization_file: str | Path | None = None,
) -> list[ClassificationSample]:
    """Load AKUDENTAL restoration/implant classification samples from the official public folder."""
    require_split_allowed(split, task="akudental_cls", allow_test=allow_test, authorization_file=authorization_file)
    root = Path(data_dir)
    coco = strict_json_loads((root / "akudental_instances.json").read_bytes())
    splits = make_akudental_public_splits(coco, seed=seed)

    rest_ids = {
        c["id"]
        for c in coco.get("categories", [])
        if any(kw in str(c.get("name", "")).lower() for kw in RESTORATION_KEYWORDS)
    }
    has_restoration = {a["image_id"] for a in coco.get("annotations", []) if a.get("category_id") in rest_ids}

    images_dir = root / "images"
    samples: list[ClassificationSample] = []
    for im in sorted(coco["images"], key=lambda x: str(x["file_name"])):
        fname = str(im["file_name"])
        if splits.get(fname) != split:
            continue
        img_arr = prepare_cls_canvas(images_dir / fname, size=size)
        samples.append(
            ClassificationSample(
                sample_id=Path(fname).stem,
                group_id=Path(fname).stem,
                image=img_arr,
                label=1 if im["id"] in has_restoration else 0,
            )
        )
    require(len(samples) > 0, f"No AKUDENTAL classification samples found for split '{split}'.")
    return samples


def load_akudental_seg_zip(
    archive_path: str | Path,
    split_plan_path: str | Path,
    *,
    split: str = "val",
    canvas: tuple[int, int] = (384, 768),
    allow_test: bool = False,
    authorization_file: str | Path | None = None,
) -> list[SegmentationSample]:
    """Legacy Study 1 ZIP loader (for reproducing the original exploratory archive)."""
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
                SegmentationSample(sample_id=cid, group_id=assignments[cid], image=img, target=target, valid=valid)
            )
    require(len(samples) == len(assignments), "Loaded AKUDENTAL sample count disagrees with split plan.")
    return samples


def verify_akudental_source(path: Path) -> dict[str, Any]:
    """Read-only verification of an official public AKUDENTAL folder or legacy ZIP."""
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
    coco_path = path / "akudental_instances.json"
    require(coco_path.is_file(), f"Missing akudental_instances.json in {path}")
    coco = strict_json_loads(coco_path.read_bytes())
    splits = make_akudental_public_splits(coco)
    counts = {s: sum(v == s for v in splits.values()) for s in ("train", "val", "test")}
    return {
        "dataset": "akudental",
        "format": "official_public_coco",
        "total_images": len(coco["images"]),
        "categories": len(coco["categories"]),
        "public_split_counts": counts,
    }
