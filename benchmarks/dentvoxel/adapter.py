"""BYOD NIfTI adapter & HU calibration audit for DentVoxel 3D CBCT (`dentvoxel_seg`, `dentvoxel_cls`).

Supports reading directly from `DentVoxel.zip` or an extracted `DentVoxel_Dataset/` folder.
Verified DentVoxel v2.0 schema:
- 100 paired scans: `image/img0001.nii.gz` <-> `label/label0001.nii.gz` ... `0100`
- Calibrated Hounsfield Units (min=-1000, p99~2100, isotropic 0.3 mm voxels)
- Labels from `dataset_DentVoxel.json`:
  - `3..34`: 32 FDI permanent teeth (`tooth_11`..`tooth_48`)
  - `37, 38`: `r_maxillary_sinus`, `l_maxillary_sinus`
"""

from __future__ import annotations

import gzip
import re
import struct
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import zoom

from wsp_core.evaluation.classification import ClassificationSample
from wsp_core.evaluation.segmentation import SegmentationSample
from wsp_core.guards import require, require_split_allowed
from wsp_core.schemas import strict_json_loads

HU_LOW = -1000.0
HU_HIGH = 3000.0
DENTVOXEL_TOOTH_IDS = tuple(range(3, 35))
DENTVOXEL_SINUS_IDS = (37, 38)


def normalize_cbct_volume(volume: np.ndarray, mode: str = "hu_window") -> np.ndarray:
    """Normalize 3D CBCT intensities to [0, 1] using dental HU windowing or percentile fallback."""
    vol = np.asarray(volume, dtype=np.float32)
    if mode == "hu_window":
        return np.clip((vol - HU_LOW) / (HU_HIGH - HU_LOW), 0.0, 1.0).astype(np.float32)
    if mode == "percentile":
        lo, hi = np.percentile(vol, (0.5, 99.5))
        if hi <= lo:
            raise ValueError("CBCT volume has no usable intensity range.")
        return np.clip((vol - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    raise ValueError(f"Unknown CBCT normalization mode: {mode}")


def audit_intensity_calibration(volumes: list[np.ndarray]) -> dict[str, Any]:
    """Audit whether CBCT volumes exhibit calibrated Hounsfield Units vs. uncalibrated grayscale."""
    require(len(volumes) > 0, "Intensity audit requires at least one CBCT volume.")
    p01_list, p50_list, p99_list, min_list, max_list = [], [], [], [], []
    for vol in volumes:
        arr = np.asarray(vol, dtype=np.float32)
        p01, p50, p99 = np.percentile(arr, (1.0, 50.0, 99.0))
        p01_list.append(float(p01))
        p50_list.append(float(p50))
        p99_list.append(float(p99))
        min_list.append(float(arr.min()))
        max_list.append(float(arr.max()))

    appears_hu = bool(np.median(p01_list) <= -500.0 and np.median(p99_list) >= 800.0)
    return {
        "n_audited": len(volumes),
        "median_min": float(np.median(min_list)),
        "median_p01": float(np.median(p01_list)),
        "median_p50": float(np.median(p50_list)),
        "median_p99": float(np.median(p99_list)),
        "median_max": float(np.median(max_list)),
        "appears_hu_calibrated": appears_hu,
        "recommended_mode": "hu_window" if appears_hu else "percentile",
    }


def make_dentvoxel_scan_splits(
    scan_ids: list[str],
    *,
    n_train: int = 60,
    n_val: int = 20,
    seed: int = 2026,
) -> dict[str, Any]:
    """Create deterministic 60/20/20 scan-level split plan (never leaks slices across splits)."""
    unique = sorted(set(scan_ids))
    require(len(unique) >= n_train + n_val + 1, "Not enough scans for requested split counts.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique).tolist()
    rows = []
    for idx, sid in enumerate(perm):
        split = "train" if idx < n_train else ("val" if idx < n_train + n_val else "test")
        rows.append({"scan_id": sid, "split": split})
    return {"dataset": "dentvoxel", "seed": seed, "rows": rows}


def prepare_cbct_3d_canvas(
    volume: np.ndarray,
    tooth_mask: np.ndarray,
    canvas: tuple[int, int, int] = (96, 128, 128),
    norm_mode: str = "hu_window",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aspect-preserving 3D resize & centering onto (D, H, W) canvas divisible by 4."""
    vol = normalize_cbct_volume(volume, mode=norm_mode)
    msk = (np.asarray(tooth_mask) > 0).astype(np.uint8)
    require(vol.shape == msk.shape and vol.ndim == 3, "CBCT volume and mask must be matching 3D arrays.")
    cd, ch, cw = map(int, canvas)
    require(cd % 4 == 0 and ch % 4 == 0 and cw % 4 == 0, "3D canvas dims must be divisible by 4.")

    ratio = min(cd / vol.shape[0], ch / vol.shape[1], cw / vol.shape[2])
    rd = max(1, min(cd, round(vol.shape[0] * ratio)))
    rh = max(1, min(ch, round(vol.shape[1] * ratio)))
    rw = max(1, min(cw, round(vol.shape[2] * ratio)))
    factors = (rd / vol.shape[0], rh / vol.shape[1], rw / vol.shape[2])

    resized_vol = zoom(vol, factors, order=1).astype(np.float32)[:cd, :ch, :cw]
    resized_msk = (zoom(msk, factors, order=0) > 0)[:cd, :ch, :cw]
    rd, rh, rw = resized_vol.shape

    dz, dy, dx = (cd - rd) // 2, (ch - rh) // 2, (cw - rw) // 2
    img_canvas = np.zeros((cd, ch, cw), dtype=np.float32)
    msk_canvas = np.zeros((cd, ch, cw), dtype=bool)
    val_canvas = np.zeros((cd, ch, cw), dtype=bool)
    img_canvas[dz : dz + rd, dy : dy + rh, dx : dx + rw] = resized_vol
    msk_canvas[dz : dz + rd, dy : dy + rh, dx : dx + rw] = resized_msk
    val_canvas[dz : dz + rd, dy : dy + rh, dx : dx + rw] = True
    return img_canvas, msk_canvas, val_canvas


def extract_axial_sinus_slices(
    scan_id: str,
    volume: np.ndarray,
    sinus_mask: np.ndarray,
    *,
    norm_mode: str = "hu_window",
    size: tuple[int, int] = (64, 64),
    stride: int = 4,
    min_sinus_voxels: int = 50,
) -> list[ClassificationSample]:
    """Extract standardized 64x64 axial slices labeled for maxillary sinus presence."""
    vol = normalize_cbct_volume(volume, mode=norm_mode)
    msk = np.asarray(sinus_mask) > 0
    require(vol.shape == msk.shape and vol.ndim == 3, "Volume and sinus mask must match in 3D.")
    samples: list[ClassificationSample] = []
    h_out, w_out = size
    for z in range(0, vol.shape[0], max(1, stride)):
        sl = vol[z]
        ratio = min(h_out / sl.shape[0], w_out / sl.shape[1])
        rh = max(1, min(h_out, round(sl.shape[0] * ratio)))
        rw = max(1, min(w_out, round(sl.shape[1] * ratio)))
        resized = np.asarray(
            Image.fromarray(sl).resize((rw, rh), resample=Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        canvas = np.zeros((h_out, w_out), dtype=np.float32)
        yy, xx = (h_out - rh) // 2, (w_out - rw) // 2
        canvas[yy : yy + rh, xx : xx + rw] = resized
        label = 1 if int(np.count_nonzero(msk[z])) >= min_sinus_voxels else 0
        samples.append(ClassificationSample(sample_id=f"{scan_id}__z{z:04d}", group_id=scan_id, image=canvas, label=label))
    return samples


def read_nii_gz_bytes(raw_gz_bytes: bytes) -> np.ndarray:
    """Read a NIfTI-1 `.nii.gz` byte payload to `(Z, Y, X)` without requiring `nibabel`."""
    buf = gzip.decompress(raw_gz_bytes)
    endian = "<" if struct.unpack("<i", buf[:4])[0] == 348 else ">"
    dims = struct.unpack(endian + "8h", buf[40:56])
    ndim = dims[0]
    shape = dims[1 : 1 + ndim]
    datatype = struct.unpack(endian + "h", buf[70:72])[0]
    vox_offset = int(struct.unpack(endian + "f", buf[108:112])[0])
    scl_slope, scl_inter = struct.unpack(endian + "2f", buf[112:120])
    dt_map = {2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8", 256: "i1", 512: "u2", 768: "u4"}
    arr = np.frombuffer(
        buf,
        dtype=np.dtype(endian + dt_map[datatype]),
        offset=vox_offset,
        count=int(np.prod(shape)),
    ).reshape(shape, order="F")
    if scl_slope not in (0.0, 1.0) or scl_inter != 0.0:
        arr = arr.astype(np.float32) * (scl_slope if scl_slope != 0.0 else 1.0) + scl_inter
    return np.transpose(arr, (2, 1, 0))


def _scan_key(filename: str) -> str:
    match = re.search(r"(\d+)", filename)
    return match.group(1) if match else Path(filename).stem


def _resolve_dentvoxel_pairs(source_path: Path) -> tuple[str, list[tuple[str, str, str]]]:
    """Pair DentVoxel `imgXXXX.nii.gz` with `labelXXXX.nii.gz` in a ZIP or extracted folder."""
    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(source_path, "r") as zf:
            names = zf.namelist()
        imgs = [n for n in names if "/image/" in n and n.endswith(".nii.gz") and not Path(n).name.startswith("._")]
        lbls = [n for n in names if "/label/" in n and n.endswith(".nii.gz") and not Path(n).name.startswith("._")]
        lbl_by_num = {_scan_key(Path(n).name): n for n in lbls}
        pairs = []
        for img_n in sorted(imgs):
            key = _scan_key(Path(img_n).name)
            if key in lbl_by_num:
                pairs.append((f"scan_{key}", img_n, lbl_by_num[key]))
        return "zip", pairs

    base = source_path / "DentVoxel_Dataset" if (source_path / "DentVoxel_Dataset").is_dir() else source_path
    img_dir = base / "image" if (base / "image").is_dir() else base / "images"
    lbl_dir = base / "label" if (base / "label").is_dir() else base / "labels"
    require(img_dir.is_dir() and lbl_dir.is_dir(), f"Missing image/ and label/ under {base}")
    lbl_by_num = {
        _scan_key(p.name): str(p)
        for p in lbl_dir.glob("*.nii.gz")
        if not p.name.startswith("._")
    }
    pairs = []
    for img_p in sorted(img_dir.glob("*.nii.gz")):
        if img_p.name.startswith("._"):
            continue
        key = _scan_key(img_p.name)
        if key in lbl_by_num:
            pairs.append((f"scan_{key}", str(img_p), lbl_by_num[key]))
    return "dir", pairs


def _read_pair(mode: str, source_path: Path, img_ref: str, lbl_ref: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "zip":
        with zipfile.ZipFile(source_path, "r") as zf:
            return read_nii_gz_bytes(zf.read(img_ref)), read_nii_gz_bytes(zf.read(lbl_ref))
    return read_nii_gz_bytes(Path(img_ref).read_bytes()), read_nii_gz_bytes(Path(lbl_ref).read_bytes())


def load_dentvoxel_seg(
    source_path: str | Path,
    split_plan_path: str | Path,
    *,
    split: str = "val",
    canvas: tuple[int, int, int] = (96, 128, 128),
    norm_mode: str = "hu_window",
    allow_test: bool = False,
    authorization_file: str | Path | None = None,
) -> list[SegmentationSample]:
    """Load scan-split DentVoxel 3D tooth-union samples from `DentVoxel.zip` or extracted folder."""
    require_split_allowed(split, task="dentvoxel_seg", allow_test=allow_test, authorization_file=authorization_file)
    src = Path(source_path)
    plan = strict_json_loads(Path(split_plan_path).read_bytes())
    allowed_scans = {r["scan_id"] for r in plan["rows"] if r["split"] == split}
    mode, pairs = _resolve_dentvoxel_pairs(src)

    samples: list[SegmentationSample] = []
    for scan_id, img_ref, lbl_ref in pairs:
        if scan_id not in allowed_scans:
            continue
        vol, lbl = _read_pair(mode, src, img_ref, lbl_ref)
        tooth_mask = np.isin(lbl, DENTVOXEL_TOOTH_IDS)
        img_c, msk_c, val_c = prepare_cbct_3d_canvas(vol, tooth_mask, canvas=canvas, norm_mode=norm_mode)
        samples.append(SegmentationSample(sample_id=scan_id, group_id=scan_id, image=img_c, target=msk_c, valid=val_c))
    require(len(samples) > 0, f"No DentVoxel 3D segmentation scans loaded for split '{split}'.")
    return samples


def load_dentvoxel_cls(
    source_path: str | Path,
    split_plan_path: str | Path,
    *,
    split: str = "val",
    norm_mode: str = "hu_window",
    stride: int = 4,
    allow_test: bool = False,
    authorization_file: str | Path | None = None,
) -> list[ClassificationSample]:
    """Load strictly scan-split axial maxillary-sinus classification slices from DentVoxel."""
    require_split_allowed(split, task="dentvoxel_cls", allow_test=allow_test, authorization_file=authorization_file)
    src = Path(source_path)
    plan = strict_json_loads(Path(split_plan_path).read_bytes())
    allowed_scans = {r["scan_id"] for r in plan["rows"] if r["split"] == split}
    mode, pairs = _resolve_dentvoxel_pairs(src)

    samples: list[ClassificationSample] = []
    for scan_id, img_ref, lbl_ref in pairs:
        if scan_id not in allowed_scans:
            continue
        vol, lbl = _read_pair(mode, src, img_ref, lbl_ref)
        sinus_mask = np.isin(lbl, DENTVOXEL_SINUS_IDS)
        samples.extend(extract_axial_sinus_slices(scan_id, vol, sinus_mask, norm_mode=norm_mode, stride=stride))
    require(len(samples) > 0, f"No DentVoxel axial classification slices loaded for split '{split}'.")
    return samples


def verify_dentvoxel_directory(source_path: Path, max_scans: int = 5) -> dict[str, Any]:
    """Verify DentVoxel ZIP or extracted folder pairs and run HU calibration audit."""
    mode, pairs = _resolve_dentvoxel_pairs(source_path)
    require(len(pairs) > 0, f"No matching DentVoxel image/label NIfTI pairs found in {source_path}")
    audited_vols = [_read_pair(mode, source_path, img_ref, lbl_ref)[0] for _, img_ref, lbl_ref in pairs[: max(1, max_scans)]]
    hu_report = audit_intensity_calibration(audited_vols)
    return {
        "dataset": "dentvoxel",
        "format": mode,
        "matched_nifti_pairs": len(pairs),
        "tooth_label_ids": f"{DENTVOXEL_TOOTH_IDS[0]}..{DENTVOXEL_TOOTH_IDS[-1]}",
        "sinus_label_ids": list(DENTVOXEL_SINUS_IDS),
        "intensity_audit": hu_report,
    }
