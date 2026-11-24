"""BYOD NIfTI adapter & HU calibration audit for DentVoxel 3D CBCT (`dentvoxel_seg`, `dentvoxel_cls`)."""

from __future__ import annotations

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

    # Calibrated dental CBCT in HU typically has air around -1000 and bright bone/enamel > +1000.
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

    resized_vol = zoom(vol, factors, order=1).astype(np.float32)
    resized_msk = zoom(msk, factors, order=0) > 0

    # Clamp in case scipy.ndimage.zoom rounds +/-1 voxel on edge shapes
    resized_vol = resized_vol[:cd, :ch, :cw]
    resized_msk = resized_msk[:cd, :ch, :cw]
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
    """Extract standardized 64x64 axial slices labeled for maxillary sinus presence.

    Every slice inherits `group_id = scan_id` to enforce strict scan-level split isolation.
    """
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
        samples.append(
            ClassificationSample(
                sample_id=f"{scan_id}__z{z:04d}",
                group_id=scan_id,
                image=canvas,
                label=label,
            )
        )
    return samples


def _load_nifti_array(path: Path) -> np.ndarray:
    import nibabel as nib

    img = nib.load(str(path))
    # Standardize orientation to RAS+ and reorder to (Z/axial, Y, X)
    canonical = nib.as_closest_canonical(img)
    arr = np.asarray(canonical.dataobj)
    return np.transpose(arr, (2, 1, 0))


def _resolve_nifti_pairs(data_dir: Path) -> list[tuple[str, Path, Path]]:
    img_dir = data_dir / "images"
    lbl_dir = data_dir / "labels"
    require(img_dir.is_dir() and lbl_dir.is_dir(), "DentVoxel directory must contain images/ and labels/.")
    pairs: list[tuple[str, Path, Path]] = []
    for img_path in sorted(list(img_dir.glob("*.nii.gz")) + list(img_dir.glob("*.nii"))):
        stem = img_path.name.replace(".nii.gz", "").replace(".nii", "")
        lbl_cand = lbl_dir / img_path.name
        if not lbl_cand.is_file():
            alt = lbl_dir / (f"{stem}.nii.gz" if img_path.name.endswith(".nii") else f"{stem}.nii")
            if alt.is_file():
                lbl_cand = alt
        if lbl_cand.is_file():
            pairs.append((stem, img_path, lbl_cand))
    return pairs


def load_dentvoxel_seg(
    data_dir: str | Path,
    split_plan_path: str | Path,
    *,
    split: str = "val",
    canvas: tuple[int, int, int] = (96, 128, 128),
    norm_mode: str = "hu_window",
    allow_test: bool = False,
    authorization_file: str | Path | None = None,
) -> list[SegmentationSample]:
    """Load scan-split DentVoxel 3D tooth-union samples from NIfTI files."""
    require_split_allowed(split, task="dentvoxel_seg", allow_test=allow_test, authorization_file=authorization_file)
    root = Path(data_dir)
    label_map = strict_json_loads((root / "label_map.json").read_bytes())
    tooth_ids = set(int(x) for x in label_map["tooth_ids"])
    plan = strict_json_loads(Path(split_plan_path).read_bytes())
    allowed_scans = {r["scan_id"] for r in plan["rows"] if r["split"] == split}

    samples: list[SegmentationSample] = []
    for scan_id, img_path, lbl_path in _resolve_nifti_pairs(root):
        if scan_id not in allowed_scans:
            continue
        vol = _load_nifti_array(img_path)
        lbl = _load_nifti_array(lbl_path)
        tooth_mask = np.isin(lbl, list(tooth_ids))
        img_c, msk_c, val_c = prepare_cbct_3d_canvas(vol, tooth_mask, canvas=canvas, norm_mode=norm_mode)
        samples.append(
            SegmentationSample(sample_id=scan_id, group_id=scan_id, image=img_c, target=msk_c, valid=val_c)
        )
    require(len(samples) > 0, f"No DentVoxel 3D segmentation scans loaded for split '{split}'.")
    return samples


def load_dentvoxel_cls(
    data_dir: str | Path,
    split_plan_path: str | Path,
    *,
    split: str = "val",
    norm_mode: str = "hu_window",
    stride: int = 4,
    allow_test: bool = False,
    authorization_file: str | Path | None = None,
) -> list[ClassificationSample]:
    """Load strictly scan-split axial maxillary-sinus classification slices from DentVoxel NIfTI scans."""
    require_split_allowed(split, task="dentvoxel_cls", allow_test=allow_test, authorization_file=authorization_file)
    root = Path(data_dir)
    label_map = strict_json_loads((root / "label_map.json").read_bytes())
    sinus_ids = set(int(x) for x in label_map["sinus_ids"])
    plan = strict_json_loads(Path(split_plan_path).read_bytes())
    allowed_scans = {r["scan_id"] for r in plan["rows"] if r["split"] == split}

    samples: list[ClassificationSample] = []
    for scan_id, img_path, lbl_path in _resolve_nifti_pairs(root):
        if scan_id not in allowed_scans:
            continue
        vol = _load_nifti_array(img_path)
        lbl = _load_nifti_array(lbl_path)
        sinus_mask = np.isin(lbl, list(sinus_ids))
        samples.extend(
            extract_axial_sinus_slices(scan_id, vol, sinus_mask, norm_mode=norm_mode, stride=stride)
        )
    require(len(samples) > 0, f"No DentVoxel axial classification slices loaded for split '{split}'.")
    return samples


def verify_dentvoxel_directory(data_dir: Path, max_scans: int = 10) -> dict[str, Any]:
    """Verify DentVoxel NIfTI pairs and run HU calibration audit on up to `max_scans` volumes."""
    pairs = _resolve_nifti_pairs(data_dir)
    require(len(pairs) > 0, f"No matching NIfTI image/label pairs found in {data_dir}")
    audited_vols = [_load_nifti_array(img_p) for _, img_p, _ in pairs[: max(1, max_scans)]]
    hu_report = audit_intensity_calibration(audited_vols)
    return {
        "dataset": "dentvoxel",
        "matched_nifti_pairs": len(pairs),
        "label_map_present": (data_dir / "label_map.json").is_file(),
        "intensity_audit": hu_report,
    }
