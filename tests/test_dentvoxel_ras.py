"""Tests for pure-NumPy NIfTI-1 canonical RAS+ reorientation and DentVoxel scan-ID integrity."""

from __future__ import annotations

import gzip
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

from benchmarks.dentvoxel.adapter import (
    _ras_orientation_plan,
    get_dentvoxel_scan_ids,
    load_dentvoxel_cls,
    load_dentvoxel_seg,
    make_dentvoxel_scan_splits,
    read_nii_gz_bytes,
    verify_dentvoxel_directory,
)
from wsp_core.guards import StudyStop


def _make_nii_gz_bytes(
    arr_ijk: np.ndarray,
    *,
    srow_x: tuple[float, float, float, float] = (0.3, 0.0, 0.0, 0.0),
    srow_y: tuple[float, float, float, float] = (0.0, 0.3, 0.0, 0.0),
    srow_z: tuple[float, float, float, float] = (0.0, 0.0, 0.3, 0.0),
    sform_code: int = 1,
    qform_code: int = 0,
    pixdim: tuple[float, float, float, float] = (1.0, 0.3, 0.3, 0.3),
    quatern_bcd: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bytes:
    """Build a minimal valid NIfTI-1 `.nii.gz` byte payload for unit testing."""
    data = np.asarray(arr_ijk, dtype=np.float32)
    assert data.ndim == 3
    hdr = bytearray(352)
    struct.pack_into("<i", hdr, 0, 348)
    struct.pack_into("<8h", hdr, 40, 3, data.shape[0], data.shape[1], data.shape[2], 1, 1, 1, 1)
    struct.pack_into("<h", hdr, 70, 16)  # float32
    struct.pack_into("<h", hdr, 72, 32)  # bitpix
    struct.pack_into("<8f", hdr, 76, pixdim[0], pixdim[1], pixdim[2], pixdim[3], 0.0, 0.0, 0.0, 0.0)
    struct.pack_into("<f", hdr, 108, 352.0)  # vox_offset
    struct.pack_into("<2f", hdr, 112, 1.0, 0.0)  # scl_slope, scl_inter
    struct.pack_into("<2h", hdr, 252, qform_code, sform_code)
    struct.pack_into("<6f", hdr, 256, quatern_bcd[0], quatern_bcd[1], quatern_bcd[2], 0.0, 0.0, 0.0)
    struct.pack_into("<4f", hdr, 280, *srow_x)
    struct.pack_into("<4f", hdr, 296, *srow_y)
    struct.pack_into("<4f", hdr, 312, *srow_z)
    hdr[344:348] = b"n+1\x00"
    raw_bytes = bytes(hdr) + np.asfortranarray(data).tobytes(order="F")
    return gzip.compress(raw_bytes)


def test_nifti_canonical_ras_reorients_lps_lip_and_qform_identically():
    # Canonical RAS+ phantom in (X: L->R, Y: P->A, Z: I->S) with distinct asymmetric landmarks
    # Shape (X=6, Y=8, Z=4)
    ras_xyz = np.full((6, 8, 4), -1000.0, dtype=np.float32)
    ras_xyz[5, 7, 3] = 2500.0  # Right, Anterior, Superior (e.g., upper-right anterior crown)
    ras_xyz[0, 0, 0] = 500.0   # Left, Posterior, Inferior (e.g., lower-left spine)
    ras_xyz[1, 6, 2] = 1800.0  # Left-anterior-mid tooth landmark

    expected_zyx = np.transpose(ras_xyz, (2, 1, 0))

    # 1. Native RAS+
    gz_ras = _make_nii_gz_bytes(ras_xyz)
    out_ras = read_nii_gz_bytes(gz_ras)
    assert np.array_equal(out_ras, expected_zyx)

    # 2. LPS+ (99/100 DentVoxel scans: X flipped, Y flipped, Z preserved)
    lps_xyz = np.flip(np.flip(ras_xyz, axis=0), axis=1)
    gz_lps = _make_nii_gz_bytes(
        lps_xyz,
        srow_x=(-0.3, 0.0, 0.0, 66.0),
        srow_y=(0.0, -0.3, 0.0, 66.0),
        srow_z=(0.0, 0.0, 0.3, -51.6),
    )
    out_lps = read_nii_gz_bytes(gz_lps)
    assert np.array_equal(out_lps, expected_zyx)
    # Confirm disabling canonical_ras produces the unflipped (mismatched) tensor
    assert not np.array_equal(read_nii_gz_bytes(gz_lps, canonical_ras=False), expected_zyx)

    # 3. LIP+ (1/100 DentVoxel scan: axis0=L, axis1=I, axis2=P -> Y and Z swapped & flipped)
    # Transform RAS(x, y, z) -> LIP(i=-x, j=-z, k=-y)
    lip_ijk = np.flip(np.flip(np.flip(np.transpose(ras_xyz, (0, 2, 1)), axis=0), axis=1), axis=2)
    gz_lip = _make_nii_gz_bytes(
        lip_ijk,
        srow_x=(-0.3, 0.0, 0.0, 66.0),
        srow_y=(0.0, 0.0, -0.3, 66.0),
        srow_z=(0.0, -0.3, 0.0, 51.6),
    )
    out_lip = read_nii_gz_bytes(gz_lip)
    assert np.array_equal(out_lip, expected_zyx)

    # 4. Qform-only LPS+ (sform_code=0, qform_code=1, quaternion b=0, c=0, d=1 -> 180 deg around Z)
    gz_qform = _make_nii_gz_bytes(
        lps_xyz,
        sform_code=0,
        qform_code=1,
        quatern_bcd=(0.0, 0.0, 1.0),
        pixdim=(1.0, 0.3, 0.3, 0.3),
    )
    out_qform = read_nii_gz_bytes(gz_qform)
    assert np.array_equal(out_qform, expected_zyx)

    # Verify reported axcodes
    assert _ras_orientation_plan(np.diag([-0.3, -0.3, 0.3]))[2] == "LPS"
    assert _ras_orientation_plan(np.array([[-0.3, 0, 0], [0, 0, -0.3], [0, -0.3, 0]]))[2] == "LIP"
    assert _ras_orientation_plan(np.diag([0.3, 0.3, 0.3]))[2] == "RAS"


def test_dentvoxel_noncontiguous_scan_ids_and_missing_scan_guard(tmp_path: Path):
    zip_path = tmp_path / "DentVoxel_mini.zip"
    # Non-contiguous IDs mirroring DentVoxel's real numbering gap (skipping 0003, including 0006)
    ids = ["0001", "0002", "0004", "0005", "0006"]
    with zipfile.ZipFile(zip_path, "w") as zf:
        for idx, num in enumerate(ids):
            img_ras = np.full((8, 8, 8), -1000.0, dtype=np.float32)
            lbl_ras = np.zeros((8, 8, 8), dtype=np.float32)
            img_ras[2:6, 5:7, 3:5] = 2200.0  # anterior tooth
            lbl_ras[2:6, 5:7, 3:5] = 11.0    # FDI tooth label (in 3..34)
            lbl_ras[2:6, 4:7, 6:8] = 37.0    # superior maxillary sinus (37, 38)
            if idx == 4:
                # Store one scan in LIP to test mixed orientation audit
                img_arr = np.flip(np.flip(np.flip(np.transpose(img_ras, (0, 2, 1)), 0), 1), 2)
                lbl_arr = np.flip(np.flip(np.flip(np.transpose(lbl_ras, (0, 2, 1)), 0), 1), 2)
                kwargs = dict(srow_x=(-0.3, 0, 0, 0), srow_y=(0, 0, -0.3, 0), srow_z=(0, -0.3, 0, 0))
            else:
                img_arr = np.flip(np.flip(img_ras, 0), 1)
                lbl_arr = np.flip(np.flip(lbl_ras, 0), 1)
                kwargs = dict(srow_x=(-0.3, 0, 0, 0), srow_y=(0, -0.3, 0, 0), srow_z=(0, 0, 0.3, 0))
            zf.writestr(f"DentVoxel_Dataset/image/img{num}.nii.gz", _make_nii_gz_bytes(img_arr, **kwargs))
            zf.writestr(f"DentVoxel_Dataset/label/label{num}.nii.gz", _make_nii_gz_bytes(lbl_arr, **kwargs))

    scan_ids = get_dentvoxel_scan_ids(zip_path)
    assert scan_ids == ["scan_0001", "scan_0002", "scan_0004", "scan_0005", "scan_0006"]

    report = verify_dentvoxel_directory(zip_path, max_scans=2)
    assert report["matched_nifti_pairs"] == 5
    assert report["raw_nifti_orientations"] == {"LIP": 1, "LPS": 4}
    assert report["intensity_audit"]["appears_hu_calibrated"] is True

    # Valid split derived from actual IDs loads every scan cleanly
    split_path = tmp_path / "splits.json"
    plan = make_dentvoxel_scan_splits(scan_ids, n_train=2, n_val=2, seed=2026)
    split_path.write_text(json.dumps(plan), encoding="utf-8")
    val_seg = load_dentvoxel_seg(zip_path, split_path, split="val", canvas=(8, 8, 8))
    val_cls = load_dentvoxel_cls(zip_path, split_path, split="val", stride=1)
    assert len(val_seg) == 2
    assert len({s.group_id for s in val_cls}) == 2
    # Confirm tooth mask sits anteriorly (high Y = indices 5..7) in canonical RAS+
    assert int(val_seg[0].target[:, 5:7, :].sum()) > 0
    assert int(val_seg[0].target[:, 0:3, :].sum()) == 0

    # Stale hardcoded split referencing nonexistent scan_0003 MUST fail loudly instead of silently dropping
    bad_split = tmp_path / "bad_splits.json"
    bad_split.write_text(
        json.dumps({"rows": [{"scan_id": "scan_0001", "split": "val"}, {"scan_id": "scan_0003", "split": "val"}]}),
        encoding="utf-8",
    )
    with pytest.raises(StudyStop, match="missing 1 scan"):
        load_dentvoxel_seg(zip_path, bad_split, split="val", canvas=(8, 8, 8))
    with pytest.raises(StudyStop, match="missing 1 scan"):
        load_dentvoxel_cls(zip_path, bad_split, split="val", stride=1)
