"""BYOD dataset adapter for DenPAR intra-oral periapical radiographs (`denpar_seg`, `denpar_cls`).

Supports reading directly from `DenPAR Radiographs Dataset.zip` or an extracted `Dataset/` directory,
using DenPAR's official `Training` (650), `Validation` (150), and sealed `Testing` (200) folders plus
`Characteristics of radiographs included.xlsx`.
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

from benchmarks.akudental.adapter import prepare_cls_canvas, prepare_panorama
from wsp_core.evaluation.classification import ClassificationSample
from wsp_core.evaluation.segmentation import SegmentationSample
from wsp_core.guards import require, require_split_allowed

SPLIT_DIR_MAP = {"train": "Training", "val": "Validation", "test": "Testing"}


def parse_denpar_xlsx(xlsx_bytes: bytes) -> dict[str, dict[str, str]]:
    """Parse `Characteristics of radiographs included.xlsx` using Python stdlib only."""
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes), "r") as xz:
        strings: list[str] = []
        if "xl/sharedStrings.xml" in xz.namelist():
            root_s = ET.fromstring(xz.read("xl/sharedStrings.xml"))
            strings = ["".join(t.text or "" for t in si.iter() if t.tag.endswith("}t")) for si in root_s]
        root_sh = ET.fromstring(xz.read("xl/worksheets/sheet1.xml"))
        ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rows: list[list[str]] = []
        for row in root_sh.findall(".//m:sheetData/m:row", ns):
            vals: list[str] = []
            for cell in row.findall("m:c", ns):
                cell_type = cell.attrib.get("t")
                v = cell.find("m:v", ns)
                if v is None or v.text is None:
                    vals.append("")
                elif cell_type == "s":
                    vals.append(strings[int(v.text)])
                else:
                    vals.append(v.text)
            rows.append(vals)

    require(len(rows) > 1, "DenPAR XLSX spreadsheet contains no data rows.")
    records: dict[str, dict[str, str]] = {}
    for row in rows[1:]:
        if len(row) < 2:
            continue
        rid = str(row[0]).replace(".jpg", "").replace(".0", "").strip()
        arch = str(row[1]).strip()
        site = str(row[2]).strip() if len(row) > 2 else ""
        records[rid] = {"arch": arch, "site": site}
    return records


def load_denpar_seg(
    source_path: str | Path,
    *,
    split: str = "val",
    canvas: tuple[int, int] = (384, 384),
    allow_test: bool = False,
    authorization_file: str | Path | None = None,
) -> list[SegmentationSample]:
    """Load DenPAR periapical tooth-union samples from the Zenodo ZIP or extracted `Dataset/` folder."""
    require_split_allowed(split, task="denpar_seg", allow_test=allow_test, authorization_file=authorization_file)
    folder_name = SPLIT_DIR_MAP[split]
    src = Path(source_path)
    samples: list[SegmentationSample] = []

    if src.is_file() and src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src, "r") as zf:
            img_prefix = f"Dataset/{folder_name}/Images/"
            msk_prefix = f"Dataset/{folder_name}/Masks (Radiograph-wise)/"
            img_entries = sorted(n for n in zf.namelist() if n.startswith(img_prefix) and n.lower().endswith(".jpg"))
            for img_entry in img_entries:
                stem = Path(img_entry).stem
                msk_entry = f"{msk_prefix}{stem}.png"
                with zf.open(img_entry) as img_f, zf.open(msk_entry) as msk_f:
                    img, target, valid = prepare_panorama(img_f, msk_f, canvas=canvas)
                samples.append(SegmentationSample(sample_id=stem, group_id=stem, image=img, target=target, valid=valid))
    else:
        base = src / "Dataset" if (src / "Dataset").is_dir() else src
        img_dir = base / folder_name / "Images"
        msk_dir = base / folder_name / "Masks (Radiograph-wise)"
        require(img_dir.is_dir() and msk_dir.is_dir(), f"Missing DenPAR Images/Masks under {base / folder_name}")
        for img_path in sorted(img_dir.glob("*.jpg")):
            stem = img_path.stem
            msk_path = msk_dir / f"{stem}.png"
            img, target, valid = prepare_panorama(img_path, msk_path, canvas=canvas)
            samples.append(SegmentationSample(sample_id=stem, group_id=stem, image=img, target=target, valid=valid))

    require(len(samples) > 0, f"No DenPAR segmentation samples found for split '{split}'.")
    return samples


def load_denpar_cls(
    source_path: str | Path,
    *,
    split: str = "val",
    size: tuple[int, int] = (64, 64),
    allow_test: bool = False,
    authorization_file: str | Path | None = None,
) -> list[ClassificationSample]:
    """Load DenPAR `Arch` (Upper=1 vs Lower=0) classification samples from ZIP or extracted folder."""
    require_split_allowed(split, task="denpar_cls", allow_test=allow_test, authorization_file=authorization_file)
    folder_name = SPLIT_DIR_MAP[split]
    src = Path(source_path)
    samples: list[ClassificationSample] = []

    if src.is_file() and src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src, "r") as zf:
            meta = parse_denpar_xlsx(zf.read("Dataset/Characteristics of radiographs included.xlsx"))
            img_prefix = f"Dataset/{folder_name}/Images/"
            img_entries = sorted(n for n in zf.namelist() if n.startswith(img_prefix) and n.lower().endswith(".jpg"))
            for img_entry in img_entries:
                stem = Path(img_entry).stem
                arch = meta[stem]["arch"].lower()
                require(arch in ("upper", "lower"), f"Unexpected DenPAR arch label: {arch}")
                with zf.open(img_entry) as img_f:
                    img = prepare_cls_canvas(img_f, size=size)
                samples.append(ClassificationSample(sample_id=stem, group_id=stem, image=img, label=1 if arch == "upper" else 0))
    else:
        base = src / "Dataset" if (src / "Dataset").is_dir() else src
        meta = parse_denpar_xlsx((base / "Characteristics of radiographs included.xlsx").read_bytes())
        img_dir = base / folder_name / "Images"
        for img_path in sorted(img_dir.glob("*.jpg")):
            stem = img_path.stem
            arch = meta[stem]["arch"].lower()
            require(arch in ("upper", "lower"), f"Unexpected DenPAR arch label: {arch}")
            img = prepare_cls_canvas(img_path, size=size)
            samples.append(ClassificationSample(sample_id=stem, group_id=stem, image=img, label=1 if arch == "upper" else 0))

    require(len(samples) > 0, f"No DenPAR classification samples found for split '{split}'.")
    return samples


def verify_denpar_directory(source_path: Path) -> dict[str, Any]:
    """Read-only verification of DenPAR ZIP or extracted folder (without scoring Testing)."""
    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(source_path, "r") as zf:
            names = zf.namelist()
            train_imgs = [n for n in names if n.startswith("Dataset/Training/Images/") and n.lower().endswith(".jpg")]
            val_imgs = [n for n in names if n.startswith("Dataset/Validation/Images/") and n.lower().endswith(".jpg")]
            meta = parse_denpar_xlsx(zf.read("Dataset/Characteristics of radiographs included.xlsx"))
        return {
            "dataset": "denpar",
            "format": "zip",
            "train_images": len(train_imgs),
            "val_images": len(val_imgs),
            "metadata_rows": len(meta),
        }
    base = source_path / "Dataset" if (source_path / "Dataset").is_dir() else source_path
    require(base.is_dir(), f"DenPAR path not found: {source_path}")
    return {
        "dataset": "denpar",
        "format": "directory",
        "train_images": len(list((base / "Training" / "Images").glob("*.jpg"))),
        "val_images": len(list((base / "Validation" / "Images").glob("*.jpg"))),
        "xlsx_present": (base / "Characteristics of radiographs included.xlsx").is_file(),
    }
