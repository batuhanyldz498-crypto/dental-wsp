# DentVoxel Dataset (3D Large-FOV Dental CBCT)

## Description & Public Access
- **Dataset:** DentVoxel — 100 large field-of-view dental CBCT scans in **NIfTI** format with **38 instance-level anatomical labels**:
  - Maxilla (1)
  - Mandible (1)
  - Bilateral maxillary sinuses (2)
  - Bilateral mandibular canals (2)
  - 32 individual permanent teeth labeled according to the FDI numbering system
- **Public Host (Figshare):** [https://figshare.com/articles/dataset/DentVoxel_a_fully_annotated_dental_CBCT_dataset_with_38_instance_anatomical_structures/31239889](https://figshare.com/articles/dataset/DentVoxel_a_fully_annotated_dental_CBCT_dataset_with_38_instance_anatomical_structures/31239889)

## Annotations Used in `dental-wsp`
- **`dentvoxel_seg` (`cbct-wsp82-v1`, 82 params):** 3D binary union of the 32 FDI tooth labels resampled/padded onto a `96 × 128 × 128` isotropic canvas. Audited by 3D A/B/C ablations, 3D paired swaps, inside/outside ROI FPs, 3D surface F1/recall, and **3D ROI excluded tooth-voxel rate**.
- **`dentvoxel_cls` (`cls-lenet2d-v1`, 261 params):** Strictly **scan-split** (60 train / 20 val / 20 test scans) `64 × 64` axial slices labeled for **bilateral maxillary sinus presence** (dark air-filled cavities bounded by thin cortical walls).

## Bring-Your-Own-Data (BYOD) Placement
Download the NIfTI volumes and labels from Figshare into `data/dentvoxel/` (outside Git):
```text
data/dentvoxel/
├── images/          # *.nii or *.nii.gz CBCT scans
├── labels/          # matching *.nii or *.nii.gz 38-label instance masks
├── label_map.json   # maps semantic names ("tooth_ids", "sinus_ids") to integer NIfTI label IDs
└── splits.private.json  # 60/20/20 scan-level split (generated deterministically)
```

Verify and run the mandatory HU calibration audit locally (requires `pip install -e ".[nifti]"`):
```bash
dental-wsp verify-dataset dentvoxel --data-dir data/dentvoxel --max-scans 10
```
