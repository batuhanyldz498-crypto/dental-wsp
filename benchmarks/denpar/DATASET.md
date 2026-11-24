# DenPAR Dataset (2D Intra-Oral Periapical Radiographs)

## Citation & Access
- **Authors:** Rasnayaka et al.
- **Paper:** *DenPAR: Annotated Intra-Oral Periapical Radiographs Dataset for Machine Learning*, *Scientific Data* (Nature Portfolio, 2025)
- **Hosting:** Publicly available on Zenodo (download directly from the official Zenodo record; do not commit raw images or masks to Git)

## Annotations Used in `dental-wsp`
- **`denpar_seg` (`periapical-wsp82-v1`, 82 params):** Binary pixel-level natural-tooth union on a standardized `384 × 384` aspect-preserving padded canvas after per-film `0.5th–99.5th` percentile clipping. Because maxillary and mandibular periapical films vary widely in orientation and framing, this task directly tests WSP when static spatial priors cannot carry segmentation.
- **`denpar_cls` (`cls-lenet2d-v1`, 261 params):** Binary radiograph-level alveolar bone loss presence (or horizontal/vertical bone loss vs. healthy crestal bone level) on a standardized `64 × 64` grayscale canvas.

## Bring-Your-Own-Data (BYOD) Placement
Download the DenPAR archive from Zenodo and place it outside Git (for example under `data/denpar/`):
```text
data/denpar/
├── images/          # Periapical radiograph files
├── masks/           # Corresponding tooth-union masks (same stem)
└── metadata.csv     # Columns: image_id, group_id, bone_loss_label (0/1), split (train/val/test)
```

Verify locally:
```bash
dental-wsp verify-dataset denpar --data-dir data/denpar
```
