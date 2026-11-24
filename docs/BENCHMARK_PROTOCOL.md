# Pre-Registered Benchmark Protocol (`dental-wsp` v0.1)

This document locks the scientific protocol for the `dental-wsp` 2 × 3 digital dentistry benchmark before collecting new Weight-Space Programming (WSP) replies or opening any held-out test split.

---

## 1. Scientific Scope & Relation to Prior Work

Lindholz et al. ([arXiv:2611.12066](https://arxiv.org/abs/2611.12066)) evaluated single-reply LLM-written weights for compact LeNet-style CNNs across seven medical **classification** tasks (test ROC-AUC), comparing frozen WSP against same-architecture Adam training on small labeled subsets.

`dental-wsp` extends that framework to **digital dentistry** across three publicly available imaging datasets—**DenPAR** (2D periapical), **AKUDENTAL** (2D panoramic), and **DentVoxel** (3D CBCT)—with two core tracks:

- **Track 1 (Classification — Lindholz Mirror):** Compact 2D LeNet-style binary classifiers (`cls-lenet2d-v1`, 261 parameters), evaluated by ROC-AUC and PR-AUC.
- **Track 2 (Segmentation — Dense 2D/3D Extension):** Exact **82-parameter** multiscale architectures (`periapical-wsp82-v1`, `pano-wsp82-v1`, `cbct-wsp82-v1`), evaluated by region Dice, pathway ablations, paired cross-group swaps, inside/outside ROI false positives, boundary/surface F1, reference-edge/surface recall, and in-box precision.
- **Derived Engineering Audit (Zero Extra Labels):** Coarse enclosing-box localization from Track 2 predictions audited by **excluded annotated tooth-pixel / tooth-voxel fraction** (directly addressing the area-vs-pixel coverage gap identified in AKUDENTAL Study 1 Cell 12).

---

## 2. Locked Architectures & Parameter Budgets

### 2.1 Segmentation Architectures (Exact 82-Parameter Parity)

| Architecture ID | Dataset | Canvas (divisible by 16/8) | Image Filters | Spatial Grid | Total Params |
|---|---|---|---|---|---:|
| `periapical-wsp82-v1` | DenPAR (2D) | `384 × 384` | 6 × `3×3` at scales `(1, 4, 16)` + 6 bias + 6 readout | `3 × 5` bilinear + 1 output bias | **82** |
| `pano-wsp82-v1` | AKUDENTAL (2D) | `384 × 768` | 6 × `3×3` at scales `(1, 4, 16)` + 6 bias + 6 readout | `3 × 5` bilinear + 1 output bias | **82** |
| `cbct-wsp82-v1` | DentVoxel (3D) | `96 × 128 × 128` | 6 × 7-tap 3D cross-stencil `(c, -z, +z, -y, +y, -x, +x)` at scales `(1, 2, 4)` + 6 bias + 6 readout | `3 × 3 × 3` trilinear + 1 output bias | **82** |

Foreground prediction uses inclusive **logit ≥ 0** restricted to valid unpadded pixels/voxels.

### 2.2 Classification Architecture (`cls-lenet2d-v1`, 261 Parameters)

- Input: standardized `64 × 64` grayscale image scaled to `[0, 1]`.
- `Conv1`: 4 filters of `5 × 5` (padding `2`), bias (`4`), ReLU, `4 × 4` average pooling → `(4, 16, 16)`.
- `Conv2`: 4 filters of `3 × 3` over 4 input channels (padding `1`), bias (`4`), ReLU → `(4, 16, 16)`.
- Global pooling: channel-wise `[global_mean, global_max]` → 8 scalar features.
- Linear head: 8 weights + 1 scalar output bias → binary logit.
- Total parameters: `100 + 4 + 144 + 4 + 8 + 1 = 261`.

---

## 3. Preprocessing, Splitting & Intensity Rules

1. **AKUDENTAL (`akudental_seg`, `akudental_cls`):**
   - Preserves the exact preprocessing (`0.5th–99.5th` per-film percentile clip → `[0, 1]`, aspect-preserving center resize on `384 × 768`, bilinear image / nearest mask) and locked split digest from Study 1.
   - `akudental_seg` target: binary union of FDI permanent tooth classes 1–32.
   - `akudental_cls` target: binary presence of any restorative class (`Implant`, `Crown/Filling`, `Bridge`) resized to `64 × 64`.
   - Provisional groups are preserved and documented transparently as not establishing patient independence.
2. **DenPAR (`denpar_seg`, `denpar_cls`):**
   - Grayscale `0.5th–99.5th` percentile clip → `[0, 1]`.
   - `denpar_seg`: aspect-preserving resize onto `384 × 384` canvas with valid-pixel mask; target = binary union of tooth masks.
   - `denpar_cls`: `64 × 64` grayscale; target = binary alveolar bone-loss presence (or documented bone-loss severity threshold).
   - Splits grouped by patient/radiograph group ID (60% train / 20% val / 20% test).
3. **DentVoxel (`dentvoxel_seg`, `dentvoxel_cls`):**
   - 100 large-FOV NIfTI CBCT scans with 38 instance anatomical labels.
   - **Intensity audit gate:** Before locking prompts, `dental-wsp verify-dataset dentvoxel` audits scan intensity quantiles across the cohort. Primary normalization uses fixed dental Hounsfield windowing `[-1000, 3000]` → `[0, 1]` if scans are HU-calibrated; if scans are uncalibrated grayscale, the protocol locks per-volume `0.5th–99.5th` percentile clipping → `[0, 1]` and records that choice in `design.lock.json`.
   - **Strict scan-level split:** 60 train / 20 validation / 20 held-out test scans. For `dentvoxel_cls`, all axial slices sampled from a scan remain strictly inside that scan's split.
   - `dentvoxel_seg`: resampled/padded to `96 × 128 × 128` with valid-voxel mask; target = binary union of the 32 FDI tooth labels.
   - `dentvoxel_cls`: standardized `64 × 64` axial slices; target = binary presence of bilateral maxillary sinuses (sinus label voxels ≥ 50 in slice).

---

## 4. Three Standardized Comparison Arms & Label Budgets

For every task, all arms use the exact same architecture, preprocessing, split, and evaluation metrics:

1. **Arm 1 — Frozen WSP (`K = 0`):**
   - Single-reply manual GPT / Claude web-chat outputs collected blind to images, without tools or code execution.
   - Fixed design: 3 independent repeats per LLM family (all attempts logged in collection order; invalid schema replies recorded transparently without silent retries).
2. **Arm 2 — Random-Init + Adam (`K` budgets):**
   - Prespecified patient/scan-level training budgets `K ∈ {1, 2, 4, 8, 16, 32, full_train}` (truncated to available training groups/scans; nested subsets per seed across 5 fixed seeds).
   - Fixed Adam optimizer (`lr = 1e-2` for 82-param segmentation / `5e-3` for 261-param classification, cosine decay or fixed 200 epochs, zero test access).
3. **Arm 3 — WSP Warm-Start + Adam (same `K` budgets):**
   - Identical Adam loop and budgets `K`, initialized from **every valid frozen WSP reply** (no validation reply selection).
4. **Control Arm (Segmentation only):**
   - **Trained Position-Only:** Optimize only `spatial_bias` and `output_bias` under the same budgets `K` to quantify the ceiling of pure spatial priors.

---

## 5. Mandatory Diagnostic & Anti-Shortcut Suite

- **Segmentation (all 3 datasets):**
  1. Exact reproduction gate on saved baseline scores.
  2. Pathway ablations (`A` intact, `B` image-only, `C` position-only) + mandatory position-invariance check.
  3. Paired cross-group swaps across all eligible pairs sharing identical valid-pixel/valid-voxel masks.
  4. Inside-vs-outside reference-ROI false positives per 1,000 total valid pixels/voxels.
  5. Full-canvas/full-volume boundary/surface F1 (inclusive 2-pixel / 2-voxel tolerance), reference-edge/surface recall, paired edge transitions (`preserved`, `gained`, `lost`, `missed`), and in-box predicted-edge precision.
  6. Coarse enclosing-box IoU, reference-box coverage, and **excluded annotated tooth-pixel / tooth-voxel rate**.
- **Classification (all 3 datasets):**
  1. ROC-AUC and PR-AUC (scan-aggregated and slice-level for `dentvoxel_cls`).
  2. **Sign-flip detector:** Automatically flags `AUC < 0.35` transparently without inverting predictions.
  3. **Margin-crop shortcut check:** Evaluates AUC when outer 10% border pixels are zeroed.

---

## 6. Sealed Test-Set Authorization Gate

No evaluation script may load `split == "test"` unless:
1. All six `design.lock.json` and `responses.lock.json` files are finalized and committed.
2. All validation results and hyperparameters are frozen.
3. An explicit authorization receipt file (`test_authorization.json` containing date, authorized tasks, and commit SHA) is passed via `--allow-test --authorization-file`.
4. No prior unauthorized test-exposure marker exists (`exposure_guard`).
