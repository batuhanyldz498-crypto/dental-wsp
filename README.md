# `dental-wsp` — Weight-Space Programming Benchmark for Digital Dentistry

An open-source research benchmark extending **Weight-Space Programming (WSP)** ([Lindholz et al., *Weight-Space Programming for Medical Imaging: Do We Still Need Training Data?*, arXiv:2611.12066](https://arxiv.org/abs/2611.12066)) to **digital dentistry** across three imaging modalities: **2D intra-oral periapical radiographs (DenPAR)**, **2D panoramic radiographs (AKUDENTAL)**, and **3D cone-beam CT volumes (DentVoxel)**.

> **Governance & Data Policy:** This repository contains **code, locked architecture/prompt specifications, verification tests, and non-private weight locks only**. Raw radiographs, CBCT NIfTI volumes, patient identifiers, private overlays, and credentials are never stored here. All held-out test sets remain programmatically sealed until pre-registered validation protocols are finalized and authorized.

---

## 2 × 3 Benchmark Matrix

| Dataset & Modality | Track 1: Classification *(Lindholz AUC Mirror)* | Track 2: Segmentation *(82-Parameter Parity)* | Derived Engineering Probe |
|---|---|---|---|
| **DenPAR** *(2D Periapical)* | `denpar_cls` — Upper (Maxillary) vs. Lower (Mandibular) arch (`cls-lenet2d-v1`) | `denpar_seg` — Natural-tooth union (`periapical-wsp82-v1`, 82 params) | 2D ROI tooth-pixel exclusion & crestal proximity |
| **AKUDENTAL** *(2D Panoramic)* | `akudental_cls` — Radiopaque restoration / implant presence (`cls-lenet2d-v1`) | `akudental_seg` — Natural-tooth union (`pano-wsp82-v1`, 82 params) | 2D arch ROI cropper audited by excluded tooth-pixel rate |
| **DentVoxel** *(3D CBCT, 100 NIfTI)* | `dentvoxel_cls` — Scan-split axial maxillary sinus presence (`cls-lenet2d-v1`) | `dentvoxel_seg` — 3D natural-tooth union (`cbct-wsp82-v1`, 82 params) | 3D jaw/tooth ROI cropper audited by excluded tooth-voxel rate |

### Why Exact 82-Parameter Parity Across All Three Segmentation Modalities?
- **`pano-wsp82-v1` (AKUDENTAL 2D)** & **`periapical-wsp82-v1` (DenPAR 2D):** 6 multiscale `3×3` filters (`54`) + 6 feature biases (`6`) + 6 readouts (`6`) + bilinear `3×5` spatial grid (`15`) + scalar output bias (`1`) = **82 parameters**.
- **`cbct-wsp82-v1` (DentVoxel 3D):** 6 multiscale **7-tap 3D cross-stencils** `(center, ±z, ±y, ±x)` (`42`) + 6 feature biases (`6`) + 6 readouts (`6`) + trilinear `3×3×3` spatial grid (`27`) + scalar output bias (`1`) = **82 parameters**.

### Three Standardized Comparison Arms
Every task evaluates the **exact same compact architecture** across:
1. **Frozen WSP (`K = 0`):** Single-reply LLM weights collected blind to data (zero optimizer updates; all valid replies reported in collection order).
2. **Random-Init + Adam (`K` budgets):** Prespecified patient/scan-level label budgets (`K = 1, 2, 4, 8, 16, ...`) across fixed seeds.
3. **WSP Warm-Start + Adam (same `K` budgets):** Exact same Adam loop initialized from *every* valid frozen WSP reply (no cherry-picking).
4. **Controls & Mechanistic Audit:** Trained position-only control, A/B/C pathway ablations, paired cross-group swaps, inside-vs-outside ROI false positives, boundary/surface F1, reference-edge/surface recall, in-box precision, and ROI pixel/voxel exclusion for segmentation; ROC-AUC, sign-flip detection, and margin-crop shortcut checks for classification.

---

## Quickstart

```bash
# Core package + unit tests
pip install -e ".[dev]"
pytest -v

# Full installation (PyTorch trainer + NIfTI CBCT loader)
pip install -e ".[all]"
```

### Offline Manual Web-Chat WSP Collection (Zero API Keys Required)
1. Lock or verify the task design receipt:
   ```bash
   dental-wsp lock-design denpar_seg --check-only
   ```
2. Open a fresh GPT or Claude chat (no web browsing, no code execution, no image uploads) and paste the exact prompt from `benchmarks/<dataset>/<seg|cls>_prompt.md`.
3. Save the raw text reply locally under `inbox/` (git-ignored) and ingest it:
   ```bash
   dental-wsp lock-reply denpar_seg --slot claude_1 --raw-file inbox/claude_1.txt --out runs/denpar_seg/responses.lock.json
   ```
   The CLI enforces strict duplicate-key rejection, exact tensor shapes, finite values in `[-1000, 1000]`, computes SHA-256 receipts, and records invalid responses transparently without silent retries.

---

## Bring-Your-Own-Data (BYOD) Datasets & Licenses

This repository's **code** is MIT-licensed. Each external dataset remains subject to its own license and citation requirements:

| Dataset | Source & Access | License / Terms | Guide |
|---|---|---|---|
| **AKUDENTAL** | Oz et al., *BMC Oral Health* (2025) — [DOI](https://doi.org/10.1186/s12903-025-07645-0) · [GitHub](https://github.com/melihoz/AKUDENTAL) | CC BY-NC-SA 4.0 | [benchmarks/akudental/DATASET.md](benchmarks/akudental/DATASET.md) |
| **DenPAR** | Rasnayaka et al., *Scientific Data* (2025) — Zenodo | See Zenodo record | [benchmarks/denpar/DATASET.md](benchmarks/denpar/DATASET.md) |
| **DentVoxel** | 100 large-FOV CBCT scans, 38 instance labels (NIfTI) — [Figshare](https://figshare.com/articles/dataset/DentVoxel_a_fully_annotated_dental_CBCT_dataset_with_38_instance_anatomical_structures/31239889) | Public Figshare dataset terms | [benchmarks/dentvoxel/DATASET.md](benchmarks/dentvoxel/DATASET.md) |

---

## Repository Structure

```text
dental-wsp/
├── wsp_core/
│   ├── schemas.py                # Strict JSON schema, bounds, SHA-256 lock & receipt verifier
│   ├── guards.py                 # Sealed test-set guard & privacy-safe error wrapper
│   ├── collection.py             # Offline manual-chat reply ingestion & responses.lock builder
│   ├── cli.py                    # `dental-wsp` CLI
│   ├── architectures/            # NumPy oracles + PyTorch modules (2D 82p, 3D 82p, 2D LeNet)
│   ├── evaluation/               # Segmentation (A/B/C, swaps, FP, boundary, ROI exclusion) & Classification
│   └── training/                 # Deterministic label-budget Adam trainer + WSP warm-start
├── benchmarks/
│   ├── akudental/                # Adapter, DATASET.md, locked prompts, task configs
│   ├── denpar/                   # Adapter, DATASET.md, locked prompts, task configs
│   └── dentvoxel/                # NIfTI adapter + HU audit, DATASET.md, locked prompts, task configs
├── notebooks/                    # Thin output-cleared Colab/Jupyter reproduction notebooks
├── docs/
│   ├── BENCHMARK_PROTOCOL.md     # Pre-registered protocol & sealed-test authorization gate
│   └── GITHUB_WORKFLOW.md        # Private-to-public GitHub publishing & CI guide
└── tests/                        # Unit, numerical-parity, synthetic end-to-end & hygiene tests
```

---

## References

- Lindholz M, Ruppel R, Milnik A, et al. *Weight-Space Programming for Medical Imaging: Do We Still Need Training Data?* arXiv:2611.12066. https://arxiv.org/abs/2611.12066
- Oz M, Sengul A, Hatipoglu M, Danisman T. *AKUDENTAL teeth instance segmentation dataset: a cross-dataset analysis.* BMC Oral Health (2025). https://doi.org/10.1186/s12903-025-07645-0
- Rasnayaka et al. *DenPAR: Annotated Intra-Oral Periapical Radiographs Dataset for Machine Learning.* Scientific Data (2025).
- DentVoxel CBCT Dataset (100 scans, 38 anatomical structures, NIfTI). Figshare: https://figshare.com/articles/dataset/DentVoxel_a_fully_annotated_dental_CBCT_dataset_with_38_instance_anatomical_structures/31239889
