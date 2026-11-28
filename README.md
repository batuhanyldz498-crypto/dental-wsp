# `dental-wsp` — Weight-Space Programming Benchmark for Digital Dentistry

> **TL;DR:** This repository tests whether blind large language models (LLMs) can directly write all weights of tiny neural networks (82 parameters for segmentation, 261 parameters for classification) for dental X-rays and 3D cone-beam CT (CBCT) without seeing training images, extending Weight-Space Programming (WSP; Lindholz et al., [arXiv:2611.12066](https://arxiv.org/abs/2611.12066)). Across six held-out test tasks, frozen zero-shot WSP achieves moderate segmentation overlap (42.4–46.6% mean Dice across 6 replies) and strong axial maxillary-sinus classification on CBCT (`0.868 ± 0.059` ROC-AUC), which beats a 3-feature global-intensity logistic regression trained on all 60 scans (`0.870` vs. `0.719` Full / `0.854` vs. `0.611` Margin AUC on validation). However, most other major claims fail or invert under fair validation controls: frozen segmentation is beaten on DenPAR by predicting all valid pixels (`53.9%` vs. `43.3%` val Dice) and a single train-selected threshold (`63.6%`), and on DentVoxel by a single HU-window threshold (`51.2%` vs. `47.4%` val Dice); `akudental_cls` is at chance for every arm (`0.47–0.55` test AUC, 16 test negatives); and frozen `denpar_cls` averages `0.467 ± 0.217` test AUC with 3/3 `opus` sign flips while a zero-parameter top-vs-bottom brightness rule reaches `0.880` validation AUC. Crucially, **the v0.1.0 low-`K` WSP warm-start advantages on both `dentvoxel_seg` and `akudental_seg` do not survive fair fixed-step and log-odds prior-bias controls on validation**: initializing `output_bias = log(p/(1-p))` eliminates 100% of 3D random-init collapses (`0/60`) and beats all-6 WSP warm-start at every tested budget on both `dentvoxel_seg` (`82.2%` vs. `77.4%` at `K = 1, S = 1,800`) and `akudental_seg` (`63.2–69.5%` vs. `60.9–68.5%` at `S = 1,050`). All deviations from the initial protocol are logged in [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md).

**Context & Non-Clinical Disclaimer:** Typical supervised deep tooth segmentation models (e.g., U-Net / nnU-Net) exceed **90% Dice**. The 82-parameter and 261-parameter networks in `dental-wsp` are **mechanistic research probes**, not competitors to supervised models and not intended for clinical or diagnostic use.

> **Governance, Provenance & Deviations:** This repository contains **code, locked architecture/prompt specifications, non-private weight locks, aggregate held-out test CSVs, aggregate validation-control CSVs, and verification tests only**. Raw radiographs, CBCT NIfTI volumes, patient identifiers, private overlays, and credentials are never stored here. Held-out test evaluation was governed by an internal Git-tracked authorization file ([`docs/test_authorization.json`](docs/test_authorization.json) — not an external registry such as OSF), completed and sealed at `2026-11-27T11:06:39Z` after one mid-load RGBA mask crash fix (`commit 34a6b16`), and its 12 aggregate test CSVs under [`results/sealed_test/`](results/sealed_test/) remain immutable (SHA-256 verified by CI). All deviations from [`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md) and the design locks are disclosed in [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md).
>
> **LLM Family Provenance:** Replies are logged in collection order across two LLM families (`astra_1..3`, `opus_1..3`; `n = 3` per family):
> - `astra`: **GPT Astra-6.2 Max** (manual web-chat replies collected `2026-11-20`; ingested into repo locks `2026-11-24`–`2026-11-25`, with `dentvoxel_seg` `astra_3` re-collected `2026-11-26T09:50:30Z`)
> - `opus`: **Opus 5.5 Max** (manual web-chat replies collected `2026-11-20`; ingested into repo locks `2026-11-24`–`2026-11-25`)

---

## Abbreviations Used

- **WSP**: Weight-Space Programming (hand-authored / LLM-generated network weights without gradient training)
- **CBCT**: Cone-beam computed tomography; **HU**: Hounsfield Units (note: CBCT values are **not** reliably calibrated HU)
- **FDI**: Fédération Dentaire Internationale permanent tooth numbering (`11–48`)
- **ROI**: Region of interest (reference tooth enclosing bounding box); **FP**: False positive
- **pp**: Percentage points; **SD**: Standard deviation; **CI**: Confidence interval
- **BF1**: Boundary F1 score (2-pixel tolerance in 2D; 3D Surface F1 at 2-voxel tolerance in 3D)
- **Edge Rec**: Reference-Edge Recall (2D) / 3D Surface Recall (3D) within 2-pixel / 2-voxel tolerance
- **IoU**: Intersection over Union; **RAS+**: Right–Anterior–Superior canonical NIfTI coordinate orientation
- **ROC-AUC** / **PR-AUC**: Area under the Receiver Operating Characteristic / Precision-Recall curve
- **Margin-Crop**: Border-zeroing ablation zeroing the outer 10% border of `64 × 64` classification inputs
- **sign flip**: Inverted score polarity automatically flagged when ROC-AUC `< 0.35` (scores are never inverted)
- **`K_max`**: Largest training label budget evaluated for a given task (note: `200` of `650` training films for `denpar_seg`)
- **BYOD**: Bring-Your-Own-Data (external datasets downloaded by the user outside Git)

---

## 2 × 3 Benchmark Matrix

| Dataset & Modality | Track 1: Classification *(261 Params, `cls-lenet2d-v1`)* | Track 2: Segmentation *(82-Parameter Parity)* | Derived Bounding-Box Audit |
|---|---|---|---|
| **DenPAR** *(2D Periapical)* | `denpar_cls` — Upper (Maxillary) vs. Lower (Mandibular) arch *(largely an image-orientation task)* | `denpar_seg` — Natural-tooth union (`periapical-wsp82-v1`, 82 params) | 2D enclosing-box IoU, valid-area retention & excluded tooth-pixel rate |
| **AKUDENTAL** *(2D Panoramic)* | `akudental_cls` — Radiopaque restoration / implant presence | `akudental_seg` — Natural-tooth union (`pano-wsp82-v1`, 82 params) | 2D enclosing-box IoU, valid-area retention & excluded tooth-pixel rate |
| **DentVoxel** *(3D CBCT, 100 NIfTI)* | `dentvoxel_cls` — Scan-split axial maxillary sinus presence | `dentvoxel_seg` — 3D natural-tooth union (`cbct-wsp82-v1`, 82 params) | 3D enclosing-box IoU, valid-volume retention & excluded tooth-voxel rate |

### Exact 82-Parameter Parity Across Segmentation Modalities
- **`pano-wsp82-v1` (AKUDENTAL 2D)** & **`periapical-wsp82-v1` (DenPAR 2D):** 6 multiscale `3 × 3` filters (`54`) + 6 feature biases (`6`) + 6 readouts (`6`) + bilinear `3 × 5` spatial grid (`15`) + scalar output bias (`1`) = **82 parameters**.
- **`cbct-wsp82-v1` (DentVoxel 3D):** 6 multiscale **7-tap 3D cross-stencils** `(center, ±z, ±y, ±x)` (`42`) + 6 feature biases (`6`) + 6 readouts (`6`) + trilinear `3 × 3 × 3` spatial grid (`27`) + scalar output bias (`1`) = **82 parameters**.

### Comparison Arms & Controls
Every task evaluates the same compact architecture across:
1. **Frozen WSP (`K = 0`):** Single-reply manual LLM outputs collected blind to images (all 6 valid replies reported in collection order without post-hoc reply selection).
2. **Random-Init + Adam (`K` budgets):** Nested patient/scan-level subsets across fixed seeds.
3. **WSP Warm-Start + Adam (same `K` budgets):** Same Adam loop initialized from *every* valid frozen WSP reply.
4. **Controls & Audits:** Trained position-only control (spatial grid + output bias only), A/B/C pathway ablations, paired cross-group swaps, inside-vs-outside ROI FPs, BF1, Edge Rec, enclosing-box valid fraction & excluded tooth rate (segmentation); ROC-AUC with Hanley–McNeil 95% CIs, sign-flip detection, and Margin-Crop border-zeroing checks (classification); plus v0.1.1 trivial/non-LLM baselines and fixed-step/prior-bias validation controls (`split == "val"` only).

---

## Held-Out Test Summary (Means ± SD Across All 6 Replies)

All headline numbers below are **means ± SD across all 6 locked replies** (`astra_1..3`, `opus_1..3`) from the sealed test CSVs in [`results/sealed_test/`](results/sealed_test/). Individual per-reply tables, Hanley–McNeil 95% CIs, validation-to-test comparisons, and SHA-256 checksums are in [`docs/RESULTS_AND_DISCUSSION.md`](docs/RESULTS_AND_DISCUSSION.md). See [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md) for all schedule, budget, and execution disclosures.

### Track 2 · Segmentation (82 Parameters · Held-Out Test)

| Task (`val` / `test`) | Frozen WSP `K = 0` Dice *(Val → Test, 6 replies)* | Grid Gain `A − B` & Outside-ROI Share of Grid-Removed FPs (A vs B)\* *(Test)* | Trained Pos-Only `K_max` *(Test)* | Low-`K` Warm-Start vs. Random-Init *(Test, epoch schedule†)* | `K_max` Warm-Start vs. Random-Init *(Test)* |
|---|---|---|---|---|---|
| `akudental_seg` (`67` / `66`, `K_max = 200/200`) | 46.98% → **46.58 ± 5.45%** | `+26.57 ± 6.84 pp`; **90.27 ± 3.19%** | **56.64 ± 1.92%** *(Edge Rec 5.12%)* | `K = 2..32`: **50.9–57.3%** vs. **34.6–46.3%** *(+11.0 to +16.3 pp Dice)* | **66.27 ± 2.63%** vs. **66.69 ± 0.93%** |
| `denpar_seg` (`150` / `200`, **`K_max = 200/650`**) | 43.35% → **42.35 ± 7.38%** | `+0.88 ± 0.51 pp`; **80.23 ± 25.58%** | **66.76 ± 2.54%** *(Edge Rec 4.66%)* | `K = 1..64`: Warm **53.2–72.6%** vs. Random **51.6–73.8%** *(tied/slightly lower Dice)* | **73.54 ± 2.28%** vs. **75.14 ± 1.27%** |
| `dentvoxel_seg` (`20` / `20` scans, RAS+, `K_max = 60/60`) | 47.42% → **45.54 ± 3.13%** | `+10.95 ± 1.72 pp`; **90.20 ± 4.09%** | **7.12 ± 0.76%** *(3D pos-only fails)* | `K = 1..4`: **56.9–57.5%** vs. **0.00%** Random collapse† | **78.87 ± 4.79%** vs. **75.83 ± 12.79%** *(n.s., 3 random seeds)* |

*\* **Outside-ROI share of grid-removed FPs (A vs B)** = `100 × (outside-ROI FPs removed by the spatial grid) / (total net FPs removed by the grid)` comparing intact arm A vs. image-only arm B. This is a signed share of net grid-removed FPs (not a percentage reduction of all outside-ROI FPs); it can exceed `100%` when the grid increases inside-ROI FPs (e.g., DenPAR `astra_2` = `128.99%`) or go negative.*
*† **Falsified by v0.1.1 validation controls:** In the v0.1.0 epoch-based schedule (`120`/`150` epochs, `output_bias = 0.0`), `dentvoxel_seg` `K = 1..4` received only `120` optimizer steps (vs. `1,800` at `K = 60`) under `<2%` foreground prevalence, and `akudental_seg` `K = 1..32` received only `150` steps (vs. `1,050` at `K = 200`). In the completed validation factorials ([`results/validation_controls/`](results/validation_controls/)), adding log-odds prior bias (`log(p/(1-p))`) and matched step budgets eliminates 100% of collapses (`0/110` runs) and `random_prior_bias` **beats** all-6 WSP warm-start at every tested `K` on both `dentvoxel_seg` (`81.9–82.6%` vs. `77.4–78.8%` at `S = 1,800`) and `akudental_seg` (`63.2–69.5%` vs. `60.9–68.5%` at `S = 1,050`).*

### Track 1 · Classification (261 Parameters · Held-Out Test, with Hanley–McNeil 95% CIs)

| Task (`test` cohort, `n_pos / n_neg`) | Frozen WSP `K = 0` Full / Margin AUC *(Mean ± SD, 6 replies; HM 95% CI)* | Low-`K` Warm-Start vs. Random-Init *(Test Full / Margin)* | `K_max` Warm-Start (All 6) vs. Random-Init *(Test Full / Margin; HM 95% CI)* |
|---|---|---|---|
| `akudental_cls` (`n = 66`, `50 / 16`) | **0.543 ± 0.041** `[0.382–0.704]` / **0.544 ± 0.141** `[0.383–0.704]` *(6/6 predict all-positive)* | At chance across all budgets (`0.46–0.55`; 95% CI ≈ `±0.16`) | **0.507 / 0.531** `[0.343–0.670]` vs. **0.470 / 0.419** `[0.305–0.635]` |
| `denpar_cls` (`n = 200`, `79 / 121`) | **0.467 ± 0.217** `[0.386–0.548]` / **0.485 ± 0.219** `[0.403–0.567]` *(3/3 `opus` sign flips)* | `K = 2..128`: Warm **0.432–0.825** vs. Random **0.460–0.922** Full | **0.810 / 0.793** `[0.746–0.875]` *(25/240 dead-ReLU)* vs. **0.977 / 0.812** `[0.953–1.000]` |
| `dentvoxel_cls` (`20` scans, `1,720` slices, `512 / 1,208`)‡ | **0.868 ± 0.059** `[0.847–0.889]` / **0.864 ± 0.077** `[0.842–0.885]` *(0 flips)* | `K = 2` scans *(~170 slices)*: **0.982 / 0.971** vs. Random **0.900 / 0.857** | **0.991 / 0.981** `[0.985–0.997]` vs. **0.965 / 0.917** `[0.953–0.976]` |

*‡ `dentvoxel_cls` slices come from `20` scans (`effective n ≈ 20` independent scans, not `1,720` independent images); slice-level Hanley–McNeil CIs therefore understate true scan-clustered uncertainty.*

### Key Negative and Mixed Results (Summary)
- **Low-`K` segmentation warm-start advantages do not survive fair controls:** In the completed validation factorials (`540` runs on `dentvoxel_seg`, `450` runs on `akudental_seg`; [`results/validation_controls/`](results/validation_controls/)), giving random initialization log-odds prior bias (`output_bias = log(p/(1-p))`) and matched optimizer steps eliminates all collapses (`0/110` runs) and beats all-6 WSP warm-start at every tested budget on both `dentvoxel_seg` (`K = 1..8`: `81.9–82.6%` vs. `77.4–78.8%` at `S = 1,800`, where even 1-filter `human_init` scores `79.9–80.7%`) and `akudental_seg` (`K = 1..64`: `63.2–69.5%` vs. `60.9–68.5%` at `S = 1,050`, plus `+5.3–6.6 pp` higher BF1 and `+7.3–11.8 pp` higher Edge Rec).
- **Frozen segmentation trails simple thresholds or position-only priors (and uses oversized boxes):** On DenPAR validation (`n = 150`), predicting **all valid pixels** gets `53.90%` Dice, zero-label **Otsu** gets `43.58%` Dice / `22.06%` BF1 / `30.47%` Edge Rec, and a **single train-selected threshold (`t* = 0.25`)** gets `63.61%` Dice — all matching or beating frozen WSP (`43.35 ± 7.11%` Dice / `12.41%` BF1 / `16.43%` Edge Rec). On DentVoxel validation (`n = 20`), a single HU-window threshold (`t* = 0.74`, `+1,960`) gets `51.18%` Dice / `58.09%` Surface F1 vs. `47.42 ± 3.19%` / `51.60%` for frozen WSP. Frozen WSP enclosing boxes retain **99.37%** (DenPAR), **67.84%** (AKUDENTAL), and **59.52%** (DentVoxel) of the valid frame/volume on validation, making near-0% excluded tooth voxels near-trivial. On held-out test, frozen 2D WSP also trails or merely matches `K = 1` trained position-only (`denpar_seg`: `42.35%` vs. `64.48%`; `akudental_seg`: `46.58%` vs. `45.98%`).
- **No consistent `K_max` warm-start benefit:** At `K_max`, random-init is slightly higher on both 2D segmentation tasks (`akudental_seg` `66.69%` vs. `66.27%`; `denpar_seg` `75.14%` vs. `73.54%`), while 3D `dentvoxel_seg` (`78.87 ± 4.79%` vs. `75.83 ± 12.79%`) is not statistically significant with 3 random seeds.
- **Chance-level restoration classification (`akudental_cls`) and orientation/dead-ReLU caveats (`denpar_cls`):** `akudental_cls` stays near chance for every arm (`16` test negatives, 95% CI ≈ `±0.16`, all 6 frozen replies predict 100% positive, and 3-feature global-intensity logistic regression beats frozen WSP on validation at all 7 budgets). On `denpar_cls`, all-6 warm-start (`0.810 / 0.793`) trails random-init (`0.977 / 0.812`) due to `25/240` dead-ReLU `opus` runs; `warm_astra` is a post-hoc `n = 3` family subset; and a zero-parameter `mean(top_half) - mean(bottom_half)` rule already achieves `0.880` Full / `0.888` Margin-Crop AUC on DenPAR validation.
- **Small-sample family differences (`n = 3`) & confounded HU-windowing hypothesis:** `astra` vs. `opus` rankings reverse across tasks (`n = 3` per family). Stronger CBCT scores (`dentvoxel_cls`, the single clearest surviving frozen WSP positive at `0.870` vs. `0.719` full-train intensity logreg on validation) occur under HU-windowed CBCT gray values (`[-1000, +3000]`; CBCT values are not reliably calibrated HU), but remain confounded by task structure (gross air-filled sinus cavities, 3D geometry), prompts, and dataset differences.

Full breakdown in [`docs/RESULTS_AND_DISCUSSION.md`](docs/RESULTS_AND_DISCUSSION.md#negative-and-mixed-results).

---

## Quickstart & Validation Controls

```bash
# Core package + unit tests (matches CI)
pip install -e ".[dev,nifti]"
pytest -v

# Full installation (adds PyTorch for training & factorial controls)
pip install -e ".[all]"
```

### Run Validation-Only Baselines & Factorial Controls (v0.1.1)
All new baseline and optimization-control commands load `split="train"` and `split="val"` **only** (held-out test splits stay sealed):

```bash
# Trivial, Otsu, global threshold, human-82p prior & classification baselines (validation only)
dental-wsp run-baselines denpar_seg --data-dir "data/DenPAR Radiographs Dataset.zip" --out-dir results/validation_controls --run-human-warmstart --device cuda
dental-wsp run-baselines denpar_cls --data-dir "data/DenPAR Radiographs Dataset.zip" --out-dir results/validation_controls
dental-wsp run-baselines akudental_seg --data-dir data/AKUDENTAL --out-dir results/validation_controls --run-human-warmstart --device cuda
dental-wsp run-baselines akudental_cls --data-dir data/AKUDENTAL --out-dir results/validation_controls
dental-wsp run-baselines dentvoxel_seg --data-dir data/DentVoxel.zip --split-file runs/dentvoxel_splits.private.json --out-dir results/validation_controls --run-human-warmstart --device cuda
dental-wsp run-baselines dentvoxel_cls --data-dir data/DentVoxel.zip --split-file runs/dentvoxel_splits.private.json --out-dir results/validation_controls

# Task (d) primary factorial control on dentvoxel_seg validation (K in {1,2,4,8}, seeds 0..4, S in {orig 120 ep, 480, 1800})
dental-wsp run-factorial-controls dentvoxel_seg \
  --data-dir data/DentVoxel.zip \
  --split-file runs/dentvoxel_splits.private.json \
  --out-dir results/validation_controls \
  --budgets 1,2,4,8 --seeds 0,1,2,3,4 --steps 480,1800 --device cuda

# Task (d) secondary factorial control on akudental_seg validation (K in {1,4,16,32,64}, seeds 0..4, S=1050)
dental-wsp run-factorial-controls akudental_seg \
  --data-dir data/AKUDENTAL \
  --out-dir results/validation_controls \
  --budgets 1,4,16,32,64 --seeds 0,1,2,3,4 --steps 1050 --device cuda
```

### Offline Manual Web-Chat WSP Collection (Zero API Keys Required)
1. Verify a task design receipt:
   ```bash
   dental-wsp lock-design denpar_seg --check-only
   ```
2. Open a fresh LLM chat (no web browsing, no code execution, no image uploads) and paste the exact prompt from `benchmarks/<dataset>/<seg|cls>_prompt.md`.
3. Save the raw text reply locally under `inbox/` (git-ignored) and ingest it:
   ```bash
   dental-wsp lock-reply denpar_seg --slot claude_1 --raw-file inbox/claude_1.txt --out runs/denpar_seg/responses.lock.json
   ```

---

## Bring-Your-Own-Data (BYOD) Datasets & Licenses

This repository's **code** is MIT-licensed. Each external dataset remains subject to its own license and citation requirements:

| Dataset | Source & Access | License / Terms | Guide |
|---|---|---|---|
| **AKUDENTAL** | Oz M, Sengul A, Hatipoglu M, Danisman T, *BMC Oral Health* (2025) — [DOI](https://doi.org/10.1186/s12903-025-07645-0) · [GitHub](https://github.com/melihoz/AKUDENTAL) | CC BY-NC-SA 4.0 | [benchmarks/akudental/DATASET.md](benchmarks/akudental/DATASET.md) |
| **DenPAR** | Rasnayaka S, Leuke Bandara D, Jayasundara A, Jayasinghe R, Wimalasiri C, Rathnayake P, Wijerathne S, Ragel R, Thambawita V, Nawinne I, *Scientific Data* 12, 1615 (2025) — [DOI](https://doi.org/10.1038/s41597-025-05906-9) · [Zenodo](https://doi.org/10.5281/zenodo.16645076) | See Zenodo / *Scientific Data* record | [benchmarks/denpar/DATASET.md](benchmarks/denpar/DATASET.md) |
| **DentVoxel** | Zhou Y, Xu Y, Tarce MA — 100 large-FOV CBCT scans, 38 instance labels (NIfTI) — [Figshare DOI](https://doi.org/10.6084/m9.figshare.31239889) | Public Figshare dataset terms | [benchmarks/dentvoxel/DATASET.md](benchmarks/dentvoxel/DATASET.md) |

---

## Repository Structure

```text
dental-wsp/
├── wsp_core/
│   ├── schemas.py                # Strict JSON schema, bounds, SHA-256 lock & receipt verifier
│   ├── guards.py                 # Sealed test-set guard & privacy-safe error wrapper
│   ├── collection.py             # Offline manual-chat reply ingestion & responses.lock builder
│   ├── cli.py                    # `dental-wsp` CLI (locks, verify, budgets, baselines, factorial controls)
│   ├── architectures/            # NumPy oracles + PyTorch modules (2D 82p, 3D 82p, 2D LeNet)
│   ├── evaluation/               # Segmentation audits, classification + Hanley–McNeil CIs, baselines.py
│   └── training/                 # Label-budget Adam trainer (epochs / max_steps / prior bias), runner & controls.py
├── benchmarks/
│   ├── akudental/                # Adapter, DATASET.md, locked prompts, task configs & locks
│   ├── denpar/                   # Adapter, DATASET.md, locked prompts, task configs & locks
│   └── dentvoxel/                # NIfTI adapter + RAS+ reorientation, DATASET.md, locked prompts, configs & locks
├── notebooks/                    # Output-cleared Colab/Jupyter validation (01–03) & archived test (04) notebooks
├── results/
│   ├── sealed_test/              # 12 immutable aggregate held-out test CSVs (SHA-256 verified)
│   └── validation_controls/      # Aggregate validation-only baseline & factorial control CSVs / SVG figures
├── docs/
│   ├── BENCHMARK_PROTOCOL.md     # Original v0.1 protocol & sealed-test authorization gate
│   ├── DEVIATIONS.md             # Full programmatic diff & commit-level deviation disclosures
│   ├── RESULTS_AND_DISCUSSION.md # Unified 3-modality results, negative findings, CIs & validation controls
│   ├── test_authorization.json   # Immutable internal Git-tracked held-out test authorization receipt
│   └── GITHUB_WORKFLOW.md        # Private-to-public GitHub publishing & CI guide
├── CHANGELOG.md                  # Release history (v0.1.0, v0.1.1 review fixes)
├── CITATION.cff                  # Citation metadata for GitHub & Zenodo
└── tests/                        # Unit, parity, regression, baseline, factorial & repo-hygiene tests
```

---

## References

- Lindholz M, Ruppel R, Milnik A, Halskov S, Eminovic S, Hamm CA, Schmidt R, Knupfer A, Hutter J, Müller JP, Dominguez Aleixo C, Nawabi J, Vogel-Claussen J, Penzkofer T. *Weight-Space Programming for Medical Imaging: Do We Still Need Training Data?* arXiv:2611.12066 (2026). https://arxiv.org/abs/2611.12066 · Code: https://github.com/MaximilianLindholz/weight-space-programming-medical
- Hanley JA, McNeil BJ. *The meaning and use of the area under a receiver operating characteristic (ROC) curve.* Radiology (1982); 143(1):29–36.
- Oz M, Sengul A, Hatipoglu M, Danisman T. *AKUDENTAL teeth instance segmentation dataset: a cross-dataset analysis.* BMC Oral Health (2025). https://doi.org/10.1186/s12903-025-07645-0 · https://github.com/melihoz/AKUDENTAL
- Rasnayaka S, Leuke Bandara D, Jayasundara A, Jayasinghe R, Wimalasiri C, Rathnayake P, Wijerathne S, Ragel R, Thambawita V, Nawinne I. *DenPAR: Annotated Intra-Oral Periapical Radiographs Dataset for Machine Learning.* Scientific Data 12, 1615 (2025). https://doi.org/10.1038/s41597-025-05906-9 · Zenodo: https://doi.org/10.5281/zenodo.16645076
- Zhou Y, Xu Y, Tarce MA. *DentVoxel: a fully annotated dental CBCT dataset with 38 instance anatomical structures.* Figshare. https://doi.org/10.6084/m9.figshare.31239889
