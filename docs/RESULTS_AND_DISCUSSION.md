# Unified 3-Modality Results & Discussion (`dental-wsp`)

> **Governance Notice:** Validation results below reflect all **6 / 6 completed validation suites** across **36 / 36 hash-unique, blind manual WSP replies** (`astra_1..3`, `opus_1..3` reported in collection order; zero schema errors, zero cherry-picking). Held-out test evaluation is pre-registered under [`docs/test_authorization.json`](test_authorization.json) (`commit_sha = 05ae8d24940397c55b2c250a50860560a89888d2`) and executed via a single pass of [`notebooks/04_sealed_test_evaluation.ipynb`](../notebooks/04_sealed_test_evaluation.ipynb). **No clinical or diagnostic deployment claims are made:** all 82-parameter and 261-parameter models are mechanistic research probes, shortcut-resistant low-label initializers, and lightweight ROI pre-localizers.

---

## 1. Cross-Modality Validation Overview (`2 × 3` Matrix)

Extending Lindholz et al. (*Weight-Space Programming for Medical Imaging*, arXiv:2611.12066) across three distinct dental imaging physics regimes — **2D panoramic (AKUDENTAL)**, **2D intra-oral periapical (DenPAR)**, and **3D cone-beam CT (DentVoxel, canonical RAS+)** — reveals a clear physical organizing principle: **calibrated voxel radiodensity (CBCT Hounsfield Units) combined with canonical anatomical coordinate alignment unlocks both zero-shot classification and collapse-free 3D volumetric segmentation at exact 82-parameter parity.**

### Table 1 · Track 2 Segmentation Validation Summary (Exact 82-Parameter Parity)

| Modality & Task (`val` size) | Arch (Params) | Frozen WSP `K=0` Intact Dice / Surf-Edge Rec | Frozen Grid Gain (`A − B`) | Frozen Swap Loss A (pp) | Frozen Outside FP Share (%) | Frozen Excl. Tooth (%) | Trained Pos-Only (`K_max`) | Low-`K` Warm-Start vs. Random-Init | Full-`K` Warm-Start vs. Random-Init |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| **AKUDENTAL** 2D Panoramic (`val=67`) | `pano-wsp82-v1` (82) | **46.98%** / 32.25% | **+25.85 pp** *(arch critical)* | 14.58 pp (`66/67`) | 90.13% | 2.13% (`opus_2`: 0.29%) | 59.56% Dice / 5.41% Rec | `K=4..32`: **+12–15 pp** Dice & +14–17 pp Rec | 68.19% *(5/6 ~69.3%)* vs. 68.60% |
| **DenPAR** 2D Periapical (`val=150`) | `periapical-wsp82-v1` (82) | **43.35%** / 16.43% | **+0.80 pp** *(filter-driven)* | 12.65 pp (`122/150`) | 79.03%* | 0.01%* | 65.37% Dice / 4.70% Rec | `K=1..64`: Dice tied; `opus` **+5.5–7.6 pp** Edge Rec | 71.97% (`opus` 73.32%) vs. 73.65% |
| **DentVoxel** 3D CBCT (`val=20` scans, RAS+) | `cbct-wsp82-v1` (82) | **47.42 ± 3.19%** (`opus_3` **53.39%**) / **96.08%** | **+10.65 pp** *(6/6 positive)* | **28.52 pp** (`opus_3` **33.74**) | **87.13%** (`opus_2` **93.65%**) | **0.00%** *(6/6 zero!)* | **7.32%** Dice / 8.93% Rec *(fails in 3D)* | `K=1..4`: **59.7–60.4%** (`opus` **64.7%**) vs. **0.00%** Random collapse | **79.21%** (`opus` **83.01 ± 0.72%**) vs. 76.21 ± 11.77% |

*\*On DenPAR, bright rounded intra-oral sensor rims cause frozen enclosing boxes to span nearly the full valid frame (`Box IoU ~68.7%`, `Excl ~0.01%`), so raw frozen box metrics reflect full-frame coverage rather than tight localization.*

### Table 2 · Track 1 Classification Validation Summary (Lindholz Mirror, `cls-lenet2d-v1`, 261 Params)

| Modality & Task (`val` size) | Target | Frozen WSP `K=0` Full ROC-AUC (Sign Flips) | Frozen WSP `K=0` Margin-Crop AUC | Low-`K` Warm-Start vs. Random-Init (Full / Margin) | Full-`K` Warm-Start vs. Random-Init (Full / Margin) | Border Shortcut Drop (`K_max`: Random vs. Best Family) |
|---|---|---:|---:|---|---|---|
| **AKUDENTAL** 2D Panoramic (`val=67`) | Restoration / implant presence | 0.5108 *(0/6)* | **0.5930** (`opus_3` **0.8768**) | Warm beats Random on Margin-Crop **7/7** budgets | 0.5338 / 0.5565 vs. 0.5500 / 0.5355 | Resolution + left-edge film strip bottleneck |
| **DenPAR** 2D Periapical (`val=150`) | Upper vs. Lower arch | 0.4279 (`astra_1` **0.6979**; `opus` **3/3 flip**) | 0.5008 (`astra_1` **0.8245**) | `astra` warm: **0.9300** Margin vs. Random 0.7960 | All-6: 0.7666 / 0.7646 *(25/240 `opus` dead-ReLU)*; `astra`: **0.9233 / 0.9300** vs. Random 0.9213 / 0.7960 | Random **−12.5 pts** vs. `astra` **−0.7 pts** *(gain)* |
| **DentVoxel** 3D CBCT Axial (`val=20` scans, `n=1720`) | Bilateral maxillary sinus presence | **0.8699 ± 0.0573** (`astra_2` **0.9670**, **0/6 flips**) | **0.8543 ± 0.0798** | `K=2` scans: **0.9812 / 0.9714** vs. Random 0.8729 / 0.8227 (**+10.8 / +14.9 pts!**) | **0.9919 / 0.9831** (`warm_astra` **0.9922 / 0.9898**) vs. Random 0.9667 / 0.9062 | Random **−6.05 pts** vs. `warm_astra` **−0.24 pts** *(>20× lower)* |

---

## 2. Per-Modality Validation Breakdown (All 6 Replies Reported in Collection Order)

### 2.1 AKUDENTAL (2D Panoramic · `train=200`, `val=67`, `test=66`)
- **`akudental_seg` (`pano-wsp82-v1`, 82 params):**
  - Frozen `K=0` Intact Dice: `astra_1` 49.74%, `astra_2` 48.93%, `astra_3` 49.21%, `opus_1` 41.94%, `opus_2` **54.51%**, `opus_3` 37.56% (mean **46.98%**, Image-only **21.13%**, Position-only **0.00%**, Swap Loss **14.58 pp**, Outside FP Share **90.13%**, Edge Recall **32.25%**, Excluded Tooth **2.13%**; `opus_2` reaches 61.02% Box IoU at 0.29% excluded pixels).
  - Learning curve (`K=1..200`, 3 seeds): Frozen WSP beats Random-Init through `K=32` (`45.32%`). Warm-Start leads Random-Init by **+12–15 pp** across `K=4..32` (`K=1` 54.05%, `K=8` 55.13%, `K=16` 58.38%, `K=32` 58.55%, `K=64` 64.45%, `K=200` 68.19% vs. Random 68.60%, Position-Only 59.56% / 5.41% Edge Recall).
- **`akudental_cls` (`cls-lenet2d-v1`, 261 params):**
  - Full-frame `64×64` stays near chance across all arms (`0.47–0.55`) due to tiny metallic restorations and left-margin film strips. Under 10% border ablation (Margin-Crop), Frozen WSP (`0.5930` mean; `opus_3` **0.8768**, `astra_1` 0.6679, `astra_2` 0.6201) outperforms every trained budget, and Warm-Start beats Random-Init at **7/7** budgets.

### 2.2 DenPAR (2D Intra-Oral Periapical · `train=650`, `val=150`, `test=200`)
- **`denpar_seg` (`periapical-wsp82-v1`, 82 params):**
  - Frozen `K=0` Intact vs. Image-only Dice: `astra_1` 51.01% vs 49.66%, `astra_2` 48.71% vs 47.44%, `astra_3` 49.65% vs 49.00%, `opus_1` 36.98% vs 36.69%, `opus_2` 36.14% vs 35.37%, `opus_3` 37.58% vs 37.12% (mean **43.35 ± 7.11%** Intact vs **42.55 ± 6.80%** Image-only, **+0.80 pp** grid dependence; Position-Only **0.00%**; Swap Loss **12.65 pp** across `122/150` eligible groups).
  - Learning curve (`K=1..200`, 3 seeds): `Random-Init` climbs `52.25% → 73.65%` (Edge Rec `17.98% → 39.36%`); `Position-Only` sits at `65.37%` Dice but only `4.70%` Edge Recall (`13.72%` excluded teeth); `Warm-Start` reaches `71.97%` overall (`opus_1..3` **73.32% / 38.76%**, leading Random-Init by **+5.5–7.6 pp Edge Recall** across `K=1..64` with `0.27–0.71%` excluded teeth vs. Random `2.46–4.69%`, while `astra_3` plateaus at `67.24% / 13.68%`).
- **`denpar_cls` (`cls-lenet2d-v1`, 261 params):**
  - Frozen `K=0` Full → Margin-Crop ROC-AUC: `astra_1` **0.6979 → 0.8245**, `astra_2` **0.5540 → 0.7464**, `astra_3` 0.4933 → 0.5252; `opus_1` 0.3375 → 0.3735 (*sign-flip*), `opus_2` 0.2375 → 0.2671 (*sign-flip*), `opus_3` 0.2474 → 0.2684 (*sign-flip*).
  - Learning curve (`K=2..650`, 5 seeds): `Random-Init` hits `0.9213` Full at `K=650` but drops **−12.5 pts** under Margin-Crop (`0.7960`), exposing letterbox-border reliance. `astra_1..3` warm-starts reach **0.9233 Full / 0.9300 Margin-Crop** (`astra_2 = 0.9521`, **0 dead runs**), whereas `opus_2` and `opus_3` exhibit **dead-ReLU collapse (`AUC = 0.5000`) on 25 / 240 warm-start runs** due to steep negative frozen conv biases (`-3.6` to `-18.2`).

### 2.3 DentVoxel (3D Large-FOV CBCT, Canonical RAS+ · `train=60`, `val=20`, `test=20` scans)
- **`dentvoxel_seg` (`cbct-wsp82-v1`, 82 params):**
  - Frozen `K=0` (all 6 unique replies): `astra_1` 44.77% (Img-only 33.27%), `astra_2` 46.69% (34.76%), `astra_3` 46.26% (34.27%), `opus_1` 45.07% (37.59%), `opus_2` 48.37% (38.99%), `opus_3` **53.39%** (41.78%). Cohort mean: **47.42 ± 3.19% 3D Dice**, **+10.65 pp** unanimous grid gain (`6/6` positive), **0.00% Position-Only**, **28.52 pp** 3D Swap Loss (`opus_3 = 33.74 pp`), **87.13%** Outside FP Share (`opus_2 = 93.65%`), **96.08%** 3D Surface Recall (**+1.39 pp** gain over Image-Only), and **0.00% Excluded Tooth Voxels** across `6/6` replies.
  - Learning curve (`K=1..60` scans, 3 seeds): `Position-Only` gets `0.00%` through `K=16` and **7.32%** at `K=60`. `Random-Init + Adam` collapses to **`0.00%` across 3/3 seeds at `K=1, 2, 4`** and 1/3 seeds at `K=8` (`48.37 ± 41.89%`), reaching `76.21 ± 11.77%` at `K=60` (Seed 2 stalled at `62.61%`). `WSP Warm-Start + Adam` has **0/126 collapsed runs**, starts at **60.41% (`opus` 64.65%) from `K=1` scan**, leads Random-Init at **7/7 budgets**, and reaches **79.21 ± 4.48% overall / 83.01 ± 0.72% (`opus_1..3`: 82.05%, 83.50%, 83.46%; 0.10% excluded voxels; 94.4% surface recall)** at `K=60`.
- **`dentvoxel_cls` (`cls-lenet2d-v1`, 261 params, `n=1,720` axial slices / `20` scans):**
  - Frozen `K=0` (0/6 sign flips): `astra_1` 0.8343, `astra_2` **0.9670** (PR-AUC 0.9477, Within-Scan 0.9713, Acc 90.5%), `astra_3` 0.8412, `opus_1` 0.8403 (Margin-Crop **0.9181**), `opus_2` 0.8237, `opus_3` **0.9129** (mean **0.8699 Full / 0.8543 Margin-Crop**).
  - Learning curve (`K=2..60` scans, 5 seeds): At `K=2` scans, Warm-Start hits **0.9812 Full / 0.9714 Margin-Crop** (`warm_astra` **0.9860 / 0.9817**) vs. Random-Init **0.8729 / 0.8227** (**+10.8 / +14.9 pts**), surpassing full-dataset `K=60` Random-Init (`0.9667 / 0.9062`) with >30× fewer labeled scans and >20× smaller border-crop drop (`0.0024` for `warm_astra` vs. `0.0605` for Random-Init).

---

## 3. Single-Shot Held-Out Test Results (`04_sealed_test_evaluation.ipynb`)

> **Status:** Pre-registered and authorized via [`docs/test_authorization.json`](test_authorization.json) (`commit_sha = 05ae8d24940397c55b2c250a50860560a89888d2`). Pending execution of `notebooks/04_sealed_test_evaluation.ipynb` on Colab GPU (`dental_wsp_sealed_test_bundle.zip`).

| Task (`test` cohort) | Frozen WSP `K=0` (Mean ± SD, 6/6) | Trained Pos-Only (`K_max`) | Random-Init Low-`K` → Full-`K` | WSP Warm-Start Low-`K` → Full-`K` | Key Audit Metric (Excl. Tooth / Margin-Crop) |
|---|---|---|---|---|---|
| `akudental_cls` (`n=66`) | *TBD from test bundle* | — | *TBD* | *TBD* | *TBD* |
| `akudental_seg` (`n=66`) | *TBD from test bundle* | *TBD* | *TBD* | *TBD* | *TBD* |
| `denpar_cls` (`n=200`) | *TBD from test bundle* | — | *TBD* | *TBD* | *TBD* |
| `denpar_seg` (`n=200`) | *TBD from test bundle* | *TBD* | *TBD* | *TBD* | *TBD* |
| `dentvoxel_cls` (`20` scans) | *TBD from test bundle* | — | *TBD* | *TBD* | *TBD* |
| `dentvoxel_seg` (`20` scans) | *TBD from test bundle* | *TBD* | *TBD* | *TBD* | *TBD* |

---

## 4. Mechanistic Discussion & Key Scientific Takeaways

1. **Physical Intensity Calibration Is the Decisive Unlock for WSP:**
   In Lindholz et al. (arXiv:2611.12066), frozen LLM weights had to guess relative grayscale thresholds on uncalibrated 2D medical images. Across our 3 dental modalities, uncalibrated 2D grayscale radiographs produce exposure-dependent threshold shifts, letterbox-border sign flips (`denpar_cls` `opus_1..3`), and sensor-rim contamination (`denpar_seg`). Moving to **calibrated CBCT Hounsfield Units (`[-1000, +3000]`)** resolves this bottleneck directly: blind LLMs translate textbook radiographic physics (air ≈ 0.0, cortical bone ≈ 0.45–0.60, dentin/enamel ≈ 0.55–1.00) into exact weights, achieving **0/6 sign flips** and `0.967` peak zero-shot sinus AUC on `dentvoxel_cls` and `53.4%` peak zero-shot 3D tooth Dice on `dentvoxel_seg`.
2. **Spatial Prior Function Shifts Systematically Across Imaging Geometries:**
   - **Panoramic (`AKUDENTAL`):** Standardized jaw layout makes the `3×5` spatial grid essential (`+25.85 pp` gain, `90.1%` outside-ROI FP reduction).
   - **Periapical (`DenPAR`):** Removing standardized panoramic geometry collapses grid dependence to `+0.80 pp` — yet frozen WSP retains `43.35%` Dice purely from radiodensity filters, refuting the hypothesis that dental WSP is a fixed-coordinate shortcut.
   - **3D CBCT (`DentVoxel`):** Volumetric separation breaks the 2D positional shortcut completely (`Trained Position-Only` drops from `59.6%`/`65.4%` in 2D to `7.32%` in 3D), while the `3×3×3` trilinear grid suppresses non-dental craniofacial bone (cervical spine & skull base: `87.1%` outside-FP share, `0.00%` excluded tooth voxels, `+10.65 pp` unanimous gain).
3. **Sparse-3D Optimization Stabilization & Anti-Shortcut Warm-Starting:**
   Two practical benefits of WSP warm-starting emerge beyond zero-shot inference:
   - **Preventing sparse-3D collapse:** Under severe 3D foreground sparsity (<2% tooth voxels), random initialization collapses to all-zero predictions across `K=1..4` scans (`0.00%` Dice), whereas WSP warm-starting starts inside the enamel/dentin basin (`0/126` collapses, `60.4%` Dice from 1 scan, `83.0 ± 0.7%` at `K=60`).
   - **Shortcut resistance:** On both `denpar_cls` (`astra` family) and `dentvoxel_cls`, WSP warm-starts rely far less on outer border/FOV artifacts than randomly initialized CNNs (`margin_auc_drop = 0.0024` vs. `0.0605` on `dentvoxel_cls`). Conversely, steep negative hand-coded biases (`denpar_cls` `opus_2..3`) can induce dead-ReLU warm-start collapse (`25/240` runs) — highlighting why per-family reporting and alive-ReLU diagnostics are essential.
4. **Methodological Lesson — Coordinate-Frame Fidelity in 3D Spatial WSP:**
   During pre-test validation auditing of `DentVoxel.zip` (prior to opening any test split), we identified that raw NIfTI arrays were stored in `99 LPS + 1 LIP` orientations rather than canonical RAS+ specified in `seg_prompt.md` (and five non-contiguous scan IDs `0101..0105` replaced gaps inside `0001..0100`). Before canonicalization, prompt-faithful `opus` posterior-spine suppressors inadvertently attenuated anterior incisors (causing up to `26.8%` excluded tooth voxels and `−31.5 pp` surface-recall loss). Enforcing strict NIfTI `sform`/`qform` canonical RAS+ reorientation lifted `opus_3` by `+13.62 pp` (`39.77% → 53.39%`), lifted Outside FP Share from `62.6% → 87.1%`, and reduced excluded tooth voxels to **`0.00%` across all 6/6 replies** — demonstrating that spatial WSP acts as a sharp anatomical coordinate audit as well as a prior.
5. **Limitations & Non-Clinical Scope:**
   These 82-parameter and 261-parameter networks are intentionally ultra-compact mechanistic probes (~330–1,044 bytes of float32 weights), not clinical diagnostic systems or tooth-instance segmenters. They cannot separate adjacent roots, detect subtle caries/restorations at `64×64`, or replace full-capacity 3D U-Net / nnU-Net pipelines in clinical practice. Their legitimate value lies in auditing LLM domain physics, providing zero-shot/few-shot warm-start priors under extreme label scarcity, and ultra-lightweight 3D tooth-band ROI localization.
