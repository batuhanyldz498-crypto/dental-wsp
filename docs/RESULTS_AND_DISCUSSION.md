# Unified 3-Modality Results & Discussion (`dental-wsp`)

> **Governance & Provenance:** All **6 / 6 held-out validation suites** and **6 / 6 single-shot held-out test suites** are complete across **36 / 36 hash-unique, blind manual WSP replies** (`astra_1..3`, `opus_1..3` reported in collection order; zero schema errors, zero cherry-picking). Held-out test evaluation was pre-registered in [`docs/test_authorization.json`](test_authorization.json) (`commit_sha = 05ae8d24940397c55b2c250a50860560a89888d2`), executed once via [`notebooks/04_sealed_test_evaluation.ipynb`](../notebooks/04_sealed_test_evaluation.ipynb), and sealed at `2026-11-27T11:06:39+00:00`. All 12 non-private aggregate test CSVs are committed under [`results/sealed_test/`](../results/sealed_test/) with SHA-256 verification below. **No clinical or diagnostic deployment claims are made:** these 82-parameter and 261-parameter networks are mechanistic research probes, shortcut-resistant low-label initializers, and lightweight ROI pre-localizers.

---

## 1. Cross-Modality Overview (`2 × 3` Matrix · Validation vs. Single-Shot Held-Out Test)

Extending Lindholz et al. (*Weight-Space Programming for Medical Imaging*, arXiv:2611.12066) across three dental imaging physics regimes — **2D panoramic (AKUDENTAL)**, **2D intra-oral periapical (DenPAR)**, and **3D cone-beam CT (DentVoxel, canonical RAS+)** — confirms every headline validation finding on the sealed held-out test splits with minimal validation-to-test drift.

### Table 1 · Track 2 Segmentation (Exact 82-Parameter Parity · Validation → Held-Out Test)

| Modality & Task (`val` / `test`) | Arch (Params) | Frozen WSP `K=0` Dice *(Val → **Test**)* | Frozen Grid Gain `A − B` *(**Test**)* | Frozen Swap Loss A *(**Test**)* | Frozen Outside FP Share *(**Test**)* | Frozen Excl. Tooth *(**Test**)* | Trained Pos-Only `K_max` *(Val → **Test**)* | Low-`K` Warm-Start vs. Random-Init *(**Test**)* | Full-`K` Warm-Start vs. Random-Init *(**Test**)* |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| **AKUDENTAL** 2D Panoramic (`67` / `66`) | `pano-wsp82-v1` (82) | 46.98% → **46.58 ± 5.45%** (`opus_2` **53.80%**) | **+26.57 pp** *(arch critical)* | **16.64 pp** (`66/66`) | **90.27%** | **2.14%** (`opus_2` 0.96%) | 59.56% → **56.64%** (Edge Rec **5.12%**) | `K=4..32`: **52.7–57.3%** vs. **39.5–46.3%** *(+11–16 pp Dice, +13–16 pp Rec)* | **66.27%** (`opus` **67.65 ± 0.55%**) vs. **66.69%** |
| **DenPAR** 2D Periapical (`150` / `200`) | `periapical-wsp82-v1` (82) | 43.35% → **42.35 ± 7.38%** (`astra_1` **50.31%**) | **+0.88 pp** *(filter-driven)* | **10.36 pp** (`166/200`) | 80.23%* | 0.22%* | 65.37% → **66.76%** (Edge Rec **4.66%**) | `K=1..64`: Dice tied; `opus` **+5–7 pp Edge Rec** over Random | **73.54%** (`opus` **74.96 ± 0.34%**) vs. **75.14%** |
| **DentVoxel** 3D CBCT (`20` / `20` scans, RAS+) | `cbct-wsp82-v1` (82) | 47.42% → **45.54 ± 3.13%** (`opus_3` **51.34%**) | **+10.95 pp** *(6/6 positive)* | **26.35 pp** (`opus_3` **31.44**) | **90.20%** (`opus_2` **95.77%**) | **0.00%** *(6/6 zero!)* | 7.32% → **7.12%** *(fails in 3D)* | `K=1..4`: **56.9–57.5%** (`opus` **61.9%**) vs. **0.00%** Random collapse *(10/21 collapsed!)* | **78.87%** (`opus` **82.96 ± 0.53%**) vs. **75.83 ± 12.79%** |

*\*On DenPAR, bright rounded intra-oral sensor borders cause frozen enclosing boxes to span nearly the full valid frame (`Box IoU ~69.3%`, `Excl ~0.22%`), so frozen box metrics reflect full-frame coverage rather than tight cropping.*

### Table 2 · Track 1 Classification (Lindholz Mirror, `cls-lenet2d-v1`, 261 Params · Validation → Held-Out Test)

| Modality & Task (`val` / `test`) | Target | Frozen WSP `K=0` Full AUC *(Val → **Test**, Flips)* | Frozen WSP `K=0` Margin-Crop AUC *(Val → **Test**)* | Low-`K` Warm-Start vs. Random-Init *(**Test** Full / Margin)* | Full-`K` Warm-Start vs. Random-Init *(**Test** Full / Margin)* | Border-Shortcut Drop at `K_max` *(**Test**: Random vs. Best Family)* |
|---|---|---:|---:|---|---|---|
| **AKUDENTAL** 2D Panoramic (`67` / `66`) | Restoration / implant presence | 0.511 → **0.543** *(0/6)* | 0.593 → **0.544** (`opus_3` **0.794**) | Margin-Crop: Warm beats Random **7 / 7** budgets | **0.507 / 0.531** vs. Random **0.470 / 0.419** | Resolution (`64×64`) + left-edge film-strip limit |
| **DenPAR** 2D Periapical (`150` / `200`) | Upper vs. Lower arch | 0.428 → **0.467** (`astra_1` **0.777**; `opus` **3/3 flip**) | 0.501 → **0.485** (`astra_1` **0.798**) | `K=64..128` Margin: `warm_astra` **0.896–0.959** vs. Random **0.643–0.719** | All-6: **0.810 / 0.793** *(25/240 `opus` dead-ReLU)*; `warm_astra`: **0.974 / 0.966** vs. Random **0.977 / 0.812** | Random **−16.5 pts** (`−26.4` at `K=128`) vs. `warm_astra` **−0.8 pts** *(20× smaller)* |
| **DentVoxel** 3D CBCT Axial (`20` / `20` scans, `n=1720`) | Bilateral maxillary sinus presence | 0.870 → **0.868 ± 0.059** (`astra_2` **0.966**, **0/6 flips**) | 0.854 → **0.864 ± 0.077** | `K=2` scans: **0.9817 / 0.9705** (`astra` **0.9865 / 0.9836**) vs. Random **0.9004 / 0.8565** | **0.9909 / 0.9806** (`warm_astra` **0.9915 / 0.9910**) vs. Random **0.9645 / 0.9170** | Random **−4.75 pts** (`−7.69` at `K=16`) vs. `warm_astra` **−0.05 pts** *(~95× smaller)* |

---

## 2. Per-Modality Held-Out Validation Breakdown (All 6 Replies in Collection Order)

### 2.1 AKUDENTAL Validation (`train=200`, `val=67`)
- **`akudental_seg` (`pano-wsp82-v1`):** Frozen `K=0` Intact Dice: `astra_1` 49.74%, `astra_2` 48.93%, `astra_3` 49.21%, `opus_1` 41.94%, `opus_2` **54.51%**, `opus_3` 37.56% (mean **46.98%**, Image-only **21.13%**, Position-only **0.00%**, Swap Loss **14.58 pp**, Outside FP Share **90.13%**, Edge Recall **32.25%**, Excluded Tooth **2.13%**). Learning curve (`K=1..200`, 3 seeds): Frozen WSP beats Random-Init through `K=32` (`45.32%`); Warm-Start leads Random-Init by **+12–15 pp** across `K=4..32` (`K=200`: Warm-Start **68.19%** vs. Random **68.60%**, Position-Only **59.56%** / 5.41% Edge Recall).
- **`akudental_cls` (`cls-lenet2d-v1`):** Full-frame `64×64` near chance (`0.47–0.55`); under 10% border zeroing (Margin-Crop), Frozen WSP (`0.5930` mean; `opus_3` **0.8768**, `astra_1` 0.6679, `astra_2` 0.6201) beats every trained budget and Warm-Start beats Random-Init at **7 / 7** budgets.

### 2.2 DenPAR Validation (`train=650`, `val=150`)
- **`denpar_seg` (`periapical-wsp82-v1`):** Frozen `K=0` Intact vs. Image-only Dice: `astra_1` 51.01% vs 49.66%, `astra_2` 48.71% vs 47.44%, `astra_3` 49.65% vs 49.00%, `opus_1` 36.98% vs 36.69%, `opus_2` 36.14% vs 35.37%, `opus_3` 37.58% vs 37.12% (mean **43.35%** vs **42.55%**, **+0.80 pp** grid gain; Position-Only **0.00%**; Swap Loss **12.65 pp**). Learning curve (`K=1..200`): Random-Init `52.25% → 73.65%`, Position-Only `65.37%` (4.70% Edge Rec), Warm-Start `71.97%` (`opus_1..3` **73.32%**, leading Random-Init by `+5.5–7.6 pp` Edge Recall across `K=1..64`; `astra_3` plateaus at `67.24%`).
- **`denpar_cls` (`cls-lenet2d-v1`):** Frozen Full → Margin AUC: `astra_1` **0.6979 → 0.8245**, `astra_2` **0.5540 → 0.7464**, `astra_3` 0.4933 → 0.5252; `opus_1..3` **3 / 3 sign-flip** (`0.2375–0.3375`). Trained `K=650`: Random-Init `0.9213 Full → 0.7960 Margin` (`−12.5 pts`), `warm_astra` **0.9233 Full / 0.9300 Margin** (`0` dead runs), `opus_2..3` **25 / 240 dead-ReLU runs (`AUC = 0.5000`)**.

### 2.3 DentVoxel Validation (`train=60`, `val=20` scans, canonical RAS+)
- **`dentvoxel_seg` (`cbct-wsp82-v1`):** Frozen `K=0` Intact (Image-Only): `astra_1` 44.77% (33.27%), `astra_2` 46.69% (34.76%), `astra_3` 46.26% (34.27%), `opus_1` 45.07% (37.59%), `opus_2` 48.37% (38.99%), `opus_3` **53.39%** (41.78%) → mean **47.42 ± 3.19%**, **+10.65 pp** grid gain (`6/6` positive), **0.00% Position-Only**, **28.52 pp** 3D Swap Loss, **87.13%** Outside FP Share, **96.08%** 3D Surface Recall, **0.00% Excluded Tooth Voxels (`6/6`)**. Learning curve (`K=1..60`): Position-Only `7.32%`; Random-Init collapses `0.00%` across `K=1..4` (`76.21 ± 11.77%` at `K=60`); Warm-Start has `0/126` collapses, hits **60.41%** (`opus` 64.65%) at `K=1` and **79.21%** (`opus` **83.01 ± 0.72%**) at `K=60`.
- **`dentvoxel_cls` (`cls-lenet2d-v1`, `n=1,720` slices):** Frozen `K=0` (`0/6` flips): `astra_1` 0.8343, `astra_2` **0.9670** (PR-AUC 0.9477, Within-Scan 0.9713), `astra_3` 0.8412, `opus_1` 0.8403 (Margin **0.9181**), `opus_2` 0.8237, `opus_3` **0.9129** → mean **0.8699 / 0.8543**. `K=2` Warm-Start (**0.9812 / 0.9714**) beats `K=60` Random-Init (`0.9667 / 0.9062`).

---

## 3. Single-Shot Held-Out Test Results (`04_sealed_test_evaluation.ipynb`)

Executed **once** on `2026-11-27T11:06:39+00:00` under [`docs/test_authorization.json`](test_authorization.json) with zero tuning or retries. Exact non-private aggregate CSVs are archived under [`results/sealed_test/`](../results/sealed_test/).

### 3.1 Held-Out Test Stage 1 Frozen WSP (`K = 0`, All 6 Replies in Collection Order)

#### Track 2 Segmentation (`K = 0` Intact Dice / Image-Only Dice / Swap Loss A / Outside FP Share / Excl. Tooth)
| Reply | `akudental_seg` (`n=66`) | `denpar_seg` (`n=200`) | `dentvoxel_seg` (`20` scans, RAS+) |
|---|---|---|---|
| `astra_1` | 48.83% / 23.62% / 17.34 pp / 93.46% / 0.00% | **50.31%** / 48.72% / 12.06 pp / 87.36% / 0.06% | 42.75% / 30.99% / 24.04 pp / 88.29% / **0.00%** |
| `astra_2` | 48.13% / 23.76% / 17.71 pp / **94.48%** / 0.00% | 47.78% / 46.36% / 11.84 pp / 128.99% / 0.14% | 44.92% / 32.63% / 24.78 pp / 87.67% / **0.00%** |
| `astra_3` | 48.07% / 18.08% / 16.72 pp / 86.88% / 0.00% | 49.02% / 48.25% / **12.35 pp** / 71.20% / 0.06% | 44.22% / 32.02% / 25.23 pp / 88.17% / **0.00%** |
| `opus_1` | 42.37% / 16.32% / 14.19 pp / 87.22% / 9.27% | 35.77% / 35.41% / 8.93 pp / 61.80% / 0.53% | 43.44% / 35.48% / 24.89 pp / 95.02% / **0.00%** |
| `opus_2` | **53.80%** / 16.52% / **19.01 pp** / 88.87% / 0.96% | 34.87% / 34.23% / 8.09 pp / 69.12% / 0.47% | 46.59% / 36.76% / 27.72 pp / **95.77%** / **0.00%** |
| `opus_3` | 38.26% / 21.74% / 14.88 pp / 90.72% / 2.60% | 36.34% / 35.85% / 8.91 pp / 62.93% / 0.05% | **51.34%** / **39.67%** / **31.44 pp** / 86.30% / **0.00%** |
| **Mean ± SD** | **46.58 ± 5.45%** / 20.01% / 16.64 pp / 90.27% / 2.14% | **42.35 ± 7.38%** / 41.47% / 10.36 pp / 80.23% / 0.22% | **45.54 ± 3.13%** / 34.59% / 26.35 pp / **90.20%** / **0.00%** |

*(Position-Only `K = 0` Dice = `0.00%` across all 18/18 frozen segmentation evaluations.)*

#### Track 1 Classification (`K = 0` Full ROC-AUC → Margin-Crop ROC-AUC)
| Reply | `akudental_cls` (`n=66`) | `denpar_cls` (`n=200`, 79 Upper) | `dentvoxel_cls` (`20` scans, `n=1,720` slices) |
|---|---|---|---|
| `astra_1` | 0.5275 → 0.6094 | **0.7774 → 0.7975** *(72.0% Acc)* | 0.8469 → 0.8712 *(Within-Scan 0.8669)* |
| `astra_2` | 0.5788 → 0.5387 | 0.5775 → 0.6829 | **0.9658 → 0.9488** *(PR 0.9253, Within-Scan **0.9755**, Acc 90.1%)* |
| `astra_3` | **0.6012** → 0.4400 | 0.6076 → 0.5158 | 0.8114 → 0.7374 *(Within-Scan 0.8072)* |
| `opus_1` | 0.5312 → 0.4388 | 0.3214 → 0.3632 *(sign-flip)* | 0.8371 → **0.9238** *(Within-Scan 0.8530)* |
| `opus_2` | 0.4875 → 0.4412 | 0.2636 → 0.2692 *(sign-flip)* | 0.8327 → 0.8173 *(Within-Scan 0.8535)* |
| `opus_3` | 0.5312 → **0.7938** | 0.2546 → 0.2823 *(sign-flip)* | **0.9139 → 0.8829** *(PR 0.8235, Within-Scan 0.9197)* |
| **Mean ± SD** | **0.5429 ± 0.0407** → **0.5436** *(0 flips)* | **0.4670 ± 0.2172** → **0.4852** *(3 flips)* | **0.8680 ± 0.0592** → **0.8636 ± 0.0766** *(**0 flips**)* |

### 3.2 Held-Out Test Learning Curves Highlights (Stages 2 & 3)
- **`dentvoxel_seg` (`20` test scans):** `Trained Position-Only` gets `0.00%` through `K=16` and **7.12%** at `K=60`. `Random-Init + Adam` collapses to **`0.00%` across 3/3 seeds at `K=1, 2, 4`** and 1/3 at `K=8` (`47.89 ± 41.47%`), reaching `75.83 ± 12.79%` at `K=60` (Seeds 0–2: `83.04%, 83.38%, 61.06%`). `WSP Warm-Start + Adam` has **`0 / 126` zero-Dice runs**, starts at **`57.46%` (`opus` 61.94 ± 0.84%) from `K=1` scan**, beats Random-Init at **7 / 7 budgets**, and reaches **`78.87 ± 4.79%` overall / `82.96 ± 0.53%` for `opus_1..3`** (`opus_1` 82.30%, `opus_2` 83.28%, `opus_3` **83.30%**, **`0.03–0.08%` excluded voxels**, **`95.9%` 3D surface recall** — `24×` tighter SD than Random-Init!).
- **`dentvoxel_cls` (`20` test scans, `1,720` slices):** At **`K = 2` scans**, WSP Warm-Start hits **`0.9817 Full / 0.9705 Margin-Crop`** (`warm_astra` **0.9865 / 0.9836**) vs. Random-Init `0.9004 / 0.8565` — surpassing full-dataset `K = 60` Random-Init (`0.9645 / 0.9170`) with **>30× fewer scans**. At `K=60`, `warm_astra` reaches **`0.9915 Full / 0.9910 Margin-Crop`** (`margin_auc_drop = 0.0005` vs. Random-Init `0.0475`, **`0 / 180` dead runs**).
- **`akudental_seg` (`66` test panoramics):** Frozen WSP (`46.58% / 33.06%`) beats Random-Init through **`K = 32`** (`45.32% / 17.94%`); Warm-Start beats Random-Init by **+11 to +16 pp Dice** across `K=2..32`, finishing at `66.27%` (`opus` **67.65 ± 0.55%**, `opus_1` 68.30%) vs. Random `66.69%` and Position-Only `56.64%` (`5.12%` Edge Recall).
- **`denpar_seg` (`200` test periapicals):** Random-Init finishes at `75.14% / 38.39%`, Position-Only at `66.76% / 4.66%` (`14.19%` excluded teeth), and Warm-Start at `73.54%` (`opus_1..3` **74.96 ± 0.34%**, leading Random-Init by `+5–7 pp` Edge Recall across `K=1..64`; `astra_3` stalls at `68.77%`).
- **`denpar_cls` (`200` test periapicals):** At `K=650`, Random-Init hits `0.9765` Full **but plunges −16.5 pts to `0.8117` under Margin-Crop** (`−26.4 pts` at `K=128`), whereas `warm_astra` hits **`0.9742 Full / 0.9659 Margin-Crop`** (`−0.83 pt` drop, `0` dead runs). `opus_2` & `opus_3` replicate the exact validation pathology (**25 / 240 dead-ReLU runs**).
- **`akudental_cls` (`66` test panoramics):** Full-Frame stays near chance (`0.46–0.55`), while on Margin-Crop both Frozen WSP (`0.5436`; `opus_3 = 0.7938`) and Warm-Start (`0.5260–0.5506`) beat Random-Init (`0.4185–0.5342`) at **7 / 7 budgets**.

### 3.3 Sealed Test Artifact SHA-256 Receipts (`results/sealed_test/`)

| Committed Artifact (`results/sealed_test/`) | SHA-256 Digest |
|---|---|
| `akudental_cls_test_stage1_frozen.csv` | `35e74c338147fab3380d4dbb3c3288815881ab8fe527b850e60890d8a5e8a68a` |
| `akudental_cls_test_summary.csv` | `442769f9333c0f446b0e321d2e582f391d7bff8d220c2b46daa73ac19ea95216` |
| `akudental_seg_test_stage1_frozen.csv` | `2b54726862804827ad6cae0849be4d7eb8ef8ed61f0d0ed6ae3b254c0888c35b` |
| `akudental_seg_test_summary.csv` | `4757b00a2cc5a97cd67378cec09aebba123cb9ca92961e59c09dd7bdcbdb478c` |
| `denpar_cls_test_stage1_frozen.csv` | `ab8b6135eeeb9e1baee2c41b71b9004f20b4000f877de90fe36563652afbb477` |
| `denpar_cls_test_summary.csv` | `bdfe4345610406549c9356d4dc0e37e4b0037e3a635e93c2ef6d10c06a9953b3` |
| `denpar_seg_test_stage1_frozen.csv` | `81d01ea55b952e61f56018cd4819c3a16662c98e262f6ef63825486e8cfc918b` |
| `denpar_seg_test_summary.csv` | `4f30835bdb02ad66be533e1b9ac876d2a6d5e8d9faee6c3024d4e7fdddf6722a` |
| `dentvoxel_cls_test_stage1_frozen.csv` | `a3b1dae18dffd51901073ca635e1dc434d3e3c019a12a42b3322b6a060dd1f9c` |
| `dentvoxel_cls_test_summary.csv` | `170a4c9532f591c11cbd970701545fb646518c4a6004bd347c56a6757b69c676` |
| `dentvoxel_seg_test_stage1_frozen.csv` | `4d3eeccd24fa1e92eb40b4f21e9e944ef3d9a9ad59950262bc60ad108bf12d80` |
| `dentvoxel_seg_test_summary.csv` | `a1eb20cd90c9afca118e0b37ce7f09e5a7916bfecdda6325fbd4a64c44bf6895` |

---

## 4. Mechanistic Discussion & Key Scientific Takeaways

1. **Physical Intensity Calibration Is the Decisive Unlock for WSP:**
   In Lindholz et al. (arXiv:2611.12066), frozen LLM weights had to guess relative thresholds on uncalibrated 2D images. Across our 3 dental modalities, uncalibrated 2D grayscale radiographs exhibit exposure-dependent shifts, letterbox-border sign flips (`denpar_cls` `opus_1..3`), and sensor-rim artifacts (`denpar_seg`). Calibrated CBCT Hounsfield Units (`[-1000, +3000]`) remove this barrier directly: blind LLMs map textbook attenuation physics (air ≈ 0.0, cortical bone ≈ 0.45–0.60, dentin/enamel ≈ 0.55–1.00) into exact weights, achieving **0 / 6 sign flips** and `0.966` peak zero-shot sinus AUC on `dentvoxel_cls` test and `51.3%` peak zero-shot 3D tooth Dice on `dentvoxel_seg` test.
2. **Spatial Prior Function Shifts Systematically Across Imaging Geometries:**
   - **Panoramic (`AKUDENTAL`):** Standardized jaw layout makes the `3×5` spatial grid essential (`+26.57 pp` test gain, `90.27%` outside-ROI FP reduction).
   - **Periapical (`DenPAR`):** Removing standardized panoramic geometry collapses grid dependence to `+0.88 pp` — yet frozen WSP retains `42.35%` test Dice purely from radiodensity filters, refuting the critique that dental WSP is merely a fixed-coordinate shortcut.
   - **3D CBCT (`DentVoxel`):** Volumetric separation breaks the 2D positional shortcut completely (`Trained Position-Only` drops from `56.6%`/`66.8%` in 2D to `7.12%` in 3D), while the `3×3×3` trilinear grid erases posterior cervical spine & skull-base false positives (`90.20%` test outside-FP share, **`0.00%` excluded tooth voxels across 6/6**, `+10.95 pp` unanimous gain).
3. **Sparse-3D Optimization Stabilization & Anti-Shortcut Warm-Starting:**
   - **Preventing sparse-3D collapse:** Under severe 3D foreground sparsity (<2% tooth voxels), random initialization collapses to all-zero predictions across `K = 1..4` scans (`0.00%` Dice, `10/21` runs collapsed on test), whereas WSP warm-starting starts inside the enamel/dentin basin (`0/126` collapses, `57.5%` overall / `61.9%` `opus` Dice from 1 scan, `82.96 ± 0.53%` at `K=60`).
   - **Shortcut resistance vs. dead-ReLU risk:** On both `denpar_cls` (`warm_astra`: `0.974 → 0.966` Margin vs. Random `0.977 → 0.812`) and `dentvoxel_cls` (`warm_astra`: `0.9915 → 0.9910` vs. Random `0.9645 → 0.9170`), WSP warm-starting cuts border-shortcut drop by **20–95×**. Conversely, steep negative hand-coded biases (`denpar_cls` `opus_2..3`) cause dead-ReLU warm-start collapse (`25/240` runs on both validation and test), underscoring why per-family transparent reporting is mandatory.
4. **Methodological Lesson — Coordinate-Frame Fidelity in 3D Spatial WSP:**
   Pre-test validation auditing of `DentVoxel.zip` uncovered `99 LPS + 1 LIP` raw NIfTI orientations (opposite `Y` from prompt-specified RAS+, plus 1 coronal-stored volume) and 5 non-contiguous scan IDs (`0101..0105`). Before canonicalization, prompt-faithful `opus` posterior-spine suppressors attenuated anterior incisors (up to `26.8%` excluded tooth voxels and `−31.5 pp` surface-recall loss). Enforcing strict NIfTI `sform`/`qform` RAS+ reorientation lifted frozen `opus_3` by `+13.62 pp`, raised Outside FP Share from `62.6% → 87.1%` (`90.2%` test), and reduced excluded tooth voxels to **`0.00%` across all 6 / 6 replies on both validation and held-out test**.
5. **Why Tooth-Union Segmentation (Not CEJ / Root-Apex Landmarking) on DenPAR — Limitations & Scope:**
   Although DenPAR also distributes point-level `Key Points Annotations` (CEJ, root apex) and `Bone Level Annotations`, `dental-wsp` intentionally evaluated **tooth-union segmentation** and **Upper vs. Lower arch classification** for three pre-registered reasons: (i) **cross-modality parity** — holding both target (natural-tooth union) and capacity (exact 82 parameters) identical across panoramic, periapical, and 3D CBCT is required to isolate imaging geometry and HU calibration cleanly; (ii) **architectural capacity** — 6 single-layer `3×3` filters cannot regress sub-millimeter anatomical intersections (CEJ/apex) amidst overlapping trabecular bone and cervical burnout; and (iii) **strict non-clinical scoping** — radiographic bone-loss (CEJ-to-crest) staging is a clinical diagnostic task outside the legitimate scope of 82p/261p mechanistic probes. Future work may evaluate ultra-lightweight 82p WSP masks strictly as coarse ROI pre-localizers feeding full-capacity landmark detectors.
