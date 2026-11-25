# `dental-wsp` Continuation Handoff (AKUDENTAL Complete → DenPAR Running → DentVoxel Next)

## Paste This Into Your Next Antigravity Chat

> Open `C:\Users\Lenovo\Desktop\dental-wsp` and read `docs/CONTINUATION_HANDOFF.md` first. All 36/36 manual WSP replies are locked across the 2 × 3 benchmark (AKUDENTAL, DenPAR, DentVoxel), and AKUDENTAL validation is complete. I am now bringing the completed **DenPAR** (`02_denpar_wsp_benchmark.ipynb`) validation results (CSVs, learning curves, and periapical overlays) to analyze, and then we will run **DentVoxel 3D CBCT** (`03_dentvoxel_cbct_wsp_benchmark.ipynb`), prepare the final authorized one-shot held-out test evaluation, draft the full multi-modality Results & Discussion, and prepare the public GitHub release. Keep all three held-out test sets sealed until explicitly authorized, and leave `C:\Users\Lenovo\Desktop\AKUDENTAL_WSP_Spatial_Ablation` untouched.

---

## 1. Repository & Data Paths

| Asset | Location | Status |
|---|---|---|
| Standalone open-source repo (`dental-wsp`, MIT) | `C:\Users\Lenovo\Desktop\dental-wsp` (GitHub: `batuhanyldz498-crypto/dental-wsp`, currently **Private**) | Clean `main`, `pytest` passing (`23 passed, 2 skipped`) |
| Legacy Study 1 folder (separate pilot) | `C:\Users\Lenovo\Desktop\AKUDENTAL_WSP_Spatial_Ablation` | Complete & archived; **do not modify or merge** |
| Local BYOD datasets (never committed to Git) | `C:\Users\Lenovo\Desktop\WSP_Dental_Data\AKUDENTAL`<br>`C:\Users\Lenovo\Desktop\WSP_Dental_Data\DenPAR Radiographs Dataset.zip`<br>`C:\Users\Lenovo\Desktop\WSP_Dental_Data\DentVoxel.zip` | All three verified via `dental-wsp verify-dataset` |
| Google Drive mirrors (for Colab GPU) | `My Drive/dental-wsp` (code)<br>`My Drive/WSP_Dental_Data/` (datasets)<br>`My Drive/dental-wsp-runs/{akudental,denpar,dentvoxel}` (permanent CSV/PNG checkpoints) | Writable Drive checkpoints survive Colab disconnects |
| Raw manual chat inbox (git-ignored) | `C:\Users\Lenovo\Desktop\dental-wsp\inbox\<task>\<slot>.txt` | All 36 raw replies saved (`astra_1–3`, `opus_1–3` × 6 tasks) |

---

## 2. Benchmark Progress Board (2 × 3 Matrix)

All **6 `design.lock.json`** and all **6 `responses.lock.json`** (**36 / 36 VALID**, zero schema errors) are locked and committed under `benchmarks/*/locks/`:

| Task | Modality & Target | Architecture (Params) | Replies Locked | Validation Status | Held-Out Test Split |
|---|---|---|---:|---|---|
| `akudental_cls` | Panoramic · Restoration/Implant presence | `cls-lenet2d-v1` (261) | 6 / 6 | ✅ Complete (`val=67`, `K=0..200`) | 🔒 Sealed (`66`) |
| `akudental_seg` | Panoramic · Natural-tooth union (FDI 11–48) | `pano-wsp82-v1` (82) | 6 / 6 | ✅ Complete (`val=67`, converged `K=0..200` + X-ray visuals) | 🔒 Sealed (`66`) |
| `denpar_cls` | Periapical · Upper vs. Lower arch (`Arch` col) | `cls-lenet2d-v1` (261) | 6 / 6 | 🔄 Running now (`02_denpar_wsp_benchmark.ipynb`, `val=150`) | 🔒 Sealed (`200`) |
| `denpar_seg` | Periapical · Tooth union (`384×384`) | `periapical-wsp82-v1` (82) | 6 / 6 | 🔄 Running now (`02_denpar_wsp_benchmark.ipynb`, `val=150`) | 🔒 Sealed (`200`) |
| `dentvoxel_cls` | 3D CBCT · Scan-split axial maxillary sinus (`HU [-1000,3000]`) | `cls-lenet2d-v1` (261) | 6 / 6 | ⬜ Ready next (`03_dentvoxel_cbct_wsp_benchmark.ipynb`, `val=20` scans) | 🔒 Sealed (`20` scans) |
| `dentvoxel_seg` | 3D CBCT · 3D tooth union (`96×128×128`, 7-tap 3D cross-stencil) | `cbct-wsp82-v1` (82) | 6 / 6 | ⬜ Ready next (`03_dentvoxel_cbct_wsp_benchmark.ipynb`, `val=20` scans) | 🔒 Sealed (`20` scans) |

---

## 3. Completed AKUDENTAL Validation Summary (Reference Anchor)

### `akudental_seg` (`train=200`, `val=67`, `test=66` sealed; converged schedule: `epochs=150`, `lr=3e-2`, `batch_size=32`, cosine LR, `feature_bias=+0.1` alive ReLU init)

- **Stage 1 Frozen WSP (`K = 0`, 6 replies):**
  - Mean Intact Dice = **46.98%** (`astra_1`: 49.74%, `astra_2`: 48.93%, `astra_3`: 49.21%, `opus_1`: 41.94%, `opus_2`: **54.51%**, `opus_3`: 37.56%).
  - Mean Image-only Dice = **21.13%**; Position-only Dice = **0.00%** across all 6 replies.
  - Mean Intact Swap Loss = **14.58 pp** (`66/67` eligible groups); Outside-box FP reduction share = **90.13%**.
  - Mean Reference-Edge Recall = **32.25%**; Mean Excluded Tooth Pixels = **2.13%** (`opus_2` achieves **61.02% Box IoU** with only **0.29%** excluded tooth pixels, vs. `opus_1` 67.03% Box IoU with 10.22% excluded).
- **Stages 2 & 3 Learning Curve (Mean Dice % / Mean Edge Recall %):**
  - `Frozen WSP (K=0)`: **46.98% / 32.25%**
  - `WSP Warm-Start + Adam`: `K=1` **54.05% / 31.45%** (`opus_1–3` hit 63.2%–64.0% from 1 film!) → `K=8` **55.13% / 30.89%** → `K=16` **58.38% / 34.05%** → `K=32` **58.55% / 34.17%** → `K=64` **64.45% / 38.51%** → `K=200` **68.19% / 45.32%** (5/6 slots average ~69.3% Dice & ~51.5% Rec; `opus_1` hits 70.57%; only `astra_1` converged into a blocky scale-16 minimum at 62.6% / 14.9%).
  - `Random-Init + Adam`: `K=1` 45.65% / 18.91% → `K=4` 38.50% / 15.08% → `K=16` 46.19% / 17.56% → `K=32` 45.32% / 17.19% → `K=64` 61.67% / 31.57% → `K=200` **68.60% / 50.56%** (true 82-param architecture ceiling).
  - `Trained Position-Only`: `K=1` 46.73% / 4.81% → `K=32` 44.24% / 4.59% → `K=200` **59.56% / 5.41%** (visually paints a central cyan diamond with ~10× worse outline recall than image-filter models).
- **Key Takeaway:** Zero-shot Frozen WSP beats Random-Init Adam through `K = 32` panoramics, reaches 68%–79% of the full-train 82-param ceiling with zero labels, and WSP Warm-Start leads Random-Init by **+12–15 pp Dice** and **+14–17 pp Edge Recall** across `K = 4..32`.

### `akudental_cls` (`train=200`, `val=67`, `test=66` sealed)
- Full-frame `64×64` ROC-AUC stays near chance across all arms even at `K=200` (Frozen WSP `0.5108`, Random-Init `0.4679–0.5500`, Warm-Start `0.5042–0.5338`) due to resolution loss on tiny restorations and bright film-margin strips visible along the left panoramic edge.
- **Margin-Crop ROC-AUC (outer 10% border zeroed):** Frozen WSP (`0.5930` mean; `opus_3 = 0.8768`, `astra_1 = 0.6679`, `astra_2 = 0.6201`) outperforms every trained budget (`Random-Init` `0.4873–0.5355`, `Warm-Start` `0.4885–0.5565`), and WSP Warm-Start beats Random-Init at **7/7 budgets**.

---

## 4. Immediate Next Steps for the New Chat

1. **Analyze completed DenPAR validation results** (`denpar_cls` Upper vs. Lower Arch + `denpar_seg` periapical tooth union without fixed arch geometry + periapical visual overlays).
2. **Run DentVoxel 3D CBCT validation** (`03_dentvoxel_cbct_wsp_benchmark.ipynb` on Colab GPU after uploading `DentVoxel.zip` to `My Drive/WSP_Dental_Data/`), then analyze 3D tooth-union segmentation (`cbct-wsp82-v1`), 3D excluded tooth-voxel cropping rate, scan-split axial maxillary sinus classification (`dentvoxel_cls`), and axial/coronal CBCT overlays.
3. **Cross-modality synthesis & sealed test gate:** Once all 6 tasks are validated, create one pre-registered `test_authorization.json`, run the single authorized held-out test evaluation across all 3 datasets, draft the unified 3-modality paper/README Results & Discussion (with curated attributed figures), verify `pytest tests/test_repo_hygiene.py`, and flip `dental-wsp` from **Private → Public**.
