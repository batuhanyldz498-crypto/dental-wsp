# Protocol Deviations, Audit Trail & Methodological Disclosures (`dental-wsp`)

This document discloses all known deviations between the initial protocol ([`docs/BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md)), the locked per-task specifications (`benchmarks/*/configs/*.yaml` and `benchmarks/*/locks/*.design.lock.json`), the internal test-authorization receipt ([`docs/test_authorization.json`](test_authorization.json)), and the code and schedules actually executed for the v0.1.0 validation and sealed held-out test runs.

The sealed held-out test CSVs under [`results/sealed_test/`](../results/sealed_test/) and [`docs/test_authorization.json`](test_authorization.json) are preserved **unchanged** (verified by SHA-256 checksums in [`tests/test_repo_hygiene.py`](../tests/test_repo_hygiene.py)). Rather than retroactively editing historical locks or rewriting Git history, every discrepancy is documented here plainly.

---

## 1. Programmatic Diff: Protocol & Design Locks vs. Executed Schedules

Across all six tasks, each `benchmarks/<dataset>/configs/<track>.yaml` is identical to the `protocol` block inside its corresponding `benchmarks/<dataset>/locks/<task>.design.lock.json` (verified by `dental-wsp lock-design <task> --check-only`). The table below lists every difference between `docs/BENCHMARK_PROTOCOL.md` / design locks and the `frozen_schedules` executed in validation notebooks `01–03` and the held-out test run (`notebooks/04_sealed_test_evaluation.ipynb`, recorded in `docs/test_authorization.json`).

| Task | Parameter | Locked / Protocol Value | Executed Value |
|---|---|---|---|
| `akudental_seg` | `lr` | Protocol `0.01` (`1e-2`) / Lock `0.01` | `0.03` |
| `akudental_seg` | `epochs` | Protocol `200` / Lock `200` | `150` |
| `akudental_seg` | `seeds` | Protocol `5` / Lock `[0, 1, 2, 3, 4]` | `[0, 1, 2]` (3 seeds) |
| `akudental_seg` | `batch_size` | Protocol & Lock unspecified (trainer default `16`) | `32` |
| `akudental_seg` | `cosine_lr` | Protocol: *"cosine decay or fixed 200 epochs"* / Lock unspecified | `true` (`CosineAnnealingLR`, `eta_min = 0.05 * lr`, stepped per epoch) |
| `akudental_seg` | `budgets` (`K`) | Protocol `{1, 2, 4, 8, 16, 32, full_train}` / Lock `[1, 2, 4, 8, 16, 32, 64]` | `[1, 2, 4, 8, 16, 32, 64, 200]` (`64` added vs. protocol; `200 = full_train` added vs. lock) |
| `akudental_seg` | `init` (`feature_bias`) | Protocol & Lock unspecified | `+0.1` (`alive ReLU` random init added in commit `855c016`; `output_bias = 0.0`) |
| `denpar_seg` | `lr` | Protocol `0.01` (`1e-2`) / Lock `0.01` | `0.03` |
| `denpar_seg` | `epochs` | Protocol `200` / Lock `200` | `150` |
| `denpar_seg` | `seeds` | Protocol `5` / Lock `[0, 1, 2, 3, 4]` | `[0, 1, 2]` (3 seeds) |
| `denpar_seg` | `batch_size` | Protocol & Lock unspecified (trainer default `16`) | `32` |
| `denpar_seg` | `cosine_lr` | Protocol: *"cosine decay or fixed 200 epochs"* / Lock unspecified | `true` (`CosineAnnealingLR`, `eta_min = 0.05 * lr`, stepped per epoch) |
| `denpar_seg` | `budgets` (`K`) | Protocol `{1, 2, 4, 8, 16, 32, full_train}` (`650` train images) / Lock `[1, 2, 4, 8, 16, 32, 64]` | `[1, 2, 4, 8, 16, 32, 64, 200]` — **`K_max = 200` of 650 training images (30.8%), NOT full training set** |
| `denpar_seg` | `init` (`feature_bias`) | Protocol & Lock unspecified | `+0.1` (`alive ReLU` random init, commit `855c016`; `output_bias = 0.0`) |
| `dentvoxel_seg` | `lr` | Protocol `0.01` (`1e-2`) / Lock `0.01` | `0.03` |
| `dentvoxel_seg` | `epochs` | Protocol `200` / Lock `150` | `120` |
| `dentvoxel_seg` | `seeds` | Protocol `5` / Lock `[0, 1, 2, 3, 4]` | `[0, 1, 2]` (3 seeds) |
| `dentvoxel_seg` | `batch_size` | Protocol & Lock unspecified (trainer hardcodes `4` for `cbct-wsp82-v1`) | `4` |
| `dentvoxel_seg` | `cosine_lr` | Protocol: *"cosine decay or fixed 200 epochs"* / Lock unspecified | `true` (`CosineAnnealingLR`, `eta_min = 0.05 * lr`, stepped per epoch) |
| `dentvoxel_seg` | `init` (`feature_bias`) | Protocol & Lock unspecified | `+0.1` (`alive ReLU` random init, commit `855c016`; `output_bias = 0.0`) |
| `akudental_cls` | `budgets` (`K`) | Protocol `{1, 2, 4, 8, 16, 32, full_train}` / Lock `[2, 4, 8, 16, 32, 64]` | `[2, 4, 8, 16, 32, 64, 200]` (starts at `K = 2` to include both classes; `64` added vs. protocol; `200 = full_train` added vs. lock) |
| `denpar_cls` | `budgets` (`K`) | Protocol `{1, 2, 4, 8, 16, 32, full_train}` / Lock `[2, 4, 8, 16, 32, 64, 128]` | `[2, 4, 8, 16, 32, 64, 128, 650]` (starts at `K = 2`; `64, 128` added vs. protocol; `650 = full_train` added vs. lock) |
| `dentvoxel_cls` | `budgets` (`K` scans) | Protocol `{1, 2, 4, 8, 16, 32, full_train}` / Lock `[2, 4, 8, 16, 32, 60]` | `[2, 4, 8, 16, 32, 60]` (starts at `K = 2` scans; matches lock) |

**Parameters that matched protocol and design locks:**
- All three classification tasks (`akudental_cls`, `denpar_cls`, `dentvoxel_cls`) matched `lr = 0.005`, `epochs = 200` (full-batch Adam, no LR decay), and `seeds = [0, 1, 2, 3, 4]`.
- `dentvoxel_seg` budgets `[1, 2, 4, 8, 16, 32, 60]` matched both the design lock and the protocol (`full_train = 60` scans).
- Architecture parameter counts (`82` for all 3 segmentation tracks; `261` for all 3 classification tracks), input canvas dimensions, intensity normalization rules, and inclusive `logit >= 0.0` thresholding matched the locks across all six tasks.

**Consequence of fixed-epoch segmentation schedules (`dentvoxel_seg` low-`K` step confound):**
Because `wsp_core/training/trainer.py` trained segmentation models for a fixed number of **epochs** (`120` in 3D at `batch_size = 4`, `150` in 2D at `batch_size = 32`), total Adam optimizer steps scaled with `K`:
- On `dentvoxel_seg`: `K ∈ {1, 2, 4}` received only **120 optimizer steps**, `K = 8` received **240 steps**, `K = 16` received **480 steps**, `K = 32` received **960 steps**, and `K = 60` received **1,800 steps**.
- Combined with `output_bias = 0.0` under `<2%` 3D foreground voxel prevalence, the `0.00%` Dice "collapse" of random initialization at `K ∈ {1, 2, 4}` is confounded by severe under-training relative to `K = 60` and an uncalibrated initial bias. Fixed-step (`max_steps`) and prior-bias (`init_output_bias="prior"`) controls are added in v0.1.1 (validation split only) to test this confound directly.

---

## 2. Chronological Audit Trail (Dates & Commits from `git log`)

All timestamps below are converted to UTC from `git log` commit headers and JSON lock metadata.

1. **`2026-11-24T11:23:15Z` (`96045d9`):** Initial repository scaffold, `docs/BENCHMARK_PROTOCOL.md`, configs, and design locks.
2. **`2026-11-24T12:46Z–14:32Z` (`b4d03b4`, `e6c6697`):** Locked 6 replies each for `akudental_cls` and `akudental_seg`.
3. **`2026-11-25T16:42:55Z` (`855c016`):** Changed segmentation training to mini-batch cosine LR (`lr = 0.03`, `epochs = 150` 2D / `120` 3D, `seeds = [0, 1, 2]`, `feature_bias = +0.1` alive ReLU init). Existing `configs/*.yaml` and `*.design.lock.json` were not updated to reflect these schedule changes.
4. **`2026-11-25T22:02:10Z–22:02:17Z` (`0f0316a`, committed `22:04:43Z`):** Locked 24 replies across DenPAR and DentVoxel. In `dentvoxel_seg.responses.lock.json`, `astra_3` was accidentally pasted as an exact duplicate of `astra_1` (identical `raw_sha256` and `parameters_sha256`).
5. **`2026-11-26T09:50:30Z` → `10:09:09Z` (`05ae8d2`):**
   - **Duplicate reply replacement after validation had begun:** `dentvoxel_seg` `astra_3` was re-collected at `2026-11-26T09:50:30Z` (recorded in `dentvoxel_seg.responses.lock.json` under `attempts.astra_3.metadata.notes`: `"Replaced copy-paste duplicate of astra_1 caught during pre-test validation audit"`). Because this replacement occurred after initial validation runs had already begun, `astra_3` was not collected strictly prior to all validation inspection.
   - **DentVoxel LPS/LIP → RAS+ reorientation after observing validation results:** Raw DentVoxel NIfTI files had `99 LPS + 1 LIP` orientations (opposite anterior–posterior `Y` axis relative to the RAS+ frame described in the prompt, plus 1 coronal-stored scan) and 5 non-contiguous scan IDs (`img0001..0105`). Commit `05ae8d2` added NIfTI `sform`/`qform` canonicalization to RAS+ and dynamic 100-scan discovery **after** poor validation scores were observed on raw LPS volumes. Consequently, the DentVoxel validation split acted partly as a development/debugging set rather than a strictly untouched validation set.
6. **`2026-11-26T13:50:00Z` vs. `2026-11-27T05:47:59Z` (`33ca670`):**
   - `docs/test_authorization.json` records `"approved_at_utc": "2026-11-26T13:50:00+00:00"` and pins `"commit_sha": "05ae8d24940397c55b2c250a50860560a89888d2"`, but the file was **first committed to Git on `2026-11-27T05:47:59Z`** (local `2026-11-27T08:47:59+03:00`) in commit `33ca670` alongside `notebooks/04_sealed_test_evaluation.ipynb` and `wsp_core/training/runner.py` (`split_label` support).
   - **"Pre-registered" clarification:** `docs/test_authorization.json` is an **internal Git-tracked authorization file**, not a time-stamped third-party pre-registration on an external registry such as OSF or AsPredicted.
7. **`2026-11-27T08:44:27Z` (`34a6b16`, local `11:44:27+03:00`):**
   - **Code changes after the pinned authorization commit (`05ae8d2`):** The final sealed test run executed code from commit `34a6b16`, which includes two post-`05ae8d2` commits:
     - `33ca670`: added `split_label` to `wsp_core/training/runner.py` (CSV naming and plot titles) and `notebooks/04_sealed_test_evaluation.ipynb`.
     - `34a6b16`: `"Fix DenPAR RGBA test mask 431.png and pre-stage <=2GB CUDA segmentation tensors"` — changed `benchmarks/akudental/adapter.py` (`prepare_panorama`, shared by DenPAR) to convert PNG masks via `.convert("L")` (and `.max(axis=-1)` on 3-channel arrays) and added a performance-only GPU tensor pre-staging optimization (`<= 2 GB` CUDA tensors) in `wsp_core/training/trainer.py`.
   - **One test-split load crash before the completed run (do not claim "zero retries"):** Because DenPAR `Testing/431.png` was stored as a 4-channel RGBA mask (causing a shape mismatch in `prepare_panorama` when loading the DenPAR test split), the held-out test split was loaded and crashed once during execution before `34a6b16` fixed RGBA mask loading and the completed sealed test run finished at `2026-11-27T11:06:39Z`. No hyperparameters, architectures, prompts, or locked weights were altered in `34a6b16`, and the crash occurred during mask loading rather than model evaluation, but the execution was **not** literally "zero retries".
8. **`2026-11-27T11:06:39Z` → `11:47:11Z` (`9c2e9b5`):** Completed held-out test run sealed locally (`final_test_exposure.private.json`) and 12 aggregate test CSVs committed to `results/sealed_test/`.

---

## 3. Dataset, Split & Optimization Disclosures

1. **AKUDENTAL provisional groups (patient-independence caveat):**
   In the official public AKUDENTAL adapter (`benchmarks/akudental/adapter.py`), each image file is assigned `group_id = image_stem` (one image per group), and the stratified 60/20/20 split is performed at the image level because public AKUDENTAL metadata does not provide verified patient IDs. These provisional groups **do not establish patient independence** across train/val/test splits. If multiple radiographs from the same patient exist in AKUDENTAL, trained arms (`random_init_adam`, `wsp_warmstart_adam`, `trained_position_only`) carry potential patient-level leakage risk (frozen `K = 0` arms do not train on the data).
2. **`denpar_seg` maximum budget (`K_max = 200` of 650):**
   Although DenPAR provides `650` official training radiographs (and `denpar_cls` trains up to `K_max = 650`), `denpar_seg` stopped at `K_max = 200` (30.8% of the training split) for compute parity with `akudental_seg` (`200`). Any earlier phrasing referring to `K = 200` on `denpar_seg` as "full-K" was inaccurate; v0.1.1 replaces "Full-K" with `K_max` throughout and states `200 / 650` explicitly.
3. **Dead-ReLU warm-start failures (`denpar_cls`):**
   In `denpar_cls`, `25 / 240` warm-start runs (concentrated in `opus_2` and `opus_3`) yielded `ROC-AUC = 0.5000` on both validation and held-out test because steeply negative hand-coded `conv1_bias`/`conv2_bias` values caused all ReLU units to output zero across training samples (zero gradient). This is a known, fixable initialization/activation pathology (e.g., LeakyReLU or positive bias shift before fine-tuning), left unfixed during the v0.1.0 evaluation to preserve the locked architecture and protocol.
4. **LLM family provenance (`astra` / `opus`):**
   The locked reply files use anonymous family labels (`astra_1..3`, `opus_1..3`) collected via manual web-chat sessions (`model_label = "astra_manual_web_chat"` / `"opus_manual_web_chat"`).
   - `astra`: `TODO(author): model name/version/date`
   - `opus`: `TODO(author): model name/version/date`
   (These placeholders must be filled by the repository author before public presentation so external readers know the exact models and collection versions.)

---

## 4. Policy for Any Future Held-Out Test Runs

All new experiments added in v0.1.1 (trivial/non-LLM baselines, human-written 82-parameter prior, and fixed-step + prior-bias factorial controls) run **exclusively on the validation split (`split == "val"`)** using training subsets (`split == "train"`). The original sealed held-out test results under `results/sealed_test/` and `docs/test_authorization.json` remain untouched. Any hypothetical future held-out test run would require a separate, explicitly labeled **`POST-HOC` addendum authorization file** and would never modify `docs/test_authorization.json` or overwrite `results/sealed_test/`.
