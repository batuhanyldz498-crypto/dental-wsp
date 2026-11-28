# Changelog

All notable changes to `dental-wsp` are documented here.

## [0.1.1] - 2026-11-28 (External Review Fixes & Validation Controls)

### Governance, Provenance & Disclosures
- Added [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md) with a programmatic diff of every `benchmarks/*/configs/*.yaml`, `benchmarks/*/locks/*.design.lock.json`, and `docs/BENCHMARK_PROTOCOL.md` against the executed schedules in `docs/test_authorization.json`, plus a commit-by-commit timeline (`git log`) disclosing:
  - segmentation hyperparameter and seed deviations (`lr = 0.03`, `150`/`120` epochs, 3 seeds, mini-batch cosine LR, `+0.1` alive-ReLU init);
  - `denpar_seg` `K_max = 200` of 650 training images (30.8%), replacing "Full-K" with `K_max`;
  - re-collection of `dentvoxel_seg` `astra_3` (`2026-11-26T09:50:30Z`, commit `05ae8d2`) after replacing a copy-paste duplicate of `astra_1`;
  - DentVoxel LPS/LIP → RAS+ canonicalization (`05ae8d2`) made after observing validation scores;
  - post-`05ae8d2` code used during the sealed test pass (`33ca670`, `34a6b16`), including the DenPAR RGBA mask crash on `Testing/431.png` fixed before completion (`2026-11-27T11:06:39Z`);
  - clarification that `docs/test_authorization.json` is an internal Git-tracked receipt (first committed `2026-11-27`, `33ca670`), not an external registry such as OSF;
  - provisional AKUDENTAL image-level grouping (no verified patient independence) and dead-ReLU `denpar_cls` warm-start failures (`25/240` runs);
  - explicit LLM family provenance (`astra` = GPT Astra-6.2 Max, `opus` = Opus 5.5 Max; collected `2026-11-20`, ingested into locks `2026-11-24`–`2026-11-25`, `dentvoxel_seg` `astra_3` re-collected `2026-11-26T09:50:30Z`).
- Preserved all files under `results/sealed_test/` and `docs/test_authorization.json` unchanged (SHA-256 digests unchanged and verified by `tests/test_repo_hygiene.py`).

### Metrics & Evaluation
- Documented `outside_fp_reduction_share_pct` in `wsp_core/evaluation/segmentation.py` (retaining the dict key for backward compatibility) and renamed its presentation label everywhere to **"Outside-ROI share of grid-removed FPs (A vs B)"**, with a one-line definition, >100% / negative footnote, and a legacy sealed-CSV header glossary.
- Added predicted bounding-box valid-area/volume retention fraction (`box_valid_vol_a_pct`, `box_valid_vol_b_pct`) to `evaluate_segmentation_paired_audit`, `score_segmentation_masks`, and `runner.py` to contextualize `excluded_tooth_a_pct`.
- Added Hanley–McNeil 95% confidence interval helper (`hanley_mcneil_auc_ci`) to `wsp_core/evaluation/classification.py` and reported 95% CIs for all held-out test classification AUCs (with effective scan-level `n ≈ 20` caveat on `dentvoxel_cls`).

### Validation Controls & Non-LLM Baselines (Validation Split Only)
- Added `wsp_core/evaluation/baselines.py` implementing all-foreground, per-image Otsu, train-selected global intensity threshold `t*`, human-written 82-parameter center-tap prior (frozen `K = 0` and warm-start), constant-score classification baseline, 3-feature global-intensity logistic regression across `K` budgets, and zero-parameter DenPAR vertical orientation / crown-edge rules.
- Added optional `TrainingConfig.max_steps` (fixed optimizer step budgets independent of `K`, cosine decay over steps) and `TrainingConfig.init_output_bias` (`"zero" | "prior"`, where `"prior"` initializes `output_bias = log(p / (1 - p))` over valid training voxels) to `wsp_core/training/trainer.py`, with bit-for-bit CPU regression tests for default behavior.
- Added `wsp_core/training/controls.py` implementing the full `init × step-budget` factorial validation control runner with scan/image-resampling 95% bootstrap CIs, collapse counts (`Dice < 1%`), paired per-(seed, K) differences, and SVG figure export.
- Added CLI subcommands `dental-wsp run-baselines` and `dental-wsp run-factorial-controls`, plus output-cleared review-control cells in validation notebooks `01`, `02`, and `03`.
- Executed all six validation baseline suites (`denpar_seg`, `denpar_cls`, `akudental_seg`, `akudental_cls`, `dentvoxel_seg`, `dentvoxel_cls`, including GPU `human_82p_warmstart` curves on all three segmentation tasks) and both optimization factorials (`dentvoxel_seg`: 540 runs across `K ∈ {1,2,4,8}`, 5 seeds, 3 schedules; `akudental_seg`: 450 runs across `K ∈ {1,4,16,32,64}`, 5 seeds, 2 schedules) strictly on validation splits, archiving 14 non-private aggregate CSVs and 2 SVG figures under `results/validation_controls/` (zero patient data).
- Updated `README.md` and `docs/RESULTS_AND_DISCUSSION.md` with the completed validation-control verdict: log-odds prior-bias initialization + matched optimizer steps eliminate 100% of random-init collapses (`0/110` runs) and outperform all-6 WSP warm-start at every tested low-`K` budget on both `dentvoxel_seg` and `akudental_seg`, while `dentvoxel_cls` frozen WSP (`0.870` vs. `0.719` full-train 3-feature global-intensity logreg) remains the single clearest surviving frozen WSP positive.

### Documentation & Claims Calibration
- Rewrote `README.md` and `docs/RESULTS_AND_DISCUSSION.md` with plain-language TL;DRs including negative results, all-6 mean ± SD headlines (removing best-of-6 headlines and ratio claims), a dedicated **Negative and Mixed Results** section, supervisor-model context (>90% Dice), abbreviation definitions on first use, and HU-windowed CBCT gray-value wording framing calibration as a confounded hypothesis.

## [0.1.0] - 2026-11-27
- Initial public release of the 2 × 3 digital dentistry WSP benchmark (`akudental`, `denpar`, `dentvoxel`), 36 locked manual replies, four notebooks, sealed held-out test CSVs, `CITATION.cff`, and repository hygiene CI tests.
