"""Fixed-step and prior-bias factorial validation controls for segmentation low-K analysis.

Addresses the epoch-based step-scaling and zero-bias initialization confounds in the
original v0.1.0 segmentation runs by comparing all four initializations under identical
optimizer step budgets on the VALIDATION split only (never loading held-out test splits):

- `random_zero_bias`: random initialization with `output_bias = 0.0` (v0.1.0 control)
- `random_prior_bias`: random initialization with `output_bias = log(p / (1 - p))`
- `wsp_warmstart`: warm-start from all 6 locked WSP replies (same step budget)
- `human_init`: warm-start from the human-written 82-param center-tap prior (same step budget)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wsp_core.evaluation.baselines import build_human_written_seg82_payload, fit_global_intensity_threshold
from wsp_core.evaluation.segmentation import (
    SegmentationSample,
    bounding_slices,
    contour_and_band,
    group_segmentation_scores,
)
from wsp_core.guards import require
from wsp_core.schemas import strict_json_loads, verify_response_lock
from wsp_core.training.runner import _precompute_seg_val_cache, _predict_masks_torch
from wsp_core.training.trainer import (
    TrainingConfig,
    sample_nested_group_budgets,
    train_segmentation_budget,
)

COLLAPSE_DICE_THRESHOLD_PCT = 1.0


@dataclass(frozen=True)
class ScheduleSpec:
    """Training schedule specification for factorial control runs."""

    name: str
    epochs: int
    max_steps: int | None


def default_factorial_schedules(task: str) -> list[ScheduleSpec]:
    """Return the prespecified Task (d) schedules for validation control experiments."""
    if task == "dentvoxel_seg":
        return [
            ScheduleSpec("original_epochs_120", epochs=120, max_steps=None),
            ScheduleSpec("fixed_steps_480", epochs=120, max_steps=480),
            ScheduleSpec("fixed_steps_1800", epochs=120, max_steps=1800),
        ]
    if task in ("akudental_seg", "denpar_seg"):
        return [
            ScheduleSpec("original_epochs_150", epochs=150, max_steps=None),
            ScheduleSpec("fixed_steps_1050", epochs=150, max_steps=1050),
        ]
    raise ValueError(f"Unsupported segmentation task for factorial controls: {task}")


def bootstrap_group_mean_ci(
    run_by_group_matrix: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 2026,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Compute percentile bootstrap CI by resampling validation scans/images (columns).

    `run_by_group_matrix` has shape `(n_runs, n_groups)`. Each bootstrap draw resamples
    columns (validation scans/images) with replacement and computes the grand mean across
    all runs and resampled scans/images.
    """
    mat = np.asarray(run_by_group_matrix, dtype=np.float64)
    require(mat.ndim == 2 and mat.shape[0] >= 1 and mat.shape[1] >= 1, "Bootstrap matrix must be non-empty 2D.")
    n_groups = mat.shape[1]
    # Grand mean over runs for each group reduces resampling to an exact O(n_boot * n_groups) vector mean.
    per_group_mean = mat.mean(axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_groups, size=(n_boot, n_groups))
    boot_means = per_group_mean[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def _score_payload_with_group_dice(
    architecture: str,
    payload: dict[str, Any],
    val_x: Any,
    val_samples: list[SegmentationSample],
    val_cache: list[dict[str, Any]],
    ordered_val_groups: list[str],
    device: str,
) -> tuple[dict[str, float], list[float]]:
    """Score one trained payload and return `(aggregate_metrics, per_group_dice_pct_list)`."""
    masks = _predict_masks_torch(architecture, payload, val_x, val_samples, device=device)
    reg, group_dice = group_segmentation_scores(val_samples, masks)

    by_group: dict[str, list[tuple[float, float, float, float, float]]] = {}
    for item in val_cache:
        s = item["sample"]
        valid = item["valid"]
        valid_area = int(np.count_nonzero(valid))
        truth = item["truth"]
        ref_box = item["ref_box"]
        ref_edge = item["ref_edge"]
        ref_band = item["ref_band"]
        ref_area = item["ref_area"]
        ref_box_vol = item["ref_box_vol"]

        pred = masks[s.sample_id]
        pred_edge, pred_band = contour_and_band(pred, valid)
        prec = float(np.count_nonzero(ref_band[pred_edge]) / pred_edge.size) if pred_edge.size else 0.0
        recall = float(pred_band[ref_edge].mean()) if ref_edge.size else 0.0
        bf1 = 2.0 * prec * recall / (prec + recall) if (prec + recall) > 0.0 else 0.0

        pred_box = bounding_slices(pred)
        proposal = np.zeros_like(valid, dtype=bool)
        if pred_box is not None:
            proposal[pred_box] = True
        prop_vol = int(np.count_nonzero(proposal))
        prop_valid_vol = int(np.count_nonzero(proposal & valid))
        box_valid_frac = prop_valid_vol / valid_area if valid_area > 0 else 0.0
        overlap = int(np.count_nonzero(proposal[ref_box])) if ref_box is not None else 0
        denom = prop_vol + ref_box_vol - overlap
        box_iou = overlap / denom if denom > 0 else 0.0
        excl_tooth = 1.0 - (int(np.count_nonzero(truth & proposal)) / ref_area) if ref_area > 0 else 0.0
        by_group.setdefault(s.group_id, []).append(
            (100.0 * bf1, 100.0 * recall, 100.0 * box_iou, 100.0 * box_valid_frac, 100.0 * excl_tooth)
        )

    group_means = np.array([np.mean(v, axis=0) for v in by_group.values()], dtype=np.float64)
    metrics = {
        "dice_pct": 100.0 * reg["dice"],
        "iou_pct": 100.0 * reg["iou"],
        "bf1_a_pct": float(group_means[:, 0].mean()),
        "edge_recall_a_pct": float(group_means[:, 1].mean()),
        "box_iou_a_pct": float(group_means[:, 2].mean()),
        "box_valid_vol_a_pct": float(group_means[:, 3].mean()),
        "excluded_tooth_a_pct": float(group_means[:, 4].mean()),
    }
    group_dice_vec = [100.0 * float(group_dice[g]) for g in ordered_val_groups]
    return metrics, group_dice_vec


def _run_key_str(schedule: str, arm: str, k_budget: int, seed: int, init_slot: str) -> str:
    return f"{schedule}|{arm}|{k_budget}|{seed}|{init_slot}"


def summarize_factorial_controls(
    runs_df: pd.DataFrame,
    group_dice_by_key: dict[str, list[float]],
    *,
    n_boot: int = 2000,
    boot_seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute aggregate summary (with scan-bootstrap 95% CIs & collapse counts) and paired diffs.

    Returns `(summary_df, paired_summary_df, paired_by_seed_df)`. None of the returned tables
    contain raw scan or patient identifiers.
    """
    summary_rows: list[dict[str, Any]] = []
    for (schedule, arm, k_budget), sub in runs_df.groupby(["schedule", "arm", "k_budget"], sort=True):
        keys = [
            _run_key_str(str(r["schedule"]), str(r["arm"]), int(r["k_budget"]), int(r["seed"]), str(r["init_slot"]))
            for _, r in sub.iterrows()
        ]
        mat = np.array([group_dice_by_key[k] for k in keys], dtype=np.float64)
        ci_lo, ci_hi = bootstrap_group_mean_ci(mat, n_boot=n_boot, seed=boot_seed)
        dices = sub["dice_pct"].to_numpy(dtype=np.float64)
        n_runs = int(len(sub))
        n_collapsed = int(np.sum(dices < COLLAPSE_DICE_THRESHOLD_PCT))
        summary_rows.append(
            {
                "task": str(sub["task"].iloc[0]),
                "schedule": str(schedule),
                "max_steps": sub["max_steps"].iloc[0],
                "arm": str(arm),
                "k_budget": int(k_budget),
                "n_runs": n_runs,
                "n_collapsed": n_collapsed,
                "collapse_rate_pct": 100.0 * n_collapsed / max(1, n_runs),
                "dice_mean": float(dices.mean()),
                "dice_std": float(dices.std(ddof=1)) if n_runs > 1 else np.nan,
                "dice_boot_ci95_low": ci_lo,
                "dice_boot_ci95_high": ci_hi,
                "bf1_mean": float(sub["bf1_a_pct"].mean()),
                "edge_recall_mean": float(sub["edge_recall_a_pct"].mean()),
                "box_valid_vol_mean": float(sub["box_valid_vol_a_pct"].mean()),
                "excluded_tooth_mean": float(sub["excluded_tooth_a_pct"].mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["schedule", "k_budget", "arm"]).reset_index(drop=True)

    # Paired differences per (schedule, k_budget, seed): arm_a minus arm_b on exact same K subset
    comparisons = [
        ("wsp_warmstart_vs_random_zero_bias", "wsp_warmstart", "random_zero_bias"),
        ("wsp_warmstart_vs_random_prior_bias", "wsp_warmstart", "random_prior_bias"),
        ("wsp_warmstart_vs_human_init", "wsp_warmstart", "human_init"),
        ("human_init_vs_random_prior_bias", "human_init", "random_prior_bias"),
        ("random_prior_bias_vs_random_zero_bias", "random_prior_bias", "random_zero_bias"),
    ]

    # Build per-(schedule, arm, k_budget, seed) mean group-dice vector (averages across 6 replies for wsp_warmstart)
    seed_group_vecs: dict[tuple[str, str, int, int], np.ndarray] = {}
    for (schedule, arm, k_budget, seed), sub in runs_df.groupby(["schedule", "arm", "k_budget", "seed"]):
        keys = [
            _run_key_str(str(r["schedule"]), str(r["arm"]), int(r["k_budget"]), int(r["seed"]), str(r["init_slot"]))
            for _, r in sub.iterrows()
        ]
        mat = np.array([group_dice_by_key[k] for k in keys], dtype=np.float64)
        seed_group_vecs[(str(schedule), str(arm), int(k_budget), int(seed))] = mat.mean(axis=0)

    paired_seed_rows: list[dict[str, Any]] = []
    paired_summary_rows: list[dict[str, Any]] = []
    task_name = str(runs_df["task"].iloc[0])

    schedules = sorted(runs_df["schedule"].unique().tolist())
    budgets = sorted(int(k) for k in runs_df["k_budget"].unique().tolist())
    seeds = sorted(int(s) for s in runs_df["seed"].unique().tolist())

    for schedule in schedules:
        for k_budget in budgets:
            for comp_name, arm_a, arm_b in comparisons:
                diff_vecs = []
                for seed in seeds:
                    va = seed_group_vecs.get((schedule, arm_a, k_budget, seed))
                    vb = seed_group_vecs.get((schedule, arm_b, k_budget, seed))
                    if va is None or vb is None:
                        continue
                    diff_vec = va - vb
                    diff_vecs.append(diff_vec)
                    paired_seed_rows.append(
                        {
                            "task": task_name,
                            "schedule": schedule,
                            "k_budget": k_budget,
                            "seed": seed,
                            "comparison": comp_name,
                            "arm_a_dice_pct": float(va.mean()),
                            "arm_b_dice_pct": float(vb.mean()),
                            "delta_dice_pp": float(diff_vec.mean()),
                        }
                    )
                if not diff_vecs:
                    continue
                diff_mat = np.array(diff_vecs, dtype=np.float64)
                per_seed_deltas = diff_mat.mean(axis=1)
                ci_lo, ci_hi = bootstrap_group_mean_ci(diff_mat, n_boot=n_boot, seed=boot_seed)
                paired_summary_rows.append(
                    {
                        "task": task_name,
                        "schedule": schedule,
                        "k_budget": k_budget,
                        "comparison": comp_name,
                        "n_seeds": int(len(per_seed_deltas)),
                        "delta_dice_mean_pp": float(per_seed_deltas.mean()),
                        "delta_dice_std_pp": float(per_seed_deltas.std(ddof=1)) if len(per_seed_deltas) > 1 else np.nan,
                        "delta_dice_min_pp": float(per_seed_deltas.min()),
                        "delta_dice_max_pp": float(per_seed_deltas.max()),
                        "delta_boot_ci95_low_pp": ci_lo,
                        "delta_boot_ci95_high_pp": ci_hi,
                    }
                )

    paired_summary_df = pd.DataFrame(paired_summary_rows)
    paired_by_seed_df = pd.DataFrame(paired_seed_rows)
    return summary_df, paired_summary_df, paired_by_seed_df


def plot_factorial_controls_svg(
    summary_df: pd.DataFrame,
    *,
    task: str,
    out_path: str | Path,
    frozen_wsp_dice_pct: float | None = None,
) -> Path:
    """Plot multi-panel validation factorial control figure (Dice % with 95% scan-bootstrap CI) as SVG."""
    schedules = list(dict.fromkeys(summary_df["schedule"].tolist()))
    n_panels = max(1, len(schedules))
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.4), sharey=True, layout="constrained", squeeze=False)
    axes_flat = axes[0]

    arm_styles = {
        "random_zero_bias": ("o--", "#6b7280", "Random (zero bias)"),
        "random_prior_bias": ("^-", "#2563eb", "Random (prior bias)"),
        "human_init": ("D-", "#d97706", "Human 82p init (c4)"),
        "wsp_warmstart": ("s-", "#0f766e", "WSP Warm-Start (6 replies)"),
    }

    for ax, sched in zip(axes_flat, schedules):
        sub_sched = summary_df[summary_df["schedule"] == sched]
        if frozen_wsp_dice_pct is not None:
            ax.axhline(
                frozen_wsp_dice_pct,
                color="#b91c1c",
                linestyle=":",
                linewidth=1.5,
                label=f"Frozen WSP K=0 ({frozen_wsp_dice_pct:.1f}%)",
            )
        for arm, (fmt, color, label) in arm_styles.items():
            sub = sub_sched[sub_sched["arm"] == arm].sort_values("k_budget")
            if sub.empty:
                continue
            ks = sub["k_budget"].to_numpy(dtype=np.float64)
            means = sub["dice_mean"].to_numpy(dtype=np.float64)
            lo = sub["dice_boot_ci95_low"].to_numpy(dtype=np.float64)
            hi = sub["dice_boot_ci95_high"].to_numpy(dtype=np.float64)
            ax.plot(ks, means, fmt, color=color, linewidth=2.0, markersize=6, label=label)
            ax.fill_between(ks, lo, hi, color=color, alpha=0.14)
        ax.set_xscale("log", base=2)
        ks_all = sorted(sub_sched["k_budget"].unique().tolist())
        ax.set_xticks(ks_all)
        ax.set_xticklabels([str(k) for k in ks_all])
        ax.set_ylim(-2.0, 100.0)
        ax.set_xlabel("Training label budget K")
        ax.set_title(sched.replace("_", " "), loc="left", fontsize=10.5)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)

    axes_flat[0].set_ylabel("Validation Dice (%) · 95% scan-bootstrap CI")
    axes_flat[-1].legend(loc="lower right", frameon=True, fontsize=8.5)
    fig.suptitle(f"{task} Validation Factorial Controls: Init × Step Budget", fontsize=12)

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_p


def run_segmentation_factorial_controls(
    *,
    task: str,
    architecture: str,
    train_samples: list[SegmentationSample],
    val_samples: list[SegmentationSample],
    design_lock_path: Path,
    responses_lock_path: Path,
    budgets: list[int],
    seeds: list[int],
    schedules: list[ScheduleSpec] | None = None,
    lr: float = 3e-2,
    batch_size: int = 4,
    cosine_lr: bool = True,
    device: str = "cpu",
    out_dir: Path | None = None,
    resume: bool = True,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Run the Task (d) full factorial validation control experiment with resume-safe checkpointing.

    All four init arms (`random_zero_bias`, `random_prior_bias`, `wsp_warmstart` across all 6
    replies, and `human_init` from baseline c4) receive the exact same step budget in every schedule.
    """
    import torch

    sched_list = default_factorial_schedules(task) if schedules is None else list(schedules)
    d_bytes = design_lock_path.read_bytes()
    frozen_models = verify_response_lock(
        strict_json_loads(d_bytes),
        d_bytes,
        strict_json_loads(responses_lock_path.read_bytes()),
        expected_architecture=architecture,
    )
    require(len(frozen_models) > 0, f"No valid frozen replies found in {responses_lock_path}")

    # Fit Baseline c3 global intensity threshold strictly on TRAIN split to construct c4 human_init
    t_star, train_thresh_dice = fit_global_intensity_threshold(train_samples)
    human_payload = build_human_written_seg82_payload(architecture, t_star)

    runs_csv = (out_dir / f"{task}_factorial_runs.csv") if out_dir is not None else None
    group_cache_path = (out_dir / f"{task}_factorial_group_dice.private.json") if out_dir is not None else None

    rows: list[dict[str, Any]] = []
    group_dice_by_key: dict[str, list[float]] = {}
    done: set[str] = set()
    if resume and runs_csv is not None and runs_csv.is_file() and group_cache_path is not None and group_cache_path.is_file():
        rows = pd.read_csv(runs_csv).to_dict(orient="records")
        group_dice_by_key = json.loads(group_cache_path.read_text(encoding="utf-8"))
        done = {
            _run_key_str(str(r["schedule"]), str(r["arm"]), int(r["k_budget"]), int(r["seed"]), str(r["init_slot"]))
            for r in rows
            if _run_key_str(str(r["schedule"]), str(r["arm"]), int(r["k_budget"]), int(r["seed"]), str(r["init_slot"]))
            in group_dice_by_key
        }
        if done:
            print(f"[{task}] Resuming factorial controls ({len(done)} completed runs skipped)")

    val_cache = _precompute_seg_val_cache(val_samples)
    ordered_val_groups = sorted({s.group_id for s in val_samples})
    val_x = torch.stack([torch.from_numpy(s.image.astype(np.float32)).unsqueeze(0) for s in val_samples])

    # Compute frozen WSP reference Dice on validation for optional SVG line
    frozen_dices = [
        _score_payload_with_group_dice(architecture, p, val_x, val_samples, val_cache, ordered_val_groups, device)[0][
            "dice_pct"
        ]
        for p in frozen_models.values()
    ]
    frozen_mean_dice = float(np.mean(frozen_dices))

    by_group: dict[str, list[SegmentationSample]] = defaultdict(list)
    for s in train_samples:
        by_group[s.group_id].append(s)
    all_groups = sorted(by_group.keys())

    def _checkpoint() -> None:
        if out_dir is None or not rows:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(runs_csv, index=False)
        group_cache_path.write_text(json.dumps(group_dice_by_key), encoding="utf-8")

    for sched in sched_list:
        for seed in seeds:
            nested = sample_nested_group_budgets(all_groups, budgets, seed=seed)
            for k, selected_groups in nested.items():
                sub_train = [s for g in selected_groups for s in by_group[g]]
                arm_specs: list[tuple[str, str, dict[str, Any] | None, str]] = [
                    ("random_zero_bias", "random_zero", None, "zero"),
                    ("random_prior_bias", "random_prior", None, "prior"),
                    ("human_init", "human_c4", human_payload, "zero"),
                ]
                for slot, payload in frozen_models.items():
                    arm_specs.append(("wsp_warmstart", slot, payload, "zero"))

                for arm, slot, init_payload, init_bias_mode in arm_specs:
                    key = _run_key_str(sched.name, arm, k, seed, slot)
                    if key in done:
                        continue
                    cfg = TrainingConfig(
                        epochs=sched.epochs,
                        lr=lr,
                        seed=seed,
                        device=device,
                        batch_size=batch_size,
                        cosine_lr=cosine_lr,
                        max_steps=sched.max_steps,
                        init_output_bias=init_bias_mode,
                    )
                    trained_payload, losses = train_segmentation_budget(
                        architecture,
                        sub_train,
                        cfg,
                        init_payload=init_payload,
                    )
                    metrics, g_vec = _score_payload_with_group_dice(
                        architecture,
                        trained_payload,
                        val_x,
                        val_samples,
                        val_cache,
                        ordered_val_groups,
                        device,
                    )
                    rows.append(
                        {
                            "task": task,
                            "schedule": sched.name,
                            "epochs": sched.epochs,
                            "max_steps": sched.max_steps if sched.max_steps is not None else -1,
                            "executed_steps": len(losses) if sched.max_steps is not None else sched.epochs * int(np.ceil(len(sub_train) / max(1, min(4 if architecture == "cbct-wsp82-v1" else batch_size, len(sub_train))))),
                            "arm": arm,
                            "k_budget": int(k),
                            "seed": int(seed),
                            "init_slot": slot,
                            "collapsed": bool(metrics["dice_pct"] < COLLAPSE_DICE_THRESHOLD_PCT),
                            **metrics,
                        }
                    )
                    group_dice_by_key[key] = g_vec
                    done.add(key)

                _checkpoint()
                print(f"[{task}] Factorial checkpointed schedule={sched.name}, seed={seed}, K={k}")

    runs_df = pd.DataFrame(rows)
    summary_df, paired_summary_df, paired_by_seed_df = summarize_factorial_controls(
        runs_df,
        group_dice_by_key,
        n_boot=n_boot,
    )

    svg_path: Path | None = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        runs_df.to_csv(out_dir / f"{task}_factorial_runs.csv", index=False)
        summary_df.to_csv(out_dir / f"{task}_factorial_summary.csv", index=False)
        paired_summary_df.to_csv(out_dir / f"{task}_factorial_paired_diffs.csv", index=False)
        paired_by_seed_df.to_csv(out_dir / f"{task}_factorial_paired_by_seed.csv", index=False)
        svg_path = plot_factorial_controls_svg(
            summary_df,
            task=task,
            out_path=out_dir / f"{task}_factorial_controls.svg",
            frozen_wsp_dice_pct=frozen_mean_dice,
        )

    return {
        "runs_df": runs_df,
        "summary_df": summary_df,
        "paired_summary_df": paired_summary_df,
        "paired_by_seed_df": paired_by_seed_df,
        "svg_path": svg_path,
        "human_threshold_t_star": t_star,
        "human_threshold_train_dice_pct": train_thresh_dice,
        "frozen_wsp_val_dice_pct": frozen_mean_dice,
    }
