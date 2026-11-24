"""Three-arm benchmark runner (Frozen WSP, Random-Init Adam, WSP Warm-Start Adam) & learning curves."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wsp_core.evaluation.classification import ClassificationSample, evaluate_classifier
from wsp_core.guards import require
from wsp_core.schemas import strict_json_loads, verify_response_lock
from wsp_core.training.trainer import (
    TrainingConfig,
    sample_stratified_cls_budgets,
    train_classification_budget,
)


def run_classification_three_arm_validation(
    *,
    task: str,
    train_samples: list[ClassificationSample],
    val_samples: list[ClassificationSample],
    design_lock_path: Path,
    responses_lock_path: Path,
    budgets: list[int],
    seeds: list[int],
    epochs: int = 200,
    lr: float = 5e-3,
    device: str = "cpu",
    out_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Arm 1 (Frozen WSP), Arm 2 (Random-Init Adam), and Arm 3 (WSP Warm-Start Adam) on validation."""
    design_bytes = design_lock_path.read_bytes()
    design = strict_json_loads(design_bytes)
    resp = strict_json_loads(responses_lock_path.read_bytes())
    frozen_models = verify_response_lock(design, design_bytes, resp, expected_architecture="cls-lenet2d-v1")
    require(len(frozen_models) > 0, f"No valid frozen replies found in {responses_lock_path}")

    rows: list[dict[str, Any]] = []

    # --- Arm 1: Frozen WSP (K = 0, zero optimizer steps) ---
    for slot, payload in frozen_models.items():
        metrics = evaluate_classifier(payload, val_samples)
        rows.append(
            {
                "task": task,
                "arm": "frozen_wsp",
                "k_budget": 0,
                "seed": -1,
                "init_slot": slot,
                **metrics,
            }
        )

    # Pre-index training samples by group_id for fast budget subsetting
    by_group: dict[str, list[ClassificationSample]] = {}
    for s in train_samples:
        by_group.setdefault(s.group_id, []).append(s)

    for seed in seeds:
        nested = sample_stratified_cls_budgets(train_samples, budgets, seed=seed)
        for k, selected_groups in nested.items():
            sub_train = [s for g in selected_groups for s in by_group[g]]
            cfg = TrainingConfig(epochs=epochs, lr=lr, seed=seed, device=device)

            # --- Arm 2: Random-Init + Adam ---
            rand_payload, _ = train_classification_budget(sub_train, cfg, init_payload=None)
            rand_metrics = evaluate_classifier(rand_payload, val_samples)
            rows.append(
                {
                    "task": task,
                    "arm": "random_init_adam",
                    "k_budget": k,
                    "seed": seed,
                    "init_slot": "random",
                    **rand_metrics,
                }
            )

            # --- Arm 3: WSP Warm-Start + Adam (every valid frozen reply, no cherry-picking) ---
            for slot, frozen_payload in frozen_models.items():
                warm_payload, _ = train_classification_budget(sub_train, cfg, init_payload=frozen_payload)
                warm_metrics = evaluate_classifier(warm_payload, val_samples)
                rows.append(
                    {
                        "task": task,
                        "arm": "wsp_warmstart_adam",
                        "k_budget": k,
                        "seed": seed,
                        "init_slot": slot,
                        **warm_metrics,
                    }
                )

    runs_df = pd.DataFrame(rows)
    summary_df = (
        runs_df.groupby(["arm", "k_budget"], as_index=False)
        .agg(
            n_runs=("roc_auc", "count"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            margin_roc_auc_mean=("margin_ablated_roc_auc", "mean"),
            margin_roc_auc_std=("margin_ablated_roc_auc", "std"),
            sign_flip_count=("sign_flip_flagged", "sum"),
        )
        .sort_values(["arm", "k_budget"])
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        runs_df.to_csv(out_dir / f"{task}_val_runs.csv", index=False)
        summary_df.to_csv(out_dir / f"{task}_val_summary.csv", index=False)
        plot_classification_learning_curve(
            summary_df,
            task=task,
            out_path=out_dir / f"{task}_learning_curve.png",
        )

    return runs_df, summary_df


def plot_classification_learning_curve(
    summary_df: pd.DataFrame,
    *,
    task: str,
    out_path: Path | None = None,
) -> Any:
    """Plot 2-panel learning curves: Full-Frame ROC-AUC and Margin-Crop ROC-AUC vs. K."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True, layout="constrained")
    panels = [
        ("roc_auc_mean", "roc_auc_std", "Full-Frame Validation ROC-AUC"),
        ("margin_roc_auc_mean", "margin_roc_auc_std", "Margin-Crop (10% Border Zeroed) ROC-AUC"),
    ]
    styles = {
        "random_init_adam": ("o-", "#263d57", "Random-Init + Adam"),
        "wsp_warmstart_adam": ("s-", "#147b72", "WSP Warm-Start + Adam"),
    }

    frozen = summary_df[summary_df["arm"] == "frozen_wsp"]
    for ax, (mean_col, std_col, title) in zip(axes, panels):
        if not frozen.empty:
            val = float(frozen[mean_col].iloc[0])
            ax.axhline(val, color="#a44624", linestyle="--", linewidth=1.8, label=f"Frozen WSP K=0 ({val:.3f})")
        for arm, (fmt, color, label) in styles.items():
            sub = summary_df[summary_df["arm"] == arm].sort_values("k_budget")
            if sub.empty:
                continue
            ks = sub["k_budget"].to_numpy()
            means = sub[mean_col].to_numpy()
            stds = np.nan_to_num(sub[std_col].to_numpy(), nan=0.0)
            ax.plot(ks, means, fmt, color=color, linewidth=2.0, markersize=6, label=label)
            ax.fill_between(ks, means - stds, means + stds, color=color, alpha=0.14)
        ax.axhline(0.5, color="#999999", linestyle=":", linewidth=1.0)
        ax.set_xscale("log", base=2)
        ax.set_ylim(0.30, 1.02)
        ax.set_xlabel("Labeled training groups (K)")
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Validation ROC-AUC")
    axes[1].legend(loc="lower right", frameon=True)
    fig.suptitle(f"{task} · Frozen WSP vs. Random-Init Adam vs. WSP Warm-Start Adam (Validation Only)", fontsize=12)

    if out_path is not None:
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    return fig
