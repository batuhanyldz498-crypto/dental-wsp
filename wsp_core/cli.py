"""Command-line interface (`dental-wsp`) for locking, verifying, and evaluating WSP tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wsp_core.collection import build_design_lock, ingest_manual_reply, write_design_lock
from wsp_core.guards import StudyStop
from wsp_core.registry import REPO_ROOT, TASKS, get_task_spec
from wsp_core.schemas import sha256_bytes, stable_json_bytes, strict_json_loads


def _cmd_list_tasks(_args: argparse.Namespace) -> int:
    for name, spec in TASKS.items():
        print(f"{name:15s}  dataset={spec.dataset:10s}  track={spec.track}  arch={spec.architecture}")
    return 0


def _cmd_lock_design(args: argparse.Namespace) -> int:
    root = Path(args.repo_root).resolve() if args.repo_root else REPO_ROOT
    spec = get_task_spec(args.task)
    cfg = spec.load_config(root)
    lock_obj = build_design_lock(
        task=spec.task,
        dataset=spec.dataset,
        track=spec.track,
        architecture=spec.architecture,
        prompt_path=spec.prompt_path(root),
        protocol=cfg["protocol"],
        scientific_sources=cfg.get("scientific_sources", []),
    )
    out_path = Path(args.out) if args.out else spec.design_lock_path(root)
    if args.check_only:
        if not out_path.is_file():
            print(f"FAIL: missing design lock at {out_path}", file=sys.stderr)
            return 1
        existing = strict_json_loads(out_path.read_bytes())
        if stable_json_bytes(existing) != stable_json_bytes(lock_obj):
            print(f"FAIL: design lock out of sync for {spec.task}", file=sys.stderr)
            return 1
        print(f"PASS: {spec.task} design lock verified ({sha256_bytes(out_path.read_bytes())[:12]})")
        return 0

    raw = write_design_lock(lock_obj, out_path)
    print(f"Wrote {out_path} (sha256={sha256_bytes(raw)[:12]})")
    return 0


def _cmd_lock_reply(args: argparse.Namespace) -> int:
    root = Path(args.repo_root).resolve() if args.repo_root else REPO_ROOT
    spec = get_task_spec(args.task)
    design_path = Path(args.design_lock) if args.design_lock else spec.design_lock_path(root)
    if not design_path.is_file():
        print(f"Error: design lock missing at {design_path}. Run `dental-wsp lock-design {spec.task}` first.", file=sys.stderr)
        return 1
    out_path = Path(args.out)
    record = ingest_manual_reply(
        design_lock_path=design_path,
        responses_lock_path=out_path,
        slot=args.slot,
        raw_reply_path=Path(args.raw_file),
        model_label=args.model_label,
        notes=args.notes,
    )
    print(json.dumps({"slot": args.slot, "status": record["status"], "error": record.get("error")}, indent=2))
    return 0 if record["status"] == "valid" else 2


def _cmd_verify_dataset(args: argparse.Namespace) -> int:
    dataset = args.dataset
    data_dir = Path(args.data_dir)
    if dataset == "akudental":
        from benchmarks.akudental.adapter import verify_akudental_source

        report = verify_akudental_source(data_dir)
    elif dataset == "denpar":
        from benchmarks.denpar.adapter import verify_denpar_directory

        report = verify_denpar_directory(data_dir)
    elif dataset == "dentvoxel":
        from benchmarks.dentvoxel.adapter import verify_dentvoxel_directory

        report = verify_dentvoxel_directory(data_dir, max_scans=args.max_scans)
    else:
        print(f"Unknown dataset: {dataset}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


def _cmd_run_cls_budgets(args: argparse.Namespace) -> int:
    root = Path(args.repo_root).resolve() if args.repo_root else REPO_ROOT
    spec = get_task_spec(args.task)
    if spec.track != "cls":
        print(f"Error: {args.task} is a segmentation task, not classification.", file=sys.stderr)
        return 1
    cfg = spec.load_config(root)
    proto = cfg["protocol"]
    data_dir = Path(args.data_dir)

    if spec.dataset == "akudental":
        from benchmarks.akudental.adapter import load_akudental_cls_public

        train_samples = load_akudental_cls_public(data_dir, split="train")
        val_samples = load_akudental_cls_public(data_dir, split="val")
    elif spec.dataset == "denpar":
        from benchmarks.denpar.adapter import load_denpar_cls

        train_samples = load_denpar_cls(data_dir, split="train")
        val_samples = load_denpar_cls(data_dir, split="val")
    else:
        from benchmarks.dentvoxel.adapter import load_dentvoxel_cls

        if not args.split_file:
            print("Error: --split-file is required for dentvoxel_cls.", file=sys.stderr)
            return 1
        train_samples = load_dentvoxel_cls(data_dir, args.split_file, split="train")
        val_samples = load_dentvoxel_cls(data_dir, args.split_file, split="val")

    budgets = [int(x) for x in args.budgets.split(",")] if args.budgets else list(proto.get("label_budgets_k", proto.get("label_budgets_k_scans")))
    if args.include_full_train:
        n_groups = len({s.group_id for s in train_samples})
        if n_groups not in budgets:
            budgets.append(n_groups)
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else list(proto["seeds"])

    resp_path = (
        Path(args.responses_lock)
        if args.responses_lock
        else spec.design_lock_path(root).with_name(f"{spec.task}.responses.lock.json")
    )
    from wsp_core.training.runner import run_classification_three_arm_validation

    _, summary_df = run_classification_three_arm_validation(
        task=spec.task,
        train_samples=train_samples,
        val_samples=val_samples,
        design_lock_path=spec.design_lock_path(root),
        responses_lock_path=resp_path,
        budgets=budgets,
        seeds=seeds,
        epochs=args.epochs or int(proto["adam_epochs"]),
        lr=args.lr or float(proto["adam_lr"]),
        device=args.device,
        out_dir=Path(args.out_dir),
    )
    print(summary_df.round(4).to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dental-wsp", description="Digital dentistry WSP benchmark CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list-tasks", help="List all 6 benchmark tasks.")
    p_list.set_defaults(func=_cmd_list_tasks)

    p_design = sub.add_parser("lock-design", help="Generate or verify a task design.lock.json.")
    p_design.add_argument("task", choices=sorted(TASKS))
    p_design.add_argument("--out", default=None)
    p_design.add_argument("--repo-root", default=None)
    p_design.add_argument("--check-only", action="store_true")
    p_design.set_defaults(func=_cmd_lock_design)

    p_reply = sub.add_parser("lock-reply", help="Validate and record a manual LLM chat reply.")
    p_reply.add_argument("task", choices=sorted(TASKS))
    p_reply.add_argument("--slot", required=True, help="Slot name like 'gpt_1' or 'claude_2'.")
    p_reply.add_argument("--raw-file", required=True, help="Path to saved raw chat text.")
    p_reply.add_argument("--out", required=True, help="Path to responses.lock.json.")
    p_reply.add_argument("--design-lock", default=None)
    p_reply.add_argument("--model-label", default="manual_web_chat")
    p_reply.add_argument("--notes", default="")
    p_reply.add_argument("--repo-root", default=None)
    p_reply.set_defaults(func=_cmd_lock_reply)

    p_ver = sub.add_parser("verify-dataset", help="Verify BYOD dataset structure and intensity audit.")
    p_ver.add_argument("dataset", choices=("akudental", "denpar", "dentvoxel"))
    p_ver.add_argument("--data-dir", required=True)
    p_ver.add_argument("--max-scans", type=int, default=10, help="Max CBCT scans to sample for HU audit.")
    p_ver.set_defaults(func=_cmd_verify_dataset)

    p_cls = sub.add_parser("run-cls-budgets", help="Run 3-arm classification validation (Frozen, Random-Init, WSP Warm-Start).")
    p_cls.add_argument("task", choices=("akudental_cls", "denpar_cls", "dentvoxel_cls"))
    p_cls.add_argument("--data-dir", required=True)
    p_cls.add_argument("--out-dir", required=True, help="Output directory for CSVs and learning-curve PNG.")
    p_cls.add_argument("--responses-lock", default=None)
    p_cls.add_argument("--split-file", default=None, help="Required for dentvoxel_cls.")
    p_cls.add_argument("--budgets", default=None, help="Comma-separated K values (defaults to task config).")
    p_cls.add_argument("--include-full-train", action="store_true", help="Also include full training-group count.")
    p_cls.add_argument("--seeds", default=None, help="Comma-separated random seeds (defaults to task config).")
    p_cls.add_argument("--epochs", type=int, default=None)
    p_cls.add_argument("--lr", type=float, default=None)
    p_cls.add_argument("--device", default="cpu")
    p_cls.add_argument("--repo-root", default=None)
    p_cls.set_defaults(func=_cmd_run_cls_budgets)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (StudyStop, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
