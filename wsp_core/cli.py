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
