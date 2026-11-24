"""Offline manual-chat WSP reply ingestion, validation, and SHA-256 lock sealing."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

from wsp_core.schemas import (
    SchemaError,
    parameter_count,
    sha256_bytes,
    stable_json_bytes,
    strict_json_loads,
    validate_parameters,
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_candidate(raw_text: str) -> str:
    """Extract a single JSON object from a raw LLM chat reply (fenced or bare)."""
    text = raw_text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    fences = _FENCE_RE.findall(text)
    if len(fences) == 1:
        return fences[0].strip()
    if len(fences) > 1:
        raise SchemaError("multiple_json_code_blocks")
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first : last + 1]
    raise SchemaError("no_json_object_found")


def build_design_lock(
    *,
    task: str,
    dataset: str,
    track: str,
    architecture: str,
    prompt_path: Path,
    protocol: dict[str, Any],
    scientific_sources: list[dict[str, str]],
) -> dict[str, Any]:
    """Create a canonical `design.lock.json` binding prompt, architecture, and protocol."""
    prompt_bytes = prompt_path.read_bytes()
    return {
        "schema_version": "dental-wsp-design-v1",
        "task": task,
        "dataset": dataset,
        "track": track,
        "architecture": architecture,
        "parameter_count": parameter_count(architecture),
        "prompt_file": prompt_path.name,
        "prompt_sha256": sha256_bytes(prompt_bytes),
        "protocol": protocol,
        "protocol_sha256": sha256_bytes(stable_json_bytes(protocol)),
        "scientific_sources": scientific_sources,
    }


def write_design_lock(lock_obj: dict[str, Any], out_path: Path) -> bytes:
    """Write `design.lock.json` deterministically and return exact file bytes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(lock_obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    out_path.write_bytes(raw)
    return raw


def ingest_manual_reply(
    *,
    design_lock_path: Path,
    responses_lock_path: Path,
    slot: str,
    raw_reply_path: Path,
    model_label: str = "manual_web_chat",
    notes: str = "",
) -> dict[str, Any]:
    """Parse, strictly validate, and record a single manual LLM chat reply into `responses.lock.json`.

    Invalid replies are recorded transparently with `status = "invalid"` and the exact error token,
    preserving full reporting hygiene (no silent retries or dropped attempts).
    """
    if not re.fullmatch(r"[a-z0-9]+_[0-9]+", slot):
        raise SchemaError(f"invalid_slot_name:{slot} (expected '<family>_<repeat>', e.g. 'claude_1')")

    design_bytes = design_lock_path.read_bytes()
    design = strict_json_loads(design_bytes)
    design_sha = sha256_bytes(design_bytes)
    expected_arch = design["architecture"]

    if responses_lock_path.is_file():
        lock = strict_json_loads(responses_lock_path.read_bytes())
        if lock.get("design_sha256") != design_sha:
            raise SchemaError("existing_responses_lock_design_sha_mismatch")
    else:
        lock = {
            "schema_version": "dental-wsp-responses-v1",
            "task": design["task"],
            "architecture": expected_arch,
            "design_sha256": design_sha,
            "attempts": {},
        }

    if slot in lock["attempts"]:
        raise SchemaError(f"slot_already_recorded:{slot}")

    raw_bytes = raw_reply_path.read_bytes()
    raw_sha = sha256_bytes(raw_bytes)
    raw_text = raw_bytes.decode("utf-8-sig")
    timestamp = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()

    metadata = {
        "collection_mode": "manual_web_chat",
        "model_label": model_label,
        "collected_at_utc": timestamp,
        "blind_to_images": True,
        "tools_enabled": False,
        "notes": notes,
    }

    try:
        candidate = extract_json_candidate(raw_text)
        payload = strict_json_loads(candidate)
        validate_parameters(payload, expected_architecture=expected_arch)
        record: dict[str, Any] = {
            "status": "valid",
            "raw_sha256": raw_sha,
            "parameters_sha256": sha256_bytes(stable_json_bytes(payload)),
            "metadata": metadata,
            "parameters": payload,
        }
    except SchemaError as exc:
        record = {
            "status": "invalid",
            "error": str(exc),
            "raw_sha256": raw_sha,
            "parameters_sha256": None,
            "metadata": metadata,
            "parameters": None,
        }

    lock["attempts"][slot] = record
    responses_lock_path.parent.mkdir(parents=True, exist_ok=True)
    responses_lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return record
