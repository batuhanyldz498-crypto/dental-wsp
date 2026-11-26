"""Tests for strict JSON schemas, 82-param parity, design locks, and offline manual-reply ingestion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wsp_core.collection import build_design_lock, ingest_manual_reply, write_design_lock
from wsp_core.registry import REPO_ROOT, TASKS
from wsp_core.schemas import (
    SchemaError,
    matches_saved_precision,
    parameter_count,
    strict_json_loads,
    validate_parameters,
    verify_response_lock,
    zero_payload,
)


def test_exact_parameter_counts():
    assert parameter_count("pano-wsp82-v1") == 82
    assert parameter_count("periapical-wsp82-v1") == 82
    assert parameter_count("cbct-wsp82-v1") == 82
    assert parameter_count("cls-lenet2d-v1") == 261


def test_duplicate_keys_and_non_finite_rejected():
    with pytest.raises(SchemaError, match="duplicate_key"):
        strict_json_loads('{"a": 1, "a": 2}')
    with pytest.raises(SchemaError, match="non_finite_constant"):
        strict_json_loads('{"a": NaN}')


@pytest.mark.parametrize("arch", ["pano-wsp82-v1", "periapical-wsp82-v1", "cbct-wsp82-v1", "cls-lenet2d-v1"])
def test_zero_payload_validates_and_rejects_bounds_or_bool(arch: str):
    payload = zero_payload(arch)
    arrays = validate_parameters(payload, expected_architecture=arch)
    assert len(arrays) > 0

    bad_bound = dict(payload)
    key = "output_bias" if "output_bias" in payload else "head_bias"
    bad_bound[key] = 1000.1
    with pytest.raises(SchemaError, match="out_of_bounds"):
        validate_parameters(bad_bound, expected_architecture=arch)

    bad_bool = dict(payload)
    bad_bool[key] = True
    with pytest.raises(SchemaError, match="invalid_value"):
        validate_parameters(bad_bool, expected_architecture=arch)


def test_matches_saved_precision():
    assert matches_saved_precision(0.56401104, "0.564011")
    assert not matches_saved_precision(0.564019, "0.564011")


def test_manual_reply_ingestion_valid_and_invalid(tmp_path: Path):
    spec = TASKS["dentvoxel_seg"]
    cfg = spec.load_config(REPO_ROOT)
    lock_obj = build_design_lock(
        task=spec.task,
        dataset=spec.dataset,
        track=spec.track,
        architecture=spec.architecture,
        prompt_path=spec.prompt_path(REPO_ROOT),
        protocol=cfg["protocol"],
        scientific_sources=cfg["scientific_sources"],
    )
    design_path = tmp_path / "design.lock.json"
    design_bytes = write_design_lock(lock_obj, design_path)

    responses_path = tmp_path / "responses.lock.json"
    valid_raw = tmp_path / "claude_1.txt"
    valid_payload = zero_payload("cbct-wsp82-v1")
    valid_raw.write_text(
        "Here is my single-reply response:\n```json\n" + json.dumps(valid_payload) + "\n```\n",
        encoding="utf-8",
    )
    rec1 = ingest_manual_reply(
        design_lock_path=design_path,
        responses_lock_path=responses_path,
        slot="claude_1",
        raw_reply_path=valid_raw,
        model_label="claude_manual",
    )
    assert rec1["status"] == "valid"

    bad_raw = tmp_path / "gpt_1.txt"
    bad_raw.write_text('{"architecture": "cbct-wsp82-v1", "stencils": []}', encoding="utf-8")
    rec2 = ingest_manual_reply(
        design_lock_path=design_path,
        responses_lock_path=responses_path,
        slot="gpt_1",
        raw_reply_path=bad_raw,
        model_label="gpt_manual",
    )
    assert rec2["status"] == "invalid"
    assert rec2["error"]

    # Duplicate slot rejected
    with pytest.raises(SchemaError, match="slot_already_recorded"):
        ingest_manual_reply(
            design_lock_path=design_path,
            responses_lock_path=responses_path,
            slot="claude_1",
            raw_reply_path=valid_raw,
        )

    # Identical paste under a different slot rejected (paste-error guard)
    with pytest.raises(SchemaError, match="duplicate_raw_reply_sha256"):
        ingest_manual_reply(
            design_lock_path=design_path,
            responses_lock_path=responses_path,
            slot="claude_2",
            raw_reply_path=valid_raw,
        )

    # Response lock verifies cleanly and returns only valid slot payloads
    loaded_lock = strict_json_loads(responses_path.read_bytes())
    valid_models = verify_response_lock(lock_obj, design_bytes, loaded_lock, "cbct-wsp82-v1")
    assert list(valid_models.keys()) == ["claude_1"]
