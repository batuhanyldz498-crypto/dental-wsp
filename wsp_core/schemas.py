"""Strict JSON schema validation, canonical hashing, and lock/receipt verification."""

from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any

import numpy as np

VALUE_MIN = -1000.0
VALUE_MAX = 1000.0

ARCHITECTURE_SHAPES: dict[str, dict[str, tuple[int, ...]]] = {
    # 6*3*3 (54) + 6 + 6 + 3*5 (15) + 1 = 82 parameters (exact Study 1 match)
    "pano-wsp82-v1": {
        "kernels": (6, 3, 3),
        "feature_bias": (6,),
        "readout": (6,),
        "spatial_bias": (3, 5),
        "output_bias": (),
    },
    # 6*3*3 (54) + 6 + 6 + 3*5 (15) + 1 = 82 parameters (periapical 2D parity)
    "periapical-wsp82-v1": {
        "kernels": (6, 3, 3),
        "feature_bias": (6,),
        "readout": (6,),
        "spatial_bias": (3, 5),
        "output_bias": (),
    },
    # 6*7 (42) + 6 + 6 + 3*3*3 (27) + 1 = 82 parameters (3D CBCT 7-tap cross-stencil parity)
    "cbct-wsp82-v1": {
        "stencils": (6, 7),
        "feature_bias": (6,),
        "readout": (6,),
        "spatial_bias": (3, 3, 3),
        "output_bias": (),
    },
    # 4*1*5*5 (100) + 4 + 4*4*3*3 (144) + 4 + 8 + 1 = 261 parameters (compact 2D LeNet classifier)
    "cls-lenet2d-v1": {
        "conv1_weights": (4, 1, 5, 5),
        "conv1_bias": (4,),
        "conv2_weights": (4, 4, 3, 3),
        "conv2_bias": (4,),
        "head_weights": (8,),
        "head_bias": (),
    },
}


class SchemaError(ValueError):
    """Raised when a WSP parameter payload or lock file violates schema or bounds."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def strict_json_loads(raw: bytes | str) -> Any:
    """Parse JSON while rejecting duplicate keys and NaN/Infinity tokens."""

    def _no_duplicate_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise SchemaError(f"duplicate_key:{key}")
            result[key] = value
        return result

    def _reject_constant(token: str) -> Any:
        raise SchemaError(f"non_finite_constant:{token}")

    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
    try:
        return json.loads(text, object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_constant)
    except SchemaError:
        raise
    except Exception as exc:
        raise SchemaError(f"invalid_json:{exc}") from None


def parameter_count(architecture: str) -> int:
    if architecture not in ARCHITECTURE_SHAPES:
        raise SchemaError(f"unknown_architecture:{architecture}")
    total = 0
    for shape in ARCHITECTURE_SHAPES[architecture].values():
        size = 1
        for dim in shape:
            size *= dim
        total += size
    return total


def validate_parameters(payload: Any, expected_architecture: str | None = None) -> dict[str, np.ndarray]:
    """Validate a raw WSP JSON payload and return float32 NumPy parameter arrays."""
    if not isinstance(payload, dict):
        raise SchemaError("payload_must_be_object")
    arch = payload.get("architecture")
    if not isinstance(arch, str) or arch not in ARCHITECTURE_SHAPES:
        raise SchemaError(f"unknown_architecture:{arch}")
    if expected_architecture is not None and arch != expected_architecture:
        raise SchemaError(f"architecture_mismatch:expected={expected_architecture},got={arch}")

    shapes = ARCHITECTURE_SHAPES[arch]
    expected_keys = {"architecture", *shapes.keys()}
    if set(payload.keys()) != expected_keys:
        missing = sorted(expected_keys - set(payload.keys()))
        extra = sorted(set(payload.keys()) - expected_keys)
        raise SchemaError(f"key_mismatch:missing={missing},extra={extra}")

    arrays: dict[str, np.ndarray] = {}
    for key, shape in shapes.items():
        raw_val = payload[key]
        arr = np.asarray(raw_val, dtype=object)
        if arr.shape != shape:
            raise SchemaError(f"shape_mismatch:{key}:expected={shape},got={arr.shape}")
        for value in arr.flat:
            # Reject booleans explicitly because isinstance(True, int) is True in Python.
            if type(value) not in (int, float) or not math.isfinite(value):
                raise SchemaError(f"invalid_value:{key}")
            if not (VALUE_MIN <= float(value) <= VALUE_MAX):
                raise SchemaError(f"out_of_bounds:{key}")
        arrays[key] = arr.astype(np.float32)
    return arrays


def zero_payload(architecture: str) -> dict[str, Any]:
    """Build a valid all-zero parameter payload for the requested architecture."""
    if architecture not in ARCHITECTURE_SHAPES:
        raise SchemaError(f"unknown_architecture:{architecture}")
    payload: dict[str, Any] = {"architecture": architecture}
    for key, shape in ARCHITECTURE_SHAPES[architecture].items():
        if shape == ():
            payload[key] = 0.0
        else:
            payload[key] = np.zeros(shape, dtype=float).tolist()
    return payload


def matches_saved_precision(value: float, token: str) -> bool:
    """Compare a float against a saved decimal string at its recorded precision."""
    expected = Decimal(token)
    observed = Decimal(str(value))
    if not expected.is_finite() or not observed.is_finite():
        return False
    unit = Decimal(1).scaleb(expected.as_tuple().exponent)
    return observed.quantize(unit, rounding=ROUND_HALF_EVEN) == expected


def verify_response_lock(
    design_lock: dict[str, Any],
    design_bytes: bytes,
    responses_lock: dict[str, Any],
    expected_architecture: str,
) -> dict[str, dict[str, Any]]:
    """Verify hash bindings between design.lock.json and responses.lock.json.

    Returns valid slot payloads keyed by slot name in stored order.
    """
    if responses_lock.get("design_sha256") != sha256_bytes(design_bytes):
        raise SchemaError("design_sha256_mismatch")
    attempts = responses_lock.get("attempts")
    if not isinstance(attempts, dict):
        raise SchemaError("missing_attempts_dict")

    valid_models: dict[str, dict[str, Any]] = {}
    for slot, record in attempts.items():
        if not isinstance(record, dict):
            raise SchemaError(f"invalid_record:{slot}")
        status = record.get("status")
        if status != "valid":
            continue
        payload = record.get("parameters")
        validate_parameters(payload, expected_architecture=expected_architecture)
        if sha256_bytes(stable_json_bytes(payload)) != record.get("parameters_sha256"):
            raise SchemaError(f"parameters_sha256_mismatch:{slot}")
        raw_sha = record.get("raw_sha256", "")
        if not isinstance(raw_sha, str) or len(raw_sha) != 64:
            raise SchemaError(f"raw_sha256_invalid:{slot}")
        valid_models[slot] = payload
    return valid_models
