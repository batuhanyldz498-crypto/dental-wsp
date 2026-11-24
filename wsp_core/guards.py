"""Sealed test-set guards, validation assertions, and privacy-safe error wrappers."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from wsp_core.schemas import strict_json_loads


class StudyStop(RuntimeError):
    """Raised when an input check, reproduction gate, or sealed-test guard halts execution."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StudyStop(message)


@contextmanager
def private_errors(message: str) -> Iterator[None]:
    """Mask low-level library exceptions so private paths or IDs never leak to outputs."""
    try:
        yield
    except StudyStop:
        raise
    except Exception:
        raise StudyStop(message) from None


def exposure_guard(run_root: str | Path) -> None:
    """Stop immediately if an unauthorized test-exposure marker exists under `run_root`."""
    root = Path(run_root)
    if not root.exists():
        return
    markers = (
        root / "final_test_exposure.private.json",
        root / "AKUDENTAL_Part1_Runs" / "akudental-controlled-init-part1-v2" / "final_test_exposure.private.json",
    )
    for marker in markers:
        if marker.exists():
            raise StudyStop(
                "An earlier final-test exposure marker was detected. Stop for provenance review."
            )


def require_split_allowed(
    split: str,
    *,
    task: str,
    allow_test: bool = False,
    authorization_file: str | Path | None = None,
) -> None:
    """Enforce that held-out test splits stay sealed unless explicitly authorized."""
    if split in ("train", "val", "validation"):
        return
    if split != "test":
        raise StudyStop(f"Unsupported split name: {split!r}")
    if not allow_test or authorization_file is None:
        raise StudyStop(
            f"Test split is sealed for task '{task}'. Provide --allow-test and a valid authorization file."
        )
    auth_path = Path(authorization_file)
    require(auth_path.is_file(), "Test authorization file does not exist.")
    auth = strict_json_loads(auth_path.read_bytes())
    authorized_tasks = auth.get("authorized_tasks", [])
    require(
        isinstance(authorized_tasks, list)
        and task in authorized_tasks
        and auth.get("approved") is True
        and bool(auth.get("commit_sha")),
        f"Test authorization file does not authorize task '{task}'.",
    )
