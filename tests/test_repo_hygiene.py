"""Repository privacy and output-hygiene checks run in pytest and GitHub Actions CI."""

from __future__ import annotations

import json
from pathlib import Path

from wsp_core.registry import REPO_ROOT, TASKS

FORBIDDEN_SUFFIXES = (".nii", ".nii.gz", ".dcm", ".mha", ".nrrd", ".zip", ".png", ".jpg", ".jpeg", ".tif", ".bmp")
SKIP_DIRS = {".git", ".pytest_cache", "__pycache__", ".venv", "venv", "dental_wsp.egg-info", "runs", "inbox"}


def test_no_raw_medical_data_or_private_files_in_repo():
    violations: list[str] = []
    for path in REPO_ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts) or not path.is_file():
            continue
        name_lower = path.name.lower()
        if name_lower.endswith(".private.json") or "final_test_exposure" in name_lower:
            violations.append(str(path.relative_to(REPO_ROOT)))
        elif any(name_lower.endswith(ext) for ext in FORBIDDEN_SUFFIXES):
            violations.append(str(path.relative_to(REPO_ROOT)))
    assert not violations, f"Forbidden raw data or private files found: {violations}"


def test_all_notebooks_have_cleared_outputs():
    notebooks = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebooks) == 3
    for nb_path in notebooks:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("outputs") == [], f"Uncleared outputs in {nb_path.name}"
                assert cell.get("execution_count") is None, f"Uncleared execution_count in {nb_path.name}"


def test_all_six_tasks_have_prompts_configs_and_design_locks():
    assert len(TASKS) == 6
    for task, spec in TASKS.items():
        assert spec.prompt_path(REPO_ROOT).is_file(), f"Missing prompt for {task}"
        assert spec.config_path(REPO_ROOT).is_file(), f"Missing config for {task}"
        assert spec.design_lock_path(REPO_ROOT).is_file(), f"Missing design.lock.json for {task}"
