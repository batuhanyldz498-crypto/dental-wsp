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
    assert len(notebooks) == 4
    for nb_path in notebooks:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("outputs") == [], f"Uncleared outputs in {nb_path.name}"
                assert cell.get("execution_count") is None, f"Uncleared execution_count in {nb_path.name}"


def test_pre_registered_test_authorization_covers_all_six_tasks():
    from wsp_core.guards import require_split_allowed

    auth_path = REPO_ROOT / "docs/test_authorization.json"
    assert auth_path.is_file(), "Missing docs/test_authorization.json"
    for task in TASKS:
        require_split_allowed("test", task=task, allow_test=True, authorization_file=auth_path)


def test_all_six_tasks_have_prompts_configs_and_design_locks():
    assert len(TASKS) == 6
    for task, spec in TASKS.items():
        assert spec.prompt_path(REPO_ROOT).is_file(), f"Missing prompt for {task}"
        assert spec.config_path(REPO_ROOT).is_file(), f"Missing config for {task}"
        assert spec.design_lock_path(REPO_ROOT).is_file(), f"Missing design.lock.json for {task}"


def test_all_six_tasks_have_six_unique_valid_locked_replies():
    from wsp_core.schemas import strict_json_loads, verify_response_lock

    for task, spec in TASKS.items():
        design_path = spec.design_lock_path(REPO_ROOT)
        resp_path = design_path.with_name(f"{task}.responses.lock.json")
        assert resp_path.is_file(), f"Missing responses.lock.json for {task}"
        design_bytes = design_path.read_bytes()
        design_obj = strict_json_loads(design_bytes)
        resp_obj = strict_json_loads(resp_path.read_bytes())
        valid_models = verify_response_lock(design_obj, design_bytes, resp_obj, spec.architecture)
        assert len(valid_models) == 6, f"Expected 6 valid locked replies for {task}, got {len(valid_models)}"
        raw_hashes = [rec["raw_sha256"] for rec in resp_obj["attempts"].values()]
        param_hashes = [rec["parameters_sha256"] for rec in resp_obj["attempts"].values()]
        assert len(set(raw_hashes)) == 6, f"Duplicate raw_sha256 detected in {task}.responses.lock.json"
        assert len(set(param_hashes)) == 6, f"Duplicate parameters_sha256 detected in {task}.responses.lock.json"


def test_no_personal_paths_or_internal_handoff_artifacts():
    forbidden_tokens = (
        "C:\\" + "Users\\",
        "C:/" + "Users/",
        "CONTINUATION_" + "HANDOFF",
        "AKUDENTAL_WSP_Spatial_" + "Ablation",
        "WSP_" + "Work",
    )
    violations: list[str] = []
    for path in REPO_ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts) or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        lower = text.lower()
        for token in forbidden_tokens:
            if token.lower() in lower:
                violations.append(f"{path.relative_to(REPO_ROOT)}:{token}")
    assert not violations, f"Personal paths or internal workspace tokens found: {violations}"


def test_sealed_test_aggregate_csvs_match_recorded_sha256_receipts():
    from wsp_core.schemas import sha256_file

    auth_path = REPO_ROOT / "docs/test_authorization.json"
    assert (
        sha256_file(auth_path)
        == "6413f557db216921564664bb5fd5873a051fd30e345a5b4c0a99b8ab246d745c"
    ), "docs/test_authorization.json digest changed after sealed test execution"

    expected_csv_digests = {
        "akudental_cls_test_stage1_frozen.csv": "35e74c338147fab3380d4dbb3c3288815881ab8fe527b850e60890d8a5e8a68a",
        "akudental_cls_test_summary.csv": "442769f9333c0f446b0e321d2e582f391d7bff8d220c2b46daa73ac19ea95216",
        "akudental_seg_test_stage1_frozen.csv": "2b54726862804827ad6cae0849be4d7eb8ef8ed61f0d0ed6ae3b254c0888c35b",
        "akudental_seg_test_summary.csv": "4757b00a2cc5a97cd67378cec09aebba123cb9ca92961e59c09dd7bdcbdb478c",
        "denpar_cls_test_stage1_frozen.csv": "ab8b6135eeeb9e1baee2c41b71b9004f20b4000f877de90fe36563652afbb477",
        "denpar_cls_test_summary.csv": "bdfe4345610406549c9356d4dc0e37e4b0037e3a635e93c2ef6d10c06a9953b3",
        "denpar_seg_test_stage1_frozen.csv": "81d01ea55b952e61f56018cd4819c3a16662c98e262f6ef63825486e8cfc918b",
        "denpar_seg_test_summary.csv": "4f30835bdb02ad66be533e1b9ac876d2a6d5e8d9faee6c3024d4e7fdddf6722a",
        "dentvoxel_cls_test_stage1_frozen.csv": "a3b1dae18dffd51901073ca635e1dc434d3e3c019a12a42b3322b6a060dd1f9c",
        "dentvoxel_cls_test_summary.csv": "170a4c9532f591c11cbd970701545fb646518c4a6004bd347c56a6757b69c676",
        "dentvoxel_seg_test_stage1_frozen.csv": "4d3eeccd24fa1e92eb40b4f21e9e944ef3d9a9ad59950262bc60ad108bf12d80",
        "dentvoxel_seg_test_summary.csv": "a1eb20cd90c9afca118e0b37ce7f09e5a7916bfecdda6325fbd4a64c44bf6895",
    }
    sealed_dir = REPO_ROOT / "results/sealed_test"
    assert sealed_dir.is_dir(), "Missing results/sealed_test directory"
    actual_files = sorted(p.name for p in sealed_dir.iterdir() if p.is_file())
    assert actual_files == sorted(expected_csv_digests), f"Unexpected files in results/sealed_test: {actual_files}"
    for name, expected_sha in expected_csv_digests.items():
        actual_sha = sha256_file(sealed_dir / name)
        assert actual_sha == expected_sha, f"SHA-256 mismatch for results/sealed_test/{name}"
