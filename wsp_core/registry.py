"""Central registry of all six `dental-wsp` 2x3 benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class TaskSpec:
    task: str
    dataset: str
    track: str  # "seg" or "cls"
    architecture: str
    prompt_relpath: str
    config_relpath: str
    design_lock_relpath: str

    def prompt_path(self, root: Path = REPO_ROOT) -> Path:
        return root / self.prompt_relpath

    def config_path(self, root: Path = REPO_ROOT) -> Path:
        return root / self.config_relpath

    def design_lock_path(self, root: Path = REPO_ROOT) -> Path:
        return root / self.design_lock_relpath

    def load_config(self, root: Path = REPO_ROOT) -> dict[str, Any]:
        with open(self.config_path(root), "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)


TASKS: dict[str, TaskSpec] = {
    "akudental_seg": TaskSpec(
        task="akudental_seg",
        dataset="akudental",
        track="seg",
        architecture="pano-wsp82-v1",
        prompt_relpath="benchmarks/akudental/seg_prompt.md",
        config_relpath="benchmarks/akudental/configs/seg.yaml",
        design_lock_relpath="benchmarks/akudental/locks/akudental_seg.design.lock.json",
    ),
    "akudental_cls": TaskSpec(
        task="akudental_cls",
        dataset="akudental",
        track="cls",
        architecture="cls-lenet2d-v1",
        prompt_relpath="benchmarks/akudental/cls_prompt.md",
        config_relpath="benchmarks/akudental/configs/cls.yaml",
        design_lock_relpath="benchmarks/akudental/locks/akudental_cls.design.lock.json",
    ),
    "denpar_seg": TaskSpec(
        task="denpar_seg",
        dataset="denpar",
        track="seg",
        architecture="periapical-wsp82-v1",
        prompt_relpath="benchmarks/denpar/seg_prompt.md",
        config_relpath="benchmarks/denpar/configs/seg.yaml",
        design_lock_relpath="benchmarks/denpar/locks/denpar_seg.design.lock.json",
    ),
    "denpar_cls": TaskSpec(
        task="denpar_cls",
        dataset="denpar",
        track="cls",
        architecture="cls-lenet2d-v1",
        prompt_relpath="benchmarks/denpar/cls_prompt.md",
        config_relpath="benchmarks/denpar/configs/cls.yaml",
        design_lock_relpath="benchmarks/denpar/locks/denpar_cls.design.lock.json",
    ),
    "dentvoxel_seg": TaskSpec(
        task="dentvoxel_seg",
        dataset="dentvoxel",
        track="seg",
        architecture="cbct-wsp82-v1",
        prompt_relpath="benchmarks/dentvoxel/seg_prompt.md",
        config_relpath="benchmarks/dentvoxel/configs/seg.yaml",
        design_lock_relpath="benchmarks/dentvoxel/locks/dentvoxel_seg.design.lock.json",
    ),
    "dentvoxel_cls": TaskSpec(
        task="dentvoxel_cls",
        dataset="dentvoxel",
        track="cls",
        architecture="cls-lenet2d-v1",
        prompt_relpath="benchmarks/dentvoxel/cls_prompt.md",
        config_relpath="benchmarks/dentvoxel/configs/cls.yaml",
        design_lock_relpath="benchmarks/dentvoxel/locks/dentvoxel_cls.design.lock.json",
    ),
}


def get_task_spec(task: str) -> TaskSpec:
    if task not in TASKS:
        raise KeyError(f"Unknown task '{task}'. Choose from: {', '.join(sorted(TASKS))}")
    return TASKS[task]
