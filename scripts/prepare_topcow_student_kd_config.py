#!/usr/bin/env python3
"""Resolve a stable teacher checkpoint for the TopCoW student KD job."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bvs.config import load_config, project_path  # noqa: E402


def clean_config(config: dict[str, Any]) -> dict[str, Any]:
    cleaned = copy.deepcopy(config)
    cleaned.pop("_config_path", None)
    cleaned.setdefault("training", {})["resume_checkpoint"] = None
    return cleaned


def read_yaml(path: Path) -> dict[str, Any] | None:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return value if isinstance(value, dict) else None


def valid_summary(path: Path, experiment_name: str, stage: str) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(value, dict)
        and value.get("experiment_name") == experiment_name
        and value.get("stage") == stage
    )


def existing_student_teacher(config: dict[str, Any]) -> Path | None:
    experiment_root = (
        project_path(config, config["output_root"]) / config["experiment_name"]
    )
    requested = clean_config(config)
    if not experiment_root.is_dir():
        return None
    for run_dir in sorted(experiment_root.iterdir(), reverse=True):
        saved = read_yaml(run_dir / "resolved_config.yaml")
        if saved is None:
            continue
        checkpoint_value = saved.get("model", {}).get("teacher_checkpoint")
        if not checkpoint_value:
            continue
        candidate = copy.deepcopy(requested)
        candidate["model"]["teacher_checkpoint"] = checkpoint_value
        if clean_config(saved) != candidate:
            continue
        resumable = (run_dir / "checkpoints/latest.pt").is_file()
        completed = valid_summary(
            run_dir / "metrics/summary.json",
            config["experiment_name"],
            config["stage"],
        )
        checkpoint = project_path(config, checkpoint_value)
        if (resumable or completed) and checkpoint.is_file():
            return checkpoint
    return None


def newest_completed_teacher(config: dict[str, Any]) -> Path:
    experiment_root = (
        project_path(config, config["output_root"]) / config["experiment_name"]
    )
    requested = clean_config(config)
    if experiment_root.is_dir():
        for run_dir in sorted(experiment_root.iterdir(), reverse=True):
            saved = read_yaml(run_dir / "resolved_config.yaml")
            checkpoint = run_dir / "checkpoints/best.pt"
            if (
                saved is not None
                and clean_config(saved) == requested
                and checkpoint.is_file()
                and valid_summary(
                    run_dir / "metrics/summary.json",
                    config["experiment_name"],
                    config["stage"],
                )
            ):
                return checkpoint.resolve()
    raise RuntimeError(
        "No completed teacher run with an identical resolved configuration and "
        f"best.pt was found under {experiment_root}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-config", required=True)
    parser.add_argument("--teacher-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    student = load_config(args.student_config)
    teacher = load_config(args.teacher_config)
    checkpoint = existing_student_teacher(student)
    if checkpoint is None:
        checkpoint = newest_completed_teacher(teacher)
    resolved = clean_config(student)
    resolved["model"]["teacher_checkpoint"] = str(checkpoint)
    Path(args.output).write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    print(checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
