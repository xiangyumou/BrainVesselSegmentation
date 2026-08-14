from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

from bvs.config import load_config

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_topcow_student_kd_config",
    ROOT / "scripts/prepare_topcow_student_kd_config.py",
)
assert SPEC is not None and SPEC.loader is not None
RESOLVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESOLVER)


def write_run(
    root: Path,
    config: dict,
    name: str,
    *,
    completed: bool,
    latest: bool = False,
) -> Path:
    run = root / config["experiment_name"] / name
    (run / "checkpoints").mkdir(parents=True)
    (run / "metrics").mkdir()
    (run / "resolved_config.yaml").write_text(
        yaml.safe_dump(RESOLVER.clean_config(config), sort_keys=False),
        encoding="utf-8",
    )
    if completed:
        (run / "metrics/summary.json").write_text(
            json.dumps(
                {
                    "experiment_name": config["experiment_name"],
                    "stage": config["stage"],
                }
            ),
            encoding="utf-8",
        )
    if latest:
        (run / "checkpoints/latest.pt").touch()
    return run


def test_selects_completed_teacher_and_reuses_student_checkpoint(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    teacher = load_config(ROOT / "configs/experiments/topcow_mra_cta_teacher.yaml")
    teacher["output_root"] = str(runs)
    old_teacher = write_run(runs, teacher, "20260101-000000", completed=True)
    old_checkpoint = old_teacher / "checkpoints/best.pt"
    old_checkpoint.touch()

    assert RESOLVER.newest_completed_teacher(teacher) == old_checkpoint.resolve()

    student = load_config(ROOT / "configs/experiments/topcow_mra_student_kd.yaml")
    student["output_root"] = str(runs)
    assert RESOLVER.existing_student_teacher(student) is None

    student["model"]["teacher_checkpoint"] = str(old_checkpoint.resolve())
    write_run(runs, student, "20260102-000000", completed=False, latest=True)

    new_teacher = write_run(runs, teacher, "20260103-000000", completed=True)
    (new_teacher / "checkpoints/best.pt").touch()
    assert RESOLVER.newest_completed_teacher(teacher) == (
        new_teacher / "checkpoints/best.pt"
    ).resolve()
    assert RESOLVER.existing_student_teacher(student) == old_checkpoint.resolve()


def test_teacher_selection_rejects_incomplete_or_changed_run(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    teacher = load_config(ROOT / "configs/experiments/topcow_mra_cta_teacher.yaml")
    teacher["output_root"] = str(runs)
    incomplete = write_run(runs, teacher, "20260101-000000", completed=False)
    (incomplete / "checkpoints/best.pt").touch()

    changed = load_config(ROOT / "configs/experiments/topcow_mra_cta_teacher.yaml")
    changed["output_root"] = str(runs)
    changed["training"]["batch_size"] += 1
    changed_run = write_run(runs, changed, "20260102-000000", completed=True)
    (changed_run / "checkpoints/best.pt").touch()

    try:
        RESOLVER.newest_completed_teacher(teacher)
    except RuntimeError as error:
        assert "No completed teacher run" in str(error)
    else:
        raise AssertionError("Expected incompatible teacher runs to be rejected")
