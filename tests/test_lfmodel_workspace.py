from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_staging_manifest_resume_conflict_and_source_unchanged(tmp_path: Path) -> None:
    staging = _load_script("stage_topcow_to_lfmodel.py")
    source = tmp_path / "release"
    source.mkdir()
    first = source / "README.txt"
    nested = source / "imagesTr" / "sample.nii.gz"
    nested.parent.mkdir()
    first.write_bytes(b"release metadata\n")
    nested.write_bytes(b"synthetic nifti bytes")
    before = {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in (first, nested)
    }

    workspace = tmp_path / "workspace"
    manifest = staging.stage(source, workspace)
    assert manifest["file_count"] == 2
    assert [item["path"] for item in manifest["files"]] == ["README.txt", "imagesTr/sample.nii.gz"]
    on_disk = json.loads((workspace / "manifests" / staging.MANIFEST_NAME).read_text())
    assert on_disk["files"] == manifest["files"]
    staging.stage(source, workspace)
    assert before == {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in (first, nested)
    }

    destination = workspace / "raw" / staging.DATASET_NAME / "README.txt"
    destination.write_bytes(b"conflict")
    with pytest.raises(FileExistsError, match="conflicts"):
        staging.stage(source, workspace)
    staging.stage(source, workspace, overwrite=True)
    assert destination.read_bytes() == first.read_bytes()


def _synthetic_pair(sitk):
    shape = (40, 40, 40)
    zz, yy, xx = np.indices(shape)
    fixed_array = np.exp(-((xx - 20) ** 2 + (yy - 18) ** 2 + (zz - 21) ** 2) / 45.0)
    fixed_array += 0.7 * np.exp(-((xx - 12) ** 2 + (yy - 27) ** 2 + (zz - 14) ** 2) / 18.0)
    fixed_array[fixed_array < 0.02] = 0
    moving_array = np.roll(fixed_array, shift=(2, -3, 4), axis=(0, 1, 2))
    fixed = sitk.GetImageFromArray(fixed_array.astype(np.float32))
    moving = sitk.GetImageFromArray(moving_array.astype(np.float32))
    fixed.SetSpacing((0.8, 0.9, 1.2))
    fixed.SetOrigin((11.0, -4.0, 2.0))
    moving.CopyInformation(fixed)
    return fixed, moving


def test_registration_outputs_qc_geometry_skip_and_overwrite(tmp_path: Path) -> None:
    sitk = pytest.importorskip("SimpleITK")
    registration = _load_script("register_topcow_cta_to_mra.py")
    workspace = tmp_path / "workspace"
    images = workspace / "raw" / registration.DATASET_NAME / "imagesTr"
    images.mkdir(parents=True)
    fixed, moving = _synthetic_pair(sitk)
    sitk.WriteImage(fixed, str(images / "topcow_mr_001_0000.nii.gz"))
    sitk.WriteImage(moving, str(images / "topcow_ct_001_0000.nii.gz"))
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)

    result = registration.register_case(workspace, "001")
    assert result["status"] == "success", result
    assert np.isfinite(result["nmi_before"])
    assert result["nmi_after"] > result["nmi_before"]
    paths = registration.output_paths(workspace, "001")
    output = sitk.ReadImage(str(paths["image"]))
    fixed_on_disk = sitk.ReadImage(str(images / "topcow_mr_001_0000.nii.gz"))
    assert registration.same_geometry(output, fixed_on_disk)
    assert all(paths[key].is_file() for key in ("image", "transform", "json", "png"))
    assert registration.register_case(workspace, "001")["status"] == "skipped"
    assert registration.register_case(workspace, "001", overwrite=True)["status"] == "success"


def test_registration_records_missing_pair_failure(tmp_path: Path) -> None:
    pytest.importorskip("SimpleITK")
    registration = _load_script("register_topcow_cta_to_mra.py")
    workspace = tmp_path / "workspace"
    result = registration.register_case(workspace, "999")
    assert result["status"] == "failed"
    assert "Missing MRA/CTA pair" in result["error"]
    assert registration.output_paths(workspace, "999")["json"].is_file()


def test_lfmodel_configs_use_only_workspace_and_release_split() -> None:
    from bvs.config import load_config

    paths = [
        ROOT / "configs/train/lingfeng_transfer_topcow_binary.yaml",
        ROOT / "configs/train/lingfeng_scratch_topcow_binary.yaml",
        ROOT / "configs/experiments/topcow_mra_cta_teacher.yaml",
        ROOT / "configs/experiments/topcow_mra_student_kd.yaml",
    ]
    for path in paths:
        config = load_config(path)
        serialized = json.dumps(config)
        assert "/home/user/xiangyu/st/LFModel" in serialized
        assert "/home/user/xiangyu/st/datasets" not in serialized
    for path in paths[2:]:
        config = load_config(path)
        assert config["data"]["split_file"] == "configs/splits/topcow2024_release_seed42.json"
        assert "cta_registered_to_mra/images" in config["data"]["modalities"]["cta"]["directory"]


def test_teacher_registration_qc_gate_requires_complete_success(
    tmp_path: Path,
) -> None:
    from bvs.config import load_config
    from bvs.training.trainer import _require_complete_registration_qc

    config = load_config(ROOT / "configs/experiments/topcow_mra_cta_teacher.yaml")
    images = tmp_path / "cta_registered_to_mra" / "images"
    images.mkdir(parents=True)
    config["data"]["modalities"]["cta"]["directory"] = str(images)
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps({"train": ["001"], "val": ["002"], "internal_test": []}),
        encoding="utf-8",
    )
    config["data"]["split_file"] = str(split_path)

    with pytest.raises(RuntimeError, match="requires registration QC summary"):
        _require_complete_registration_qc(config)
    qc = images.parent / "qc"
    qc.mkdir()
    summary_path = qc / "summary.json"
    summary_path.write_text(
        json.dumps({"failed": 1, "cases": [{"case_id": "001", "status": "success"}]}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="blocked"):
        _require_complete_registration_qc(config)
    summary_path.write_text(
        json.dumps(
            {
                "failed": 0,
                "cases": [
                    {"case_id": "001", "status": "success"},
                    {"case_id": "002", "status": "success"},
                ],
            }
        ),
        encoding="utf-8",
    )
    _require_complete_registration_qc(config)
