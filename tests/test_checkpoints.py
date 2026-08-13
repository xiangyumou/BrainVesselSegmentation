from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bvs.checkpoints import (
    build_lingfeng_from_spec,
    convert_legacy_checkpoint,
    load_lingfeng_student_checkpoint,
    load_prediction_checkpoint,
    verify_lingfeng_equivalence,
)
from bvs.config import load_config
from bvs.models import LingfengLegacyModel, LingfengMRAStudent
from bvs.models import StandardUNet3D

CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "artifacts/checkpoints/lingfeng/student_best_checkpoint_multimodaltune9.pt"
)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Lingfeng checkpoint is not installed")
def test_checkpoint_loads_all_required_student_keys() -> None:
    report = load_lingfeng_student_checkpoint(LingfengMRAStudent(), CHECKPOINT)
    assert len(report["loaded_keys"]) == 25
    assert report["ignored_keys"]
    assert report["sha256"] == "ccecc4b52ffa3832ebf2580945b19e71315f2c26c7f0149f6ecd099ca0997a22"


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Lingfeng checkpoint is not installed")
def test_legacy_student_logits_are_equivalent() -> None:
    report = verify_lingfeng_equivalence(CHECKPOINT, "cpu", patch_size=16)
    assert report["equivalent"]
    assert report["max_abs_error"] <= 1e-5


def test_missing_student_key_fails_loudly(tmp_path: Path) -> None:
    model = LingfengMRAStudent()
    state = model.state_dict()
    state.pop(next(iter(state)))
    checkpoint = tmp_path / "incomplete.pt"
    torch.save({"model": state}, checkpoint)
    with pytest.raises(RuntimeError, match="missing="):
        load_lingfeng_student_checkpoint(model, checkpoint)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Lingfeng checkpoint is not installed")
def test_real_student_checkpoint_converts_to_unified_schema(tmp_path: Path) -> None:
    config = load_config(
        Path(__file__).resolve().parents[1]
        / "configs/reproduction/lingfeng_student_kd_legacy_code.yaml"
    )
    output = tmp_path / "unified.pt"
    report = convert_legacy_checkpoint(CHECKPOINT, config, output, verify=True)
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert report["verification"]["max_abs_error"] <= 1e-5
    assert payload["schema_version"] == 1
    assert payload["student_projection_state"]["weight"].shape == (8, 16)
    assert payload["teacher_projection_state"] is None
    model = build_lingfeng_from_spec(payload["model_spec"])
    model.load_state_dict(payload["model_state"], strict=True)


def test_conversion_rejects_wrong_shape_without_writing_output(tmp_path: Path) -> None:
    source = LingfengLegacyModel().state_dict()
    first = next(iter(source))
    source[first] = source[first][0:1]
    checkpoint = tmp_path / "bad.pt"
    output = tmp_path / "should_not_exist.pt"
    torch.save({"model": source}, checkpoint)
    config = load_config(
        Path(__file__).resolve().parents[1]
        / "configs/reproduction/lingfeng_teacher_legacy_code.yaml"
    )
    with pytest.raises(ValueError, match="Shape mismatch"):
        convert_legacy_checkpoint(checkpoint, config, output)
    assert not output.exists()


@pytest.mark.parametrize("wrapper", ["model_state", "model", "bare"])
def test_standard_prediction_checkpoint_formats(
    tmp_path: Path, wrapper: str
) -> None:
    config = load_config(
        Path(__file__).resolve().parents[1]
        / "configs/train/unet3d_topcow_binary.yaml"
    )
    config["model"]["base_channels"] = 2
    model = StandardUNet3D(base_channels=2)
    state = model.state_dict()
    if wrapper == "model_state":
        payload = {
            "schema_version": 1,
            "model_spec": {
                "name": "standard_unet3d",
                "in_channels": 1,
                "num_classes": 2,
                "base_channels": 2,
            },
            "model_state": state,
        }
    elif wrapper == "model":
        payload = {"model": state}
    else:
        payload = state
    checkpoint = tmp_path / f"{wrapper}.pt"
    torch.save(payload, checkpoint)
    loaded = load_prediction_checkpoint(config, checkpoint, "cpu")
    assert isinstance(loaded, StandardUNet3D)


def test_prediction_checkpoint_spec_mismatch_is_explicit(tmp_path: Path) -> None:
    config = load_config(
        Path(__file__).resolve().parents[1]
        / "configs/train/unet3d_topcow_binary.yaml"
    )
    config["model"]["base_channels"] = 2
    checkpoint = tmp_path / "bad_spec.pt"
    torch.save(
        {
            "schema_version": 1,
            "model_spec": {
                "name": "standard_unet3d",
                "in_channels": 1,
                "num_classes": 3,
                "base_channels": 2,
            },
            "model_state": StandardUNet3D(
                out_channels=3, base_channels=2
            ).state_dict(),
        },
        checkpoint,
    )
    with pytest.raises(RuntimeError, match="checkpoint=.*config="):
        load_prediction_checkpoint(config, checkpoint, "cpu")
