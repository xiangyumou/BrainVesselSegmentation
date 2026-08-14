from __future__ import annotations

import copy
from pathlib import Path

import pytest

from bvs.config import load_config, validate_config

CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/train/unet3d_topcow_binary.yaml"
)


def _config() -> dict:
    config = load_config(CONFIG)
    config.pop("_config_path", None)
    return config


@pytest.mark.parametrize(
    ("location", "value", "message"),
    [
        (("data", "positive_probability"), 1.1, "positive_probability"),
        (("data", "cache_max_cases"), -1, "cache_max_cases"),
        (("training", "epochs"), 0, "epochs"),
        (("training", "batch_size"), 0, "batch_size"),
        (("training", "gradient_accumulation"), 0, "gradient_accumulation"),
        (("training", "amp"), "sometimes", "training.amp"),
        (("inference", "branch"), "both", "inference.branch"),
        (("inference", "compatibility_mode"), "blend", "compatibility_mode"),
        (("data", "normalization"), "none", "normalization"),
    ],
)
def test_configuration_range_validation(
    location: tuple[str, str], value: object, message: str
) -> None:
    config = _config()
    config[location[0]][location[1]] = value
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_inference_overlap_must_be_smaller_than_window() -> None:
    config = _config()
    config["inference"]["overlap"] = [8, 8, 8]
    config["inference"]["window_size"] = [8, 8, 8]
    with pytest.raises(ValueError, match="smaller"):
        validate_config(config)


def test_pattern_directory_requires_case_id_placeholder() -> None:
    config = _config()
    config["data"]["modalities"]["mra"]["pattern"] = "mra.nii.gz"
    with pytest.raises(ValueError, match="case_id"):
        validate_config(config)


def test_topcow_adapter_remains_supported() -> None:
    config = _config()
    config["data"]["adapter"] = "topcow"
    assert validate_config(config)["data"]["adapter"] == "topcow"


def test_crop_or_pad_size_can_be_disabled() -> None:
    config = _config()
    config["data"]["crop_or_pad_size"] = None
    assert validate_config(config)["data"]["crop_or_pad_size"] is None


@pytest.mark.parametrize(
    "field",
    ["sampler", "queue_length", "patch_overlap", "test_root"],
)
def test_removed_data_fields_fail(field: str) -> None:
    config = _config()
    config["data"][field] = None
    with pytest.raises(ValueError, match="Unknown fields"):
        validate_config(config)


def test_removed_model_out_channels_fails() -> None:
    config = _config()
    config["model"]["out_channels"] = 2
    with pytest.raises(ValueError, match="Unknown fields"):
        validate_config(config)


def test_freeze_encoder_is_transfer_only() -> None:
    config = _config()
    config["model"]["freeze_encoder"] = True
    with pytest.raises(ValueError, match="only allowed"):
        validate_config(config)


def test_freeze_encoder_transfer_is_valid() -> None:
    config = load_config(
        CONFIG.parents[1] / "train/lingfeng_transfer_topcow_binary.yaml"
    )
    assert validate_config(copy.deepcopy(config))["model"]["freeze_encoder"] is False


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("training", "learning_rate", 0, "learning_rate"),
        ("training", "weight_decay", -1, "weight_decay"),
        ("training", "learning_rate", float("nan"), "learning_rate"),
        ("training", "gradient_clip_norm", 0, "gradient_clip_norm"),
        ("training", "logits_clip", -1, "logits_clip"),
    ],
)
def test_training_numeric_boundaries(
    section: str, field: str, value: object, message: str
) -> None:
    config = _config()
    config[section][field] = value
    with pytest.raises(ValueError, match=message):
        validate_config(config)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("segmentation", "dice_weight", -1, "dice_weight"),
        ("logit_distillation", "temperature", 0, "temperature"),
        ("logit_distillation", "weight", -1, "weight"),
        ("feature_distillation", "margin", -1, "margin"),
        ("feature_distillation", "projection_dim", 0, "projection_dim"),
    ],
)
def test_loss_numeric_boundaries(
    section: str, field: str, value: object, message: str
) -> None:
    config = _config()
    config["loss"][section][field] = value
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_unsupported_optimizer_and_scheduler_fail_during_config_loading() -> None:
    config = _config()
    config["training"]["optimizer"] = "sgd"
    with pytest.raises(ValueError, match="optimizer"):
        validate_config(config)
    config = _config()
    config["training"]["scheduler"]["name"] = "cosine"
    with pytest.raises(ValueError, match="scheduler.name"):
        validate_config(config)


def test_standard_unet_rejects_teacher_stage_and_branch() -> None:
    config = _config()
    config["stage"] = "teacher"
    with pytest.raises(ValueError, match="supervised"):
        validate_config(config)
    config = _config()
    config["inference"]["branch"] = "teacher"
    with pytest.raises(ValueError, match="branch=teacher"):
        validate_config(config)


def test_all_repository_configs_validate() -> None:
    root = CONFIG.parents[2]
    for path in sorted((root / "configs").rglob("*.yaml")):
        load_config(path)
