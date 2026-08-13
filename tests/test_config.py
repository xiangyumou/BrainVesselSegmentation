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
