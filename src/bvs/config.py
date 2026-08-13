from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

TOP_LEVEL_KEYS = {
    "schema_version",
    "experiment_name",
    "stage",
    "seed",
    "device",
    "output_root",
    "model",
    "data",
    "training",
    "loss",
    "inference",
    "_config_path",
}
MODEL_KEYS = {
    "name",
    "modalities",
    "student_modality",
    "in_channels",
    "out_channels",
    "num_classes",
    "base_channels",
    "pretrained_checkpoint",
    "teacher_checkpoint",
    "freeze_encoder",
}
DATA_KEYS = {
    "adapter",
    "root",
    "train_root",
    "val_root",
    "test_root",
    "split_file",
    "modalities",
    "label",
    "crop_or_pad_size",
    "patch_size",
    "patch_overlap",
    "normalization",
    "sampler",
    "samples_per_volume",
    "queue_length",
    "validation_samples_per_volume",
    "augmentation",
    "num_workers",
    "positive_probability",
}
TRAINING_KEYS = {
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "optimizer",
    "scheduler",
    "early_stopping_patience",
    "gradient_accumulation",
    "gradient_clip_norm",
    "logits_clip",
    "amp",
    "resume_checkpoint",
}
LOSS_KEYS = {"segmentation", "logit_distillation", "feature_distillation"}
INFERENCE_KEYS = {"branch", "window_size", "overlap", "compatibility_mode"}

DEFAULTS: dict[str, Any] = {
    "schema_version": 1,
    "seed": 42,
    "device": "auto",
    "output_root": "runs",
    "training": {
        "epochs": 100,
        "batch_size": 4,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "optimizer": "adam",
        "scheduler": {"name": "step_lr", "step_size": 10, "gamma": 0.8},
        "early_stopping_patience": 20,
        "gradient_accumulation": 1,
        "gradient_clip_norm": None,
        "logits_clip": None,
        "amp": "auto",
        "resume_checkpoint": None,
    },
    "loss": {
        "segmentation": {
            "cross_entropy_weight": 1.0,
            "dice_weight": 1.0,
            "dice_variant": "legacy_multiclass_squared",
        },
        "logit_distillation": {
            "enabled": False,
            "temperature": 10.0,
            "weight": 0.5,
        },
        "feature_distillation": {
            "enabled": False,
            "weight": 0.5,
            "margin": 1.0,
            "projection_dim": 8,
        },
    },
    "inference": {
        "branch": "student",
        "window_size": [48, 48, 48],
        "overlap": [4, 4, 4],
        "compatibility_mode": "torchio",
    },
}


def _merge(defaults: dict[str, Any], supplied: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(defaults)
    for key, value in supplied.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _unknown(mapping: dict[str, Any], allowed: set[str], location: str) -> None:
    keys = sorted(set(mapping) - allowed)
    if keys:
        raise ValueError(f"Unknown fields in {location}: {keys}")


def _triple(value: Any, location: str, *, positive: bool = True) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{location} must be a three-element list")
    if positive and any(int(item) <= 0 for item in value):
        raise ValueError(f"{location} values must be positive")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    _unknown(config, TOP_LEVEL_KEYS, "configuration")
    required = {"experiment_name", "stage", "model", "data"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing required configuration fields: {missing}")
    if config.get("schema_version", 1) != 1:
        raise ValueError("Only schema_version 1 is supported")
    if config["stage"] not in {"teacher", "student_kd", "supervised"}:
        raise ValueError("stage must be teacher, student_kd, or supervised")
    if not isinstance(config["experiment_name"], str) or not config["experiment_name"]:
        raise ValueError("experiment_name must be a non-empty string")

    model = config["model"]
    data = config["data"]
    training = config.get("training", {})
    loss = config.get("loss", {})
    inference = config.get("inference", {})
    for value, name in (
        (model, "model"),
        (data, "data"),
        (training, "training"),
        (loss, "loss"),
        (inference, "inference"),
    ):
        if not isinstance(value, dict):
            raise TypeError(f"{name} must be a mapping")
    _unknown(model, MODEL_KEYS, "model")
    _unknown(data, DATA_KEYS, "data")
    _unknown(training, TRAINING_KEYS, "training")
    _unknown(loss, LOSS_KEYS, "loss")
    _unknown(inference, INFERENCE_KEYS, "inference")
    if "scheduler" in training:
        _unknown(
            training["scheduler"],
            {"name", "step_size", "gamma"},
            "training.scheduler",
        )
    if "segmentation" in loss:
        _unknown(
            loss["segmentation"],
            {"cross_entropy_weight", "dice_weight", "dice_variant"},
            "loss.segmentation",
        )
    if "logit_distillation" in loss:
        _unknown(
            loss["logit_distillation"],
            {"enabled", "temperature", "weight"},
            "loss.logit_distillation",
        )
    if "feature_distillation" in loss:
        _unknown(
            loss["feature_distillation"],
            {"enabled", "weight", "margin", "projection_dim"},
            "loss.feature_distillation",
        )
    if "augmentation" in data:
        _unknown(data["augmentation"], {"enabled"}, "data.augmentation")

    model_name = model.get("name")
    if model_name not in {
        "configurable_lingfeng",
        "standard_unet3d",
        "lingfeng_student_transfer",
    }:
        raise ValueError(f"Unknown model name: {model_name}")
    if model_name in {"configurable_lingfeng", "lingfeng_student_transfer"}:
        for key in ("modalities", "student_modality", "in_channels", "num_classes"):
            if key not in model:
                raise ValueError(f"model.{key} is required for {model_name}")
        modalities = model["modalities"]
        if not isinstance(modalities, list) or not modalities:
            raise ValueError("model.modalities must be a non-empty list")
        if len(modalities) != len(set(modalities)):
            raise ValueError("model.modalities must not contain duplicates")
        if model["student_modality"] not in modalities:
            raise ValueError("model.student_modality must be in model.modalities")
        channels = model["in_channels"]
        if not isinstance(channels, dict) or set(channels) != set(modalities):
            raise ValueError("model.in_channels keys must exactly match model.modalities")
        if any(int(value) < 1 for value in channels.values()):
            raise ValueError("model.in_channels values must be positive")
        configured_modalities = data.get("modalities")
        if not isinstance(configured_modalities, dict):
            raise ValueError("data.modalities must be a mapping")
        if set(configured_modalities) != set(modalities):
            raise ValueError(
                "data.modalities keys must exactly match model.modalities"
            )
        if not data.get("label"):
            raise ValueError("data.label is required")
    if data.get("adapter") not in {"lingfeng_case_directory", "topcow"}:
        raise ValueError("data.adapter must be lingfeng_case_directory or topcow")
    for key in ("patch_size", "patch_overlap", "crop_or_pad_size"):
        if key in data:
            _triple(data[key], f"data.{key}", positive=key != "patch_overlap")
    if "patch_size" in data and "patch_overlap" in data:
        if any(
            int(overlap) >= int(size)
            for overlap, size in zip(data["patch_overlap"], data["patch_size"])
        ):
            raise ValueError("data.patch_overlap must be smaller than data.patch_size")
    if "window_size" in inference:
        _triple(inference["window_size"], "inference.window_size")
    if "overlap" in inference:
        _triple(inference["overlap"], "inference.overlap", positive=False)
    if config["stage"] == "student_kd":
        if not model.get("teacher_checkpoint"):
            raise ValueError("student_kd requires model.teacher_checkpoint")
        if not loss.get("logit_distillation", {}).get("enabled"):
            raise ValueError("student_kd requires logit_distillation.enabled=true")
        if not loss.get("feature_distillation", {}).get("enabled"):
            raise ValueError("student_kd requires feature_distillation.enabled=true")
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    supplied = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(supplied, dict):
        raise TypeError(f"Configuration must be a mapping: {config_path}")
    config = _merge(DEFAULTS, supplied)
    config["_config_path"] = str(config_path)
    return validate_config(config)


def model_spec_from_config(config: dict[str, Any]) -> dict[str, Any]:
    model = config["model"]
    if model["name"] == "lingfeng_student_transfer":
        name = "configurable_lingfeng"
    else:
        name = model["name"]
    return {
        "name": name,
        "modalities": list(model["modalities"]),
        "student_modality": model["student_modality"],
        "in_channels": dict(model["in_channels"]),
        "num_classes": int(model["num_classes"]),
        "base_channels": int(model.get("base_channels", 16)),
    }


def project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    if path.is_absolute():
        return path.resolve()
    config_path = config.get("_config_path")
    root = Path(config_path).parents[2] if config_path else Path.cwd()
    return (root / path).resolve()


def resolve_data_root(config: dict[str, Any], split: str | None = None) -> Path:
    data = config.get("data", {})
    configured = data.get(f"{split}_root") if split else None
    configured = configured or data.get("root")
    value = configured or os.environ.get("BVS_DATA_ROOT")
    if not value:
        raise RuntimeError(
            f"Set data.{split + '_root' if split else 'root'} (or data.root) "
            "or define BVS_DATA_ROOT"
        )
    return project_path(config, value)
