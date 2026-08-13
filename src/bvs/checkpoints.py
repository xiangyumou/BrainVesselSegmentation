from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import model_spec_from_config, project_path
from .models.lingfeng import (
    ConfigurableLingfengModel,
    LingfengLegacyModel,
    LingfengMRAStudent,
)
from .models.unet3d import StandardUNet3D

LEGACY_PREFIX_MAP = {
    "input_mra_encoder.": "encoders.mra.",
    "ce_t1.": "encoders.t1.",
    "ce_t2.": "encoders.t2.",
    "ce_pd.": "encoders.pd.",
    "att_c1.": "attention.s1.",
    "att_c2.": "attention.s2.",
    "att_c3.": "attention.s3.",
    "att_c4.": "attention.s4.",
    "fusion_c1.": "fusion.s1.",
    "fusion_c2.": "fusion.s2.",
    "fusion_c3.": "fusion.s3.",
    "fusion_c4.": "fusion.s4.",
    "mask_de_prs.": "student_decoder.",
    "mask_de_abs.": "teacher_decoder.",
    "metric_prs.": "student_metric.",
    "metric_abs.": "teacher_metric.",
}
STUDENT_LEGACY_PREFIXES = ("input_mra_encoder.", "mask_de_prs.", "metric_prs.")
STUDENT_UNIFIED_PREFIXES = (
    "encoders.mra.",
    "student_decoder.",
    "student_metric.",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_state(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    clean: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            raise TypeError(f"Checkpoint state value is not a tensor: {key}")
        normalized = key[7:] if key.startswith("module.") else key
        if normalized in clean:
            raise ValueError(f"Duplicate key after removing module prefix: {normalized}")
        clean[normalized] = value
    return clean


def _load_checkpoint(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must be a mapping")
    state = payload.get("model_state", payload.get("model", payload.get("state_dict", payload)))
    if not isinstance(state, dict):
        raise TypeError("Checkpoint model state must be a mapping")
    return _clean_state(state), payload


def build_lingfeng_from_spec(spec: dict[str, Any]) -> ConfigurableLingfengModel:
    if spec.get("name") != "configurable_lingfeng":
        raise ValueError(f"Not a configurable Lingfeng model spec: {spec.get('name')}")
    return ConfigurableLingfengModel(
        modalities=spec["modalities"],
        student_modality=spec["student_modality"],
        in_channels=spec["in_channels"],
        num_classes=int(spec["num_classes"]),
        base_channels=int(spec.get("base_channels", 16)),
    )


def build_model_from_spec(spec: dict[str, Any]) -> torch.nn.Module:
    if spec.get("name") == "standard_unet3d":
        return StandardUNet3D(
            in_channels=int(spec["in_channels"]),
            out_channels=int(spec["num_classes"]),
            base_channels=int(spec.get("base_channels", 32)),
        )
    return build_lingfeng_from_spec(spec)


def _require_matching_spec(
    payload: dict[str, Any], requested_spec: dict[str, Any]
) -> None:
    checkpoint_spec = payload.get("model_spec")
    if checkpoint_spec != requested_spec:
        raise RuntimeError(
            "Checkpoint model spec is incompatible: "
            f"checkpoint={checkpoint_spec}, config={requested_spec}"
        )


def load_prediction_checkpoint(
    config: dict[str, Any],
    checkpoint: str | Path,
    device: torch.device | str,
) -> torch.nn.Module:
    """Build and load the configured prediction model.

    Unified checkpoints are checked against the exact configured model spec.
    Historical state dictionaries remain supported only for model families
    whose architecture can be reconstructed from the configuration.
    """

    checkpoint_path = project_path(config, checkpoint)
    state, payload = _load_checkpoint(checkpoint_path)
    requested_spec = model_spec_from_config(config)
    name = config["model"]["name"]

    if name == "configurable_lingfeng":
        if (
            payload.get("schema_version") != 1
            or "model_spec" not in payload
            or "model_state" not in payload
        ):
            raise RuntimeError(
                "configurable_lingfeng prediction requires a schema_version 1 "
                "checkpoint containing model_spec and model_state"
            )
        _require_matching_spec(payload, requested_spec)
        model = build_lingfeng_from_spec(requested_spec)
        model.load_state_dict(state, strict=True)
    elif name == "standard_unet3d":
        if payload.get("schema_version") == 1 and "model_spec" in payload:
            _require_matching_spec(payload, requested_spec)
        model = build_model_from_spec(requested_spec)
        model.load_state_dict(state, strict=True)
    elif name == "lingfeng_student_transfer":
        model = build_lingfeng_from_spec(requested_spec)
        if payload.get("schema_version") == 1 and "model_spec" in payload:
            _require_matching_spec(payload, requested_spec)
            model.load_state_dict(state, strict=True)
        else:
            load_lingfeng_student_checkpoint(model, checkpoint_path)
    else:
        raise ValueError(f"Unknown model name: {name}")
    return model.to(device).eval()


def load_unified_checkpoint(
    model: torch.nn.Module, checkpoint: str | Path, strict: bool = True
) -> dict[str, Any]:
    state, payload = _load_checkpoint(checkpoint)
    model.load_state_dict(state, strict=strict)
    return payload


def _map_legacy_key(key: str) -> str:
    for source, target in LEGACY_PREFIX_MAP.items():
        if key.startswith(source):
            return target + key[len(source) :]
    raise KeyError(f"Unknown legacy model key: {key}")


def map_legacy_state(
    source_state: dict[str, torch.Tensor],
    model: ConfigurableLingfengModel,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    mapped: dict[str, torch.Tensor] = {}
    key_map: dict[str, str] = {}
    expected = model.state_dict()
    for source_key, value in source_state.items():
        target_key = _map_legacy_key(source_key)
        if target_key in mapped:
            raise ValueError(f"Multiple source keys map to {target_key}")
        if target_key not in expected:
            raise KeyError(f"Mapped target key does not exist: {source_key} -> {target_key}")
        if tuple(value.shape) != tuple(expected[target_key].shape):
            raise ValueError(
                f"Shape mismatch for {source_key} -> {target_key}: "
                f"source={tuple(value.shape)}, target={tuple(expected[target_key].shape)}"
            )
        mapped[target_key] = value
        key_map[source_key] = target_key
    missing = sorted(set(expected) - set(mapped))
    if missing:
        raise RuntimeError(f"Legacy checkpoint does not provide target keys: {missing}")
    return mapped, key_map


def _max_error(expected: torch.Tensor, actual: torch.Tensor) -> float:
    return float(torch.max(torch.abs(expected - actual)).cpu())


@torch.inference_mode()
def verify_converted_state(
    legacy_state: dict[str, torch.Tensor],
    unified_state: dict[str, torch.Tensor],
    model_spec: dict[str, Any],
    patch_size: int = 16,
) -> dict[str, Any]:
    required_spec = {
        "modalities": ["mra", "t1", "t2", "pd"],
        "student_modality": "mra",
        "in_channels": {"mra": 1, "t1": 1, "t2": 1, "pd": 1},
        "num_classes": 2,
        "base_channels": 16,
    }
    for key, expected in required_spec.items():
        if model_spec.get(key) != expected:
            raise ValueError(
                "Legacy numerical verification requires the canonical four-modality "
                f"model; model_spec.{key}={model_spec.get(key)!r}"
            )
    legacy = LingfengLegacyModel().eval()
    legacy.load_state_dict(legacy_state, strict=True)
    unified = build_lingfeng_from_spec(model_spec).eval()
    unified.load_state_dict(unified_state, strict=True)
    generator = torch.Generator(device="cpu").manual_seed(1729)
    inputs = {
        name: torch.randn(
            1, 1, patch_size, patch_size, patch_size, generator=generator
        )
        for name in model_spec["modalities"]
    }
    legacy_inputs = {
        "source": inputs["mra"],
        "input_t1": inputs["t1"],
        "input_t2": inputs["t2"],
        "input_pd": inputs["pd"],
    }
    old = legacy(legacy_inputs, is_training=False)
    teacher = unified.forward_teacher(inputs)
    student = unified.forward_student(inputs)
    errors = {
        "teacher_logits": _max_error(old["seg_logit_abs"], teacher["logits"]),
        "teacher_probabilities": _max_error(
            old["seg_pred_abs"], teacher["probabilities"]
        ),
        "teacher_decoder_feature": _max_error(
            old["d1_out_abs"], teacher["decoder_feature"]
        ),
        "teacher_metric_feature": _max_error(
            old["feat_abs"], teacher["metric_feature"]
        ),
        "student_logits": _max_error(old["seg_logit_prs"], student["logits"]),
        "student_probabilities": _max_error(
            old["seg_pred_prs"], student["probabilities"]
        ),
        "student_decoder_feature": _max_error(
            old["d1_out_prs"], student["decoder_feature"]
        ),
        "student_metric_feature": _max_error(
            old["feat_prs"], student["metric_feature"]
        ),
    }
    maximum = max(errors.values())
    if maximum > 1e-5:
        raise AssertionError(
            f"Converted checkpoint differs from legacy by {maximum}; errors={errors}"
        )
    return {"max_abs_error": maximum, "outputs": errors, "tolerance": 1e-5}


def make_unified_checkpoint(
    *,
    stage: str,
    model: torch.nn.Module,
    model_spec: dict[str, Any],
    resolved_config: dict[str, Any],
    epoch: int = 0,
    best_validation_loss: float = float("inf"),
    best_validation_dice: float | None = None,
    best_validation_cldice: float | None = None,
    best_epoch: int | None = None,
    stale_epochs: int = 0,
    student_projection: torch.nn.Module | None = None,
    teacher_projection: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    source: dict[str, Any] | None = None,
    conversion_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": stage,
        "model_spec": model_spec,
        "model_state": model.state_dict(),
        "student_projection_state": (
            student_projection.state_dict() if student_projection is not None else None
        ),
        "teacher_projection_state": (
            teacher_projection.state_dict() if teacher_projection is not None else None
        ),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "best_validation_loss": (
            float(best_validation_loss)
            if best_validation_loss is not None
            else None
        ),
        "best_validation_dice": best_validation_dice,
        "best_validation_cldice": best_validation_cldice,
        "best_epoch": best_epoch,
        "stale_epochs": int(stale_epochs),
        "selection_metric": "mean_dice",
        "resolved_config": resolved_config,
        "random_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "source": source,
        "conversion_report": conversion_report,
    }


def convert_legacy_checkpoint(
    source: str | Path,
    config: dict[str, Any],
    output: str | Path,
    verify: bool = False,
) -> dict[str, Any]:
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {output_path}")
    source_state, payload = _load_checkpoint(source_path)
    spec = model_spec_from_config(config)
    model = build_lingfeng_from_spec(spec)
    mapped, key_map = map_legacy_state(source_state, model)
    model.load_state_dict(mapped, strict=True)
    projection_state = payload.get("proj_head")
    if projection_state is not None:
        projection_state = _clean_state(dict(projection_state))
        if set(projection_state) == {"0.weight"}:
            projection_state = {"weight": projection_state["0.weight"]}
        projection_dim = int(
            config["loss"]["feature_distillation"]["projection_dim"]
        )
        expected_projection = torch.nn.Linear(16, projection_dim, bias=False)
        expected_projection.load_state_dict(projection_state, strict=True)
    stage = "student_kd" if projection_state is not None else "teacher"
    report: dict[str, Any] = {
        "source_keys": len(source_state),
        "target_keys": len(mapped),
        "key_map": key_map,
        "student_projection_restored": projection_state is not None,
        "teacher_projection_restored": False,
        "unrecoverable": [
            "Legacy checkpoints did not save the randomly initialized frozen "
            "teacher projection head."
        ],
    }
    if verify:
        report["verification"] = verify_converted_state(source_state, mapped, spec)
    clean_config = {key: value for key, value in config.items() if key != "_config_path"}
    checkpoint = {
        "schema_version": 1,
        "stage": stage,
        "model_spec": spec,
        "model_state": mapped,
        "student_projection_state": projection_state,
        "teacher_projection_state": None,
        "optimizer_state": payload.get("optim"),
        "scheduler_state": None,
        "epoch": int(payload.get("epoch", 0)),
        "best_validation_loss": float(payload.get("best_val_loss", float("inf"))),
        "best_validation_dice": payload.get("best_val_dice"),
        "resolved_config": clean_config,
        "source": {
            "format": "lingfeng_legacy",
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        },
        "conversion_report": report,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output_path)
    return {
        "output": str(output_path),
        "stage": stage,
        "source_sha256": checkpoint["source"]["sha256"],
        **report,
    }


def load_lingfeng_student_checkpoint(
    model: LingfengMRAStudent | ConfigurableLingfengModel, checkpoint: str | Path
) -> dict[str, Any]:
    state, payload = _load_checkpoint(checkpoint)
    original_keys = set(state)
    if any(key.startswith("encoders.") for key in state):
        prefixes = STUDENT_UNIFIED_PREFIXES
    else:
        state = {
            _map_legacy_key(key): value
            for key, value in state.items()
            if key.startswith(STUDENT_LEGACY_PREFIXES)
        }
        prefixes = STUDENT_UNIFIED_PREFIXES
    required = {
        key: value
        for key, value in model.state_dict().items()
        if key.startswith(STUDENT_UNIFIED_PREFIXES)
    }
    selected = {key: value for key, value in state.items() if key.startswith(prefixes)}
    missing = sorted(set(required) - set(selected))
    unexpected = sorted(set(selected) - set(required))
    mismatches = sorted(
        key
        for key in set(required) & set(selected)
        if tuple(required[key].shape) != tuple(selected[key].shape)
    )
    if missing or unexpected or mismatches:
        raise RuntimeError(
            "Lingfeng student checkpoint is incompatible: "
            f"missing={missing}, unexpected={unexpected}, shape_mismatches={mismatches}"
        )
    current = model.state_dict()
    current.update(selected)
    model.load_state_dict(current, strict=True)
    return {
        "checkpoint": str(Path(checkpoint)),
        "sha256": sha256_file(checkpoint),
        "loaded_keys": sorted(selected),
        "ignored_keys": sorted(
            original_keys
            - {
                key
                for key in original_keys
                if key.startswith(STUDENT_LEGACY_PREFIXES)
                or key.startswith(STUDENT_UNIFIED_PREFIXES)
            }
        ),
        "epoch": payload.get("epoch"),
        "best_val_loss": payload.get(
            "best_validation_loss", payload.get("best_val_loss")
        ),
        "best_val_dice": payload.get(
            "best_validation_dice", payload.get("best_val_dice")
        ),
    }


def load_lingfeng_legacy_checkpoint(
    model: LingfengLegacyModel, checkpoint: str | Path
) -> dict[str, Any]:
    state, payload = _load_checkpoint(checkpoint)
    model.load_state_dict(state, strict=True)
    return {
        "checkpoint": str(Path(checkpoint)),
        "sha256": sha256_file(checkpoint),
        "loaded_keys": sorted(state),
        "ignored_keys": [],
        "epoch": payload.get("epoch"),
    }


@torch.inference_mode()
def verify_lingfeng_equivalence(
    checkpoint: str | Path, device: str | torch.device = "cpu", patch_size: int = 16
) -> dict[str, Any]:
    target = torch.device(device)
    legacy = LingfengLegacyModel().to(target).eval()
    student = LingfengMRAStudent().to(target).eval()
    load_lingfeng_legacy_checkpoint(legacy, checkpoint)
    student_report = load_lingfeng_student_checkpoint(student, checkpoint)
    image = torch.randn(1, 1, patch_size, patch_size, patch_size, device=target)
    expected = legacy(
        {
            "source": image,
            "input_t1": torch.zeros_like(image),
            "input_t2": torch.zeros_like(image),
            "input_pd": torch.zeros_like(image),
        },
        is_training=False,
    )["seg_logit_prs"]
    actual = student(image)["logits"]
    error = _max_error(expected, actual)
    tolerance = 1e-4 if target.type == "mps" else 1e-5
    if error > tolerance:
        raise AssertionError(f"Legacy/student logits differ by {error}, tolerance={tolerance}")
    return {
        **student_report,
        "device": target.type,
        "max_abs_error": error,
        "tolerance": tolerance,
        "equivalent": True,
    }


def write_inspection_report(
    checkpoint: str | Path, output: str | Path | None = None
) -> dict[str, Any]:
    _, payload = _load_checkpoint(checkpoint)
    if payload.get("schema_version") == 1 and "model_spec" in payload:
        report = {
            "checkpoint": str(Path(checkpoint)),
            "sha256": sha256_file(checkpoint),
            "schema_version": payload["schema_version"],
            "stage": payload.get("stage"),
            "model_spec": payload["model_spec"],
            "epoch": payload.get("epoch"),
            "best_validation_loss": payload.get("best_validation_loss"),
            "best_validation_dice": payload.get("best_validation_dice"),
            "best_validation_cldice": payload.get("best_validation_cldice"),
            "best_epoch": payload.get("best_epoch"),
            "stale_epochs": payload.get("stale_epochs", 0),
            "selection_metric": payload.get("selection_metric"),
            "source": payload.get("source"),
            "conversion_report": payload.get("conversion_report"),
        }
    else:
        report = load_lingfeng_student_checkpoint(LingfengMRAStudent(), checkpoint)
    if output:
        Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
