from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .models.lingfeng import LingfengLegacyModel, LingfengMRAStudent

STUDENT_PREFIXES = ("input_mra_encoder.", "mask_de_prs.", "metric_prs.")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must be a mapping")
    state = payload.get("model", payload.get("state_dict", payload))
    if not isinstance(state, dict):
        raise TypeError("Checkpoint model/state_dict entry must be a mapping")
    clean = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    return clean, payload


def load_lingfeng_student_checkpoint(
    model: LingfengMRAStudent, checkpoint: str | Path
) -> dict[str, Any]:
    state, payload = _load_checkpoint(checkpoint)
    selected = {
        key: value for key, value in state.items() if key.startswith(STUDENT_PREFIXES)
    }
    required = model.state_dict()
    missing = sorted(set(required) - set(selected))
    unexpected_student = sorted(set(selected) - set(required))
    shape_mismatches = sorted(
        key
        for key in set(required) & set(selected)
        if tuple(required[key].shape) != tuple(selected[key].shape)
    )
    if missing or unexpected_student or shape_mismatches:
        raise RuntimeError(
            "Lingfeng student checkpoint is incompatible: "
            f"missing={missing}, unexpected={unexpected_student}, "
            f"shape_mismatches={shape_mismatches}"
        )
    model.load_state_dict(selected, strict=True)
    ignored = sorted(set(state) - set(selected))
    return {
        "checkpoint": str(Path(checkpoint)),
        "sha256": sha256_file(checkpoint),
        "loaded_keys": sorted(selected),
        "ignored_keys": ignored,
        "epoch": payload.get("epoch"),
        "best_val_loss": payload.get("best_val_loss"),
        "best_val_dice": payload.get("best_val_dice"),
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
    legacy_inputs = {
        "source": image,
        "input_t1": torch.zeros_like(image),
        "input_t2": torch.zeros_like(image),
        "input_pd": torch.zeros_like(image),
    }
    expected = legacy(legacy_inputs, is_training=False)["seg_logit_prs"]
    actual = student(image)["logits"]
    max_abs_error = float(torch.max(torch.abs(expected - actual)).cpu())
    tolerance = 1e-4 if target.type == "mps" else 1e-5
    if max_abs_error > tolerance:
        raise AssertionError(
            f"Legacy/student logits differ by {max_abs_error}, tolerance={tolerance}"
        )
    return {
        **student_report,
        "device": target.type,
        "max_abs_error": max_abs_error,
        "tolerance": tolerance,
        "equivalent": True,
    }


def write_inspection_report(checkpoint: str | Path, output: str | Path | None = None) -> dict:
    model = LingfengMRAStudent()
    report = load_lingfeng_student_checkpoint(model, checkpoint)
    if output:
        Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

