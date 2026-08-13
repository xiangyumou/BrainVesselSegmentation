from __future__ import annotations

import csv
import json
import platform
import random
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..checkpoints import make_unified_checkpoint
from ..config import model_spec_from_config, project_path, resolve_data_root
from ..data.dataset import MultimodalPatchDataset, TopCoWPatchDataset, discover_cases
from ..data.topcow import cases_by_id, discover_topcow_cases
from ..devices import seed_everything, select_device
from .stages import StageRuntime, build_stage


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    """Compatibility helper used by callers that only need model construction."""
    return build_stage(config, torch.device("cpu")).model


def _load_split(config: dict[str, Any]) -> dict[str, list[str]]:
    split_value = config["data"].get("split_file")
    if not split_value:
        raise ValueError("data.split_file is required when one root contains all splits")
    return json.loads(project_path(config, split_value).read_text(encoding="utf-8"))


def _multimodal_dataset(
    config: dict[str, Any], split_name: str
) -> MultimodalPatchDataset:
    data = config["data"]
    root = resolve_data_root(config, split_name)
    cases = discover_cases(
        data["adapter"], root, data["modalities"], data["label"]
    )
    if data.get("split_file") and not data.get(f"{split_name}_root"):
        split = _load_split(config)
        key = "internal_test" if split_name == "test" else split_name
        requested = split[key]
        indexed = {case.case_id: case for case in cases}
        missing = sorted(set(requested) - set(indexed))
        if missing:
            raise ValueError(f"Split references missing cases: {missing}")
        cases = [indexed[case_id] for case_id in requested]
    label_spec = data["label"]
    strategy = (
        label_spec.get("strategy", "nonzero_to_foreground")
        if isinstance(label_spec, dict)
        else "nonzero_to_foreground"
    )
    samples = (
        data["validation_samples_per_volume"]
        if split_name == "val"
        else data["samples_per_volume"]
    )
    return MultimodalPatchDataset(
        cases=cases,
        modalities=config["model"]["modalities"],
        student_modality=config["model"]["student_modality"],
        patch_size=tuple(data["patch_size"]),
        samples_per_volume=int(samples),
        crop_or_pad_size=tuple(data["crop_or_pad_size"]),
        normalization=data["normalization"],
        augmentation=data["augmentation"] if split_name == "train" else {"enabled": False},
        label_strategy=strategy,
    )


def _loaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    data = config["data"]
    batch_size = int(config["training"]["batch_size"])
    workers = int(data.get("num_workers", 0))
    if config["model"]["name"] == "standard_unet3d":
        root = resolve_data_root(config)
        split = _load_split(config)
        indexed = cases_by_id(discover_topcow_cases(root))
        missing = sorted((set(split["train"]) | set(split["val"])) - set(indexed))
        if missing:
            raise ValueError(f"Split references missing cases: {missing}")
        train_dataset = TopCoWPatchDataset(
            [indexed[case_id] for case_id in split["train"]],
            tuple(data["patch_size"]),
            float(data.get("positive_probability", 0.7)),
            int(data["samples_per_volume"]),
        )
        val_dataset = TopCoWPatchDataset(
            [indexed[case_id] for case_id in split["val"]],
            tuple(data["patch_size"]),
            1.0,
            int(data["validation_samples_per_volume"]),
        )
    else:
        train_dataset = _multimodal_dataset(config, "train")
        val_dataset = _multimodal_dataset(config, "val")
    return (
        DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers
        ),
        DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers
        ),
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result = dict(batch)
    for key in ("image", "student_image", "label"):
        if key in result and torch.is_tensor(result[key]):
            result[key] = result[key].to(device)
    if "inputs" in result:
        result["inputs"] = {
            name: value.to(device) for name, value in result["inputs"].items()
        }
    return result


def _run_epoch(
    runtime: StageRuntime,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    accumulation_steps: int,
    amp_enabled: bool,
    gradient_clip_norm: float | None,
) -> dict[str, float]:
    training = optimizer is not None
    runtime.train(training)
    totals: dict[str, float] = {"loss": 0.0}
    if training:
        optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    for step, raw_batch in enumerate(loader):
        batch = _move_batch(raw_batch, device)
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with torch.set_grad_enabled(training), amp_context:
            loss, components = runtime.loss(batch)
            scaled_loss = loss / accumulation_steps
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss encountered: {float(loss)}")
        if training:
            scaler.scale(scaled_loss).backward()
            update = (step + 1) % accumulation_steps == 0 or step + 1 == len(loader)
            if update:
                if gradient_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(
                        runtime.trainable_parameters, gradient_clip_norm
                    )
                    if not torch.isfinite(norm):
                        raise FloatingPointError(f"Non-finite gradient norm: {float(norm)}")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        totals["loss"] += float(loss.detach().cpu())
        for name, value in components.items():
            totals[name] = totals.get(name, 0.0) + value
    if not len(loader):
        raise RuntimeError("DataLoader contains no batches")
    return {name: value / len(loader) for name, value in totals.items()}


def _restore_random_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _resume(
    runtime: StageRuntime,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: dict[str, Any],
) -> tuple[int, float, float | None]:
    value = config["training"].get("resume_checkpoint")
    if not value:
        return 0, float("inf"), None
    path = project_path(config, value)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise RuntimeError("Resume checkpoint must use unified schema_version 1")
    runtime.model.load_state_dict(payload["model_state"], strict=True)
    if runtime.student_projection is not None:
        state = payload.get("student_projection_state")
        if state is None:
            raise RuntimeError("Resume checkpoint is missing student projection state")
        runtime.student_projection.load_state_dict(state, strict=True)
    if runtime.teacher_projection is not None:
        state = payload.get("teacher_projection_state")
        if state is None:
            raise RuntimeError("Resume checkpoint is missing teacher projection state")
        runtime.teacher_projection.load_state_dict(state, strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    _restore_random_state(payload.get("random_state"))
    return (
        int(payload["epoch"]),
        float(payload["best_validation_loss"]),
        payload.get("best_validation_dice"),
    )


def train_from_config(config: dict[str, Any]) -> Path:
    seed_everything(int(config["seed"]))
    policy = select_device(str(config["device"]))
    train_loader, val_loader = _loaders(config)
    runtime = build_stage(config, policy.device)
    training = config["training"]
    if training["optimizer"] != "adam":
        raise ValueError("Only optimizer=adam is currently supported")
    optimizer = torch.optim.Adam(
        runtime.trainable_parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler_config = training["scheduler"]
    if scheduler_config["name"] != "step_lr":
        raise ValueError("Only scheduler.name=step_lr is currently supported")
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(scheduler_config["step_size"]),
        gamma=float(scheduler_config["gamma"]),
    )
    start_epoch, best_loss, best_dice = _resume(
        runtime, optimizer, scheduler, config
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (
        project_path(config, config["output_root"])
        / str(config["experiment_name"])
        / timestamp
    )
    checkpoint_dir = run_dir / "checkpoints"
    metrics_dir = run_dir / "metrics"
    checkpoint_dir.mkdir(parents=True)
    metrics_dir.mkdir()
    clean_config = {key: value for key, value in config.items() if key != "_config_path"}
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(clean_config, sort_keys=False), encoding="utf-8"
    )
    (run_dir / "environment.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "device": policy.device.type,
                "amp_enabled": policy.amp_enabled,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    history_path = metrics_dir / "history.csv"
    history: list[dict[str, float | int]] = []
    stale_epochs = 0
    max_epochs = int(training["epochs"])
    accumulation = int(training["gradient_accumulation"])
    gradient_clip = training["gradient_clip_norm"]
    gradient_clip = float(gradient_clip) if gradient_clip is not None else None

    for epoch in range(start_epoch + 1, max_epochs + 1):
        train_metrics = _run_epoch(
            runtime,
            train_loader,
            policy.device,
            optimizer,
            accumulation,
            policy.amp_enabled and training["amp"] != False,
            gradient_clip,
        )
        val_metrics = _run_epoch(
            runtime, val_loader, policy.device, None, accumulation, False, None
        )
        scheduler.step()
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
            for name, value in metrics.items():
                if name != "loss":
                    row[f"{prefix}_{name}"] = value
        history.append(row)
        for name, value in row.items():
            if name != "epoch":
                writer.add_scalar(name.replace("_", "/", 1), value, epoch)
        improved = val_metrics["loss"] < best_loss
        if improved:
            best_loss = val_metrics["loss"]
            stale_epochs = 0
        else:
            stale_epochs += 1
        state = make_unified_checkpoint(
            stage=config["stage"],
            model=runtime.model,
            model_spec=(
                model_spec_from_config(config)
                if config["model"]["name"] != "standard_unet3d"
                else {
                    "name": "standard_unet3d",
                    "in_channels": int(config["model"].get("in_channels", 1)),
                    "num_classes": int(config["model"].get("num_classes", 2)),
                    "base_channels": int(config["model"].get("base_channels", 32)),
                }
            ),
            resolved_config=clean_config,
            epoch=epoch,
            best_validation_loss=best_loss,
            best_validation_dice=best_dice,
            student_projection=runtime.student_projection,
            teacher_projection=runtime.teacher_projection,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        torch.save(state, checkpoint_dir / "latest.pt")
        if improved:
            torch.save(state, checkpoint_dir / "best.pt")
        with history_path.open("w", newline="", encoding="utf-8") as stream:
            writer_csv = csv.DictWriter(stream, fieldnames=list(row))
            writer_csv.writeheader()
            writer_csv.writerows(history)
        if stale_epochs >= int(training["early_stopping_patience"]):
            break
    writer.close()
    summary = {
        "experiment_name": config["experiment_name"],
        "stage": config["stage"],
        "device": policy.device.type,
        "amp_enabled": policy.amp_enabled,
        "epochs_completed": len(history),
        "last_epoch": history[-1]["epoch"] if history else start_epoch,
        "best_validation_loss": best_loss,
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
    }
    (metrics_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return run_dir
