from __future__ import annotations

import csv
import json
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..checkpoints import load_lingfeng_student_checkpoint
from ..config import resolve_data_root
from ..data.dataset import TopCoWPatchDataset
from ..data.topcow import cases_by_id, discover_topcow_cases
from ..devices import seed_everything, select_device
from ..models import LingfengMRAStudent, StandardUNet3D
from .losses import CombinedSegmentationLoss


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    model_config = config["model"]
    name = model_config["name"]
    if name == "standard_unet3d":
        return StandardUNet3D(
            in_channels=int(model_config.get("in_channels", 1)),
            out_channels=int(model_config.get("out_channels", 2)),
            base_channels=int(model_config.get("base_channels", 32)),
        )
    if name == "lingfeng_student_transfer":
        model = LingfengMRAStudent(num_classes=int(model_config.get("out_channels", 2)))
        checkpoint = model_config.get("pretrained_checkpoint")
        if not checkpoint:
            raise ValueError("lingfeng_student_transfer requires pretrained_checkpoint")
        checkpoint = Path(checkpoint)
        if not checkpoint.is_absolute() and config.get("_config_path"):
            checkpoint = Path(config["_config_path"]).parents[2] / checkpoint
        load_lingfeng_student_checkpoint(model, checkpoint)
        return model
    raise ValueError(f"Unknown model name: {name}")


def _load_split(config: dict[str, Any]) -> dict[str, list[str]]:
    split_path = Path(config["data"]["split_file"])
    if not split_path.is_absolute():
        split_path = Path(config["_config_path"]).parents[2] / split_path
    return json.loads(split_path.resolve().read_text(encoding="utf-8"))


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    accumulation_steps: int,
    amp_enabled: bool,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    if training:
        optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    for step, batch in enumerate(loader):
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with torch.set_grad_enabled(training), amp_context:
            loss = loss_function(model(image)["logits"], label)
            scaled_loss = loss / accumulation_steps
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss encountered: {float(loss)}")
        if training:
            scaler.scale(scaled_loss).backward()
            if (step + 1) % accumulation_steps == 0 or step + 1 == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        total += float(loss.detach().cpu())
    if not len(loader):
        raise RuntimeError("DataLoader contains no batches")
    return total / len(loader)


def train_from_config(config: dict[str, Any]) -> Path:
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    policy = select_device(str(config.get("device", "auto")))
    root = resolve_data_root(config)
    split = _load_split(config)
    indexed = cases_by_id(discover_topcow_cases(root))
    missing = sorted((set(split["train"]) | set(split["val"])) - set(indexed))
    if missing:
        raise ValueError(f"Split references missing cases: {missing}")

    data_config = config["data"]
    patch_size = tuple(int(value) for value in data_config.get("patch_size", [48] * 3))
    train_dataset = TopCoWPatchDataset(
        [indexed[case_id] for case_id in split["train"]],
        patch_size,
        float(data_config.get("positive_probability", 0.7)),
        int(data_config.get("samples_per_case", 4)),
    )
    val_dataset = TopCoWPatchDataset(
        [indexed[case_id] for case_id in split["val"]],
        patch_size,
        1.0,
        int(data_config.get("validation_samples_per_case", 1)),
    )
    workers = int(data_config.get("num_workers", 0))
    batch_size = int(config["training"].get("batch_size", 1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    model = build_model(config).to(policy.device)
    training = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-5)),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(training.get("scheduler_step_size", 10)),
        gamma=float(training.get("scheduler_gamma", 0.8)),
    )
    loss_function = CombinedSegmentationLoss(
        float(training.get("ce_weight", 1.0)), float(training.get("dice_weight", 1.0))
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (
        Path(config.get("output_root", "runs"))
        / str(config["experiment_name"])
        / timestamp
    ).resolve()
    checkpoint_dir = run_dir / "checkpoints"
    metrics_dir = run_dir / "metrics"
    checkpoint_dir.mkdir(parents=True)
    metrics_dir.mkdir()
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump({key: value for key, value in config.items() if key != "_config_path"}, sort_keys=False),
        encoding="utf-8",
    )
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    history_path = metrics_dir / "history.csv"
    best_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    max_epochs = int(training.get("epochs", 200))
    accumulation = int(training.get("gradient_accumulation", 4))
    patience = int(training.get("early_stopping_patience", 20))

    for epoch in range(1, max_epochs + 1):
        train_loss = _run_epoch(
            model, train_loader, loss_function, policy.device, optimizer, accumulation, policy.amp_enabled
        )
        val_loss = _run_epoch(
            model, val_loader, loss_function, policy.device, None, accumulation, False
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/validation", val_loss, epoch)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": min(best_loss, val_loss),
            "config": {key: value for key, value in config.items() if key != "_config_path"},
        }
        torch.save(state, checkpoint_dir / "latest.pt")
        if val_loss < best_loss:
            best_loss = val_loss
            stale_epochs = 0
            torch.save(state, checkpoint_dir / "best.pt")
        else:
            stale_epochs += 1
        with history_path.open("w", newline="", encoding="utf-8") as stream:
            writer_csv = csv.DictWriter(stream, fieldnames=list(row))
            writer_csv.writeheader()
            writer_csv.writerows(history)
        if stale_epochs >= patience:
            break
    writer.close()
    summary = {
        "experiment_name": config["experiment_name"],
        "device": policy.device.type,
        "amp_enabled": policy.amp_enabled,
        "epochs_completed": len(history),
        "best_validation_loss": best_loss,
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir
