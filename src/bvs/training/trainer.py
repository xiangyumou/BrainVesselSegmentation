from __future__ import annotations

import csv
import copy
import json
import platform
import random
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import nibabel as nib
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..checkpoints import make_unified_checkpoint
from ..config import model_spec_from_config, project_path, resolve_data_root
from ..data.dataset import MultimodalPatchDataset, TopCoWPatchDataset, discover_cases
from ..data.topcow import TopCoWCase, cases_by_id, discover_topcow_cases
from ..devices import seed_everything, select_device
from ..evaluation import segmentation_metrics
from ..inference import InferenceCase, load_inference_case, sliding_window_inference
from .stages import StageRuntime, build_stage


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def _clean_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in config.items()
        if key != "_config_path"
    }


def _config_for_continue(config: dict[str, Any]) -> dict[str, Any]:
    comparable = _clean_config(config)
    comparable.setdefault("training", {})["resume_checkpoint"] = None
    return comparable


def find_continue_run(config: dict[str, Any]) -> tuple[Path, Path]:
    """Return the newest run whose saved configuration exactly matches config."""
    if config["training"].get("resume_checkpoint"):
        raise ValueError(
            "--continue cannot be combined with training.resume_checkpoint"
        )
    experiment_root = (
        project_path(config, config["output_root"]) / str(config["experiment_name"])
    )
    requested = _config_for_continue(config)
    if experiment_root.is_dir():
        for run_dir in sorted(experiment_root.iterdir(), reverse=True):
            saved_config = run_dir / "resolved_config.yaml"
            latest = run_dir / "checkpoints/latest.pt"
            if (
                not run_dir.is_dir()
                or not saved_config.is_file()
                or not latest.is_file()
            ):
                continue
            saved = yaml.safe_load(saved_config.read_text(encoding="utf-8"))
            if isinstance(saved, dict) and _config_for_continue(saved) == requested:
                return run_dir, latest
    raise FileNotFoundError(
        "No previous run with an identical resolved configuration and latest.pt "
        f"was found under {experiment_root}"
    )


def _load_history(path: Path) -> list[dict[str, float | int]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    history: list[dict[str, float | int]] = []
    for raw in rows:
        history.append(
            {
                key: int(value) if key in {"epoch", "is_best"} else float(value)
                for key, value in raw.items()
                if value is not None and value != ""
            }
        )
    return history


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    """Compatibility helper used by callers that only need model construction."""
    return build_stage(config, torch.device("cpu")).model


def _load_split(config: dict[str, Any]) -> dict[str, list[str]]:
    split_value = config["data"].get("split_file")
    if not split_value:
        raise ValueError("data.split_file is required when one root contains all splits")
    return json.loads(project_path(config, split_value).read_text(encoding="utf-8"))


def _multimodal_cases(config: dict[str, Any], split_name: str) -> list[Any]:
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
    return cases


def _multimodal_dataset(
    config: dict[str, Any], split_name: str
) -> MultimodalPatchDataset:
    data = config["data"]
    cases = _multimodal_cases(config, split_name)
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
        crop_or_pad_size=(
            tuple(data["crop_or_pad_size"])
            if data.get("crop_or_pad_size") is not None
            else None
        ),
        normalization=data["normalization"],
        augmentation=data["augmentation"] if split_name == "train" else {"enabled": False},
        label_strategy=strategy,
        positive_probability=float(data["positive_probability"]),
        seed=int(config["seed"]),
        cache_max_cases=int(data["cache_max_cases"]),
    )


def _require_complete_registration_qc(config: dict[str, Any]) -> None:
    if config["stage"] not in {"teacher", "student_kd"}:
        return
    cta_spec = config["data"].get("modalities", {}).get("cta")
    if not isinstance(cta_spec, dict):
        return
    directory_value = cta_spec.get("directory")
    if not directory_value or "cta_registered_to_mra" not in str(directory_value):
        return
    cta_directory = Path(str(directory_value)).expanduser()
    if not cta_directory.is_absolute():
        cta_directory = resolve_data_root(config) / cta_directory
    summary_path = cta_directory.resolve().parent / "qc" / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(
            f"Teacher/KD training requires registration QC summary: {summary_path}"
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Registration QC summary is unreadable: {summary_path}") from error
    split = _load_split(config)
    expected_ids = {
        str(case_id)
        for key in ("train", "val", "internal_test")
        for case_id in split.get(key, [])
    }
    successful_ids = {
        str(record.get("case_id"))
        for record in summary.get("cases", [])
        if record.get("status") == "success"
    }
    failed = int(summary.get("failed", -1))
    if failed != 0 or successful_ids != expected_ids:
        missing = sorted(expected_ids - successful_ids)
        unexpected = sorted(successful_ids - expected_ids)
        missing_preview = missing[:10] + (
            [f"... ({len(missing)} total)"] if len(missing) > 10 else []
        )
        unexpected_preview = unexpected[:10] + (
            [f"... ({len(unexpected)} total)"] if len(unexpected) > 10 else []
        )
        raise RuntimeError(
            "Teacher/KD training is blocked until registration QC passes for the "
            f"complete split: expected={len(expected_ids)}, successful={len(successful_ids)}, "
            f"failed={failed}, missing={missing_preview}, unexpected={unexpected_preview}"
        )


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _loaders(config: dict[str, Any]) -> tuple[DataLoader, list[Any]]:
    data = config["data"]
    _require_complete_registration_qc(config)
    batch_size = int(config["training"]["batch_size"])
    workers = int(data.get("num_workers", 0))
    if config["model"]["name"] == "standard_unet3d":
        root = resolve_data_root(config)
        split = _load_split(config)
        if data["adapter"] == "pattern_directory":
            generic_cases = discover_cases(
                data["adapter"], root, data["modalities"], data["label"]
            )
            modality = next(iter(data["modalities"]))
            indexed = cases_by_id(
                [
                    TopCoWCase(
                        case.case_id, case.modalities[modality], case.label
                    )
                    for case in generic_cases
                ]
            )
        else:
            indexed = cases_by_id(discover_topcow_cases(root))
        missing = sorted((set(split["train"]) | set(split["val"])) - set(indexed))
        if missing:
            raise ValueError(f"Split references missing cases: {missing}")
        train_dataset = TopCoWPatchDataset(
            [indexed[case_id] for case_id in split["train"]],
            tuple(data["patch_size"]),
            float(data.get("positive_probability", 0.7)),
            int(data["samples_per_volume"]),
            str(data["normalization"]),
            int(config["seed"]),
            int(data["cache_max_cases"]),
        )
        val_cases = [indexed[case_id] for case_id in split["val"]]
    else:
        train_dataset = _multimodal_dataset(config, "train")
        val_cases = _multimodal_cases(config, "val")
    generator = torch.Generator().manual_seed(int(config["seed"]))
    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            generator=generator,
            worker_init_fn=_seed_worker,
        ),
        val_cases,
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
    show_progress: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    runtime.train(training)
    totals: dict[str, float] = {"loss": 0.0}
    if training:
        optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    progress_interval = max(1, len(loader) // 20)
    for step, raw_batch in enumerate(loader):
        batch = _move_batch(raw_batch, device)
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with torch.set_grad_enabled(training), amp_context:
            loss, components = runtime.loss(batch)
            group_start = (step // accumulation_steps) * accumulation_steps
            group_size = min(accumulation_steps, len(loader) - group_start)
            scaled_loss = loss / group_size
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
        completed = step + 1
        if show_progress and (
            completed == 1
            or completed == len(loader)
            or completed % progress_interval == 0
        ):
            progress = (
                f"Training batch {completed}/{len(loader)} "
                f"({100.0 * completed / len(loader):.0f}%), "
                f"epoch running mean loss={totals['loss'] / completed:.6f}"
            )
            if optimizer is not None:
                progress += (
                    f", learning rate={optimizer.param_groups[0]['lr']:.8g}"
                )
            _log(progress)
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


def _validation_case(
    case: Any, config: dict[str, Any]
) -> tuple[InferenceCase, Path]:
    if config["model"]["name"] == "standard_unet3d":
        return (
            InferenceCase(case.case_id, {"image": case.image}, case.image),
            case.label,
        )
    return (
        InferenceCase(case.case_id, dict(case.modalities), next(iter(case.modalities.values()))),
        case.label,
    )


def _validation_loss_from_probabilities(
    probabilities: torch.Tensor,
    label: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    settings = config["loss"]["segmentation"]
    ce = torch.nn.functional.nll_loss(
        probabilities.clamp_min(1e-7).log(), label.long()
    )
    variant = str(settings["dice_variant"])
    smooth = 1e-5
    if variant in {"foreground", "foreground_linear"}:
        score = probabilities[:, 1]
        truth = (label == 1).to(score.dtype)
        dimensions = tuple(range(1, score.ndim))
        intersection = torch.sum(score * truth, dim=dimensions)
        denominator = torch.sum(score, dim=dimensions) + torch.sum(
            truth, dim=dimensions
        )
        dice = 1.0 - torch.mean(
            (2.0 * intersection + smooth) / (denominator + smooth)
        )
    elif variant == "legacy_multiclass_squared":
        one_hot = torch.nn.functional.one_hot(
            label.long(), num_classes=probabilities.shape[1]
        ).movedim(-1, 1).to(probabilities.dtype)
        dice = probabilities.new_zeros(())
        for index in range(probabilities.shape[1]):
            score = probabilities[:, index]
            truth = one_hot[:, index]
            intersection = torch.sum(score * truth)
            denominator = torch.sum(score.square()) + torch.sum(truth.square())
            dice = dice + 1.0 - (
                2.0 * intersection + smooth
            ) / (denominator + smooth)
        dice = dice / probabilities.shape[1]
    else:
        raise ValueError(f"Unknown dice variant: {variant}")
    return (
        float(settings["cross_entropy_weight"]) * ce
        + float(settings["dice_weight"]) * dice
    )


@torch.inference_mode()
def validate_full_volumes(
    runtime: StageRuntime,
    cases: list[Any],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    if not cases:
        raise RuntimeError("Validation set contains no cases")
    was_training = runtime.model.training
    runtime.train(False)
    branch = "teacher" if config["stage"] == "teacher" else "student"
    metric_rows: list[dict[str, float]] = []
    losses: list[float] = []
    try:
        for case_index, raw_case in enumerate(cases, start=1):
            case, label_path = _validation_case(raw_case, config)
            _log(
                f"Validating case {case_index}/{len(cases)}: {case.case_id}"
            )
            tensors, _ = load_inference_case(
                case, str(config["data"]["normalization"])
            )
            inference_input: torch.Tensor | dict[str, torch.Tensor] = (
                tensors
                if branch == "teacher"
                else tensors[
                    str(config["model"].get("student_modality", "image"))
                ]
            )
            probabilities = sliding_window_inference(
                inference_input,
                runtime.model,
                tuple(config["inference"]["window_size"]),
                tuple(config["inference"]["overlap"]),
                device,
                branch,
                str(config["inference"]["compatibility_mode"]),
            ).to(device)
            label = (
                torch.from_numpy(
                    (nib.load(str(label_path)).get_fdata() > 0).astype(np.int64)
                )
                .unsqueeze(0)
                .to(device)
            )
            loss = _validation_loss_from_probabilities(
                probabilities, label, config
            )
            losses.append(float(loss.cpu()))
            prediction = torch.argmax(probabilities, dim=1).squeeze(0).cpu().numpy()
            metric_rows.append(
                segmentation_metrics(
                    prediction, label.squeeze(0).cpu().numpy()
                )
            )
            _log(
                f"Validated case {case.case_id}: "
                f"Dice={metric_rows[-1]['dice']:.6f}, "
                f"clDice={metric_rows[-1]['cldice']:.6f}"
            )
    finally:
        runtime.train(was_training)
    return {
        "loss": float(np.mean(losses)),
        **{
            name: float(np.mean([row[name] for row in metric_rows]))
            for name in ("dice", "iou", "precision", "recall", "cldice")
        },
    }


def validation_improved(
    metrics: dict[str, float],
    best_dice: float | None,
    best_cldice: float | None,
    best_loss: float | None,
    tolerance: float = 1e-6,
) -> bool:
    if best_dice is None:
        return True
    dice_delta = metrics["dice"] - best_dice
    if dice_delta > tolerance:
        return True
    if abs(dice_delta) > tolerance:
        return False
    if best_cldice is None:
        return True
    cldice_delta = metrics["cldice"] - best_cldice
    if cldice_delta > tolerance:
        return True
    if abs(cldice_delta) > tolerance:
        return False
    return best_loss is None or metrics["loss"] < best_loss - tolerance


def _training_epochs(start_epoch: int, configured_epochs: int) -> range:
    return range(start_epoch + 1, configured_epochs + 1)


def _early_stopping_triggered(
    stale_epochs: int, patience: int | None
) -> bool:
    return patience is not None and stale_epochs >= patience


def _resume(
    runtime: StageRuntime,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: dict[str, Any],
) -> tuple[int, float | None, float | None, float | None, int | None, int]:
    value = config["training"].get("resume_checkpoint")
    if not value:
        return 0, None, None, None, None, 0
    path = project_path(config, value)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    prefix = "Resume checkpoint is incompatible:"
    if not isinstance(payload, dict):
        raise RuntimeError(f"{prefix} checkpoint must be a mapping")
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"{prefix} schema_version must be 1")
    if payload.get("stage") != config["stage"]:
        raise RuntimeError(
            f"{prefix} stage checkpoint={payload.get('stage')}, "
            f"config={config['stage']}"
        )
    requested_spec = model_spec_from_config(config)
    if payload.get("model_spec") != requested_spec:
        raise RuntimeError(
            f"{prefix} model_spec checkpoint={payload.get('model_spec')}, "
            f"config={requested_spec}"
        )
    for key in ("model_state", "optimizer_state", "scheduler_state"):
        state = payload.get(key)
        if not isinstance(state, dict) or not state:
            raise RuntimeError(f"{prefix} missing required {key}")
    if payload.get("epoch") is None:
        raise RuntimeError(f"{prefix} missing required epoch")
    projection_runtimes = {
        "student_projection_state": runtime.student_projection,
        "teacher_projection_state": runtime.teacher_projection,
    }
    for key, projection in projection_runtimes.items():
        state = payload.get(key)
        if projection is not None and state is None:
            raise RuntimeError(f"{prefix} missing required {key}")
        if projection is None and state is not None:
            raise RuntimeError(f"{prefix} unexpected {key} for stage {runtime.stage}")
    try:
        runtime.model.load_state_dict(payload["model_state"], strict=True)
        if runtime.student_projection is not None:
            runtime.student_projection.load_state_dict(
                payload["student_projection_state"], strict=True
            )
        if runtime.teacher_projection is not None:
            runtime.teacher_projection.load_state_dict(
                payload["teacher_projection_state"], strict=True
            )
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError(f"{prefix} state restoration failed: {error}") from error
    _restore_random_state(payload.get("random_state"))
    return (
        int(payload["epoch"]),
        (
            float(payload["best_validation_loss"])
            if payload.get("best_validation_loss") is not None
            else None
        ),
        payload.get("best_validation_dice"),
        payload.get("best_validation_cldice"),
        payload.get("best_epoch"),
        int(payload.get("stale_epochs", 0)),
    )


def train_from_config(
    config: dict[str, Any], *, continue_run: bool = False
) -> Path:
    _log(
        f"Starting experiment '{config['experiment_name']}' "
        f"(stage={config['stage']}, continue={continue_run})"
    )
    clean_config = _clean_config(config)
    continued_run: Path | None = None
    if continue_run:
        _log("Searching for the newest run with an identical resolved configuration")
        try:
            continued_run, resume_checkpoint = find_continue_run(config)
        except FileNotFoundError:
            _log(
                "No compatible latest.pt was found; starting a new run from "
                "the configured initialization"
            )
        else:
            _log(f"Found continuation checkpoint: {resume_checkpoint}")
            config = copy.deepcopy(config)
            config["training"]["resume_checkpoint"] = str(resume_checkpoint)
    seed_everything(int(config["seed"]))
    policy = select_device(str(config["device"]))
    device_description = policy.device.type
    if policy.device.type == "cuda":
        device_description = f"cuda ({torch.cuda.get_device_name(policy.device)})"
    _log(
        f"Environment ready: torch={torch.__version__}, device={device_description}, "
        f"AMP available={policy.amp_enabled}, seed={config['seed']}"
    )
    _log(f"Discovering data under {resolve_data_root(config)}")
    train_loader, val_cases = _loaders(config)
    train_cases = len(getattr(train_loader.dataset, "cases", []))
    _log(
        f"Data ready: train cases={train_cases}, validation cases={len(val_cases)}, "
        f"batches per epoch={len(train_loader)}"
    )
    model_config = config["model"]
    if model_config.get("pretrained_checkpoint"):
        _log(
            "Loading pretrained checkpoint: "
            f"{project_path(config, model_config['pretrained_checkpoint'])}"
        )
    _log(f"Building model '{model_config['name']}'")
    runtime = build_stage(config, policy.device)
    trainable_parameters = sum(
        parameter.numel() for parameter in runtime.trainable_parameters
    )
    _log(f"Model ready: trainable parameters={trainable_parameters:,}")
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
    start_epoch, best_loss, best_dice, best_cldice, best_epoch, stale_epochs = _resume(
        runtime, optimizer, scheduler, config
    )
    if start_epoch:
        _log(
            f"Resume state restored at epoch {start_epoch}: "
            f"best epoch={best_epoch}, best Dice={best_dice}"
        )
    resume_value = training.get("resume_checkpoint")
    resume_path = (
        str(project_path(config, resume_value)) if resume_value else None
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = continued_run or (
        project_path(config, config["output_root"])
        / str(config["experiment_name"])
        / timestamp
    )
    resuming_in_place = continued_run is not None
    checkpoint_dir = run_dir / "checkpoints"
    metrics_dir = run_dir / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=resuming_in_place)
    metrics_dir.mkdir(exist_ok=resuming_in_place)
    _log(f"Run directory: {run_dir}")
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
                "resume_checkpoint": resume_path,
                "resume_from_epoch": start_epoch,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    history_path = metrics_dir / "history.csv"
    history = _load_history(history_path) if resuming_in_place else []
    max_epochs = int(training["epochs"])
    accumulation = int(training["gradient_accumulation"])
    gradient_clip = training["gradient_clip_norm"]
    gradient_clip = float(gradient_clip) if gradient_clip is not None else None
    amp_enabled = policy.amp_enabled and str(training["amp"]).lower() not in {
        "false",
        "0",
    }
    _log(
        f"Training plan: epochs={start_epoch + 1}-{max_epochs}, "
        f"batch size={training['batch_size']}, accumulation={accumulation}, "
        f"AMP={amp_enabled}, early stopping patience="
        f"{training['early_stopping_patience']}"
    )

    for epoch in _training_epochs(start_epoch, max_epochs):
        current_learning_rate = optimizer.param_groups[0]["lr"]
        _log(
            f"Epoch {epoch}/{max_epochs} started; "
            f"learning rate={current_learning_rate:.8g}"
        )
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        if train_loader.generator is not None:
            train_loader.generator.manual_seed(int(config["seed"]) + epoch)
        train_metrics = _run_epoch(
            runtime,
            train_loader,
            policy.device,
            optimizer,
            accumulation,
            amp_enabled,
            gradient_clip,
            show_progress=True,
        )
        _log(
            f"Epoch {epoch} training complete: "
            f"loss={train_metrics['loss']:.6f}"
        )
        _log(f"Epoch {epoch} full-volume validation started")
        val_metrics = validate_full_volumes(
            runtime, val_cases, config, policy.device
        )
        _log(
            f"Epoch {epoch} validation complete: "
            f"loss={val_metrics['loss']:.6f}, Dice={val_metrics['dice']:.6f}, "
            f"clDice={val_metrics['cldice']:.6f}, IoU={val_metrics['iou']:.6f}"
        )
        scheduler.step()
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_segmentation": train_metrics["segmentation"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_cldice": val_metrics["cldice"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if config["stage"] == "student_kd":
            row["train_logit_distillation"] = train_metrics[
                "logit_distillation"
            ]
            row["train_feature_distillation"] = train_metrics[
                "feature_distillation"
            ]
        improved = validation_improved(
            val_metrics, best_dice, best_cldice, best_loss
        )
        row["is_best"] = int(improved)
        history.append(row)
        for name, value in row.items():
            if name != "epoch":
                writer.add_scalar(name.replace("_", "/", 1), value, epoch)
        if improved:
            best_loss = val_metrics["loss"]
            best_dice = val_metrics["dice"]
            best_cldice = val_metrics["cldice"]
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        state = make_unified_checkpoint(
            stage=config["stage"],
            model=runtime.model,
            model_spec=model_spec_from_config(config),
            resolved_config=clean_config,
            epoch=epoch,
            best_validation_loss=best_loss,
            best_validation_dice=best_dice,
            best_validation_cldice=best_cldice,
            best_epoch=best_epoch,
            stale_epochs=stale_epochs,
            student_projection=runtime.student_projection,
            teacher_projection=runtime.teacher_projection,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        torch.save(state, checkpoint_dir / "latest.pt")
        _log(f"Saved latest checkpoint: {checkpoint_dir / 'latest.pt'}")
        if improved:
            torch.save(state, checkpoint_dir / "best.pt")
            _log(
                f"New best model at epoch {epoch}; saved: "
                f"{checkpoint_dir / 'best.pt'}"
            )
        with history_path.open("w", newline="", encoding="utf-8") as stream:
            writer_csv = csv.DictWriter(stream, fieldnames=list(row))
            writer_csv.writeheader()
            writer_csv.writerows(history)
        if _early_stopping_triggered(
            stale_epochs, training["early_stopping_patience"]
        ):
            _log(
                f"Early stopping at epoch {epoch}: no improvement for "
                f"{stale_epochs} epochs"
            )
            break
    writer.close()
    summary = {
        "experiment_name": config["experiment_name"],
        "stage": config["stage"],
        "device": policy.device.type,
        "amp_enabled": policy.amp_enabled,
        "start_epoch": start_epoch,
        "epochs_completed": len(history),
        "last_epoch": history[-1]["epoch"] if history else start_epoch,
        "best_validation_loss": best_loss,
        "best_validation_dice": best_dice,
        "best_validation_cldice": best_cldice,
        "best_epoch": best_epoch,
        "selection_metric": "mean_dice",
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
    }
    (metrics_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _log(
        f"Training finished: last epoch={summary['last_epoch']}, "
        f"best epoch={best_epoch}, best Dice={best_dice}, run={run_dir}"
    )
    return run_dir
