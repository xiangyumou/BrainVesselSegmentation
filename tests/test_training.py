from __future__ import annotations

from pathlib import Path

import torch
import pytest
import yaml

from bvs.cli import build_parser
from bvs.config import load_config
from bvs.models import ConfigurableLingfengModel, StandardUNet3D
from bvs.training.losses import (
    CombinedSegmentationLoss,
    LegacyMulticlassSquaredDiceLoss,
    MetricContrastiveLoss,
    TemperatureKLLoss,
)
from bvs.training.stages import build_stage
from bvs.training.trainer import (
    _resume,
    _run_epoch,
    find_continue_run,
    validation_improved,
)


def test_train_continue_cli_aliases() -> None:
    parser = build_parser()
    for option in ("-c", "--c", "--continue"):
        args = parser.parse_args(["train", "--config", "config.yaml", option])
        assert args.continue_run is True


def test_continue_selects_newest_identical_run(tmp_path: Path) -> None:
    config = load_config(
        Path(__file__).resolve().parents[1]
        / "configs/train/unet3d_topcow_binary.yaml"
    )
    config["output_root"] = str(tmp_path / "runs")
    experiment_root = Path(config["output_root"]) / config["experiment_name"]
    clean = {key: value for key, value in config.items() if key != "_config_path"}
    expected = None
    for timestamp in ("20260101-000000", "20260102-000000"):
        run = experiment_root / timestamp
        (run / "checkpoints").mkdir(parents=True)
        (run / "resolved_config.yaml").write_text(
            yaml.safe_dump(clean, sort_keys=False), encoding="utf-8"
        )
        expected = run
        (run / "checkpoints/latest.pt").touch()

    assert expected is not None
    run, checkpoint = find_continue_run(config)
    assert run == expected
    assert checkpoint == expected / "checkpoints/latest.pt"

    config["training"]["batch_size"] += 1
    with pytest.raises(FileNotFoundError, match="identical resolved configuration"):
        find_continue_run(config)


def test_single_training_and_validation_step() -> None:
    model = StandardUNet3D(base_channels=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    image = torch.randn(1, 1, 16, 16, 16)
    label = (image[:, 0] > 1).long()
    loss_function = CombinedSegmentationLoss()
    loss = loss_function(model(image)["logits"], label)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        validation_loss = loss_function(model(image)["logits"], label)
    assert torch.isfinite(validation_loss)


def test_legacy_dice_matches_direct_formula() -> None:
    logits = torch.tensor(
        [[[[[0.2, -0.4]]], [[[0.7, 0.1]]]]], dtype=torch.float64
    )
    target = torch.tensor([[[[1, 0]]]])
    probabilities = torch.softmax(logits, dim=1)
    one_hot = torch.nn.functional.one_hot(target, 2).movedim(-1, 1)
    expected = sum(
        1
        - (2 * (probabilities[:, i] * one_hot[:, i]).sum() + 1e-5)
        / (probabilities[:, i].square().sum() + one_hot[:, i].square().sum() + 1e-5)
        for i in range(2)
    ) / 2
    assert torch.allclose(
        LegacyMulticlassSquaredDiceLoss(2)(logits, target), expected
    )


def test_temperature_kl_and_contrastive_match_legacy_formulas() -> None:
    student = torch.randn(3, 2, 2, 2, 2)
    teacher = torch.randn_like(student)
    temperature = 10.0
    expected_kl = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student / temperature, dim=1),
        torch.nn.functional.softmax(teacher / temperature, dim=1),
        reduction="sum",
    ) / student.numel() * temperature**2
    assert torch.allclose(TemperatureKLLoss(temperature)(student, teacher), expected_kl)
    a = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
    b = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
    distance_sq = torch.relu(
        a.square().sum(1, keepdim=True)
        + b.square().sum(1, keepdim=True).T
        - 2 * a @ b.T
    )
    identity = torch.eye(3)
    expected_contrast = torch.mean(
        torch.sum(
            0.5
            * (
                identity * distance_sq
                + (1 - identity)
                * torch.relu(1 - torch.sqrt(distance_sq + 1e-8)).square()
            ),
            dim=-1,
        )
    )
    assert torch.allclose(MetricContrastiveLoss(1)(a, b), expected_contrast)


def test_kd_stage_missing_teacher_fails_loudly(tmp_path) -> None:
    config = {
        "stage": "student_kd",
        "model": {
            "name": "configurable_lingfeng",
            "modalities": ["mra", "cta"],
            "student_modality": "mra",
            "in_channels": {"mra": 1, "cta": 1},
            "num_classes": 2,
            "base_channels": 4,
            "teacher_checkpoint": str(tmp_path / "missing.pt"),
        },
        "loss": {
            "segmentation": {
                "cross_entropy_weight": 1,
                "dice_weight": 1,
                "dice_variant": "legacy_multiclass_squared",
            },
            "logit_distillation": {"temperature": 10, "weight": 0.5},
            "feature_distillation": {"projection_dim": 8, "margin": 1, "weight": 0.5},
        },
    }
    with pytest.raises(FileNotFoundError, match="does not exist"):
        build_stage(config, torch.device("cpu"))


def test_validation_selection_uses_dice_cldice_then_loss() -> None:
    assert validation_improved(
        {"dice": 0.6, "cldice": 0.2, "loss": 3.0}, 0.5, 0.9, 0.1
    )
    assert validation_improved(
        {"dice": 0.5, "cldice": 0.6, "loss": 3.0}, 0.5, 0.5, 0.1
    )
    assert validation_improved(
        {"dice": 0.5, "cldice": 0.5, "loss": 0.09}, 0.5, 0.5, 0.1
    )
    assert not validation_improved(
        {"dice": 0.49, "cldice": 1.0, "loss": 0.0}, 0.5, 0.5, 0.1
    )


def test_transfer_freeze_encoder_excludes_encoder_gradients(tmp_path) -> None:
    source = ConfigurableLingfengModel(
        ["mra"], "mra", {"mra": 1}, 2, base_channels=2
    )
    checkpoint = tmp_path / "student.pt"
    torch.save({"model": source.state_dict()}, checkpoint)
    config = {
        "stage": "supervised",
        "model": {
            "name": "lingfeng_student_transfer",
            "modalities": ["mra"],
            "student_modality": "mra",
            "in_channels": {"mra": 1},
            "num_classes": 2,
            "base_channels": 2,
            "pretrained_checkpoint": str(checkpoint),
            "freeze_encoder": True,
        },
        "training": {"logits_clip": None},
        "loss": {
            "segmentation": {
                "cross_entropy_weight": 1,
                "dice_weight": 1,
                "dice_variant": "foreground",
            }
        },
    }
    runtime = build_stage(config, torch.device("cpu"))
    image = torch.randn(1, 1, 16, 16, 16)
    label = (image[:, 0] > 0).long()
    loss, _ = runtime.loss(
        {"inputs": {"mra": image}, "student_image": image, "label": label}
    )
    loss.backward()
    model = runtime.model
    assert isinstance(model, ConfigurableLingfengModel)
    assert model.encoders["mra"].e1_c1.conv.weight.grad is None
    assert model.student_decoder.seg_logit.conv.weight.grad is not None


class _AccumulationRuntime:
    def __init__(self) -> None:
        self.parameter = torch.nn.Parameter(torch.tensor(0.0))
        self.trainable_parameters = [self.parameter]

    def train(self, mode: bool = True) -> None:
        pass

    def loss(self, batch):
        loss = self.parameter * batch["image"].float().mean()
        return loss, {"segmentation": float(loss.detach())}


def test_gradient_accumulation_scales_partial_final_group() -> None:
    runtime = _AccumulationRuntime()
    optimizer = torch.optim.SGD(runtime.trainable_parameters, lr=1.0)
    loader = [{"image": torch.tensor(float(value))} for value in range(1, 6)]
    _run_epoch(runtime, loader, torch.device("cpu"), optimizer, 4, False, None)
    assert torch.allclose(runtime.parameter, torch.tensor(-7.5))


def _supervised_resume_fixture(tmp_path):
    from bvs.checkpoints import make_unified_checkpoint
    from bvs.config import load_config, model_spec_from_config

    config = load_config(
        Path(__file__).resolve().parents[1]
        / "configs/train/unet3d_topcow_binary.yaml"
    )
    config["model"]["base_channels"] = 1
    runtime = build_stage(config, torch.device("cpu"))
    optimizer = torch.optim.Adam(runtime.trainable_parameters, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    payload = make_unified_checkpoint(
        stage="supervised",
        model=runtime.model,
        model_spec=model_spec_from_config(config),
        resolved_config={},
        epoch=3,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    checkpoint = tmp_path / "resume.pt"
    torch.save(payload, checkpoint)
    config["training"]["resume_checkpoint"] = str(checkpoint)
    return config, runtime, optimizer, scheduler, checkpoint


def test_resume_restores_complete_unified_checkpoint(tmp_path) -> None:
    config, runtime, optimizer, scheduler, _ = _supervised_resume_fixture(tmp_path)
    result = _resume(runtime, optimizer, scheduler, config)
    assert result[0] == 3


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("stage", "teacher", "stage"),
        ("scheduler_state", None, "scheduler_state"),
        ("optimizer_state", {}, "optimizer_state"),
        (
            "student_projection_state",
            {"weight": torch.ones(1, 1)},
            "unexpected student_projection_state",
        ),
    ],
)
def test_resume_rejects_incompatible_checkpoint(
    tmp_path, field, value, message
) -> None:
    config, runtime, optimizer, scheduler, checkpoint = _supervised_resume_fixture(
        tmp_path
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload[field] = value
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match=message):
        _resume(runtime, optimizer, scheduler, config)
