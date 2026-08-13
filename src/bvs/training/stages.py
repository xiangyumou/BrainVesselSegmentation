from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ..checkpoints import build_lingfeng_from_spec, load_unified_checkpoint
from ..config import model_spec_from_config, project_path
from ..models import ConfigurableLingfengModel, StandardUNet3D
from .losses import CombinedSegmentationLoss, MetricContrastiveLoss, TemperatureKLLoss


@dataclass
class StageRuntime:
    stage: str
    model: nn.Module
    segmentation_loss: CombinedSegmentationLoss
    trainable_parameters: list[nn.Parameter]
    teacher_model: ConfigurableLingfengModel | None = None
    student_projection: nn.Module | None = None
    teacher_projection: nn.Module | None = None
    kd_loss: TemperatureKLLoss | None = None
    feature_loss: MetricContrastiveLoss | None = None
    kd_weight: float = 0.0
    feature_weight: float = 0.0
    logits_clip: float | None = None

    def train(self, mode: bool = True) -> None:
        self.model.train(mode)
        if self.student_projection is not None:
            self.student_projection.train(mode)
        if self.teacher_model is not None:
            self.teacher_model.eval()
        if self.teacher_projection is not None:
            self.teacher_projection.eval()

    def loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        label = batch["label"]
        if self.stage == "teacher":
            assert isinstance(self.model, ConfigurableLingfengModel)
            output = self.model.forward_teacher(batch["inputs"])
            logits = output["logits"]
            if self.logits_clip is not None:
                logits = torch.clamp(logits, -self.logits_clip, self.logits_clip)
            segmentation = self.segmentation_loss(logits, label)
            return segmentation, {"segmentation": float(segmentation.detach())}
        if self.stage == "student_kd":
            assert isinstance(self.model, ConfigurableLingfengModel)
            assert self.teacher_model is not None
            assert self.student_projection is not None
            assert self.teacher_projection is not None
            assert self.kd_loss is not None and self.feature_loss is not None
            student = self.model.forward_student(batch["inputs"])
            with torch.no_grad():
                teacher = self.teacher_model.forward_teacher(batch["inputs"])
                teacher_projected = F.normalize(
                    self.teacher_projection(teacher["metric_feature"]), dim=1
                )
            student_projected = F.normalize(
                self.student_projection(student["metric_feature"]), dim=1
            )
            segmentation = self.segmentation_loss(student["logits"], label)
            kd = self.kd_loss(student["logits"], teacher["logits"])
            feature = self.feature_loss(student_projected, teacher_projected)
            total = segmentation + self.kd_weight * kd + self.feature_weight * feature
            return total, {
                "segmentation": float(segmentation.detach()),
                "logit_distillation": float(kd.detach()),
                "feature_distillation": float(feature.detach()),
            }
        if isinstance(self.model, ConfigurableLingfengModel):
            logits = self.model.forward_student(batch["inputs"])["logits"]
        else:
            image = batch.get("image", batch.get("student_image"))
            logits = self.model(image)["logits"]
        segmentation = self.segmentation_loss(logits, label)
        return segmentation, {"segmentation": float(segmentation.detach())}


def _segmentation_loss(config: dict[str, Any]) -> CombinedSegmentationLoss:
    settings = config["loss"]["segmentation"]
    num_classes = int(
        config["model"].get(
            "num_classes", config["model"].get("out_channels", 2)
        )
    )
    return CombinedSegmentationLoss(
        ce_weight=float(settings["cross_entropy_weight"]),
        dice_weight=float(settings["dice_weight"]),
        dice_variant=str(settings["dice_variant"]),
        num_classes=num_classes,
    )


def _lingfeng_model(config: dict[str, Any]) -> ConfigurableLingfengModel:
    return build_lingfeng_from_spec(model_spec_from_config(config))


def _student_parameters(model: ConfigurableLingfengModel) -> list[nn.Parameter]:
    return [
        *model.encoders[model.student_modality].parameters(),
        *model.student_decoder.parameters(),
        *model.student_metric.parameters(),
    ]


def _load_teacher(
    config: dict[str, Any], device: torch.device
) -> tuple[ConfigurableLingfengModel, dict[str, Any]]:
    checkpoint_value = config["model"].get("teacher_checkpoint")
    if not checkpoint_value:
        raise FileNotFoundError("student_kd requires model.teacher_checkpoint")
    checkpoint = project_path(config, checkpoint_value)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Teacher checkpoint does not exist: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError(
            "Teacher checkpoint must use unified schema_version 1; convert it first"
        )
    teacher_spec = payload.get("model_spec")
    requested_spec = model_spec_from_config(config)
    if teacher_spec != requested_spec:
        raise RuntimeError(
            f"Teacher model spec is incompatible: checkpoint={teacher_spec}, "
            f"requested={requested_spec}"
        )
    teacher = build_lingfeng_from_spec(teacher_spec)
    teacher.load_state_dict(payload["model_state"], strict=True)
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher, payload


def build_stage(config: dict[str, Any], device: torch.device) -> StageRuntime:
    stage = config["stage"]
    model_config = config["model"]
    segmentation = _segmentation_loss(config)
    if model_config["name"] == "standard_unet3d":
        if stage != "supervised":
            raise ValueError("standard_unet3d only supports supervised stage")
        model = StandardUNet3D(
            in_channels=int(model_config.get("in_channels", 1)),
            out_channels=int(model_config.get("num_classes", model_config.get("out_channels", 2))),
            base_channels=int(model_config.get("base_channels", 32)),
        ).to(device)
        return StageRuntime(stage, model, segmentation, list(model.parameters()))

    model = _lingfeng_model(config).to(device)
    if model_config["name"] == "lingfeng_student_transfer":
        checkpoint_value = model_config.get("pretrained_checkpoint")
        if not checkpoint_value:
            raise ValueError("lingfeng_student_transfer requires pretrained_checkpoint")
        from ..checkpoints import load_lingfeng_student_checkpoint

        load_lingfeng_student_checkpoint(model, project_path(config, checkpoint_value))
    if stage == "teacher":
        parameters = [
            *model.encoders.parameters(),
            *model.attention.parameters(),
            *model.fusion.parameters(),
            *model.teacher_decoder.parameters(),
            *model.teacher_metric.parameters(),
        ]
        clip = config["training"].get("logits_clip")
        return StageRuntime(
            stage,
            model,
            segmentation,
            parameters,
            logits_clip=float(clip) if clip is not None else None,
        )
    if stage == "student_kd":
        teacher, teacher_payload = _load_teacher(config, device)
        feature = config["loss"]["feature_distillation"]
        logit = config["loss"]["logit_distillation"]
        projection_dim = int(feature["projection_dim"])
        student_projection = nn.Linear(
            model.base_channels, projection_dim, bias=False
        ).to(device)
        teacher_projection = nn.Linear(
            teacher.base_channels, projection_dim, bias=False
        ).to(device)
        saved_teacher_projection = teacher_payload.get("teacher_projection_state")
        if saved_teacher_projection is not None:
            teacher_projection.load_state_dict(saved_teacher_projection, strict=True)
        for parameter in teacher_projection.parameters():
            parameter.requires_grad = False
        parameters = _student_parameters(model) + list(student_projection.parameters())
        return StageRuntime(
            stage,
            model,
            segmentation,
            parameters,
            teacher_model=teacher,
            student_projection=student_projection,
            teacher_projection=teacher_projection,
            kd_loss=TemperatureKLLoss(float(logit["temperature"])),
            feature_loss=MetricContrastiveLoss(float(feature["margin"])),
            kd_weight=float(logit["weight"]),
            feature_weight=float(feature["weight"]),
        )
    if stage == "supervised":
        parameters = _student_parameters(model)
        return StageRuntime(stage, model, segmentation, parameters)
    raise ValueError(f"Unknown stage: {stage}")
