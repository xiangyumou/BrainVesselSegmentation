from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from bvs.models.lingfeng import (
    N_BASE_FILTERS,
    ConfigurableLingfengModel,
    LingfengMRAStudent,
)


def _require_5d(data: torch.Tensor) -> None:
    if data.ndim != 5:
        raise ValueError(f"Expected [B,C,D,H,W] input, received shape {tuple(data.shape)}")


def _freeze(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


class TeacherNetwork(nn.Module):
    """Two-channel nnU-Net adapter for the Lingfeng MRA+CTA teacher."""

    def __init__(
        self,
        num_input_channels: int = 2,
        num_classes: int = 2,
        base_channels: int = N_BASE_FILTERS,
    ) -> None:
        super().__init__()
        if num_input_channels != 2:
            raise ValueError(
                f"TeacherNetwork requires exactly 2 channels (MRA, CTA), got {num_input_channels}"
            )
        self.model = ConfigurableLingfengModel(
            modalities=("mra", "cta"),
            student_modality="mra",
            in_channels={"mra": 1, "cta": 1},
            num_classes=num_classes,
            base_channels=base_channels,
        )
        # These branches are not part of teacher training.
        _freeze(self.model.student_decoder)
        _freeze(self.model.student_metric)

    @staticmethod
    def modality_mapping(data: torch.Tensor) -> dict[str, torch.Tensor]:
        _require_5d(data)
        if data.shape[1] != 2:
            raise ValueError(
                f"TeacherNetwork requires exactly 2 channels (MRA, CTA), got {data.shape[1]}"
            )
        return {"mra": data[:, 0:1], "cta": data[:, 1:2]}

    def forward_with_features(self, data: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model.forward_teacher(self.modality_mapping(data))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(data)["logits"]


class StudentNetwork(nn.Module):
    """MRA-only Lingfeng student adapter; any additional channels are ignored."""

    def __init__(
        self,
        num_input_channels: int = 1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if num_input_channels < 1:
            raise ValueError("StudentNetwork requires at least one MRA channel")
        self.num_input_channels = int(num_input_channels)
        self.model = LingfengMRAStudent(num_classes=num_classes)
        # LingfengMRAStudent preserves the complete configurable-model API. Only
        # the MRA encoder, student decoder, and student metric are trainable here.
        _freeze(self.model.attention)
        _freeze(self.model.fusion)
        _freeze(self.model.teacher_decoder)
        _freeze(self.model.teacher_metric)

    @staticmethod
    def mra(data: torch.Tensor) -> torch.Tensor:
        _require_5d(data)
        if data.shape[1] < 1:
            raise ValueError("StudentNetwork input does not contain an MRA channel")
        return data[:, 0:1]

    def forward_with_features(self, data: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model.forward_student(self.mra(data))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(data)["logits"]


class KDNetwork(nn.Module):
    """Student inference network plus a frozen teacher used only during KD training."""

    def __init__(
        self,
        num_input_channels: int = 2,
        num_classes: int = 2,
        projection_dim: int = 8,
    ) -> None:
        super().__init__()
        if num_input_channels != 2:
            raise ValueError(
                f"KDNetwork requires exactly 2 channels (MRA, CTA), got {num_input_channels}"
            )
        self.student = StudentNetwork(num_input_channels, num_classes)
        self.teacher = TeacherNetwork(num_input_channels, num_classes)
        self.student_projection = nn.Linear(
            N_BASE_FILTERS, projection_dim, bias=False
        )
        self.teacher_projection = nn.Linear(
            N_BASE_FILTERS, projection_dim, bias=False
        )
        _freeze(self.teacher)
        _freeze(self.teacher_projection)
        self.teacher.eval()
        self.teacher_projection.eval()

    def train(self, mode: bool = True) -> KDNetwork:
        super().train(mode)
        self.teacher.eval()
        self.teacher_projection.eval()
        return self

    def load_teacher_checkpoint(self, checkpoint: Mapping[str, object]) -> None:
        trainer_name = checkpoint.get("trainer_name")
        if trainer_name != "nnUNetTrainerBVSTeacher":
            raise RuntimeError(
                "BVS_TEACHER_CHECKPOINT must come from "
                f"nnUNetTrainerBVSTeacher, got {trainer_name!r}"
            )
        state = checkpoint.get("network_weights")
        if not isinstance(state, Mapping):
            raise RuntimeError("Teacher checkpoint is missing network_weights")
        try:
            self.teacher.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise RuntimeError(f"Teacher checkpoint architecture is incompatible: {error}") from error
        _freeze(self.teacher)
        self.teacher.eval()

    def forward(
        self, data: torch.Tensor, return_distillation: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        student = self.student.forward_with_features(data)
        if not return_distillation:
            return student["logits"]
        with torch.no_grad():
            teacher = self.teacher.forward_with_features(data)
            teacher_projected = F.normalize(
                self.teacher_projection(teacher["metric_feature"]), dim=1
            )
        student_projected = F.normalize(
            self.student_projection(student["metric_feature"]), dim=1
        )
        return {
            "student_logits": student["logits"],
            "teacher_logits": teacher["logits"],
            "student_features": student_projected,
            "teacher_features": teacher_projected,
        }
