from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ForegroundDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        foreground = (target == 1).to(probabilities.dtype)
        dimensions = tuple(range(1, probabilities.ndim))
        intersection = torch.sum(probabilities * foreground, dim=dimensions)
        denominator = torch.sum(probabilities, dim=dimensions) + torch.sum(
            foreground, dim=dimensions
        )
        score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - score.mean()


class LegacyMulticlassSquaredDiceLoss(nn.Module):
    """Exact active Dice formula from legacy ``loss_function.py``."""

    def __init__(self, num_classes: int, smooth: float = 1e-5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == logits.ndim and target.shape[1] == 1:
            target = target[:, 0]
        one_hot = F.one_hot(target.long(), num_classes=self.num_classes)
        one_hot = one_hot.movedim(-1, 1).to(logits.dtype)
        probabilities = torch.softmax(logits, dim=1)
        loss = logits.new_zeros(())
        for index in range(self.num_classes):
            score = probabilities[:, index]
            truth = one_hot[:, index]
            intersection = torch.sum(score * truth)
            denominator = torch.sum(score.square()) + torch.sum(truth.square())
            loss = loss + 1.0 - (
                2.0 * intersection + self.smooth
            ) / (denominator + self.smooth)
        return loss / self.num_classes


# Preserve the original bvs public name.
DiceLoss = ForegroundDiceLoss


class CombinedSegmentationLoss(nn.Module):
    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        dice_variant: str = "foreground",
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        if dice_variant == "legacy_multiclass_squared":
            self.dice = LegacyMulticlassSquaredDiceLoss(num_classes)
        elif dice_variant in {"foreground", "foreground_linear"}:
            self.dice = ForegroundDiceLoss()
        else:
            raise ValueError(f"Unknown dice variant: {dice_variant}")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == logits.ndim and target.shape[1] == 1:
            target = target[:, 0]
        return self.ce_weight * F.cross_entropy(
            logits, target.long()
        ) + self.dice_weight * self.dice(logits, target)


class TemperatureKLLoss(nn.Module):
    def __init__(self, temperature: float = 10.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        temperature = self.temperature
        divergence = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="sum",
        )
        return divergence / student_logits.numel() * (temperature * temperature)


class MetricContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self, student_features: torch.Tensor, teacher_features: torch.Tensor
    ) -> torch.Tensor:
        batch = student_features.shape[0]
        student_sq = torch.sum(student_features.square(), dim=1, keepdim=True)
        teacher_sq = torch.sum(teacher_features.square(), dim=1, keepdim=True).T
        distances_sq = torch.relu(
            student_sq + teacher_sq - 2.0 * student_features @ teacher_features.T
        )
        positives = torch.eye(
            batch, dtype=student_features.dtype, device=student_features.device
        )
        positive_loss = positives * distances_sq
        distances = torch.sqrt(distances_sq + 1e-8)
        negative_loss = (1.0 - positives) * torch.relu(
            self.margin - distances
        ).square()
        return torch.mean(torch.sum(0.5 * (positive_loss + negative_loss), dim=-1))
