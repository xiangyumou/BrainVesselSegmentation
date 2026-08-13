from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
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


class CombinedSegmentationLoss(nn.Module):
    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * F.cross_entropy(logits, target) + self.dice_weight * self.dice(
            logits, target
        )

