from __future__ import annotations

import torch

from bvs.models import StandardUNet3D
from bvs.training.losses import CombinedSegmentationLoss


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

