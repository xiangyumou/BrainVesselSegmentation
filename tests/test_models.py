from __future__ import annotations

import torch

from bvs.models import LingfengMRAStudent, StandardUNet3D


def test_standard_unet_shape() -> None:
    model = StandardUNet3D(base_channels=4).eval()
    output = model(torch.randn(1, 1, 16, 16, 16))
    assert output["logits"].shape == (1, 2, 16, 16, 16)
    assert output["probabilities"].shape == output["logits"].shape


def test_lingfeng_student_shape() -> None:
    model = LingfengMRAStudent().eval()
    output = model(torch.randn(1, 1, 16, 16, 16))
    assert output["logits"].shape == (1, 2, 16, 16, 16)
    assert output["features"].shape == (1, 16)


def test_standard_unet_parameter_scale() -> None:
    parameters = sum(parameter.numel() for parameter in StandardUNet3D().parameters())
    assert 22_000_000 < parameters < 23_500_000

