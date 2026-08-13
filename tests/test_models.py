from __future__ import annotations

import torch
import pytest

from bvs.models import (
    ConfigurableLingfengModel,
    LingfengLegacyModel,
    LingfengMRAStudent,
    StandardUNet3D,
    StudentInferenceView,
)


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


def test_four_modality_parameter_shapes_map_to_legacy() -> None:
    legacy = LingfengLegacyModel()
    configurable = ConfigurableLingfengModel(
        ["mra", "t1", "t2", "pd"],
        "mra",
        {"mra": 1, "t1": 1, "t2": 1, "pd": 1},
        2,
    )
    assert len(legacy.state_dict()) == len(configurable.state_dict()) == 86
    assert configurable.attention["s1"].conv.weight.shape == (4, 64, 1, 1, 1)
    assert configurable.fusion["s4"].conv.weight.shape == (128, 512, 1, 1, 1)


def test_dynamic_two_modality_teacher_and_student_backward() -> None:
    model = ConfigurableLingfengModel(
        ["mra", "cta"], "mra", {"mra": 1, "cta": 1}, 2, base_channels=4
    )
    inputs = {
        "mra": torch.randn(1, 1, 16, 16, 16),
        "cta": torch.randn(1, 1, 16, 16, 16),
    }
    teacher = model.forward_teacher(inputs)
    student = model.forward_student({"mra": inputs["mra"]})
    (teacher["logits"].mean() + student["logits"].mean()).backward()
    assert len(model.encoders) == 2
    assert model.attention["s1"].conv.weight.shape == (2, 8, 1, 1, 1)
    assert model.fusion["s1"].conv.weight.shape == (4, 8, 1, 1, 1)
    assert model.encoders["cta"].e1_c1.conv.weight.grad is not None


def test_student_view_shares_parameters_and_does_not_require_teacher_modalities() -> None:
    model = ConfigurableLingfengModel(
        ["mra", "cta"], "mra", {"mra": 1, "cta": 1}, 2, base_channels=4
    )
    view = StudentInferenceView(model)
    output = view(torch.randn(1, 1, 16, 16, 16))
    assert output["logits"].shape == (1, 2, 16, 16, 16)
    assert next(view.parameters()) is next(model.parameters())


def test_invalid_branch_and_missing_teacher_modality_fail() -> None:
    model = ConfigurableLingfengModel(
        ["mra", "cta"], "mra", {"mra": 1, "cta": 1}, 2, base_channels=4
    )
    with pytest.raises(KeyError, match="cta"):
        model.forward_teacher({"mra": torch.randn(1, 1, 16, 16, 16)})
    with pytest.raises(ValueError, match="branch"):
        model({"mra": torch.randn(1, 1, 16, 16, 16)}, branch="invalid")
