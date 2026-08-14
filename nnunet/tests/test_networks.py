from __future__ import annotations

import pytest
import torch

from bvs_nnunet.networks import KDNetwork, StudentNetwork, TeacherNetwork


def test_student_is_mra_only_and_output_shape_matches() -> None:
    torch.manual_seed(1)
    network = StudentNetwork(num_input_channels=2, num_classes=2).eval()
    mra = torch.randn(1, 1, 16, 16, 16)
    first_cta = torch.randn_like(mra)
    second_cta = torch.randn_like(mra)
    with torch.no_grad():
        first = network(torch.cat((mra, first_cta), dim=1))
        second = network(torch.cat((mra, second_cta), dim=1))
    assert first.shape == (1, 2, 16, 16, 16)
    assert torch.equal(first, second)


def test_teacher_requires_mra_and_cta() -> None:
    network = TeacherNetwork()
    with pytest.raises(ValueError, match="exactly 2 channels"):
        network(torch.randn(1, 1, 16, 16, 16))
    output = network(torch.randn(1, 2, 16, 16, 16))
    assert output.shape == (1, 2, 16, 16, 16)


def test_kd_teacher_is_frozen_and_student_receives_gradients() -> None:
    torch.manual_seed(2)
    network = KDNetwork().train()
    output = network(torch.randn(2, 2, 16, 16, 16), return_distillation=True)
    assert isinstance(output, dict)
    loss = output["student_logits"].mean() + output["student_features"].mean()
    loss.backward()
    assert not network.teacher.training
    assert all(parameter.grad is None for parameter in network.teacher.parameters())
    assert any(
        parameter.grad is not None
        for parameter in network.student.parameters()
        if parameter.requires_grad
    )
    assert output["student_features"].shape == (2, 8)
    assert output["teacher_features"].shape == (2, 8)


def test_teacher_checkpoint_must_have_expected_trainer() -> None:
    network = KDNetwork()
    with pytest.raises(RuntimeError, match="nnUNetTrainerBVSTeacher"):
        network.load_teacher_checkpoint(
            {
                "trainer_name": "nnUNetTrainer",
                "network_weights": network.teacher.state_dict(),
            }
        )
