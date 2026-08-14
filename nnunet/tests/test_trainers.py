from __future__ import annotations

import json
from pathlib import Path

import torch

from bvs.training.losses import (
    CombinedSegmentationLoss,
    MetricContrastiveLoss,
    TemperatureKLLoss,
)
from bvs_nnunet.export_student import export_student
from bvs_nnunet.networks import KDNetwork, StudentNetwork
from bvs_nnunet.trainers.trainers import (
    nnUNetTrainerBVSKD,
    nnUNetTrainerBVSStudent,
    nnUNetTrainerBVSTeacher,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def test_all_trainers_are_native_nnunet_trainers() -> None:
    assert issubclass(nnUNetTrainerBVSTeacher, nnUNetTrainer)
    assert issubclass(nnUNetTrainerBVSStudent, nnUNetTrainer)
    assert issubclass(nnUNetTrainerBVSKD, nnUNetTrainer)


def test_kd_objective_matches_locked_formula() -> None:
    trainer = object.__new__(nnUNetTrainerBVSKD)
    trainer.loss = CombinedSegmentationLoss()
    trainer.logit_distillation_loss = TemperatureKLLoss(10)
    trainer.feature_distillation_loss = MetricContrastiveLoss(1)
    student_logits = torch.randn(2, 2, 4, 4, 4)
    teacher_logits = torch.randn(2, 2, 4, 4, 4)
    student_features = torch.nn.functional.normalize(torch.randn(2, 8), dim=1)
    teacher_features = torch.nn.functional.normalize(torch.randn(2, 8), dim=1)
    target = torch.randint(0, 2, (2, 1, 4, 4, 4))
    output = {
        "student_logits": student_logits,
        "teacher_logits": teacher_logits,
        "student_features": student_features,
        "teacher_features": teacher_features,
    }
    actual, _ = trainer._kd_objective(output, target)
    expected = (
        trainer.loss(student_logits, target)
        + 0.5 * trainer.logit_distillation_loss(student_logits, teacher_logits)
        + 0.5
        * trainer.feature_distillation_loss(student_features, teacher_features)
    )
    assert torch.equal(actual, expected)


def test_export_student_creates_dataset501_inference_checkpoint(
    tmp_path: Path,
) -> None:
    kd = KDNetwork()
    checkpoint = {
        "network_weights": kd.state_dict(),
        "optimizer_state": {"unused": True},
        "grad_scaler_state": None,
        "logging": {},
        "_best_ema": 0.5,
        "current_epoch": 1,
        "init_args": {"configuration": "3d_fullres"},
        "trainer_name": "nnUNetTrainerBVSKD",
        "inference_allowed_mirroring_axes": (0, 1, 2),
    }
    kd_checkpoint = tmp_path / "kd.pth"
    torch.save(checkpoint, kd_checkpoint)
    preprocessed = tmp_path / "Dataset501_BrainVesselMRA"
    preprocessed.mkdir()
    plans = {"configurations": {"3d_fullres": {}}}
    dataset_json = {
        "channel_names": {"0": "MRA"},
        "labels": {"background": 0, "vessel": 1},
        "numTraining": 100,
        "file_ending": ".nii.gz",
    }
    (preprocessed / "nnUNetPlans.json").write_text(
        json.dumps(plans), encoding="utf-8"
    )
    (preprocessed / "dataset.json").write_text(
        json.dumps(dataset_json), encoding="utf-8"
    )
    (preprocessed / "dataset_fingerprint.json").write_text(
        "{}", encoding="utf-8"
    )
    output = export_student(kd_checkpoint, preprocessed, tmp_path / "model")
    exported = torch.load(output, map_location="cpu", weights_only=False)
    assert exported["trainer_name"] == "nnUNetTrainerBVSStudent"
    assert exported["bvs_export"]["logits_verified_equal"] is True
    student = StudentNetwork()
    student.load_state_dict(exported["network_weights"], strict=True)
    assert not any(
        key.startswith("teacher.") for key in exported["network_weights"]
    )
