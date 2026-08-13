from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import pytest

from bvs.evaluation import evaluate_directories, segmentation_metrics
from bvs.inference import (
    TEACHER_INPUT_ERROR,
    discover_inference_cases,
    predict_nifti,
    sliding_window_inference,
)
from bvs.models import ConfigurableLingfengModel


class ForegroundModel(torch.nn.Module):
    def forward(self, image: torch.Tensor):
        logits = torch.cat((-image, image), dim=1)
        return {"logits": logits, "probabilities": torch.softmax(logits, dim=1), "features": image}


class ThreeClassModel(torch.nn.Module):
    def forward(self, image: torch.Tensor):
        logits = torch.cat((-image, torch.zeros_like(image), image), dim=1)
        return {"logits": logits, "probabilities": torch.softmax(logits, dim=1)}


def test_sliding_window_shape() -> None:
    probabilities = sliding_window_inference(
        torch.randn(1, 1, 19, 21, 23),
        ForegroundModel(),
        (16, 16, 16),
        (8, 8, 8),
    )
    assert probabilities.shape == (1, 2, 19, 21, 23)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones_like(probabilities[:, 0]), atol=1e-5)


def test_nifti_round_trip_preserves_shape_and_affine(tmp_path: Path) -> None:
    affine = np.diag([0.4, 0.5, 0.7, 1.0])
    source = tmp_path / "source.nii.gz"
    output = tmp_path / "prediction.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((12, 13, 14), dtype=np.float32), affine), source)
    predict_nifti(ForegroundModel(), source, output, "cpu", (8, 8, 8), (4, 4, 4))
    result = nib.load(output)
    assert result.shape == (12, 13, 14)
    assert np.allclose(result.affine, affine)


def test_metrics_are_one_for_identical_foreground() -> None:
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[2:6, 3:5, 1:7] = 1
    metrics = segmentation_metrics(mask, mask)
    assert all(abs(value - 1.0) < 1e-8 for value in metrics.values())


def test_multimodal_teacher_sliding_window() -> None:
    model = ConfigurableLingfengModel(
        ["mra", "cta"], "mra", {"mra": 1, "cta": 1}, 2, base_channels=2
    )
    probabilities = sliding_window_inference(
        {
            "mra": torch.randn(1, 1, 17, 18, 19),
            "cta": torch.randn(1, 1, 17, 18, 19),
        },
        model,
        (16, 16, 16),
        (8, 8, 8),
        branch="teacher",
    )
    assert probabilities.shape == (1, 2, 17, 18, 19)


def test_sliding_window_supports_dynamic_output_channels() -> None:
    probabilities = sliding_window_inference(
        torch.randn(1, 1, 9, 10, 11),
        ThreeClassModel(),
        (8, 8, 8),
        (4, 4, 4),
    )
    assert probabilities.shape == (1, 3, 9, 10, 11)


def test_torchio_mode_preserves_shape() -> None:
    pytest.importorskip("torchio")
    probabilities = sliding_window_inference(
        torch.randn(1, 1, 9, 10, 11),
        ForegroundModel(),
        (8, 8, 8),
        (4, 4, 4),
        compatibility_mode="torchio",
    )
    assert probabilities.shape == (1, 2, 9, 10, 11)


def test_teacher_discovery_rejects_single_file(tmp_path: Path) -> None:
    source = tmp_path / "source.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((8, 8, 8)), np.eye(4)), source)
    config = {
        "model": {"modalities": ["mra", "cta"]},
        "data": {
            "adapter": "lingfeng_case_directory",
            "modalities": {"mra": "mra.nii.gz", "cta": "cta.nii.gz"},
        },
    }
    with pytest.raises(ValueError, match=TEACHER_INPUT_ERROR):
        discover_inference_cases(config, source, "teacher")


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4)),
        path,
    )


def test_evaluation_rejects_missing_predictions_by_default(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    labels = tmp_path / "labels"
    _write_mask(predictions / "one.nii.gz")
    _write_mask(labels / "one.nii.gz")
    _write_mask(labels / "two.nii.gz")
    with pytest.raises(ValueError, match=r"missing_predictions=\['two\.nii\.gz'\]"):
        evaluate_directories(predictions, labels, tmp_path / "output")


def test_evaluation_rejects_unexpected_predictions_by_default(
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "predictions"
    labels = tmp_path / "labels"
    _write_mask(predictions / "one.nii.gz")
    _write_mask(predictions / "two.nii.gz")
    _write_mask(labels / "one.nii.gz")
    with pytest.raises(
        ValueError, match=r"unexpected_predictions=\['two\.nii\.gz'\]"
    ):
        evaluate_directories(predictions, labels, tmp_path / "output")


def test_partial_evaluation_reports_case_set_differences(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    labels = tmp_path / "labels"
    _write_mask(predictions / "one.nii.gz")
    _write_mask(predictions / "unexpected.nii.gz")
    _write_mask(labels / "one.nii.gz")
    _write_mask(labels / "missing.nii.gz")
    report = evaluate_directories(
        predictions, labels, tmp_path / "output", strict=False
    )
    assert report["case_count"] == 1
    assert report["prediction_case_count"] == 2
    assert report["label_case_count"] == 2
    assert report["missing_predictions"] == ["missing.nii.gz"]
    assert report["unexpected_predictions"] == ["unexpected.nii.gz"]
