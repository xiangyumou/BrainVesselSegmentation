from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from bvs.evaluation import segmentation_metrics
from bvs.inference import predict_nifti, sliding_window_inference
from bvs.models import ConfigurableLingfengModel


class ForegroundModel(torch.nn.Module):
    def forward(self, image: torch.Tensor):
        logits = torch.cat((-image, image), dim=1)
        return {"logits": logits, "probabilities": torch.softmax(logits, dim=1), "features": image}


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
