from __future__ import annotations

from itertools import product
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F

from .data.topcow import normalize_mra


def _starts(length: int, window: int, overlap: int) -> list[int]:
    if overlap >= window:
        raise ValueError("Overlap must be smaller than window size")
    if length <= window:
        return [0]
    stride = window - overlap
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def gaussian_importance_map(
    window_size: tuple[int, int, int], sigma_scale: float = 0.125
) -> torch.Tensor:
    axes = [
        torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        for size in window_size
    ]
    grids = torch.meshgrid(*axes, indexing="ij")
    exponent = sum(
        (grid / max(size * sigma_scale, 1e-6)) ** 2
        for grid, size in zip(grids, window_size)
    )
    weights = torch.exp(-0.5 * exponent)
    return weights.clamp_min(1e-4)


@torch.inference_mode()
def sliding_window_inference(
    image: torch.Tensor,
    model: torch.nn.Module,
    window_size: tuple[int, int, int] = (48, 48, 48),
    overlap: tuple[int, int, int] = (24, 24, 24),
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    if image.ndim != 5 or image.shape[0] != 1:
        raise ValueError("Expected image shape [1, C, D, H, W]")
    original_shape = tuple(image.shape[2:])
    padding = [max(window - length, 0) for length, window in zip(original_shape, window_size)]
    image = F.pad(image, (0, padding[2], 0, padding[1], 0, padding[0]))
    spatial = tuple(image.shape[2:])
    output = torch.zeros((1, 2, *spatial), dtype=torch.float32, device=device)
    normalizer = torch.zeros((1, 1, *spatial), dtype=torch.float32, device=device)
    weights = gaussian_importance_map(window_size).to(device)[None, None]
    positions = [
        _starts(length, window, overlap_value)
        for length, window, overlap_value in zip(spatial, window_size, overlap)
    ]
    model.eval()
    for z, y, x in product(*positions):
        patch = image[:, :, z : z + window_size[0], y : y + window_size[1], x : x + window_size[2]]
        probabilities = model(patch.to(device))["probabilities"].float()
        output[:, :, z : z + window_size[0], y : y + window_size[1], x : x + window_size[2]] += (
            probabilities * weights
        )
        normalizer[:, :, z : z + window_size[0], y : y + window_size[1], x : x + window_size[2]] += weights
    output = output / normalizer.clamp_min(1e-7)
    return output[:, :, : original_shape[0], : original_shape[1], : original_shape[2]]


def predict_nifti(
    model: torch.nn.Module,
    input_path: str | Path,
    output_path: str | Path,
    device: torch.device | str,
    window_size: tuple[int, int, int] = (48, 48, 48),
    overlap: tuple[int, int, int] = (24, 24, 24),
) -> Path:
    source = nib.load(str(input_path))
    normalized = normalize_mra(source.get_fdata(dtype=np.float32))
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    probabilities = sliding_window_inference(tensor, model, window_size, overlap, device)
    prediction = torch.argmax(probabilities, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    header = source.header.copy()
    header.set_data_dtype(np.uint8)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(prediction, source.affine, header), str(destination))
    return destination

