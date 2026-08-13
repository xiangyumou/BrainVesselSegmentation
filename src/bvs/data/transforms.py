from __future__ import annotations

import numpy as np
import torch

from .topcow import binary_label, normalize_mra


def load_training_arrays(image_path: str, label_path: str) -> tuple[np.ndarray, np.ndarray]:
    import nibabel as nib

    image = normalize_mra(nib.load(image_path).get_fdata(dtype=np.float32))
    label = binary_label(nib.load(label_path).get_fdata())
    return image, label


def _crop_with_padding(array: np.ndarray, start: np.ndarray, size: np.ndarray) -> np.ndarray:
    before = np.maximum(-start, 0)
    after = np.maximum(start + size - np.asarray(array.shape), 0)
    padded = np.pad(array, tuple(zip(before, after)), mode="constant")
    adjusted = start + before
    slices = tuple(slice(int(s), int(s + length)) for s, length in zip(adjusted, size))
    return padded[slices]


def sample_patch(
    image: np.ndarray,
    label: np.ndarray,
    patch_size: tuple[int, int, int] = (48, 48, 48),
    positive_probability: float = 0.7,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = rng or np.random.default_rng()
    size = np.asarray(patch_size)
    positive = np.argwhere(label > 0)
    if len(positive) and rng.random() < positive_probability:
        center = positive[rng.integers(len(positive))]
    else:
        center = np.asarray([rng.integers(max(length, 1)) for length in image.shape])
    start = center - size // 2
    image_patch = _crop_with_padding(image, start, size)
    label_patch = _crop_with_padding(label, start, size)
    return (
        torch.from_numpy(image_patch.copy()).unsqueeze(0).float(),
        torch.from_numpy(label_patch.copy()).long(),
    )

