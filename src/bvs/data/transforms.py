from __future__ import annotations

import numpy as np
import torch

from .topcow import binary_label


def preprocess_volume(array: np.ndarray, normalization: str) -> np.ndarray:
    image = np.asarray(array, dtype=np.float32)
    if normalization == "precomputed":
        return image.copy()
    if normalization != "nonzero_zscore":
        raise ValueError(f"Unknown normalization: {normalization}")
    foreground = image != 0
    result = np.zeros_like(image, dtype=np.float32)
    if foreground.any():
        values = image[foreground]
        mean = float(values.mean())
        standard_deviation = float(values.std())
        if standard_deviation < 1e-8:
            result[foreground] = values - mean
        else:
            result[foreground] = (values - mean) / standard_deviation
    return result


def load_training_arrays(
    image_path: str, label_path: str, normalization: str
) -> tuple[np.ndarray, np.ndarray]:
    import nibabel as nib

    image = preprocess_volume(
        nib.load(image_path).get_fdata(dtype=np.float32), normalization
    )
    label = binary_label(nib.load(label_path).get_fdata())
    return image, label


def _crop_with_padding(array: np.ndarray, start: np.ndarray, size: np.ndarray) -> np.ndarray:
    before = np.maximum(-start, 0)
    after = np.maximum(start + size - np.asarray(array.shape), 0)
    padded = np.pad(array, tuple(zip(before, after)), mode="constant")
    adjusted = start + before
    slices = tuple(slice(int(s), int(s + length)) for s, length in zip(adjusted, size))
    return padded[slices]


def crop_or_pad_array(
    array: np.ndarray, target_size: tuple[int, int, int]
) -> np.ndarray:
    size = np.asarray(target_size)
    shape = np.asarray(array.shape)
    start = (shape - size) // 2
    return _crop_with_padding(array, start, size)


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


def sample_multimodal_patch(
    images: dict[str, np.ndarray],
    label: np.ndarray,
    patch_size: tuple[int, int, int],
    positive_probability: float,
    rng: np.random.Generator,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if not images:
        raise ValueError("At least one modality is required")
    shapes = {array.shape for array in images.values()} | {label.shape}
    if len(shapes) != 1:
        raise ValueError(f"Modalities and label must share a shape: {sorted(shapes)}")
    size = np.asarray(patch_size)
    positive = np.argwhere(label > 0)
    if len(positive) and rng.random() < positive_probability:
        center = positive[rng.integers(len(positive))]
    else:
        center = np.asarray(
            [rng.integers(max(length, 1)) for length in label.shape]
        )
    start = center - size // 2
    patches = {
        name: torch.from_numpy(
            _crop_with_padding(array, start, size).copy()
        ).unsqueeze(0).float()
        for name, array in images.items()
    }
    target = torch.from_numpy(
        _crop_with_padding(label, start, size).copy()
    ).long()
    return patches, target
