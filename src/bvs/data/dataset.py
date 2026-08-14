from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from .topcow import (
    TopCoWCase,
    binary_label,
    discover_topcow_multimodal_cases,
)
from .pattern_directory import discover_pattern_cases
from .transforms import (
    crop_or_pad_array,
    load_training_arrays,
    preprocess_volume,
    sample_multimodal_patch,
    sample_patch,
)


@dataclass(frozen=True)
class MultimodalCase:
    case_id: str
    modalities: dict[str, Path]
    label: Path


def _filename(spec: str | dict[str, Any], field: str) -> str:
    if isinstance(spec, str):
        return spec
    unknown = set(spec) - {"filename", "directory", "pattern", "strategy"}
    if unknown:
        raise ValueError(f"Unknown fields in {field}: {sorted(unknown)}")
    value = spec.get("filename")
    if not value:
        raise ValueError(f"{field}.filename is required")
    return str(value)


def discover_lingfeng_cases(
    root: str | Path,
    modality_specs: dict[str, str | dict[str, Any]],
    label_spec: str | dict[str, Any],
) -> list[MultimodalCase]:
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Case directory root does not exist: {base}")
    if not modality_specs:
        raise ValueError("data.modalities must configure every model modality")
    cases: list[MultimodalCase] = []
    for directory in sorted(path for path in base.iterdir() if path.is_dir()):
        modalities = {
            name: directory / _filename(spec, f"data.modalities.{name}")
            for name, spec in modality_specs.items()
        }
        label = directory / _filename(label_spec, "data.label")
        missing = [name for name, path in modalities.items() if not path.is_file()]
        if not label.is_file():
            missing.append("label")
        if missing:
            raise FileNotFoundError(
                f"Case {directory.name} is incomplete; missing files for {missing}"
            )
        cases.append(MultimodalCase(directory.name, modalities, label))
    if not cases:
        raise ValueError(f"No case directories found in {base}")
    return cases


def discover_cases(
    adapter: str,
    root: str | Path,
    modality_specs: dict[str, str | dict[str, Any]],
    label_spec: str | dict[str, Any],
) -> list[MultimodalCase]:
    if adapter == "lingfeng_case_directory":
        return discover_lingfeng_cases(root, modality_specs, label_spec)
    if adapter == "topcow":
        return [
            MultimodalCase(case_id, modalities, label)
            for case_id, modalities, label in discover_topcow_multimodal_cases(
                root, modality_specs, label_spec
            )
        ]
    if adapter == "pattern_directory":
        return [
            MultimodalCase(case.case_id, case.modalities, case.label)
            for case in discover_pattern_cases(root, modality_specs, label_spec)
        ]
    raise ValueError(f"Unknown data adapter: {adapter}")


def validate_multimodal_case(
    case: MultimodalCase, affine_tolerance: float = 1e-5
) -> dict[str, Any]:
    images = {name: nib.load(str(path)) for name, path in case.modalities.items()}
    label = nib.load(str(case.label))
    references = {**images, "label": label}
    first_name = next(iter(images))
    reference = images[first_name]
    reference_orientation = nib.aff2axcodes(reference.affine)
    errors: list[str] = []
    for name, image in references.items():
        if image.shape != reference.shape:
            errors.append(
                f"shape mismatch: {first_name}={reference.shape}, {name}={image.shape}"
            )
        if not np.allclose(image.affine, reference.affine, atol=affine_tolerance):
            errors.append(f"affine mismatch: {first_name} vs {name}")
        orientation = nib.aff2axcodes(image.affine)
        if orientation != reference_orientation:
            errors.append(
                f"orientation mismatch: {first_name}={reference_orientation}, "
                f"{name}={orientation}"
            )
    if errors:
        raise ValueError(f"Case {case.case_id}: " + "; ".join(errors))
    return {
        "case_id": case.case_id,
        "shape": list(reference.shape),
        "orientation": list(reference_orientation),
        "modalities": {name: str(path) for name, path in case.modalities.items()},
        "label": str(case.label),
    }


class _CaseCache:
    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.values: OrderedDict[tuple[Any, ...], Any] = OrderedDict()

    def get(self, key: tuple[Any, ...]) -> Any | None:
        if key not in self.values:
            return None
        value = self.values.pop(key)
        self.values[key] = value
        return value

    def put(self, key: tuple[Any, ...], value: Any) -> None:
        if self.maximum == 0:
            return
        self.values.pop(key, None)
        self.values[key] = value
        while len(self.values) > self.maximum:
            self.values.popitem(last=False)


class MultimodalPatchDataset(Dataset):
    def __init__(
        self,
        cases: list[MultimodalCase],
        modalities: list[str],
        student_modality: str,
        patch_size: tuple[int, int, int] = (48, 48, 48),
        samples_per_volume: int = 30,
        crop_or_pad_size: tuple[int, int, int] | None = None,
        normalization: str = "nonzero_zscore",
        augmentation: dict[str, Any] | None = None,
        label_strategy: str = "nonzero_to_foreground",
        positive_probability: float = 0.7,
        seed: int = 42,
        cache_max_cases: int = 2,
    ) -> None:
        if not cases:
            raise ValueError("MultimodalPatchDataset requires at least one case")
        self.cases = cases
        self.modalities = tuple(modalities)
        self.student_modality = student_modality
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume
        self.crop_or_pad_size = crop_or_pad_size
        self.normalization = normalization
        self.augmentation = augmentation
        self.label_strategy = label_strategy
        self.positive_probability = positive_probability
        self.seed = int(seed)
        self.epoch = 0
        self._cache = _CaseCache(cache_max_cases)
        for case in cases:
            missing = sorted(set(self.modalities) - set(case.modalities))
            if missing:
                raise ValueError(f"Case {case.case_id} is missing modalities: {missing}")
            validate_multimodal_case(case)
        if augmentation and augmentation.get("enabled"):
            raise ValueError(
                "Random augmentation is not implemented in the deterministic "
                "multimodal patch pipeline"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _load_case(
        self, case: MultimodalCase
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        key = (
            case.case_id,
            tuple((name, str(case.modalities[name])) for name in self.modalities),
            str(case.label),
            self.normalization,
            self.crop_or_pad_size,
            self.label_strategy,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        images = {
            name: preprocess_volume(
                nib.load(str(case.modalities[name])).get_fdata(dtype=np.float32),
                self.normalization,
            )
            for name in self.modalities
        }
        raw_label = nib.load(str(case.label)).get_fdata()
        if self.label_strategy == "nonzero_to_foreground":
            label = (raw_label > 0).astype(np.uint8)
        elif self.label_strategy == "identity":
            label = np.asarray(raw_label, dtype=np.int64)
        else:
            raise ValueError(f"Unknown label strategy: {self.label_strategy}")
        if self.crop_or_pad_size:
            images = {
                name: crop_or_pad_array(array, self.crop_or_pad_size)
                for name, array in images.items()
            }
            label = crop_or_pad_array(label, self.crop_or_pad_size)
        for array in [*images.values(), label]:
            array.setflags(write=False)
        value = (images, label)
        self._cache.put(key, value)
        return value

    def __len__(self) -> int:
        return len(self.cases) * self.samples_per_volume

    def __getitem__(self, index: int) -> dict[str, Any]:
        case = self.cases[index % len(self.cases)]
        images, full_label = self._load_case(case)
        rng = np.random.default_rng(
            self.seed + self.epoch * 1_000_003 + int(index)
        )
        inputs, label = sample_multimodal_patch(
            images,
            full_label,
            self.patch_size,
            self.positive_probability,
            rng,
        )
        return {
            "inputs": inputs,
            "student_image": inputs[self.student_modality],
            "label": label,
            "case_id": case.case_id,
        }


class TopCoWPatchDataset(Dataset):
    """Compatibility dataset for existing single-MRA baseline configurations."""

    def __init__(
        self,
        cases: list[TopCoWCase],
        patch_size: tuple[int, int, int] = (48, 48, 48),
        positive_probability: float = 0.7,
        samples_per_case: int = 4,
        normalization: str = "nonzero_zscore",
        seed: int = 42,
        cache_max_cases: int = 2,
    ) -> None:
        self.cases = cases
        self.patch_size = patch_size
        self.positive_probability = positive_probability
        self.samples_per_case = samples_per_case
        self.normalization = normalization
        self.seed = int(seed)
        self.epoch = 0
        self._cache = _CaseCache(cache_max_cases)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _load_case(self, case: TopCoWCase) -> tuple[np.ndarray, np.ndarray]:
        key = (
            case.case_id,
            str(case.image),
            str(case.label),
            self.normalization,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        image, label = load_training_arrays(
            str(case.image), str(case.label), self.normalization
        )
        image.setflags(write=False)
        label.setflags(write=False)
        value = (image, label)
        self._cache.put(key, value)
        return value

    def __len__(self) -> int:
        return len(self.cases) * self.samples_per_case

    def __getitem__(self, index: int):
        case = self.cases[index % len(self.cases)]
        image, label = self._load_case(case)
        rng = np.random.default_rng(
            self.seed + self.epoch * 1_000_003 + int(index)
        )
        image_patch, label_patch = sample_patch(
            image,
            binary_label(label),
            self.patch_size,
            self.positive_probability,
            rng,
        )
        return {"image": image_patch, "label": label_patch, "case_id": case.case_id}
