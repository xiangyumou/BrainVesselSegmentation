from __future__ import annotations

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
from .transforms import load_training_arrays, sample_patch


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


def _torchio_subject(
    case: MultimodalCase,
    crop_or_pad_size: tuple[int, int, int] | None,
    normalization: str,
    augmentation: dict[str, Any] | None,
) -> Any:
    try:
        import torchio as tio
    except ImportError as error:
        raise RuntimeError(
            "The multimodal patch pipeline requires TorchIO; install project dependencies"
        ) from error
    subject = tio.Subject(
        **{name: tio.ScalarImage(path) for name, path in case.modalities.items()},
        label=tio.LabelMap(case.label),
    )
    transforms: list[tio.Transform] = []
    if crop_or_pad_size:
        transforms.append(tio.CropOrPad(crop_or_pad_size, padding_mode="reflect"))
    if normalization == "torchio_zscore":
        transforms.append(tio.ZNormalization())
    elif normalization not in {"none", None}:
        raise ValueError(f"Unknown normalization: {normalization}")
    if augmentation and augmentation.get("enabled"):
        unknown = set(augmentation) - {"enabled"}
        if unknown:
            raise ValueError(f"Unknown augmentation fields: {sorted(unknown)}")
        transforms.extend(
            [
                tio.RandomBiasField(),
                tio.RandomNoise(),
                tio.RandomFlip(axes=(0,)),
                tio.OneOf({tio.RandomAffine(): 0.8, tio.RandomElasticDeformation(): 0.2}),
            ]
        )
    return tio.Compose(transforms)(subject) if transforms else subject


class MultimodalPatchDataset(Dataset):
    def __init__(
        self,
        cases: list[MultimodalCase],
        modalities: list[str],
        student_modality: str,
        patch_size: tuple[int, int, int] = (48, 48, 48),
        samples_per_volume: int = 30,
        crop_or_pad_size: tuple[int, int, int] | None = None,
        normalization: str = "torchio_zscore",
        augmentation: dict[str, Any] | None = None,
        label_strategy: str = "nonzero_to_foreground",
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
        for case in cases:
            missing = sorted(set(self.modalities) - set(case.modalities))
            if missing:
                raise ValueError(f"Case {case.case_id} is missing modalities: {missing}")
            validate_multimodal_case(case)

    def __len__(self) -> int:
        return len(self.cases) * self.samples_per_volume

    def __getitem__(self, index: int) -> dict[str, Any]:
        case = self.cases[index % len(self.cases)]
        subject = _torchio_subject(
            case,
            self.crop_or_pad_size,
            self.normalization,
            self.augmentation,
        )
        spatial = np.asarray(subject.spatial_shape)
        patch = np.asarray(self.patch_size)
        if np.any(spatial < patch):
            raise ValueError(
                f"Case {case.case_id} shape {tuple(spatial)} is smaller than patch "
                f"{self.patch_size}"
            )
        starts = [
            int(torch.randint(0, int(length - size + 1), (1,)).item())
            for length, size in zip(spatial, patch)
        ]
        slices = tuple(slice(start, start + size) for start, size in zip(starts, patch))
        inputs = {
            name: subject[name].data[(slice(None), *slices)].float()
            for name in self.modalities
        }
        label = subject["label"].data[(0, *slices)]
        if self.label_strategy == "nonzero_to_foreground":
            label = (label > 0).long()
        elif self.label_strategy == "identity":
            label = label.long()
        else:
            raise ValueError(f"Unknown label strategy: {self.label_strategy}")
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
    ) -> None:
        self.cases = cases
        self.patch_size = patch_size
        self.positive_probability = positive_probability
        self.samples_per_case = samples_per_case

    def __len__(self) -> int:
        return len(self.cases) * self.samples_per_case

    def __getitem__(self, index: int):
        case = self.cases[index % len(self.cases)]
        image, label = load_training_arrays(str(case.image), str(case.label))
        image_patch, label_patch = sample_patch(
            image, binary_label(label), self.patch_size, self.positive_probability
        )
        return {"image": image_patch, "label": label_patch, "case_id": case.case_id}
