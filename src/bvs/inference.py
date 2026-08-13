from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F

from .data.topcow import topcow_release_root
from .data.transforms import preprocess_volume

TEACHER_INPUT_ERROR = (
    "Teacher prediction requires a case directory or dataset root containing "
    "every configured modality."
)


@dataclass(frozen=True)
class InferenceCase:
    case_id: str
    modalities: dict[str, Path]
    reference: Path


def _filename(spec: str | dict[str, Any], field: str) -> str:
    if isinstance(spec, str):
        return spec
    value = spec.get("filename")
    if not value:
        raise ValueError(f"{field}.filename is required")
    return str(value)


def _discover_lingfeng_inference_cases(
    root: Path, modality_specs: dict[str, str | dict[str, Any]]
) -> list[InferenceCase]:
    def build(directory: Path) -> InferenceCase | None:
        modalities = {
            name: directory / _filename(spec, f"data.modalities.{name}")
            for name, spec in modality_specs.items()
        }
        if all(path.is_file() for path in modalities.values()):
            return InferenceCase(directory.name, modalities, next(iter(modalities.values())))
        return None

    direct = build(root)
    if direct is not None:
        return [direct]
    cases = []
    incomplete = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        case = build(directory)
        if case is None:
            incomplete.append(directory.name)
        else:
            cases.append(case)
    if incomplete:
        raise FileNotFoundError(
            f"Incomplete teacher cases missing configured modalities: {incomplete}"
        )
    if not cases:
        raise FileNotFoundError(TEACHER_INPUT_ERROR)
    return cases


def _discover_topcow_inference_cases(
    root: Path, modality_specs: dict[str, str | dict[str, Any]]
) -> list[InferenceCase]:
    release = topcow_release_root(root)
    indexed: dict[str, dict[str, Path]] = {}
    all_ids: set[str] = set()
    for name, spec in modality_specs.items():
        if isinstance(spec, str):
            directory, pattern = release / "imagesTr", spec
        else:
            pattern = spec.get("pattern", spec.get("filename"))
            if not pattern:
                raise ValueError(f"data.modalities.{name} requires pattern")
            directory = release / spec.get("directory", "imagesTr")
        if "{case_id}" not in pattern:
            raise ValueError(f"TopCoW pattern must contain {{case_id}}: {pattern}")
        if not directory.is_dir():
            raise FileNotFoundError(f"Required directory does not exist: {directory}")
        regex = re.compile(
            "^"
            + re.escape(pattern).replace(
                re.escape("{case_id}"), r"(?P<case_id>.+)"
            )
            + "$"
        )
        values: dict[str, Path] = {}
        for path in sorted(directory.glob("*.nii.gz")):
            match = regex.match(path.name)
            if match:
                values[match.group("case_id")] = path
        indexed[name] = values
        all_ids |= set(values)
    missing = {
        name: sorted(all_ids - set(values))
        for name, values in indexed.items()
        if all_ids - set(values)
    }
    if missing:
        raise FileNotFoundError(f"TopCoW teacher modalities are incomplete: {missing}")
    if not all_ids:
        raise FileNotFoundError(TEACHER_INPUT_ERROR)
    return [
        InferenceCase(
            case_id,
            {name: values[case_id] for name, values in indexed.items()},
            next(iter(indexed.values()))[case_id],
        )
        for case_id in sorted(all_ids)
    ]


def discover_inference_cases(
    config: dict[str, Any], input_path: str | Path, branch: str
) -> list[InferenceCase]:
    source = Path(input_path).expanduser().resolve()
    if branch == "student":
        modality = str(config["model"].get("student_modality", "image"))
        if source.is_file():
            if not source.name.endswith(".nii.gz"):
                raise ValueError(f"Input is not a .nii.gz file: {source}")
            return [InferenceCase(source.name.removesuffix(".nii.gz"), {modality: source}, source)]
        if not source.is_dir():
            raise FileNotFoundError(f"Input does not exist: {source}")
        files = sorted(source.glob("*.nii.gz"))
        if not files:
            raise FileNotFoundError(f"No NIfTI files found: {source}")
        return [
            InferenceCase(path.name.removesuffix(".nii.gz"), {modality: path}, path)
            for path in files
        ]
    if branch != "teacher":
        raise ValueError("branch must be student or teacher")
    if source.is_file():
        raise ValueError(TEACHER_INPUT_ERROR)
    if not source.is_dir():
        raise FileNotFoundError(TEACHER_INPUT_ERROR)
    data = config["data"]
    if data["adapter"] == "lingfeng_case_directory":
        return _discover_lingfeng_inference_cases(source, data["modalities"])
    if data["adapter"] == "topcow":
        return _discover_topcow_inference_cases(source, data["modalities"])
    raise ValueError(f"Unknown data adapter: {data['adapter']}")


def _validate_window(
    window_size: tuple[int, int, int], overlap: tuple[int, int, int]
) -> None:
    if len(window_size) != 3 or len(overlap) != 3:
        raise ValueError("window_size and overlap must contain three values")
    if any(size <= 0 for size in window_size):
        raise ValueError("Window size values must be positive")
    if any(value < 0 or value >= size for value, size in zip(overlap, window_size)):
        raise ValueError("Overlap must satisfy 0 <= overlap < window size")


def _starts(length: int, window: int, overlap: int) -> list[int]:
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
    return torch.exp(-0.5 * exponent).clamp_min(1e-4)


def _predict_patch(
    patches: dict[str, torch.Tensor],
    model: torch.nn.Module,
    branch: str,
    scalar_input: bool,
) -> torch.Tensor:
    if hasattr(model, "forward_student") and branch == "student":
        result = model.forward_student(
            next(iter(patches.values())) if scalar_input else patches
        )
    elif hasattr(model, "forward_teacher") and branch == "teacher":
        if scalar_input:
            raise ValueError(
                "Teacher sliding-window inference requires a multimodal input mapping"
            )
        result = model.forward_teacher(patches)
    else:
        result = model(next(iter(patches.values())))
    probabilities = result["probabilities"]
    if probabilities.ndim != 5:
        raise ValueError(
            f"Model probabilities must have shape [B,C,D,H,W], got {probabilities.shape}"
        )
    return probabilities.float()


@torch.inference_mode()
def _gaussian_inference(
    tensors: dict[str, torch.Tensor],
    model: torch.nn.Module,
    window_size: tuple[int, int, int],
    overlap: tuple[int, int, int],
    device: torch.device | str,
    branch: str,
    scalar_input: bool,
) -> torch.Tensor:
    original_shape = tuple(next(iter(tensors.values())).shape[2:])
    padding = [
        max(window - length, 0)
        for length, window in zip(original_shape, window_size)
    ]
    padded = {
        name: F.pad(tensor, (0, padding[2], 0, padding[1], 0, padding[0]))
        for name, tensor in tensors.items()
    }
    spatial = tuple(next(iter(padded.values())).shape[2:])
    normalizer = torch.zeros((1, 1, *spatial), dtype=torch.float32, device=device)
    weights = gaussian_importance_map(window_size).to(device)[None, None]
    positions = [
        _starts(length, window, overlap_value)
        for length, window, overlap_value in zip(spatial, window_size, overlap)
    ]
    output: torch.Tensor | None = None
    for z, y, x in product(*positions):
        patches = {
            name: tensor[
                :,
                :,
                z : z + window_size[0],
                y : y + window_size[1],
                x : x + window_size[2],
            ].to(device)
            for name, tensor in padded.items()
        }
        probabilities = _predict_patch(patches, model, branch, scalar_input)
        if output is None:
            output = torch.zeros(
                (1, probabilities.shape[1], *spatial),
                dtype=torch.float32,
                device=device,
            )
        output[
            :,
            :,
            z : z + window_size[0],
            y : y + window_size[1],
            x : x + window_size[2],
        ] += probabilities * weights
        normalizer[
            :,
            :,
            z : z + window_size[0],
            y : y + window_size[1],
            x : x + window_size[2],
        ] += weights
    assert output is not None
    output = output / normalizer.clamp_min(1e-7)
    return output[:, :, : original_shape[0], : original_shape[1], : original_shape[2]]


@torch.inference_mode()
def _torchio_inference(
    tensors: dict[str, torch.Tensor],
    model: torch.nn.Module,
    window_size: tuple[int, int, int],
    overlap: tuple[int, int, int],
    device: torch.device | str,
    branch: str,
    scalar_input: bool,
) -> torch.Tensor:
    import torchio as tio

    subject = tio.Subject(
        **{
            name: tio.ScalarImage(tensor=tensor.squeeze(0).cpu())
            for name, tensor in tensors.items()
        }
    )
    sampler = tio.GridSampler(
        subject, patch_size=window_size, patch_overlap=overlap
    )
    aggregator = tio.GridAggregator(sampler, overlap_mode="average")
    loader = torch.utils.data.DataLoader(sampler, batch_size=1)
    for batch in loader:
        patches = {
            name: batch[name][tio.DATA].to(device) for name in tensors
        }
        probabilities = _predict_patch(patches, model, branch, scalar_input)
        aggregator.add_batch(probabilities.cpu(), batch[tio.LOCATION])
    return aggregator.get_output_tensor().unsqueeze(0)


@torch.inference_mode()
def sliding_window_inference(
    image: torch.Tensor | Mapping[str, torch.Tensor],
    model: torch.nn.Module,
    window_size: tuple[int, int, int] = (48, 48, 48),
    overlap: tuple[int, int, int] = (24, 24, 24),
    device: torch.device | str = "cpu",
    branch: str = "student",
    compatibility_mode: str = "gaussian",
) -> torch.Tensor:
    _validate_window(window_size, overlap)
    scalar_input = torch.is_tensor(image)
    tensors = {"image": image} if scalar_input else dict(image)
    if not tensors:
        raise ValueError("At least one input modality is required")
    for name, tensor in tensors.items():
        if tensor.ndim != 5 or tensor.shape[0] != 1:
            raise ValueError(
                f"Expected {name} shape [1,C,D,H,W], got {tuple(tensor.shape)}"
            )
    shapes = {tuple(tensor.shape[2:]) for tensor in tensors.values()}
    if len(shapes) != 1:
        raise ValueError(f"All inference modalities must share a shape: {sorted(shapes)}")
    model.eval()
    if compatibility_mode == "gaussian":
        return _gaussian_inference(
            tensors, model, window_size, overlap, device, branch, scalar_input
        )
    if compatibility_mode == "torchio":
        return _torchio_inference(
            tensors, model, window_size, overlap, device, branch, scalar_input
        )
    raise ValueError("compatibility_mode must be gaussian or torchio")


def load_inference_case(
    case: InferenceCase, normalization: str
) -> tuple[dict[str, torch.Tensor], nib.Nifti1Image]:
    images = {name: nib.load(str(path)) for name, path in case.modalities.items()}
    reference = nib.load(str(case.reference))
    tensors: dict[str, torch.Tensor] = {}
    for name, image in images.items():
        if image.shape != reference.shape:
            raise ValueError(
                f"Case {case.case_id} shape mismatch: "
                f"reference={reference.shape}, {name}={image.shape}"
            )
        if not np.allclose(image.affine, reference.affine, atol=1e-5):
            raise ValueError(f"Case {case.case_id} affine mismatch for {name}")
        array = preprocess_volume(
            image.get_fdata(dtype=np.float32), normalization
        )
        tensors[name] = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    return tensors, reference


def predict_case(
    model: torch.nn.Module,
    case: InferenceCase,
    output_path: str | Path,
    device: torch.device | str,
    normalization: str,
    window_size: tuple[int, int, int] = (48, 48, 48),
    overlap: tuple[int, int, int] = (24, 24, 24),
    branch: str = "student",
    compatibility_mode: str = "gaussian",
) -> Path:
    tensors, reference = load_inference_case(case, normalization)
    inference_input: torch.Tensor | dict[str, torch.Tensor]
    inference_input = (
        next(iter(tensors.values())) if branch == "student" else tensors
    )
    probabilities = sliding_window_inference(
        inference_input,
        model,
        window_size,
        overlap,
        device,
        branch,
        compatibility_mode,
    )
    prediction = (
        torch.argmax(probabilities, dim=1)
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(prediction, reference.affine, header), str(destination)
    )
    return destination


def predict_nifti(
    model: torch.nn.Module,
    input_path: str | Path,
    output_path: str | Path,
    device: torch.device | str,
    window_size: tuple[int, int, int] = (48, 48, 48),
    overlap: tuple[int, int, int] = (24, 24, 24),
    branch: str = "student",
    normalization: str = "nonzero_zscore",
    compatibility_mode: str = "gaussian",
) -> Path:
    source = Path(input_path)
    case = InferenceCase(
        source.name.removesuffix(".nii.gz"), {"image": source}, source
    )
    return predict_case(
        model,
        case,
        output_path,
        device,
        normalization,
        window_size,
        overlap,
        branch,
        compatibility_mode,
    )
