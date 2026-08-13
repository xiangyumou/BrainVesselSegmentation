from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

IMAGE_RE = re.compile(r"^topcow_mr_(?P<id>.+)_0000\.nii\.gz$")
LABEL_RE = re.compile(r"^topcow_mr_(?P<id>.+)\.nii\.gz$")


@dataclass(frozen=True)
class TopCoWCase:
    case_id: str
    image: Path
    label: Path


def topcow_release_root(data_root: str | Path) -> Path:
    root = Path(data_root).expanduser().resolve()
    candidate = root / "TopCoW2024_Data_Release"
    return candidate if candidate.exists() else root


def _indexed_files(directory: Path, pattern: re.Pattern[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Required directory does not exist: {directory}")
    result: dict[str, Path] = {}
    for path in sorted(directory.glob("*.nii.gz")):
        match = pattern.match(path.name)
        if not match:
            continue
        case_id = match.group("id")
        if case_id in result:
            raise ValueError(f"Duplicate case ID '{case_id}' in {directory}")
        result[case_id] = path
    return result


def discover_topcow_cases(data_root: str | Path) -> list[TopCoWCase]:
    release = topcow_release_root(data_root)
    images = _indexed_files(release / "imagesTr", IMAGE_RE)
    labels = _indexed_files(release / "cow_seg_labelsTr", LABEL_RE)
    missing_labels = sorted(set(images) - set(labels))
    missing_images = sorted(set(labels) - set(images))
    if missing_labels or missing_images:
        raise ValueError(
            f"TopCoW image/label pairing failed: missing_labels={missing_labels}, "
            f"missing_images={missing_images}"
        )
    return [TopCoWCase(case_id, images[case_id], labels[case_id]) for case_id in sorted(images)]


def binary_label(label: np.ndarray) -> np.ndarray:
    return (np.asarray(label) > 0).astype(np.uint8)


def normalize_mra(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    foreground = image != 0
    result = np.zeros_like(image, dtype=np.float32)
    if foreground.any():
        values = image[foreground]
        standard_deviation = float(values.std())
        if standard_deviation < 1e-8:
            result[foreground] = values - float(values.mean())
        else:
            result[foreground] = (values - float(values.mean())) / standard_deviation
    return result


def validate_case(case: TopCoWCase, affine_tolerance: float = 1e-5) -> dict:
    image = nib.load(str(case.image))
    label = nib.load(str(case.label))
    errors: list[str] = []
    if image.shape != label.shape:
        errors.append(f"shape mismatch: image={image.shape}, label={label.shape}")
    if not np.allclose(image.affine, label.affine, atol=affine_tolerance):
        errors.append("affine mismatch")
    image_orientation = nib.aff2axcodes(image.affine)
    label_orientation = nib.aff2axcodes(label.affine)
    if image_orientation != label_orientation:
        errors.append(
            f"orientation mismatch: image={image_orientation}, label={label_orientation}"
        )
    if errors:
        raise ValueError(f"Case {case.case_id}: " + "; ".join(errors))
    return {
        "case_id": case.case_id,
        "shape": list(image.shape),
        "orientation": list(image_orientation),
        "image": str(case.image),
        "label": str(case.label),
    }


def validate_topcow_dataset(data_root: str | Path, expected_cases: int = 125) -> dict:
    cases = discover_topcow_cases(data_root)
    if expected_cases and len(cases) != expected_cases:
        raise ValueError(f"Expected {expected_cases} paired MRA cases, found {len(cases)}")
    return {"case_count": len(cases), "cases": [validate_case(case) for case in cases]}


def create_fixed_split(
    case_ids: list[str],
    output: str | Path,
    seed: int = 42,
    train_count: int = 80,
    val_count: int = 20,
    test_count: int = 25,
) -> dict:
    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(f"Split file already exists and will not be overwritten: {output_path}")
    ids = sorted(case_ids)
    expected = train_count + val_count + test_count
    if len(ids) != expected or len(ids) != len(set(ids)):
        raise ValueError(f"Expected {expected} unique IDs, received {len(ids)}")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    split = {
        "dataset": "TopCoW2024 MRA binary",
        "seed": seed,
        "train": ids[:train_count],
        "val": ids[train_count : train_count + val_count],
        "internal_test": ids[train_count + val_count :],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    return split


def cases_by_id(cases: list[TopCoWCase]) -> dict[str, TopCoWCase]:
    return {case.case_id: case for case in cases}

