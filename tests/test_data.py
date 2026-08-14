from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from bvs.data.topcow import (
    binary_label,
    create_fixed_split,
    discover_topcow_cases,
    validate_case,
)
from bvs.data.dataset import (
    MultimodalCase,
    MultimodalPatchDataset,
    TopCoWPatchDataset,
    discover_cases,
    discover_lingfeng_cases,
    validate_multimodal_case,
)
from bvs.data.topcow import TopCoWCase
from bvs.data.transforms import preprocess_volume


def _write_case(root: Path, case_id: str, affine: np.ndarray | None = None) -> None:
    affine = np.eye(4) if affine is None else affine
    image_dir = root / "imagesTr"
    label_dir = root / "cow_seg_labelsTr"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.float32), affine),
        image_dir / f"topcow_mr_{case_id}_0000.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.uint8), affine),
        label_dir / f"topcow_mr_{case_id}.nii.gz",
    )


def test_binary_label_maps_all_nonzero_values() -> None:
    source = np.asarray([0, 1, 2, 12, 15])
    assert np.array_equal(binary_label(source), np.asarray([0, 1, 1, 1, 1]))


def test_fixed_split_sizes_cover_unique_ids(tmp_path: Path) -> None:
    output = tmp_path / "split.json"
    split = create_fixed_split([f"{index:03d}" for index in range(1, 126)], output)
    assert [len(split[key]) for key in ("train", "val", "internal_test")] == [80, 20, 25]
    all_ids = split["train"] + split["val"] + split["internal_test"]
    assert len(set(all_ids)) == 125
    with pytest.raises(FileExistsError):
        create_fixed_split([f"{index:03d}" for index in range(1, 126)], output)


def test_missing_label_is_reported(tmp_path: Path) -> None:
    image_dir = tmp_path / "imagesTr"
    (tmp_path / "cow_seg_labelsTr").mkdir()
    image_dir.mkdir()
    nib.save(
        nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4)),
        image_dir / "topcow_mr_001_0000.nii.gz",
    )
    with pytest.raises(ValueError, match="missing_labels"):
        discover_topcow_cases(tmp_path)


def test_affine_mismatch_is_reported(tmp_path: Path) -> None:
    _write_case(tmp_path, "001")
    case = discover_topcow_cases(tmp_path)[0]
    nib.save(
        nib.Nifti1Image(np.ones((8, 8, 8)), np.diag([2, 1, 1, 1])),
        case.label,
    )
    with pytest.raises(ValueError, match="affine mismatch"):
        validate_case(case)


def test_lingfeng_case_discovery_rejects_missing_teacher_modality(tmp_path: Path) -> None:
    case = tmp_path / "case001"
    case.mkdir()
    affine = np.eye(4)
    for filename in ("mra.nii.gz", "cta.nii.gz", "label.nii.gz"):
        nib.save(nib.Nifti1Image(np.ones((8, 8, 8)), affine), case / filename)
    with pytest.raises(FileNotFoundError, match="t1"):
        discover_lingfeng_cases(
            tmp_path,
            {"mra": "mra.nii.gz", "cta": "cta.nii.gz", "t1": "t1.nii.gz"},
            "label.nii.gz",
        )


def test_multimodal_affine_mismatch_is_reported(tmp_path: Path) -> None:
    paths = {}
    for name, affine in (("mra", np.eye(4)), ("cta", np.diag([2, 1, 1, 1]))):
        path = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((8, 8, 8)), affine), path)
        paths[name] = path
    label = tmp_path / "label.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((8, 8, 8)), np.eye(4)), label)
    with pytest.raises(ValueError, match="affine mismatch"):
        validate_multimodal_case(MultimodalCase("case", paths, label))


def test_pattern_directory_training_discovery_is_not_topcow_specific(
    tmp_path: Path,
) -> None:
    image = tmp_path / "scans/subject-a_tof.nii.gz"
    label = tmp_path / "annotations/vessels_subject-a.nii.gz"
    image.parent.mkdir()
    label.parent.mkdir()
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4)), np.eye(4)), image)
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4)), np.eye(4)), label)

    cases = discover_cases(
        "pattern_directory",
        tmp_path,
        {
            "tof": {
                "directory": "scans",
                "pattern": "{case_id}_tof.nii.gz",
            }
        },
        {
            "directory": "annotations",
            "pattern": "vessels_{case_id}.nii.gz",
        },
    )

    assert len(cases) == 1
    assert cases[0].case_id == "subject-a"
    assert cases[0].modalities == {"tof": image}
    assert cases[0].label == label


def test_preprocess_volume_modes() -> None:
    source = np.asarray([[[0.0, 1.0, 3.0]]], dtype=np.float32)
    normalized = preprocess_volume(source, "nonzero_zscore")
    assert normalized[0, 0, 0] == 0
    assert np.allclose(normalized[normalized != 0], [-1, 1])
    assert np.array_equal(preprocess_volume(source, "precomputed"), source)


def test_patch_sampling_is_deterministic_by_epoch(tmp_path: Path) -> None:
    _write_case(tmp_path, "001")
    image_path = tmp_path / "imagesTr/topcow_mr_001_0000.nii.gz"
    gradient = np.arange(8**3, dtype=np.float32).reshape(8, 8, 8) + 1
    nib.save(nib.Nifti1Image(gradient, np.eye(4)), image_path)
    case = TopCoWCase(
        "001",
        image_path,
        tmp_path / "cow_seg_labelsTr/topcow_mr_001.nii.gz",
    )
    dataset = TopCoWPatchDataset(
        [case],
        patch_size=(4, 4, 4),
        positive_probability=0.0,
        samples_per_case=2,
        seed=17,
        cache_max_cases=1,
    )
    first = dataset[0]["image"].clone()
    assert np.array_equal(first.numpy(), dataset[0]["image"].numpy())
    dataset.set_epoch(1)
    changed = dataset[0]["image"]
    assert not np.array_equal(first.numpy(), changed.numpy())


def test_no_center_crop_samples_foreground_from_full_volume_edge(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "mra.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    image = np.zeros((20, 20, 20), dtype=np.float32)
    label = np.zeros_like(image, dtype=np.uint8)
    image[1, 1, 1] = 5.0
    label[1, 1, 1] = 1
    nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
    nib.save(nib.Nifti1Image(label, np.eye(4)), label_path)
    dataset = MultimodalPatchDataset(
        [MultimodalCase("edge", {"mra": image_path}, label_path)],
        modalities=["mra"],
        student_modality="mra",
        patch_size=(6, 6, 6),
        samples_per_volume=1,
        crop_or_pad_size=None,
        positive_probability=1.0,
        seed=42,
        cache_max_cases=1,
    )

    sample = dataset[0]

    assert sample["label"].shape == (6, 6, 6)
    assert sample["student_image"].shape == (1, 6, 6, 6)
    assert sample["label"].sum().item() == 1
    _, full_label, positive_coordinates = dataset._load_case(dataset.cases[0])
    assert full_label.shape == (20, 20, 20)
    assert positive_coordinates.tolist() == [[1, 1, 1]]
