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

