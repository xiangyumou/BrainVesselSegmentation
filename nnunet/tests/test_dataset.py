from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from bvs_nnunet.dataset import DATASET_501, DATASET_502, prepare_datasets


def _write_nifti(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((4, 4, 4), value, dtype=np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def _workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    cta = tmp_path / "registered" / "images"
    split: dict[str, object] = {
        "dataset": "test",
        "seed": 42,
        "train": [],
        "val": [],
        "internal_test": [],
    }
    case_ids = [f"{index:03d}" for index in range(1, 126)]
    split["train"] = case_ids[:80]
    split["val"] = case_ids[80:100]
    split["internal_test"] = case_ids[100:]
    for index, case_id in enumerate(case_ids, 1):
        _write_nifti(
            source / "imagesTr" / f"topcow_mr_{case_id}_0000.nii.gz",
            float(index),
        )
        label_path = source / "cow_seg_labelsTr" / f"topcow_mr_{case_id}.nii.gz"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label = np.zeros((4, 4, 4), dtype=np.int16)
        label[0, 0, 0] = 3
        nib.save(nib.Nifti1Image(label, np.eye(4)), str(label_path))
        _write_nifti(cta / f"topcow_ct_{case_id}_0000.nii.gz", float(index + 1))
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    qc = {
        "failed": 0,
        "cases": [
            {"case_id": case_id, "status": "success"} for case_id in case_ids
        ],
    }
    qc_path = cta.parent / "qc" / "summary.json"
    qc_path.parent.mkdir(parents=True)
    qc_path.write_text(json.dumps(qc), encoding="utf-8")
    return source, cta, split_path


def test_prepare_datasets_channels_labels_split_and_test_exclusion(
    tmp_path: Path,
) -> None:
    source, cta, split = _workspace(tmp_path)
    raw = tmp_path / "nnunet_raw"
    preprocessed = tmp_path / "nnunet_preprocessed"
    prepare_datasets(source, cta, split, raw, preprocessed)

    dataset501 = raw / DATASET_501
    dataset502 = raw / DATASET_502
    assert len(list((dataset501 / "imagesTr").glob("*_0000.nii.gz"))) == 100
    assert not list((dataset501 / "imagesTr").glob("*_0001.nii.gz"))
    assert len(list((dataset502 / "imagesTr").glob("*_0000.nii.gz"))) == 100
    assert len(list((dataset502 / "imagesTr").glob("*_0001.nii.gz"))) == 100
    assert len(list((dataset501 / "imagesTs").glob("*_0000.nii.gz"))) == 25
    assert len(list((dataset501 / "labelsTr").glob("*.nii.gz"))) == 100
    label = np.asanyarray(
        nib.load(str(dataset501 / "labelsTr" / "bvs_001.nii.gz")).dataobj
    )
    assert label.dtype == np.uint8
    assert set(np.unique(label)) == {0, 1}

    split501 = json.loads(
        (preprocessed / DATASET_501 / "splits_final.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(split501) == 1
    assert len(split501[0]["train"]) == 80
    assert len(split501[0]["val"]) == 20
    assert not (
        set(split501[0]["train"])
        & {f"bvs_{index:03d}" for index in range(101, 126)}
    )


def test_prepare_dataset502_rejects_incomplete_registration_qc(
    tmp_path: Path,
) -> None:
    source, cta, split = _workspace(tmp_path)
    summary_path = cta.parent / "qc" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["cases"][0]["status"] = "failed"
    summary["failed"] = 1
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    try:
        prepare_datasets(
            source,
            cta,
            split,
            tmp_path / "raw",
            tmp_path / "preprocessed",
        )
    except RuntimeError as error:
        assert "complete successful registration QC" in str(error)
    else:
        raise AssertionError("Incomplete registration QC was accepted")
