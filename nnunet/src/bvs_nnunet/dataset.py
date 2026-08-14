from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from bvs.data.topcow import binary_label, discover_topcow_cases, topcow_release_root

DATASET_501 = "Dataset501_BrainVesselMRA"
DATASET_502 = "Dataset502_BrainVesselMRACTA"
EXPECTED_SPLIT_SIZES = {"train": 80, "val": 20, "internal_test": 25}


def load_split(path: Path) -> dict[str, list[str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, list[str]] = {}
    for key, expected in EXPECTED_SPLIT_SIZES.items():
        ids = [str(item) for item in value.get(key, [])]
        if len(ids) != expected or len(ids) != len(set(ids)):
            raise ValueError(f"Split '{key}' must contain {expected} unique case IDs")
        result[key] = ids
    all_ids = [item for key in EXPECTED_SPLIT_SIZES for item in result[key]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("train, val, and internal_test splits overlap")
    return result


def require_registration_qc(
    registered_cta_root: Path, expected_ids: set[str]
) -> None:
    summary_path = registered_cta_root.parent / "qc" / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Registration QC summary does not exist: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Registration QC summary is unreadable: {summary_path}") from error
    successful_ids = {
        str(record.get("case_id"))
        for record in summary.get("cases", [])
        if record.get("status") == "success"
    }
    failed = int(summary.get("failed", -1))
    if failed != 0 or successful_ids != expected_ids:
        missing = sorted(expected_ids - successful_ids)
        unexpected = sorted(successful_ids - expected_ids)
        raise RuntimeError(
            "Dataset502 requires complete successful registration QC: "
            f"expected={len(expected_ids)}, successful={len(successful_ids)}, "
            f"failed={failed}, missing={missing}, unexpected={unexpected}"
        )


def _link_or_copy(source: Path, destination: Path, copy_images: bool) -> None:
    if copy_images:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def _write_binary_label(source: Path, destination: Path) -> None:
    image = nib.load(str(source))
    data = binary_label(np.asanyarray(image.dataobj))
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    output = nib.Nifti1Image(data, image.affine, header)
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    output.set_qform(qform, int(qcode))
    output.set_sform(sform, int(scode))
    nib.save(output, str(destination))


def _same_geometry(left: Path, right: Path, tolerance: float = 1e-5) -> bool:
    left_image = nib.load(str(left))
    right_image = nib.load(str(right))
    return left_image.shape == right_image.shape and np.allclose(
        left_image.affine, right_image.affine, atol=tolerance
    )


def _dataset_json(
    name: str, channels: dict[str, str], training_count: int
) -> dict[str, Any]:
    return {
        "channel_names": channels,
        "labels": {"background": 0, "vessel": 1},
        "numTraining": training_count,
        "file_ending": ".nii.gz",
        "name": name,
        "description": "TopCoW MRA brain vessel segmentation for the BVS nnU-Net baseline",
        "licence": "See the source TopCoW dataset terms",
        "converted_by": "BrainVesselSegmentation",
    }


def _prepare_one_dataset(
    destination: Path,
    channels: dict[str, str],
    train_ids: list[str],
    test_ids: list[str],
    mra_by_id: dict[str, Path],
    label_by_id: dict[str, Path],
    cta_by_id: dict[str, Path] | None,
    copy_images: bool,
) -> None:
    images_tr = destination / "imagesTr"
    labels_tr = destination / "labelsTr"
    images_ts = destination / "imagesTs"
    images_tr.mkdir(parents=True)
    labels_tr.mkdir()
    images_ts.mkdir()
    for case_id in train_ids:
        _link_or_copy(mra_by_id[case_id], images_tr / f"bvs_{case_id}_0000.nii.gz", copy_images)
        if cta_by_id is not None:
            _link_or_copy(cta_by_id[case_id], images_tr / f"bvs_{case_id}_0001.nii.gz", copy_images)
        _write_binary_label(label_by_id[case_id], labels_tr / f"bvs_{case_id}.nii.gz")
    for case_id in test_ids:
        _link_or_copy(mra_by_id[case_id], images_ts / f"bvs_{case_id}_0000.nii.gz", copy_images)
        if cta_by_id is not None:
            _link_or_copy(cta_by_id[case_id], images_ts / f"bvs_{case_id}_0001.nii.gz", copy_images)
    dataset_json = _dataset_json(destination.name, channels, len(train_ids))
    (destination / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8"
    )


def prepare_datasets(
    source_root: Path,
    registered_cta_root: Path,
    split_file: Path,
    nnunet_raw: Path,
    nnunet_preprocessed: Path,
    *,
    overwrite: bool = False,
    copy_images: bool = False,
) -> tuple[Path, Path]:
    release = topcow_release_root(source_root)
    split = load_split(split_file)
    expected_ids = {
        case_id for key in EXPECTED_SPLIT_SIZES for case_id in split[key]
    }
    discovered = {case.case_id: case for case in discover_topcow_cases(release)}
    if set(discovered) != expected_ids:
        raise ValueError(
            "TopCoW source cases do not match the fixed split: "
            f"missing={sorted(expected_ids - set(discovered))}, "
            f"unexpected={sorted(set(discovered) - expected_ids)}"
        )
    require_registration_qc(registered_cta_root, expected_ids)
    cta_by_id = {
        case_id: registered_cta_root / f"topcow_ct_{case_id}_0000.nii.gz"
        for case_id in expected_ids
    }
    missing_cta = sorted(case_id for case_id, path in cta_by_id.items() if not path.is_file())
    if missing_cta:
        raise FileNotFoundError(f"Registered CTA files are missing for cases: {missing_cta}")
    geometry_errors = sorted(
        case_id
        for case_id, cta_path in cta_by_id.items()
        if not _same_geometry(discovered[case_id].image, cta_path)
    )
    if geometry_errors:
        raise ValueError(
            f"Registered CTA geometry does not match MRA for cases: {geometry_errors}"
        )

    destinations = (nnunet_raw / DATASET_501, nnunet_raw / DATASET_502)
    for destination in destinations:
        if destination.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Dataset already exists (pass --overwrite to replace): {destination}"
                )
            shutil.rmtree(destination)

    train_ids = split["train"] + split["val"]
    test_ids = split["internal_test"]
    mra_by_id = {case_id: case.image for case_id, case in discovered.items()}
    label_by_id = {case_id: case.label for case_id, case in discovered.items()}
    _prepare_one_dataset(
        destinations[0],
        {"0": "MRA"},
        train_ids,
        test_ids,
        mra_by_id,
        label_by_id,
        None,
        copy_images,
    )
    _prepare_one_dataset(
        destinations[1],
        {"0": "MRA", "1": "CTA"},
        train_ids,
        test_ids,
        mra_by_id,
        label_by_id,
        cta_by_id,
        copy_images,
    )

    nnunet_split = [
        {
            "train": [f"bvs_{case_id}" for case_id in split["train"]],
            "val": [f"bvs_{case_id}" for case_id in split["val"]],
        }
    ]
    for dataset_name in (DATASET_501, DATASET_502):
        preprocessed_folder = nnunet_preprocessed / dataset_name
        preprocessed_folder.mkdir(parents=True, exist_ok=True)
        (preprocessed_folder / "splits_final.json").write_text(
            json.dumps(nnunet_split, indent=2) + "\n", encoding="utf-8"
        )
    return destinations


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare independent Dataset501/502 nnU-Net raw datasets."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--registered-cta-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument(
        "--nnunet-raw",
        type=Path,
        default=os.environ.get("nnUNet_raw"),
        required="nnUNet_raw" not in os.environ,
    )
    parser.add_argument(
        "--nnunet-preprocessed",
        type=Path,
        default=os.environ.get("nnUNet_preprocessed"),
        required="nnUNet_preprocessed" not in os.environ,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy image volumes instead of creating absolute symlinks.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    destinations = prepare_datasets(
        args.source_root.expanduser().resolve(),
        args.registered_cta_root.expanduser().resolve(),
        args.split_file.expanduser().resolve(),
        args.nnunet_raw.expanduser().resolve(),
        args.nnunet_preprocessed.expanduser().resolve(),
        overwrite=args.overwrite,
        copy_images=args.copy_images,
    )
    print(f"Prepared {destinations[0]}")
    print(f"Prepared {destinations[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
