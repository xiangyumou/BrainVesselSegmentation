#!/usr/bin/env python3
"""Rigid and affine registration of TopCoW CTA volumes to the MRA grid."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk


DATASET_NAME = "Dataset001_BrainVesselSegmentation"
SEED = 42


def geometry(image: sitk.Image) -> dict[str, list[int] | list[float]]:
    return {
        "size": list(image.GetSize()),
        "spacing": list(image.GetSpacing()),
        "origin": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
    }


def same_geometry(left: sitk.Image, right: sitk.Image) -> bool:
    return (
        left.GetSize() == right.GetSize()
        and left.GetSpacing() == right.GetSpacing()
        and left.GetOrigin() == right.GetOrigin()
        and left.GetDirection() == right.GetDirection()
    )


def normalize_nonzero(image: sitk.Image) -> sitk.Image:
    values = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    finite = np.isfinite(values)
    mask = finite & (values != 0)
    result = np.zeros(values.shape, dtype=np.float32)
    if np.count_nonzero(mask) < 2:
        raise ValueError("Image has fewer than two finite non-zero voxels")
    low, high = np.percentile(values[mask], [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("Image non-zero robust intensity range is invalid")
    result[mask] = np.clip((values[mask] - low) / (high - low), 0.0, 1.0)
    normalized = sitk.GetImageFromArray(result)
    normalized.CopyInformation(image)
    return normalized


def normalized_mutual_information(fixed: sitk.Image, moving_on_fixed: sitk.Image) -> float:
    x = sitk.GetArrayViewFromImage(fixed).ravel()
    y = sitk.GetArrayViewFromImage(moving_on_fixed).ravel()
    mask = np.isfinite(x) & np.isfinite(y) & ((x != 0) | (y != 0))
    if np.count_nonzero(mask) < 2:
        return float("nan")
    histogram, _, _ = np.histogram2d(x[mask], y[mask], bins=64, range=((0, 1), (0, 1)))
    total = histogram.sum()
    if total <= 0:
        return float("nan")
    joint = histogram / total
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    entropy = lambda p: float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    joint_entropy = entropy(joint.ravel())
    return (entropy(px) + entropy(py)) / joint_entropy if joint_entropy > 0 else float("nan")


def registration_method(
    fixed: sitk.Image, iterations: int, learning_rate: float
) -> tuple[sitk.ImageRegistrationMethod, float]:
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    voxel_count = int(np.prod(fixed.GetSize()))
    sampling_percentage = min(0.2, max(0.02, 100_000 / voxel_count))
    method.SetMetricSamplingPercentage(sampling_percentage, SEED)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsGradientDescent(
        learningRate=learning_rate,
        numberOfIterations=iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return method, sampling_percentage


def estimate_transform(fixed: sitk.Image, moving: sitk.Image) -> tuple[sitk.Transform, dict[str, Any]]:
    initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    rigid_method, sampling_percentage = registration_method(fixed, 150, 0.5)
    rigid_method.SetInitialTransform(initial, inPlace=False)
    rigid = rigid_method.Execute(fixed, moving)
    rigid_metric = float(rigid_method.GetMetricValue())
    if not math.isfinite(rigid_metric):
        raise RuntimeError("Rigid registration produced a non-finite metric")

    rigid_base = rigid.GetBackTransform() if isinstance(rigid, sitk.CompositeTransform) else rigid
    affine_initial = sitk.AffineTransform(3)
    affine_initial.SetCenter(rigid_base.GetCenter())
    affine_initial.SetMatrix(rigid_base.GetMatrix())
    affine_initial.SetTranslation(rigid_base.GetTranslation())
    affine_method, _ = registration_method(fixed, 200, 0.1)
    affine_method.SetInitialTransform(affine_initial, inPlace=False)
    affine = affine_method.Execute(fixed, moving)
    affine_metric = float(affine_method.GetMetricValue())
    if not math.isfinite(affine_metric):
        raise RuntimeError("Affine registration produced a non-finite metric")
    details = {
        "seed": SEED,
        "metric": "MattesMutualInformation",
        "sampling_percentage": sampling_percentage,
        "shrink_factors": [4, 2, 1],
        "smoothing_sigmas_mm": [2, 1, 0],
        "rigid": {
            "metric_value": rigid_metric,
            "iterations": rigid_method.GetOptimizerIteration(),
            "stop_condition": rigid_method.GetOptimizerStopConditionDescription(),
        },
        "affine": {
            "metric_value": affine_metric,
            "iterations": affine_method.GetOptimizerIteration(),
            "stop_condition": affine_method.GetOptimizerStopConditionDescription(),
        },
        "final_parameters": list(affine.GetParameters()),
        "final_fixed_parameters": list(affine.GetFixedParameters()),
    }
    return affine, details


def atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = "".join(path.suffixes) or ".tmp"
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=suffix, dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json(path: Path, value: object) -> None:
    def writer(temporary: Path) -> None:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    atomic_write(path, writer)


def write_nifti_with_reference_geometry(
    path: Path, image: sitk.Image, reference_path: Path
) -> None:
    """Write SITK voxel values while retaining the reference NIfTI transforms exactly."""
    reference = nib.load(str(reference_path))
    values_xyz = np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0))
    header = reference.header.copy()
    header.set_data_dtype(values_xyz.dtype)
    header.set_data_shape(values_xyz.shape)
    header.set_slope_inter(1.0, 0.0)
    output = nib.Nifti1Image(values_xyz, reference.affine, header=header)
    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)
    output.set_qform(qform, int(qcode))
    output.set_sform(sform, int(scode))
    atomic_write(path, lambda temporary: nib.save(output, str(temporary)))


def checkerboard_png(fixed: sitk.Image, moving: sitk.Image, path: Path) -> None:
    fixed_array = sitk.GetArrayViewFromImage(fixed)
    moving_array = sitk.GetArrayViewFromImage(moving)
    views = [
        (fixed_array[fixed_array.shape[0] // 2], moving_array[moving_array.shape[0] // 2], "Axial"),
        (fixed_array[:, fixed_array.shape[1] // 2, :], moving_array[:, moving_array.shape[1] // 2, :], "Coronal"),
        (fixed_array[:, :, fixed_array.shape[2] // 2], moving_array[:, :, moving_array.shape[2] // 2], "Sagittal"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, (base, overlay, title) in zip(axes, views):
        board = np.zeros((*base.shape, 3), dtype=np.float32)
        tile = 24
        yy, xx = np.indices(base.shape)
        choose_moving = ((yy // tile) + (xx // tile)) % 2 == 1
        displayed = np.where(choose_moving, overlay, base)
        board[..., 0] = displayed
        board[..., 1] = displayed
        board[..., 2] = displayed
        axis.imshow(np.flipud(board), vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    atomic_write(path, lambda temporary: figure.savefig(temporary, dpi=140))
    plt.close(figure)


def output_paths(workspace: Path, case_id: str) -> dict[str, Path]:
    base = workspace / "preprocessed" / DATASET_NAME / "cta_registered_to_mra"
    return {
        "image": base / "images" / f"topcow_ct_{case_id}_0000.nii.gz",
        "transform": base / "transforms" / f"topcow_ct_{case_id}_to_mra.tfm",
        "json": base / "qc" / f"topcow_{case_id}.json",
        "png": base / "qc" / f"topcow_{case_id}_checkerboard.png",
        "qc_dir": base / "qc",
    }


def is_complete(paths: dict[str, Path], fixed_path: Path) -> bool:
    required = (paths["image"], paths["transform"], paths["json"], paths["png"])
    if not all(path.is_file() for path in required):
        return False
    try:
        record = json.loads(paths["json"].read_text(encoding="utf-8"))
        return record.get("status") == "success" and same_geometry(
            sitk.ReadImage(str(paths["image"])), sitk.ReadImage(str(fixed_path))
        )
    except Exception:
        return False


def register_case(workspace: Path, case_id: str, overwrite: bool = False) -> dict[str, Any]:
    raw_images = workspace / "raw" / DATASET_NAME / "imagesTr"
    fixed_path = raw_images / f"topcow_mr_{case_id}_0000.nii.gz"
    moving_path = raw_images / f"topcow_ct_{case_id}_0000.nii.gz"
    paths = output_paths(workspace, case_id)
    if not overwrite and is_complete(paths, fixed_path):
        existing = json.loads(paths["json"].read_text(encoding="utf-8"))
        return {**existing, "status": "skipped", "reason": "complete result exists"}

    started = time.monotonic()
    record: dict[str, Any] = {
        "case_id": case_id,
        "status": "failed",
        "fixed_mra": str(fixed_path),
        "moving_cta": str(moving_path),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        if not fixed_path.is_file() or not moving_path.is_file():
            raise FileNotFoundError(f"Missing MRA/CTA pair: fixed={fixed_path.is_file()}, moving={moving_path.is_file()}")
        fixed_original = sitk.ReadImage(str(fixed_path), sitk.sitkFloat32)
        moving_original = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)
        fixed = normalize_nonzero(fixed_original)
        moving = normalize_nonzero(moving_original)
        identity_resampled = sitk.Resample(moving, fixed, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkLinear, 0.0)
        nmi_before = normalized_mutual_information(fixed, identity_resampled)
        transform, optimization = estimate_transform(fixed, moving)
        normalized_registered = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0)
        nmi_after = normalized_mutual_information(fixed, normalized_registered)
        if not math.isfinite(nmi_before) or not math.isfinite(nmi_after):
            raise RuntimeError("Registration QC produced a non-finite NMI")
        if nmi_after <= nmi_before:
            raise RuntimeError(
                f"Registration NMI did not improve: before={nmi_before:.6f}, after={nmi_after:.6f}"
            )
        registered = sitk.Resample(moving_original, fixed_original, transform, sitk.sitkLinear, 0.0, moving_original.GetPixelID())
        if not same_geometry(registered, fixed_original):
            raise RuntimeError("Registered CTA geometry does not exactly match MRA")

        write_nifti_with_reference_geometry(paths["image"], registered, fixed_path)
        persisted = sitk.ReadImage(str(paths["image"]))
        if not same_geometry(persisted, fixed_original):
            raise RuntimeError("Persisted registered CTA geometry does not exactly match MRA")
        atomic_write(paths["transform"], lambda path: sitk.WriteTransform(transform, str(path)))
        checkerboard_png(fixed, normalized_registered, paths["png"])
        record.update(
            {
                "status": "success",
                "elapsed_seconds": time.monotonic() - started,
                "input_geometry": {"mra": geometry(fixed_original), "cta": geometry(moving_original)},
                "output_geometry": geometry(persisted),
                "interpolation": "linear",
                "background_value": 0.0,
                "nmi_before": nmi_before,
                "nmi_after": nmi_after,
                "optimization": optimization,
                "outputs": {key: str(value) for key, value in paths.items() if key != "qc_dir"},
            }
        )
        write_json(paths["json"], record)
        return record
    except Exception as error:
        record.update({"elapsed_seconds": time.monotonic() - started, "error": f"{type(error).__name__}: {error}"})
        write_json(paths["json"], record)
        return record


def discover_case_ids(workspace: Path) -> list[str]:
    directory = workspace / "raw" / DATASET_NAME / "imagesTr"
    prefix, suffix = "topcow_mr_", "_0000.nii.gz"
    return sorted(path.name[len(prefix) : -len(suffix)] for path in directory.glob(f"{prefix}*{suffix}"))


def write_summary(qc_dir: Path, records: list[dict[str, Any]]) -> None:
    persisted: list[dict[str, Any]] = []
    for path in sorted(qc_dir.glob("topcow_*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and "case_id" in value and "status" in value:
                persisted.append(value)
        except (OSError, json.JSONDecodeError):
            continue
    summary = {
        "schema_version": 1,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "total": len(persisted),
        "success": sum(record["status"] == "success" for record in persisted),
        "failed": sum(record["status"] == "failed" for record in persisted),
        "cases": persisted,
        "last_run": {
            "total": len(records),
            "success": sum(record["status"] == "success" for record in records),
            "skipped": sum(record["status"] == "skipped" for record in records),
            "failed": sum(record["status"] == "failed" for record in records),
            "case_ids": [record["case_id"] for record in records],
        },
    }
    write_json(qc_dir / "summary.json", summary)
    fields = ["case_id", "status", "elapsed_seconds", "nmi_before", "nmi_after", "error", "reason"]
    def csv_writer(path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(persisted)
    atomic_write(qc_dir / "summary.csv", csv_writer)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--case-id", help="Three-digit case ID; omit to process every discovered MRA")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    if args.threads < 1:
        print("ERROR: --threads must be positive", file=sys.stderr)
        return 2
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(args.threads)
    case_ids = [args.case_id] if args.case_id else discover_case_ids(workspace)
    if not case_ids:
        print("ERROR: no cases found", file=sys.stderr)
        return 1
    records = []
    for index, case_id in enumerate(case_ids, 1):
        print(f"[{index}/{len(case_ids)}] Registering case {case_id}", flush=True)
        record = register_case(workspace, case_id, args.overwrite)
        records.append(record)
        print(f"  {record['status']}: {record.get('error', record.get('reason', 'done'))}", flush=True)
    qc_dir = output_paths(workspace, case_ids[0])["qc_dir"]
    write_summary(qc_dir, records)
    return 1 if any(record["status"] == "failed" for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
