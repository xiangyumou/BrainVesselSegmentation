from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.morphology import skeletonize


def segmentation_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.asarray(prediction) > 0
    truth = np.asarray(target) > 0
    tp = float(np.logical_and(pred, truth).sum())
    fp = float(np.logical_and(pred, ~truth).sum())
    fn = float(np.logical_and(~pred, truth).sum())
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    precision = tp / (tp + fp) if (tp + fp) else float(not truth.any())
    recall = tp / (tp + fn) if (tp + fn) else float(not pred.any())
    pred_skeleton = skeletonize(pred)
    truth_skeleton = skeletonize(truth)
    topology_precision = (
        float(truth[pred_skeleton].mean()) if pred_skeleton.any() else float(not truth.any())
    )
    topology_sensitivity = (
        float(pred[truth_skeleton].mean()) if truth_skeleton.any() else float(not pred.any())
    )
    cldice = (
        2 * topology_precision * topology_sensitivity / (topology_precision + topology_sensitivity)
        if topology_precision + topology_sensitivity
        else 0.0
    )
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "cldice": cldice,
    }


def _bootstrap_ci(values: np.ndarray, seed: int = 42, samples: int = 2000) -> list[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, size=(samples, len(values)), replace=True), axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def evaluate_directories(
    predictions: str | Path,
    labels: str | Path,
    output: str | Path,
    *,
    strict: bool = True,
) -> dict:
    prediction_dir, label_dir, output_dir = Path(predictions), Path(labels), Path(output)
    prediction_by_name = {
        path.name: path for path in sorted(prediction_dir.glob("*.nii.gz"))
    }
    label_by_name = {
        path.name: path for path in sorted(label_dir.glob("*.nii.gz"))
    }
    if not prediction_by_name:
        raise FileNotFoundError(f"No .nii.gz predictions found in {prediction_dir}")
    prediction_names = set(prediction_by_name)
    label_names = set(label_by_name)
    missing_predictions = sorted(label_names - prediction_names)
    unexpected_predictions = sorted(prediction_names - label_names)
    if strict and (missing_predictions or unexpected_predictions):
        raise ValueError(
            "Prediction/label case sets do not match: "
            f"missing_predictions={missing_predictions}, "
            f"unexpected_predictions={unexpected_predictions}"
        )
    matched_names = sorted(prediction_names & label_names)
    if not matched_names:
        raise ValueError("Prediction and label directories contain no matching cases")
    rows = []
    for name in matched_names:
        prediction_path = prediction_by_name[name]
        label_path = label_by_name[name]
        prediction_image = nib.load(str(prediction_path))
        label_image = nib.load(str(label_path))
        if prediction_image.shape != label_image.shape:
            raise ValueError(f"Shape mismatch for {prediction_path.name}")
        if not np.allclose(prediction_image.affine, label_image.affine, atol=1e-5):
            raise ValueError(f"Affine mismatch for {prediction_path.name}")
        rows.append(
            {
                "case_id": prediction_path.name.removesuffix(".nii.gz"),
                **segmentation_metrics(prediction_image.get_fdata(), label_image.get_fdata()),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_case_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "case_count": len(rows),
        "label_case_count": len(label_by_name),
        "prediction_case_count": len(prediction_by_name),
        "missing_predictions": missing_predictions,
        "unexpected_predictions": unexpected_predictions,
        "metrics": {},
    }
    for metric in ("dice", "iou", "precision", "recall", "cldice"):
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        summary["metrics"][metric] = {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "bootstrap_95_ci": _bootstrap_ci(values),
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
