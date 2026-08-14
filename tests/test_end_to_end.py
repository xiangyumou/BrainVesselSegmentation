from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from bvs.checkpoints import load_prediction_checkpoint
from bvs.config import load_config
from bvs.evaluation import evaluate_directories
from bvs.inference import discover_inference_cases, predict_case
from bvs.training.trainer import train_from_config


def _write_case(root: Path, case_id: str, offset: int) -> None:
    images = root / "imagesTr"
    labels = root / "cow_seg_labelsTr"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    grid = np.indices((16, 16, 16)).sum(axis=0).astype(np.float32)
    label = np.zeros((16, 16, 16), dtype=np.uint8)
    label[4 + offset : 10 + offset, 5:11, 6:12] = 1
    image = grid / grid.max() + label.astype(np.float32) * 2
    affine = np.diag([0.6, 0.7, 0.8, 1.0])
    nib.save(
        nib.Nifti1Image(image, affine),
        images / f"topcow_mr_{case_id}_0000.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(label, affine),
        labels / f"topcow_mr_{case_id}.nii.gz",
    )


def test_synthetic_train_predict_evaluate(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_case(data_root, "001", 0)
    _write_case(data_root, "002", 1)
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps({"train": ["001"], "val": ["002"], "internal_test": []}),
        encoding="utf-8",
    )
    template = (
        Path(__file__).resolve().parents[1]
        / "configs/train/unet3d_topcow_binary.yaml"
    )
    config = load_config(template)
    config["experiment_name"] = "synthetic"
    config["device"] = "cpu"
    config["output_root"] = str(tmp_path / "runs")
    config["model"]["base_channels"] = 1
    config["data"]["root"] = str(data_root)
    config["data"]["split_file"] = str(split)
    config["data"]["crop_or_pad_size"] = [16, 16, 16]
    config["data"]["patch_size"] = [16, 16, 16]
    config["data"]["samples_per_volume"] = 1
    config["data"]["validation_samples_per_volume"] = 1
    config["data"]["cache_max_cases"] = 1
    config["training"]["epochs"] = 1
    config["training"]["batch_size"] = 1
    config["training"]["gradient_accumulation"] = 1
    config["training"]["early_stopping_patience"] = 1
    config["inference"]["window_size"] = [16, 16, 16]
    config["inference"]["overlap"] = [0, 0, 0]
    config["inference"]["compatibility_mode"] = "gaussian"

    # --continue is also safe for the first invocation: it starts a new run.
    run_dir = train_from_config(config, continue_run=True)
    checkpoint = run_dir / "checkpoints/best.pt"
    history = run_dir / "metrics/history.csv"
    summary = run_dir / "metrics/summary.json"
    environment = run_dir / "environment.json"
    assert (
        checkpoint.is_file()
        and history.is_file()
        and summary.is_file()
        and environment.is_file()
    )
    environment_report = json.loads(environment.read_text(encoding="utf-8"))
    assert environment_report["resume_checkpoint"] is None
    assert environment_report["resume_from_epoch"] == 0
    training_report = json.loads(summary.read_text(encoding="utf-8"))
    assert training_report["start_epoch"] == 0
    assert training_report["epochs_completed"] == 1
    assert training_report["last_epoch"] == 1
    with history.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert all(np.isfinite(float(row[key])) for key in row if key != "is_best")

    continued_run = train_from_config(config, continue_run=True)
    assert continued_run == run_dir
    continued_environment = json.loads(environment.read_text(encoding="utf-8"))
    assert continued_environment["resume_checkpoint"] is None
    assert continued_environment["resume_from_epoch"] == 0
    with history.open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 1

    model = load_prediction_checkpoint(config, checkpoint, "cpu")
    predictions = tmp_path / "predictions"
    cases = discover_inference_cases(
        config, data_root / "imagesTr", branch="student"
    )
    for case in cases:
        assert case.output_name is not None
        destination = predictions / case.output_name
        predict_case(
            model,
            case,
            destination,
            "cpu",
            config["data"]["normalization"],
            (16, 16, 16),
            (0, 0, 0),
            "student",
            "gaussian",
        )
    report = evaluate_directories(
        predictions, data_root / "cow_seg_labelsTr", tmp_path / "evaluation"
    )
    assert report["case_count"] == 2
    assert (tmp_path / "evaluation/per_case_metrics.csv").is_file()
    assert (tmp_path / "evaluation/summary.json").is_file()
    assert all(
        np.isfinite(values["mean"])
        for values in report["metrics"].values()
    )
