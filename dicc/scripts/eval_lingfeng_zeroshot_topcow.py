#!/home/user/xiangyu/.conda/envs/mu/bin/python
#SBATCH --job-name=bvs-lf-zs-topcow
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --chdir=/home/user/xiangyu/Projects/BrainVesselSegmentation
#SBATCH --output=/home/user/xiangyu/Projects/BrainVesselSegmentation/dicc/logs/%x_%j.log
#SBATCH --error=/home/user/xiangyu/Projects/BrainVesselSegmentation/dicc/logs/%x_%j.log

"""Zero-shot evaluation of the legacy Lingfeng student on fixed TopCoW splits."""

from __future__ import annotations

import json
import os
import socket
import sys
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


PROJECT_ROOT = Path("/home/user/xiangyu/Projects/BrainVesselSegmentation")
DATA_ROOT = Path(
    "/home/user/xiangyu/st/datasets/TopCoW/TopCoW2024_Data_Release"
)
SPLIT_FILE = PROJECT_ROOT / "configs/splits/topcow2024_release_seed42.json"
CONFIG_FILE = PROJECT_ROOT / "configs/train/lingfeng_transfer_topcow_binary.yaml"
CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts/checkpoints/lingfeng-student_best_checkpoint_multimodaltune9.pt"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "ccecc4b52ffa3832ebf2580945b19e71315f2c26c7f0149f6ecd099ca0997a22"
)
EXPECTED_SPLIT_SIZES = {"train": 80, "val": 20, "internal_test": 25}

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bvs.checkpoints import load_prediction_checkpoint, sha256_file  # noqa: E402
from bvs.config import load_config  # noqa: E402
from bvs.devices import select_device  # noqa: E402
from bvs.evaluation import _evaluate_indexed  # noqa: E402
from bvs.inference import InferenceCase, predict_case  # noqa: E402


def require_inputs() -> dict[str, list[str]]:
    for path in (DATA_ROOT, SPLIT_FILE, CONFIG_FILE, CHECKPOINT):
        if not path.exists():
            raise FileNotFoundError(f"Required input does not exist: {path}")
    checkpoint_hash = sha256_file(CHECKPOINT)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Legacy checkpoint SHA256 mismatch: "
            f"expected={EXPECTED_CHECKPOINT_SHA256}, actual={checkpoint_hash}"
        )

    payload = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    split = {name: list(payload[name]) for name in EXPECTED_SPLIT_SIZES}
    for name, expected_size in EXPECTED_SPLIT_SIZES.items():
        if len(split[name]) != expected_size or len(set(split[name])) != expected_size:
            raise ValueError(
                f"Split {name!r} must contain {expected_size} unique case IDs"
            )
    all_ids = [case_id for name in EXPECTED_SPLIT_SIZES for case_id in split[name]]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("train, val, and internal_test splits overlap")
    for case_id in all_ids:
        image_path, label_path = case_paths(case_id)
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(
                f"Missing TopCoW MRA image or label for case {case_id}: "
                f"image={image_path}, label={label_path}"
            )
    return split


def case_paths(case_id: str) -> tuple[Path, Path]:
    return (
        DATA_ROOT / "imagesTr" / f"topcow_mr_{case_id}_0000.nii.gz",
        DATA_ROOT / "cow_seg_labelsTr" / f"topcow_mr_{case_id}.nii.gz",
    )


def prediction_is_complete(prediction: Path, reference: Path) -> bool:
    if not prediction.is_file():
        return False
    try:
        prediction_image = nib.load(str(prediction))
        reference_image = nib.load(str(reference))
        return prediction_image.shape == reference_image.shape and np.allclose(
            prediction_image.affine, reference_image.affine, atol=1e-5
        )
    except (OSError, ValueError):
        return False


def main() -> int:
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError(
            "This is a compute job script. Submit it with sbatch; do not run it "
            "directly on the login node."
        )

    split = require_inputs()
    output_root = (
        PROJECT_ROOT
        / "runs/lingfeng_zeroshot_topcow"
        / f"legacy_student_{os.environ['SLURM_JOB_ID']}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    config = load_config(CONFIG_FILE)
    config["inference"] = {
        "branch": "student",
        "window_size": [48, 48, 48],
        "overlap": [4, 4, 4],
        "compatibility_mode": "torchio",
    }
    config["data"]["normalization"] = "nonzero_zscore"
    policy = select_device("auto")
    if policy.device.type != "cuda":
        raise RuntimeError(f"A CUDA allocation is required, got {policy.device}")
    model = load_prediction_checkpoint(config, CHECKPOINT, policy.device)

    metadata = {
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "host": socket.gethostname(),
        "started": datetime.now().astimezone().isoformat(),
        "zero_shot": True,
        "dataset": str(DATA_ROOT),
        "split_file": str(SPLIT_FILE),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "normalization": "nonzero_zscore",
        "window_size": [48, 48, 48],
        "overlap": [4, 4, 4],
        "compatibility_mode": "torchio",
        "device": str(policy.device),
    }
    (output_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    summaries: dict[str, dict] = {}
    for split_name, case_ids in split.items():
        prediction_dir = output_root / split_name / "predictions"
        prediction_dir.mkdir(parents=True)
        for index, case_id in enumerate(case_ids, 1):
            image_path, label_path = case_paths(case_id)
            output_path = prediction_dir / label_path.name
            if prediction_is_complete(output_path, image_path):
                print(
                    f"[{split_name} {index}/{len(case_ids)}] reuse {case_id}",
                    flush=True,
                )
                continue
            print(
                f"[{split_name} {index}/{len(case_ids)}] predict {case_id}",
                flush=True,
            )
            predict_case(
                model,
                InferenceCase(case_id, {"mra": image_path}, image_path),
                output_path,
                policy.device,
                "nonzero_zscore",
                (48, 48, 48),
                (4, 4, 4),
                "student",
                "torchio",
            )

        prediction_by_name = {
            f"topcow_mr_{case_id}.nii.gz":
            prediction_dir / f"topcow_mr_{case_id}.nii.gz"
            for case_id in case_ids
        }
        label_by_name = {
            f"topcow_mr_{case_id}.nii.gz": case_paths(case_id)[1]
            for case_id in case_ids
        }
        summary = _evaluate_indexed(
            prediction_by_name,
            label_by_name,
            output_root / split_name / "metrics",
            strict=True,
        )
        summaries[split_name] = summary
        print(
            f"{split_name}: cases={summary['case_count']}, "
            f"mean_dice={summary['metrics']['dice']['mean']:.6f}",
            flush=True,
        )

    report = {
        "zero_shot": True,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "splits": {
            name: {
                "case_count": summary["case_count"],
                "dice": summary["metrics"]["dice"],
            }
            for name, summary in summaries.items()
        },
    }
    (output_root / "dice_by_split.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Results: {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
