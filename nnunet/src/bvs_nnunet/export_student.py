from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from bvs_nnunet.networks import KDNetwork, StudentNetwork


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _extract_student_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "student."
    extracted = {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not extracted:
        raise RuntimeError("KD checkpoint contains no student.* network weights")
    return extracted


def export_student(
    kd_checkpoint: Path,
    dataset501_preprocessed: Path,
    output_model_folder: Path,
    *,
    checkpoint_name: str = "checkpoint_final.pth",
    fold: int = 0,
    overwrite: bool = False,
) -> Path:
    if not kd_checkpoint.is_file():
        raise FileNotFoundError(f"KD checkpoint does not exist: {kd_checkpoint}")
    plans_path = dataset501_preprocessed / "nnUNetPlans.json"
    dataset_json_path = dataset501_preprocessed / "dataset.json"
    fingerprint_path = dataset501_preprocessed / "dataset_fingerprint.json"
    for required in (plans_path, dataset_json_path, fingerprint_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"Dataset501 must be planned and preprocessed before export: {required}"
            )
    if output_model_folder.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output model folder exists (pass --overwrite to replace): {output_model_folder}"
            )
        shutil.rmtree(output_model_folder)

    checkpoint = torch.load(kd_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("KD checkpoint must be a mapping")
    if checkpoint.get("trainer_name") != "nnUNetTrainerBVSKD":
        raise RuntimeError(
            "Input checkpoint must come from nnUNetTrainerBVSKD, got "
            f"{checkpoint.get('trainer_name')!r}"
        )
    state = checkpoint.get("network_weights")
    if not isinstance(state, dict):
        raise RuntimeError("KD checkpoint is missing network_weights")
    student_state = _extract_student_state(state)

    kd = KDNetwork(2, 2)
    kd.load_state_dict(state, strict=True)
    student = StudentNetwork(1, 2)
    student.load_state_dict(student_state, strict=True)
    kd.eval()
    student.eval()
    generator = torch.Generator().manual_seed(42)
    mra = torch.randn((1, 1, 16, 16, 16), generator=generator)
    cta = torch.randn((1, 1, 16, 16, 16), generator=generator)
    with torch.no_grad():
        before = kd(torch.cat((mra, cta), dim=1))
        after = student(mra)
    if not torch.equal(before, after):
        raise RuntimeError("Exported student logits differ from the KD student logits")

    plans = _load_json(plans_path)
    dataset_json = _load_json(dataset_json_path)
    configuration = str(
        checkpoint.get("init_args", {}).get("configuration", "3d_fullres")
    )
    if configuration not in plans.get("configurations", {}):
        raise RuntimeError(
            f"Dataset501 plans do not contain configuration {configuration!r}"
        )

    fold_folder = output_model_folder / f"fold_{fold}"
    fold_folder.mkdir(parents=True)
    shutil.copy2(plans_path, output_model_folder / "plans.json")
    shutil.copy2(dataset_json_path, output_model_folder / "dataset.json")
    shutil.copy2(fingerprint_path, output_model_folder / "dataset_fingerprint.json")
    exported = dict(checkpoint)
    exported["network_weights"] = student_state
    exported["trainer_name"] = "nnUNetTrainerBVSStudent"
    exported["init_args"] = {
        **dict(checkpoint.get("init_args", {})),
        "plans": plans,
        "configuration": configuration,
        "fold": fold,
        "dataset_json": dataset_json,
    }
    # Optimizer state belongs to KDNetwork and is deliberately not resumable as
    # a supervised student training run. The exported folder is for inference.
    exported["optimizer_state"] = {}
    exported["bvs_export"] = {
        "schema_version": 1,
        "source_checkpoint": str(kd_checkpoint.resolve()),
        "purpose": "mra_only_inference",
        "logits_verified_equal": True,
    }
    output_checkpoint = fold_folder / checkpoint_name
    torch.save(exported, output_checkpoint)
    return output_checkpoint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an nnUNetTrainerBVSKD student for MRA-only nnU-Net prediction."
    )
    parser.add_argument("--kd-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset501-preprocessed", type=Path, required=True)
    parser.add_argument("--output-model-folder", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="checkpoint_final.pth")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = export_student(
        args.kd_checkpoint.expanduser().resolve(),
        args.dataset501_preprocessed.expanduser().resolve(),
        args.output_model_folder.expanduser().resolve(),
        checkpoint_name=args.checkpoint_name,
        fold=args.fold,
        overwrite=args.overwrite,
    )
    print(f"Exported MRA-only student checkpoint: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
