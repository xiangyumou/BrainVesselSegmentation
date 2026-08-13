from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .checkpoints import load_lingfeng_student_checkpoint, write_inspection_report
from .config import load_config
from .data.topcow import create_fixed_split, discover_topcow_cases, validate_topcow_dataset
from .devices import select_device
from .evaluation import evaluate_directories
from .inference import predict_nifti
from .models import LingfengMRAStudent, StandardUNet3D
from .smoke import run_smoke_test
from .training.trainer import train_from_config


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _build_prediction_model(config: dict, checkpoint: str, device: torch.device):
    name = config["model"]["name"]
    if name == "standard_unet3d":
        model = StandardUNet3D(
            base_channels=int(config["model"].get("base_channels", 32))
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("model", payload), strict=True)
    elif name == "lingfeng_student_transfer":
        model = LingfengMRAStudent()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model", payload)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            load_lingfeng_student_checkpoint(model, checkpoint)
    else:
        raise ValueError(f"Unknown model name: {name}")
    return model.to(device).eval()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bvs")
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    validate = data_commands.add_parser("validate")
    validate.add_argument("--data-root", default=os.environ.get("BVS_DATA_ROOT"))
    validate.add_argument("--expected-cases", type=int, default=125)
    split = data_commands.add_parser("split")
    split.add_argument("--data-root", default=os.environ.get("BVS_DATA_ROOT"))
    split.add_argument("--output", required=True)

    checkpoint = commands.add_parser("checkpoint")
    checkpoint_commands = checkpoint.add_subparsers(dest="checkpoint_command", required=True)
    inspect = checkpoint_commands.add_parser("inspect")
    inspect.add_argument("--checkpoint", required=True)
    inspect.add_argument("--output")

    train = commands.add_parser("train")
    train.add_argument("--config", required=True)

    predict = commands.add_parser("predict")
    predict.add_argument("--config", required=True)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--device", default="auto")

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--labels", required=True)
    evaluate.add_argument("--output", required=True)

    smoke = commands.add_parser("smoke-test")
    smoke.add_argument("--device", default="auto")
    smoke.add_argument(
        "--checkpoint",
        default="artifacts/checkpoints/lingfeng/student_best_checkpoint_multimodaltune9.pt",
    )
    smoke.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "data" and args.data_command == "validate":
        if not args.data_root:
            raise SystemExit("--data-root or BVS_DATA_ROOT is required")
        _print(validate_topcow_dataset(args.data_root, args.expected_cases))
    elif args.command == "data" and args.data_command == "split":
        if not args.data_root:
            raise SystemExit("--data-root or BVS_DATA_ROOT is required")
        cases = discover_topcow_cases(args.data_root)
        _print(create_fixed_split([case.case_id for case in cases], args.output))
    elif args.command == "checkpoint":
        _print(write_inspection_report(args.checkpoint, args.output))
    elif args.command == "train":
        config = load_config(args.config)
        _print({"run_directory": str(train_from_config(config))})
    elif args.command == "predict":
        config = load_config(args.config)
        policy = select_device(args.device)
        model = _build_prediction_model(config, args.checkpoint, policy.device)
        input_path, output_path = Path(args.input), Path(args.output)
        files = [input_path] if input_path.is_file() else sorted(input_path.glob("*.nii.gz"))
        if not files:
            raise FileNotFoundError(f"No NIfTI files found: {input_path}")
        outputs = []
        for source in files:
            output_name = source.name.replace("_0000.nii.gz", ".nii.gz")
            destination = output_path if input_path.is_file() else output_path / output_name
            outputs.append(
                str(
                    predict_nifti(
                        model,
                        source,
                        destination,
                        policy.device,
                        tuple(config["inference"].get("window_size", [48] * 3)),
                        tuple(config["inference"].get("overlap", [24] * 3)),
                    )
                )
            )
        _print({"device": policy.device.type, "predictions": outputs})
    elif args.command == "evaluate":
        _print(evaluate_directories(args.predictions, args.labels, args.output))
    elif args.command == "smoke-test":
        _print(run_smoke_test(args.device, args.checkpoint, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
