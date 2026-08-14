from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .checkpoints import (
    convert_legacy_checkpoint,
    load_prediction_checkpoint,
    write_inspection_report,
)
from .config import load_config
from .data.topcow import create_fixed_split, discover_topcow_cases, validate_topcow_dataset
from .devices import select_device
from .evaluation import evaluate_dataset, evaluate_directories
from .inference import discover_inference_cases, predict_case
from .smoke import run_smoke_test
from .training.trainer import train_from_config


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, default=str))


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
    convert = checkpoint_commands.add_parser("convert")
    convert.add_argument("--source", required=True)
    convert.add_argument("--config", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--verify", action="store_true")

    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument(
        "-c",
        "--c",
        "--continue",
        dest="continue_run",
        action="store_true",
        help="continue the newest run with an identical resolved configuration",
    )

    predict = commands.add_parser("predict")
    predict.add_argument("--config", required=True)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--device", default="auto")

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--predictions", required=True)
    evaluate_source = evaluate.add_mutually_exclusive_group(required=True)
    evaluate_source.add_argument("--labels")
    evaluate_source.add_argument("--data-root")
    evaluate.add_argument("--config")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--allow-partial", action="store_true")

    smoke = commands.add_parser("smoke-test")
    smoke.add_argument("--config")
    smoke.add_argument("--device", default="auto")
    smoke.add_argument(
        "--checkpoint",
        default="artifacts/checkpoints/lingfeng-student_best_checkpoint_multimodaltune9.pt",
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
    elif args.command == "checkpoint" and args.checkpoint_command == "inspect":
        _print(write_inspection_report(args.checkpoint, args.output))
    elif args.command == "checkpoint" and args.checkpoint_command == "convert":
        _print(
            convert_legacy_checkpoint(
                args.source, load_config(args.config), args.output, args.verify
            )
        )
    elif args.command == "train":
        config = load_config(args.config)
        _print(
            {
                "run_directory": str(
                    train_from_config(config, continue_run=args.continue_run)
                )
            }
        )
    elif args.command == "predict":
        config = load_config(args.config)
        policy = select_device(args.device)
        model = load_prediction_checkpoint(
            config, args.checkpoint, policy.device
        )
        input_path, output_path = Path(args.input), Path(args.output)
        branch = str(config["inference"]["branch"])
        cases = discover_inference_cases(config, input_path, branch)
        outputs = []
        for case in cases:
            if input_path.is_file():
                destination = output_path
            elif case.output_name:
                destination = output_path / case.output_name
            elif branch == "teacher":
                destination = output_path / f"{case.case_id}.nii.gz"
            else:
                source_name = case.reference.name
                output_name = source_name.replace("_0000.nii.gz", ".nii.gz")
                destination = output_path / output_name
            outputs.append(
                str(
                    predict_case(
                        model,
                        case,
                        destination,
                        policy.device,
                        str(config["data"]["normalization"]),
                        tuple(config["inference"]["window_size"]),
                        tuple(config["inference"]["overlap"]),
                        branch,
                        str(config["inference"]["compatibility_mode"]),
                    )
                )
            )
        _print({"device": policy.device.type, "predictions": outputs})
    elif args.command == "evaluate":
        if args.data_root:
            if not args.config:
                raise SystemExit("--config is required with --data-root")
            _print(
                evaluate_dataset(
                    args.predictions,
                    args.data_root,
                    load_config(args.config),
                    args.output,
                    strict=not args.allow_partial,
                )
            )
        else:
            _print(
                evaluate_directories(
                    args.predictions,
                    args.labels,
                    args.output,
                    strict=not args.allow_partial,
                )
            )
    elif args.command == "smoke-test":
        _print(
            run_smoke_test(
                args.device, args.checkpoint, args.output, args.config
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
