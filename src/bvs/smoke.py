from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .checkpoints import verify_lingfeng_equivalence
from .data.transforms import load_training_arrays, sample_patch
from .devices import seed_everything, select_device
from .inference import predict_nifti
from .models import LingfengMRAStudent
from .training.losses import CombinedSegmentationLoss


def run_smoke_test(
    requested_device: str = "auto",
    checkpoint: str | Path = "artifacts/checkpoints/lingfeng/student_best_checkpoint_multimodaltune9.pt",
    output: str | Path | None = None,
) -> dict:
    seed_everything(42)
    policy = select_device(requested_device)
    checkpoint_path = Path(checkpoint).resolve()
    equivalence = verify_lingfeng_equivalence(checkpoint_path, policy.device, patch_size=16)
    model = LingfengMRAStudent().to(policy.device)
    from .checkpoints import load_lingfeng_student_checkpoint

    load_lingfeng_student_checkpoint(model, checkpoint_path)
    parent = (
        Path(output).resolve()
        if output
        else (Path.cwd() / "runs" / "smoke-test" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    )
    synthetic_dir = parent / "synthetic"
    checkpoint_dir = parent / "checkpoints"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    affine = np.diag([0.5, 0.5, 0.8, 1.0])
    volume = np.zeros((16, 16, 16), dtype=np.float32)
    volume[3:13, 4:12, 5:11] = np.random.default_rng(42).normal(
        10, 2, size=(10, 8, 6)
    )
    inputs = []
    labels = []
    outputs = []
    for index in range(2):
        input_path = synthetic_dir / f"synthetic_{index}.nii.gz"
        label_path = synthetic_dir / f"synthetic_{index}_label.nii.gz"
        output_path = synthetic_dir / f"synthetic_{index}_prediction.nii.gz"
        nib.save(nib.Nifti1Image(volume + index, affine), str(input_path))
        binary = (volume > 0).astype(np.uint8)
        nib.save(nib.Nifti1Image(binary, affine), str(label_path))
        inputs.append(str(input_path))
        labels.append(str(label_path))
        outputs.append(str(output_path))

    train_image, train_label = load_training_arrays(inputs[0], labels[0])
    image_patch, label_patch = sample_patch(
        train_image,
        train_label,
        (16, 16, 16),
        positive_probability=1.0,
        rng=np.random.default_rng(42),
    )
    image = image_patch.unsqueeze(0).to(policy.device)
    label = label_patch.unsqueeze(0).to(policy.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = CombinedSegmentationLoss()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_loss = loss_function(model(image)["logits"], label)
    train_loss.backward()
    optimizer.step()

    validation_image, validation_label_array = load_training_arrays(inputs[1], labels[1])
    validation_tensor = (
        torch.from_numpy(validation_image).unsqueeze(0).unsqueeze(0).to(policy.device)
    )
    validation_label = (
        torch.from_numpy(validation_label_array).unsqueeze(0).long().to(policy.device)
    )
    model.eval()
    with torch.inference_mode():
        validation_loss = loss_function(model(validation_tensor)["logits"], validation_label)

    for input_name, output_name in zip(inputs, outputs):
        predict_nifti(
            model,
            input_name,
            output_name,
            policy.device,
            window_size=(16, 16, 16),
            overlap=(8, 8, 8),
        )
        predicted = nib.load(output_name)
        if predicted.shape != volume.shape or not np.allclose(predicted.affine, affine):
            raise AssertionError("Synthetic NIfTI round-trip did not preserve geometry")
    smoke_checkpoint = checkpoint_dir / "smoke_checkpoint.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, smoke_checkpoint)
    report = {
        "device": policy.device.type,
        "amp_enabled": policy.amp_enabled,
        "train_loss": float(train_loss.detach().cpu()),
        "validation_loss": float(validation_loss.detach().cpu()),
        "equivalence": equivalence,
        "synthetic_inputs": inputs,
        "synthetic_labels": labels,
        "synthetic_predictions": outputs,
        "checkpoint": str(smoke_checkpoint),
        "checkpoint_created": smoke_checkpoint.exists(),
    }
    destination = parent / "smoke_test_metrics.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["metrics_path"] = str(destination)
    return report
