# nnU-Net baseline

This directory is an independent nnU-Net v2 baseline. It does not replace the
existing BVS training or evaluation code.

The integration reuses the Lingfeng architecture and loss implementations from
`src/bvs`. It does not load public, external, or existing-baseline pretrained
weights:

- the teacher is randomly initialized and trained from scratch;
- the supervised student is randomly initialized and trained independently;
- the KD student is randomly initialized and learns from a frozen teacher
  checkpoint produced by this nnU-Net baseline.

## Install and configure

From the repository root:

```bash
pip install -e .
pip install -e ./nnunet

export PROJECT_ROOT="$(pwd)"
export nnUNet_raw=/home/user/xiangyu/st/LFModel/nnunet/raw
export nnUNet_preprocessed=/home/user/xiangyu/st/LFModel/nnunet/preprocessed
export nnUNet_results=/home/user/xiangyu/st/LFModel/nnunet/results
export nnUNet_extTrainer="$PROJECT_ROOT/nnunet/src/bvs_nnunet/trainers"
```

`nnUNet_extTrainer` exposes these trainer names:

- `nnUNetTrainerBVSTeacher`
- `nnUNetTrainerBVSStudent`
- `nnUNetTrainerBVSKD`

## Prepare Dataset501 and Dataset502

Dataset501 contains MRA only. Dataset502 contains MRA in channel 0 and the
registered CTA in channel 1. The fixed release split is installed as fold 0:
80 training cases and 20 validation cases. The 25 internal-test cases are
written to `imagesTs` and cannot enter training.

```bash
bvs-nnunet-prepare \
  --source-root /home/user/xiangyu/st/LFModel/raw/Dataset001_BrainVesselSegmentation \
  --registered-cta-root /home/user/xiangyu/st/LFModel/preprocessed/Dataset001_BrainVesselSegmentation/cta_registered_to_mra/images \
  --split-file "$PROJECT_ROOT/configs/splits/topcow2024_release_seed42.json"

nnUNetv2_plan_and_preprocess -d 501 502 --verify_dataset_integrity
```

Images are absolute symlinks by default to avoid duplicating the source
volumes. Add `--copy-images` if the nnU-Net workspace must be self-contained.
Labels are always materialized as binary `uint8` NIfTI files.

Dataset502 preparation is blocked unless `qc/summary.json` reports successful
registration for all 125 fixed-split cases and every CTA has the same geometry
as its MRA.

## Train from scratch

Teacher, using MRA and CTA:

```bash
nnUNetv2_train 502 3d_fullres 0 -tr nnUNetTrainerBVSTeacher
```

Independent supervised student, using only MRA and no teacher:

```bash
nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainerBVSStudent
```

Both commands use nnU-Net's native checkpointing. Continue either run by
repeating the same command with `--c`.

## Train the KD student

First train the teacher. Then explicitly select that nnU-Net teacher
checkpoint:

```bash
export BVS_TEACHER_CHECKPOINT="$nnUNet_results/Dataset502_BrainVesselMRACTA/nnUNetTrainerBVSTeacher__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"

nnUNetv2_train 502 3d_fullres 0 -tr nnUNetTrainerBVSKD
```

The teacher is frozen and runs in evaluation mode. The KD student starts from
random initialization; it does not inherit weights from the independently
trained supervised student.

The objective is:

```text
CombinedSegmentationLoss
+ 0.5 * TemperatureKLLoss(temperature=10)
+ 0.5 * MetricContrastiveLoss(margin=1, projection_dim=8)
```

Continue the KD run with the native command:

```bash
nnUNetv2_train 502 3d_fullres 0 -tr nnUNetTrainerBVSKD --c
```

Keep `BVS_TEACHER_CHECKPOINT` set to the same teacher checkpoint when
continuing.

## Export the KD student for MRA-only prediction

Dataset502 metadata expects two input channels. Exporting rewrites the KD
student into a Dataset501 model folder so standard nnU-Net prediction accepts a
single MRA:

```bash
bvs-nnunet-export-student \
  --kd-checkpoint "$nnUNet_results/Dataset502_BrainVesselMRACTA/nnUNetTrainerBVSKD__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth" \
  --dataset501-preprocessed "$nnUNet_preprocessed/Dataset501_BrainVesselMRA" \
  --output-model-folder "$nnUNet_results/Dataset501_BrainVesselMRA/nnUNetTrainerBVSStudentKDExport__nnUNetPlans__3d_fullres"

nnUNetv2_predict \
  -i /path/to/mra_input_folder \
  -o /path/to/predictions \
  -d 501 \
  -c 3d_fullres \
  -tr nnUNetTrainerBVSStudent \
  -f 0 \
  -chk checkpoint_final.pth
```

The exporter verifies exact equality between the KD student's logits and the
exported student's logits on the same MRA tensor. Exported checkpoints are for
inference, not for continuing supervised training.

## Test

```bash
pytest -q tests
pytest -q nnunet/tests
```
