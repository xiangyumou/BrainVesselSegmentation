# BrainVesselSegmentation

Maintainable training, inference, and evaluation code for binary vessel segmentation on
TopCoW 2024 MRA. All non-zero CoW anatomy labels are merged into one foreground class.

This is a cross-dataset transfer experiment. The archived Lingfeng work targets whole-brain
TOF-MRA vessel segmentation, while TopCoW labels the Circle of Willis; its reported Dice
scores are therefore not directly comparable with this project.

## Project layout

```text
src/bvs/                    active Python package
configs/train/              baseline and transfer configurations
configs/splits/             immutable generated patient split
artifacts/checkpoints/      local, Git-ignored weights plus tracked manifest
tests/                      unit and integration tests
```

Legacy source drops and model weights are retained locally for conversion and parity checks,
but are not distributed in this public repository. Legacy code is not imported by the active
package.

## Installation

Python 3.11 is required. Docker is not used.

macOS (Apple Silicon, MPS):

```bash
conda env create -f environment.yml
conda activate bvs
pip install -e .
bvs smoke-test --device mps
```

Windows or Linux with an NVIDIA driver compatible with CUDA 12.1:

```bash
conda create -n bvs python=3.11 -y
conda activate bvs
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
bvs smoke-test --device cuda
```

Device selection is CUDA first, then MPS, then CPU. AMP is enabled only on CUDA.
The global default random seed is `42`.

## Data

Set one root containing:

```text
$BVS_DATA_ROOT/
├── TopCoW2024_Data_Release/
│   ├── imagesTr/topcow_mr_{case_id}_0000.nii.gz
│   └── cow_seg_labelsTr/topcow_mr_{case_id}.nii.gz
├── MRA_IXI_HH/
└── MRA_Lausanne/
```

Validate all 125 paired TopCoW MRA cases and generate the fixed 80/20/25 split:

```bash
export BVS_DATA_ROOT=/path/to/data
bvs data validate --data-root "$BVS_DATA_ROOT"
bvs data split --data-root "$BVS_DATA_ROOT" \
  --output configs/splits/topcow_binary_seed42.json
```

An existing split file is never overwritten. The split algorithm sorts patient IDs, shuffles
them with `numpy.random.default_rng(42)`, and slices 80 train, 20 validation, and 25 internal
test cases.

MRA is Z-score normalized over non-zero voxels while background stays zero. Training samples
`48³` patches with a 0.7 probability of centering on vessel foreground. Labels use
`label > 0`, including the non-contiguous class value 15.

The download helper enforces at least 40 GB of free disk space before transfer:

```bash
python scripts/download_topcow.py --destination /data \
  --dataset topcow --url OFFICIAL_ARCHIVE_URL --sha256 EXPECTED_SHA256
```

Use `--md5 EXPECTED_MD5` instead when the official release publishes MD5. URLs and checksums
are explicit arguments so a changed challenge mirror cannot silently alter the downloaded
dataset.

## Checkpoint

Place the archived checkpoint at:

```text
artifacts/checkpoints/lingfeng/student_best_checkpoint_multimodaltune9.pt
```

Its expected SHA256 is:

```text
ccecc4b52ffa3832ebf2580945b19e71315f2c26c7f0149f6ecd099ca0997a22
```

Inspect the mapping:

```bash
bvs checkpoint inspect --checkpoint artifacts/checkpoints/lingfeng/student_best_checkpoint_multimodaltune9.pt
```

Only `input_mra_encoder`, `mask_de_prs`, and `metric_prs` are loaded into
`LingfengMRAStudent`. Missing required keys, unexpected student keys, and shape mismatches
are fatal. Teacher and T1/T2/PD keys are listed as ignored.

## Training

```bash
bvs train --config configs/train/unet3d_topcow_binary.yaml
bvs train --config configs/train/lingfeng_transfer_topcow_binary.yaml
```

Both configurations use Adam, learning rate 0.001, weight decay `1e-5`, StepLR
(`step_size=10`, `gamma=0.8`), CE + foreground Dice loss, batch size 1, gradient
accumulation 4, up to 200 epochs, and early stopping after 20 validation epochs without
improvement. The transfer model fine-tunes every loaded layer.

Each run writes:

```text
runs/{experiment_name}/{timestamp}/
├── resolved_config.yaml
├── checkpoints/{best.pt,latest.pt}
├── metrics/{history.csv,summary.json}
└── tensorboard/
```

## Prediction and evaluation

Prediction uses a `48³` sliding window, `24³` overlap, Gaussian blending, and writes the
original NIfTI shape, affine, and copied header:

```bash
bvs predict --config configs/train/unet3d_topcow_binary.yaml \
  --checkpoint runs/.../checkpoints/best.pt \
  --input /path/to/nifti_or_directory --output predictions/internal

bvs evaluate --predictions predictions/internal \
  --labels /path/to/labels --output reports/internal
```

Run evaluation separately for internal test, IXI, and Lausanne. Reports include per-case
foreground Dice, IoU, precision, recall, clDice, and aggregate mean, sample standard
deviation, and bootstrap 95% confidence interval.

## Verification

```bash
python -m compileall src tests
pytest -q
bvs smoke-test --device auto
```

Tests that require the real Lingfeng checkpoint are skipped when the weight is not installed.
The smoke test itself requires the local checkpoint: it checks legacy/student logits, runs one
training and validation step, performs complete sliding-window inference on two synthetic
NIfTI cases, and writes `smoke_test_metrics.json`. A full 200-epoch TopCoW training run is
intentionally not part of the Apple M3 smoke test.

## Unified Lingfeng reproduction

The active package now provides one configuration-driven path for the Lingfeng
multimodal teacher, MRA student, knowledge distillation, and TopCoW extensions.

### Checkpoint conversion

Convert the archived four-modality teacher once:

```bash
bvs checkpoint convert \
  --source legacy/source_drop/KD-xia-1/tf_dir/teacher_model_multimodal_old/teacher_best_checkpoint_multimodal_tune.pt \
  --config configs/reproduction/lingfeng_teacher_legacy_code.yaml \
  --output artifacts/checkpoints/converted/lingfeng_teacher_4modal_v1.pt \
  --verify
```

Convert the archived KD student checkpoint:

```bash
bvs checkpoint convert \
  --source artifacts/checkpoints/lingfeng/student_best_checkpoint_multimodaltune9.pt \
  --config configs/reproduction/lingfeng_student_kd_legacy_code.yaml \
  --output artifacts/checkpoints/converted/lingfeng_student_kd_4modal_v1.pt \
  --verify
```

Conversion is strict: unknown keys, missing keys, shape mismatches, or CPU
output error above `1e-5` abort before the final file is written.

### Teacher and KD training

All stages use the same command:

```bash
bvs train --config configs/reproduction/lingfeng_teacher_legacy_code.yaml
bvs train --config configs/reproduction/lingfeng_student_kd_legacy_code.yaml
bvs train --config configs/train/unet3d_topcow_binary.yaml
```

Set `data.train_root` and `data.val_root` for separate Lingfeng case-directory
splits. For TopCoW, set `data.root` or `BVS_DATA_ROOT`. A KD run requires a
unified teacher checkpoint and fails if it is missing or incompatible.

Each run writes the resolved YAML, environment metadata, best/latest unified
checkpoints, CSV/JSON metrics, and TensorBoard logs under
`runs/{experiment_name}/{timestamp}`.

### Unified inference and smoke test

```bash
bvs predict \
  --config configs/reproduction/lingfeng_student_eval_legacy_code.yaml \
  --checkpoint artifacts/checkpoints/converted/lingfeng_student_kd_4modal_v1.pt \
  --input INPUT.nii.gz \
  --output PREDICTION.nii.gz

bvs smoke-test \
  --config configs/reproduction/lingfeng_student_kd_legacy_code.yaml \
  --device mps
```

The student inference view accesses only the configured student modality.

The unified implementation preserves the four independent modality encoders, sigmoid modality
attention, per-scale fusion, separate teacher/student decoder and metric layers, legacy Dice
and KD formulas, and configurable teacher-logit and gradient clipping. Exact historical
training trajectories cannot be recovered because legacy global seeding was disabled. The
historical teacher checkpoint also omitted the frozen teacher projection head used during KD;
converted checkpoints record that missing state, while new unified runs save both projection
heads. Original private case IDs, preprocessing provenance, and train/validation splits are
not embedded in the checkpoints, so the reported paper Dice cannot be claimed as reproduced
without those inputs.

Archived teacher and student checkpoints were converted on CPU during development. Their
logits, probabilities, decoder features, and metric features matched the legacy models with
maximum absolute error `0.0` against a required tolerance of `1e-5`. The TopCoW MRA/CTA
profiles are a new paired two-modality extension; they do not reinterpret CTA as T1, T2, or
PD, and require matching shape, affine, and orientation.
