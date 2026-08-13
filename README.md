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
legacy/source_drop/         untouched archived experiment directories (Git-ignored)
legacy/MANIFEST.sha256      archive integrity hashes
tests/                      unit and integration tests
```

The two original `KD-xia*` directories were moved intact under `legacy/source_drop/`.
They are not imported by active code.

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

The smoke test loads the real checkpoint, checks legacy/student logits, runs one training and
validation step, performs complete sliding-window inference on two synthetic NIfTI cases,
and writes `smoke_test_metrics.json`. A full 200-epoch TopCoW training run is intentionally
not part of the Apple M3 smoke test.
