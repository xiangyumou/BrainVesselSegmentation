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

Verify the interpreter and installed environment before running experiments:

```bash
python --version
python -m pip check
python -c "import torch, torchio, nibabel; print(torch.__version__, torchio.__version__, nibabel.__version__)"
python -m pip install -e ".[test]"
```

macOS (Apple Silicon, MPS):

```bash
conda env create -f environment.yml
conda activate bvs
pip install -e ".[test]"
bvs smoke-test --device mps
```

Windows or Linux with an NVIDIA driver compatible with CUDA 12.1:

```bash
conda create -n bvs python=3.11 -y
conda activate bvs
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[test]"
bvs smoke-test --device cuda
```

Device selection is CUDA first, then MPS, then CPU. AMP is enabled only on CUDA.
The global default random seed is `42`.

## Data

The project uses an immutable LFModel data workspace. See the bilingual
[data workspace and CTA-to-MRA registration guide](docs/data_workspace_and_registration.md)
for staging, provenance verification, registration, recovery, and QC acceptance commands.

The TopCoW training profiles use this staged release directory:

```text
/home/user/xiangyu/st/LFModel/raw/Dataset001_BrainVesselSegmentation/
├── imagesTr/
│   ├── topcow_mr_{case_id}_0000.nii.gz
│   └── topcow_ct_{case_id}_0000.nii.gz
└── cow_seg_labelsTr/
    ├── topcow_mr_{case_id}.nii.gz
    └── topcow_ct_{case_id}.nii.gz
```

Validate all 125 paired TopCoW MRA cases and generate the fixed 80/20/25 split:

```bash
export BVS_DATA_ROOT=/home/user/xiangyu/st/LFModel/raw/Dataset001_BrainVesselSegmentation
bvs data validate --data-root "$BVS_DATA_ROOT"
bvs data split --data-root "$BVS_DATA_ROOT" \
  --output configs/splits/topcow2024_release_seed42.json
```

An existing split file is never overwritten. The split algorithm sorts patient IDs, shuffles
them with `numpy.random.default_rng(42)`, and slices 80 train, 20 validation, and 25 internal
test cases.

MRA is Z-score normalized over non-zero voxels while background stays zero. Training samples
`48³` patches across the complete original volume, with a 0.7 probability of centering on
vessel foreground and a 0.3 probability of choosing a random image location. Patches that
cross a volume boundary are zero-padded. Labels use
`label > 0`, including the non-contiguous class value 15.

The three Lingfeng legacy reproduction profiles use `precomputed` normalization because
their configured filenames already contain mean/std-normalized volumes. `CropOrPad` applies
only before training patch sampling; whole-volume validation and prediction retain original
shape, affine, and header. Each Dataset worker keeps an independent LRU of two preprocessed
cases by default (`data.cache_max_cases: 2`; use `0` to disable).

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
artifacts/checkpoints/lingfeng-student_best_checkpoint_multimodaltune9.pt
```

Its expected SHA256 is:

```text
ccecc4b52ffa3832ebf2580945b19e71315f2c26c7f0149f6ecd099ca0997a22
```

Inspect the mapping:

```bash
bvs checkpoint inspect --checkpoint artifacts/checkpoints/lingfeng-student_best_checkpoint_multimodaltune9.pt
```

Only `input_mra_encoder`, `mask_de_prs`, and `metric_prs` are loaded into
`LingfengMRAStudent`. Missing required keys, unexpected student keys, and shape mismatches
are fatal. Teacher and T1/T2/PD keys are listed as ignored.

## Training

```bash
bvs train --config configs/train/unet3d_topcow_binary.yaml
bvs train --config configs/train/lingfeng_transfer_topcow_binary.yaml
bvs train --config configs/train/lingfeng_scratch_topcow_binary.yaml
```

To continue the newest run whose complete resolved configuration is identical, use:

```bash
bvs train --config configs/train/lingfeng_transfer_topcow_binary.yaml --c
```

`--c` (also available as `-c` or `--continue`) restores `latest.pt` in place and appends
the existing CSV and TensorBoard history. Only a run with an identical resolved configuration
is eligible. If no eligible `latest.pt` exists, `--c` safely starts a new timestamped run from
the configured initialization. For the Lingfeng transfer profile, that means a new fine-tune
from the archived pretrained weights rather than random model initialization. Without `--c`,
training always creates a new timestamped run and never overwrites an earlier run. A manual
`training.resume_checkpoint` remains available for starting a new run from an explicitly
selected checkpoint and cannot be combined with `--c`.

### DICC Slurm

Submit the Lingfeng TopCoW fine-tuning job from the repository root:

```bash
sbatch dicc/scripts/train_lingfeng_transfer_topcow.sh
```

Submit the architecture-matched random-initialization experiment:

```bash
sbatch dicc/scripts/train_lingfeng_scratch_topcow.sh
```

The job requests one A100 GPU, 4 CPU cores, and 32 GB RAM, while leaving the time limit to
the cluster default. It uses the `mu` Conda environment directly, sets the staged LFModel
raw copy as `BVS_DATA_ROOT`, and always passes
`--c`: a compatible interrupted run resumes in place, while the first submission starts a
new run using the selected configuration. Slurm stdout and stderr are written under `dicc/logs/`.
Both streams are merged into one `bvs-topcow-ft_<job-id>.log` or
`bvs-topcow-scratch_<job-id>.log` file so initialization, progress, warnings, and errors
remain in chronological order.

The fine-tune and scratch configurations use the same Lingfeng MRA student architecture,
split, preprocessing, loss, optimizer, and 20-epoch schedule. Early stopping is disabled in
these two profiles so both complete all 20 epochs. The transfer profile loads the archived
checkpoint and fine-tunes the complete model; the scratch profile uses random initialization.
Other profiles retain their configured early-stopping behavior. The transfer profile also
supports `model.freeze_encoder: true`; when enabled, only the student decoder and metric head
are optimized.

Training logs initialization, device selection, data and model discovery, batch progress at
5% intervals, the current learning rate, every full-volume validation case, epoch metrics,
checkpoint writes, resume state, and early stopping to the terminal in real time. Batch loss
is explicitly reported as the running mean for the current epoch. The final JSON result
remains on standard output so it can still be consumed by scripts.

Starting a new epoch does not reset the model, optimizer, or scheduler. Each epoch uses a
different deterministic patch sequence and batch order. Its first running loss reflects only
the first few newly sampled patches, while the preceding epoch's final loss averages all
batches, so a temporary increase at the boundary is expected. Compare complete-epoch
`train_loss`, `val_loss`, and `val_dice` values when assessing the two experiments.

`best.pt` and early stopping use a deterministic lexicographic rule: mean validation Dice,
then mean clDice when Dice differs by at most `1e-6`, then validation loss when both metrics
are tied. `history.csv` contains training loss components, whole-volume Dice/IoU/precision/
recall/clDice, learning rate, and `is_best`. Checkpoints and `summary.json` record the best
epoch, loss, Dice, clDice, and `selection_metric: mean_dice`. Resuming restores those values
and the consecutive stale-epoch count.

Each run writes:

```text
runs/{experiment_name}/{timestamp}/
├── resolved_config.yaml
├── checkpoints/{best.pt,latest.pt}
├── metrics/{history.csv,summary.json}
└── tensorboard/
```

## Prediction and evaluation

Prediction uses the configured sliding window and writes the original NIfTI shape, affine,
and copied header. `compatibility_mode: gaussian` uses Gaussian blending;
`compatibility_mode: torchio` uses TorchIO `GridSampler`/`GridAggregator` with average
overlap. Every overlap value must satisfy `0 <= overlap < window_size`.

For `data.adapter: pattern_directory`, modality and label discovery is driven by each
configured `directory` and `pattern`. A `{case_id}` placeholder is required. Passing either
the dataset root or the configured student-modality directory filters out files belonging to
other modalities, and batch prediction names outputs using `data.label.pattern`.

For example, the transfer profile selects MRA from a directory that also contains CTA:

```yaml
data:
  adapter: pattern_directory
  modalities:
    mra:
      directory: imagesTr
      pattern: topcow_mr_{case_id}_0000.nii.gz
  label:
    directory: cow_seg_labelsTr
    pattern: topcow_mr_{case_id}.nii.gz
```

The directory and filename patterns are dataset-defined; they do not need to use TopCoW
names. Training, prediction, and config-driven evaluation share the same discovery rules.

```bash
bvs predict --config configs/train/unet3d_topcow_binary.yaml \
  --checkpoint runs/.../checkpoints/best.pt \
  --input /path/to/dataset_root --output predictions/internal

bvs evaluate --config configs/train/unet3d_topcow_binary.yaml \
  --predictions predictions/internal --data-root /path/to/dataset_root \
  --output reports/internal
```

Evaluation requires prediction and label filenames to match exactly by default.
Use `--allow-partial` only when intentionally evaluating their intersection;
the report will list missing and unexpected cases.
The legacy `--labels /path/to/labels` evaluation form remains available when no dataset
configuration is needed.

### Pretrained MRA zero-shot inference

The archived Lingfeng student checkpoint can be evaluated on all configured TopCoW MRA
cases without creating a filtered directory:

```bash
mkdir -p predictions/pretrained_zero_shot

bvs predict \
  --config configs/train/lingfeng_transfer_topcow_binary.yaml \
  --checkpoint artifacts/checkpoints/lingfeng-student_best_checkpoint_multimodaltune9.pt \
  --input "$BVS_DATA_ROOT" \
  --output predictions/pretrained_zero_shot \
  --device cuda

bvs evaluate \
  --config configs/train/lingfeng_transfer_topcow_binary.yaml \
  --predictions predictions/pretrained_zero_shot \
  --data-root "$BVS_DATA_ROOT" \
  --output reports/pretrained_zero_shot
```

With the release above, discovery selects the 125 `topcow_mr_*` inputs and ignores the 125
`topcow_ct_*` inputs. Each prediction is saved separately using the configured label name,
for example `topcow_mr_055.nii.gz`.

Student prediction accepts one NIfTI, a configured modality directory, or a dataset root.
Teacher prediction requires either a Lingfeng case directory containing every configured
modality, a root whose direct child directories are complete cases, or a pattern-directory
dataset root:

```bash
bvs predict \
  --config configs/reproduction/lingfeng_teacher_legacy_code.yaml \
  --checkpoint runs/.../checkpoints/best.pt \
  --input /path/to/case_directories \
  --output predictions/teacher

bvs predict \
  --config configs/experiments/topcow_mra_cta_teacher.yaml \
  --checkpoint runs/.../checkpoints/best.pt \
  --input "$BVS_DATA_ROOT" \
  --output predictions/topcow_teacher
```

Run evaluation separately for internal test, IXI, and Lausanne. Reports include per-case
foreground Dice, IoU, precision, recall, clDice, and aggregate mean, sample standard
deviation, and bootstrap 95% confidence interval.

## Verification

```bash
python -m compileall src tests
pytest -q
bvs smoke-test --device cpu \
  --config configs/reproduction/lingfeng_student_kd_legacy_code.yaml
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
  --source artifacts/checkpoints/lingfeng-student_best_checkpoint_multimodaltune9.pt \
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
splits. For TopCoW, set `data.root` or `BVS_DATA_ROOT`. Full training cannot start until one
of those roots is configured. A KD run requires a
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
For unified Lingfeng checkpoints, the complete architecture is reconstructed from
the checkpoint model specification so a converted four-modality checkpoint can
be used with the single-modality student inference configuration above. The
student modality, input channels, class count, and base channel count must still
match exactly.

Converted legacy checkpoints are intended for prediction or pretrained
initialization. They are not complete training-resume checkpoints because the
legacy files do not contain every optimizer, scheduler, random, and projection
state required by `training.resume_checkpoint`.

Valid high-level configuration fields are `model`, `data`, `training`, `loss`, and
`inference`; unknown fields fail during loading. Removed no-op fields include
`data.sampler`, `data.queue_length`, `data.patch_overlap`, `data.test_root`, and
`model.out_channels`.

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
