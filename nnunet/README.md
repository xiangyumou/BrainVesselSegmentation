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

## 中文使用说明

以下命令应在项目根目录执行：

```bash
conda activate bvs
cd /home/user/xiangyu/Projects/BrainVesselSegmentation
```

- `conda activate bvs`：进入项目的 Python 3.11 环境。如果实际环境名是
  `mu`，使用 `conda activate mu`。不要使用 Python 3.12，本项目要求
  Python `>=3.11,<3.12`。
- `cd ...`：进入项目根目录，保证后续 `$PWD` 和相对路径正确。

### 安装

```bash
pip install -e .
pip install -e ./nnunet
```

- `pip install -e .`：以 editable 模式安装主项目，提供 BVS 网络、损失函数和
  数据代码；修改源码后不需要重新安装。
- `pip install -e ./nnunet`：安装本项目的 nnU-Net 扩展，提供数据准备、模型
  导出命令和三个自定义 Trainer。

可以只检查命令是否安装成功，不会启动训练：

```bash
command -v bvs-nnunet-prepare
command -v nnUNetv2_train
```

### 环境变量

```bash
export PROJECT_ROOT="$PWD"
export nnUNet_raw="/home/user/xiangyu/st/LFModel/nnunet/raw"
export nnUNet_preprocessed="/home/user/xiangyu/st/LFModel/nnunet/preprocessed"
export nnUNet_results="/home/user/xiangyu/st/LFModel/nnunet/results"
export nnUNet_extTrainer="$PROJECT_ROOT/nnunet/src/bvs_nnunet/trainers"
```

- `PROJECT_ROOT`：项目根目录。
- `nnUNet_raw`：转换后的 nnU-Net 原始数据目录。
- `nnUNet_preprocessed`：指纹、plans 和预处理数据目录。
- `nnUNet_results`：训练日志与 checkpoint 输出目录。
- `nnUNet_extTrainer`：让 nnU-Net 找到本项目自定义 Trainer 的源码目录。

这些变量只在当前终端有效，新开终端后需要重新执行。

### 准备 Dataset501/502

推荐使用下面的单行命令，避免复制多行命令时漏掉反斜杠：

```bash
bvs-nnunet-prepare --source-root "/home/user/xiangyu/st/LFModel/raw/Dataset001_BrainVesselSegmentation" --registered-cta-root "/home/user/xiangyu/st/LFModel/preprocessed/Dataset001_BrainVesselSegmentation/cta_registered_to_mra/images" --split-file "$PROJECT_ROOT/configs/splits/topcow2024_release_seed42.json" --nnunet-raw "$nnUNet_raw" --nnunet-preprocessed "$nnUNet_preprocessed"
```

该命令验证病例、配准 QC 和 MRA/CTA 几何信息，然后生成：

- Dataset501：只有 MRA；
- Dataset502：channel 0 为 MRA，channel 1 为配准后的 CTA；
- fold 0：80 例训练、20 例验证；
- `imagesTs`：25 例 internal test，不参与训练；
- 二值 `uint8` 标签。图像默认使用绝对符号链接，避免重复占用磁盘。

参数说明：

- `--source-root`：原始 TopCoW 数据目录。
- `--registered-cta-root`：已经配准到 MRA 空间的 CTA 目录。
- `--split-file`：固定数据划分 JSON。
- `--nnunet-raw`：Dataset501/502 输出目录。
- `--nnunet-preprocessed`：写入固定 fold 文件的目录。
- `--copy-images`：可选，复制图像而不是建立符号链接。
- `--overwrite`：可选，删除并重建已有 Dataset501/502，使用前应确认确实要覆盖。

如果使用多行命令，每行末尾的 `\` 后面不能有空格，否则 Bash 会把下一行
当成一条新命令。

### 规划和预处理

```bash
nnUNetv2_plan_and_preprocess -d 501 502 --verify_dataset_integrity
```

- `nnUNetv2_plan_and_preprocess`：分析 spacing、尺寸和强度，生成 plans，并创建
  训练使用的预处理数据。
- `-d 501 502`：同时处理 Dataset501 和 Dataset502。
- `--verify_dataset_integrity`：预处理前检查图像、标签、通道和几何完整性。

有时 SimpleITK 会报告 `Direction mismatch`。本项目现有 100 个训练病例已经检查：
MRA 与标签的 shape、spacing、origin 全部一致，最大 direction 差为
`9.94e-7`，只是 NIfTI 方向矩阵的浮点精度差，可以继续预处理。若未来数据的
差异明显大于 `1e-5`，或 shape/spacing/origin 不一致，则应先运行
下面的命令检查标签与图像是否对齐：

```bash
nnUNetv2_plot_overlay_pngs -d 501 -o /tmp/bvs_dataset501_overlays --use_raw
```

- `-d 501`：检查 Dataset501。
- `-o`：将叠加图写入指定目录。
- `--use_raw`：直接读取 `nnUNet_raw` 中的原始图像和标签，不依赖预处理结果。

### 训练监督 Student

```bash
nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainerBVSStudent
```

- `501`：使用 MRA-only Dataset501。
- `3d_fullres`：使用三维全分辨率 plans。
- `0`：使用固定 fold 0。
- `-tr nnUNetTrainerBVSStudent`：使用本项目的 Lingfeng MRA Student。

这是最适合首先运行的训练任务，因为它不依赖 Teacher checkpoint。

### 训练 Teacher

```bash
nnUNetv2_train 502 3d_fullres 0 -tr nnUNetTrainerBVSTeacher
```

该命令使用 Dataset502 的 MRA+CTA，从随机初始化训练 Teacher。训练结果写入
`$nnUNet_results`。

### 断点续训

在原训练命令最后添加 `--c`：

```bash
nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainerBVSStudent --c
nnUNetv2_train 502 3d_fullres 0 -tr nnUNetTrainerBVSTeacher --c
```

`--c` 从同一结果目录的 `checkpoint_latest.pth` 继续训练。数据集、配置、fold
和 Trainer 名称必须与原任务一致。

### 训练 KD Student

首先指定训练完成的 Teacher checkpoint，并检查文件存在：

```bash
export BVS_TEACHER_CHECKPOINT="$nnUNet_results/Dataset502_BrainVesselMRACTA/nnUNetTrainerBVSTeacher__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"
test -f "$BVS_TEACHER_CHECKPOINT" && echo "Teacher checkpoint found"
```

然后启动 KD 训练：

```bash
nnUNetv2_train 502 3d_fullres 0 -tr nnUNetTrainerBVSKD
```

KD 训练使用冻结的 MRA+CTA Teacher 指导 MRA-only Student。继续训练时仍须保留
相同的 `BVS_TEACHER_CHECKPOINT`：

```bash
nnUNetv2_train 502 3d_fullres 0 -tr nnUNetTrainerBVSKD --c
```

### 导出 KD Student

```bash
bvs-nnunet-export-student \
  --kd-checkpoint "$nnUNet_results/Dataset502_BrainVesselMRACTA/nnUNetTrainerBVSKD__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth" \
  --dataset501-preprocessed "$nnUNet_preprocessed/Dataset501_BrainVesselMRA" \
  --output-model-folder "$nnUNet_results/Dataset501_BrainVesselMRA/nnUNetTrainerBVSStudentKDExport__nnUNetPlans__3d_fullres"
```

- `--kd-checkpoint`：KD 训练得到的 checkpoint。
- `--dataset501-preprocessed`：Dataset501 的 plans 和元数据目录。
- `--output-model-folder`：导出的单 MRA nnU-Net 模型目录。

导出器会去除 Teacher 权重，并验证导出前后的 Student logits 完全一致。导出的
checkpoint 用于推理，不用于继续训练。

### 预测

```bash
nnUNetv2_predict \
  -i /path/to/mra_input_folder \
  -o /path/to/predictions \
  -d 501 \
  -c 3d_fullres \
  -tr nnUNetTrainerBVSStudent \
  -f 0 \
  -chk checkpoint_final.pth
```

- `-i`：输入 MRA 目录，文件名应为 `CASE_0000.nii.gz`。
- `-o`：分割结果目录。
- `-d 501`：使用单通道 Dataset501 元数据。
- `-c 3d_fullres`：使用三维全分辨率配置。
- `-tr`：指定恢复 Student 网络所用的 Trainer。
- `-f 0`：加载 fold 0。
- `-chk`：指定 checkpoint 文件名。

### 测试

```bash
pytest -q tests
pytest -q nnunet/tests
```

- 第一条运行主项目测试。
- 第二条运行 nnU-Net 数据转换、网络、Trainer、KD 和导出测试。
- `-q` 只减少输出，不改变测试内容。

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
