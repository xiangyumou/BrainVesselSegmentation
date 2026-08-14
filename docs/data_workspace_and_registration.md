# LFModel data workspace and CTA-to-MRA registration

## English

### Purpose and provenance

The TopCoW 2024 release is copied once from
`/home/user/xiangyu/st/datasets/TopCoW/TopCoW2024_Data_Release` into the independent
`/home/user/xiangyu/st/LFModel` workspace. The staging command only reads the release. It
does not change source permissions, names, contents, or timestamps. Each copied file is
verified with SHA256 and recorded with its size in
`manifests/Dataset001_BrainVesselSegmentation_raw.json`. Registration and training only use the staged copy.

```text
/home/user/xiangyu/st/LFModel/
├── raw/Dataset001_BrainVesselSegmentation/
├── manifests/Dataset001_BrainVesselSegmentation_raw.json
└── preprocessed/Dataset001_BrainVesselSegmentation/cta_registered_to_mra/
    ├── images/
    ├── transforms/
    └── qc/
```

Stage and verify the complete release:

```bash
python scripts/stage_topcow_to_lfmodel.py \
  --source /home/user/xiangyu/st/datasets/TopCoW/TopCoW2024_Data_Release \
  --workspace /home/user/xiangyu/st/LFModel
```

The command uses temporary files and atomic renames. An identical destination is skipped,
so an interrupted run can be repeated. A destination with different bytes is rejected;
use `--overwrite` only after investigating and intentionally replacing it.

### Registration

MRA is fixed and CTA is moving. Finite non-zero intensities are clipped to the 1st and 99th
percentiles and scaled to `[0, 1]` only for transform estimation. Geometry-centered Euler3D
rigid registration is followed by affine refinement. Both stages use Mattes mutual
information, deterministic random sampling with seed 42, linear interpolation, and the
`[4, 2, 1]` pyramid. No label is read, no label is transformed, and no deformable transform
is used. The final transform resamples the original CTA intensities with linear interpolation
and background 0 onto the exact MRA size, spacing, origin, and direction.

Run one case or all 125 cases sequentially:

```bash
python scripts/register_topcow_cta_to_mra.py \
  --workspace /home/user/xiangyu/st/LFModel --case-id 001 --threads 4

python scripts/register_topcow_cta_to_mra.py \
  --workspace /home/user/xiangyu/st/LFModel --threads 4
```

A complete case is skipped on rerun. Incomplete or failed cases are retried; `--overwrite`
forces a complete case to be recomputed. Failures such as missing pairs, unreadable NIfTI,
non-finite metrics, interrupted writes, or geometry mismatch are recorded and make the
command exit non-zero.

Each case has a JSON record, a SimpleITK `.tfm`, and an axial/coronal/sagittal checkerboard
PNG. `qc/summary.csv` is convenient for sorting `nmi_before`, `nmi_after`, runtime, and
failures; `qc/summary.json` preserves all persistent case records and describes the most
recent invocation under `last_run`. Inspect every checkerboard
for plausible anatomy and smooth boundary alignment. NMI should generally improve, but it is
not by itself proof of correct anatomy.

Before teacher or KD training, require exactly 125 registered images, `failed: 0` in the
full-run summary, exact output/MRA geometry for every case, and visual acceptance of all QC
images. The MRA fine-tune and scratch profiles can use raw data immediately. Teacher and KD
must wait for full registration acceptance. Set the KD `model.teacher_checkpoint` to the
accepted teacher run's `best.pt` before starting KD. Training outputs remain under `runs/`.

## 中文

### 用途与数据溯源

TopCoW 2024 发布数据只在首次复制时从
`/home/user/xiangyu/st/datasets/TopCoW/TopCoW2024_Data_Release` 读取，并完整复制到独立的
`/home/user/xiangyu/st/LFModel` 工作区。staging 不修改源目录权限、文件名、内容或时间戳。
每个目标文件都用 SHA256 复核，文件大小和哈希写入
`manifests/Dataset001_BrainVesselSegmentation_raw.json`。后续配准和训练只访问工作区副本。

完整复制命令见上文。命令以临时文件加原子 rename 写入；目标内容相同时跳过，所以中断后
可直接重跑。目标内容不同时默认报错，只有确认需要主动替换后才使用 `--overwrite`。

### 配准与验收

配准以 MRA 为 fixed、CTA 为 moving。只为估计变换而对有限非零强度做 1%/99% 百分位裁剪
并缩放至 `[0, 1]`。先做 geometry-centered Euler3D 刚性配准，再做仿射细化；两阶段均使用
Mattes mutual information、随机种子 42、线性插值和 `[4, 2, 1]` 多分辨率金字塔。不读取或
变换任何标签，也不使用非线性形变。最终用原 CTA 强度重采样到 MRA 网格，背景为 0，输出
的 size、spacing、origin、direction 必须与 MRA 完全一致。

上文第一条配准命令只处理 `001`；省略 `--case-id` 后按编号顺序处理全部 125 例。完整结果
重跑时默认跳过，不完整或失败结果会重试，`--overwrite` 强制重算。缺失配对、NIfTI 损坏、
metric 非有限、写入中断或几何不一致都会记录失败，并使批处理返回非零退出码。

每例生成 JSON 指标、`.tfm` 变换文件以及 axial/coronal/sagittal 三视图 checkerboard PNG。
可在 `qc/summary.csv` 中检查配准前后 NMI、耗时和失败项，并逐例查看 checkerboard 的解剖
合理性和边界连续性。NMI 通常应改善，但不能代替人工 QC。

teacher/KD 训练前必须确认：已生成 125 个 CTA、全量 summary 的 `failed` 为 0、每例输出
几何与 MRA 严格相同，并完成人工 QC。fine-tune/scratch 可直接读取 raw；teacher/KD 必须等
全量验收后再启动。KD 启动前还需把配置中的 `model.teacher_checkpoint` 改为已验收 teacher
运行的 `best.pt`。训练产物仍写入项目 `runs/`，不得由数据准备流程修改或删除。
