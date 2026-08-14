#!/usr/bin/env bash
#SBATCH --job-name=bvs-topcow-scratch
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/home/user/xiangyu/Projects/BrainVesselSegmentation
#SBATCH --output=/home/user/xiangyu/Projects/BrainVesselSegmentation/dicc/logs/%x_%j.log
#SBATCH --error=/home/user/xiangyu/Projects/BrainVesselSegmentation/dicc/logs/%x_%j.log

set -euo pipefail

project_root=/home/user/xiangyu/Projects/BrainVesselSegmentation
bvs_command=/home/user/xiangyu/.conda/envs/mu/bin/bvs
config_path="$project_root/configs/train/lingfeng_scratch_topcow_binary.yaml"
export BVS_DATA_ROOT="${BVS_DATA_ROOT:-/home/user/xiangyu/st/datasets/TopCoW/TopCoW2024_Data_Release}"
export PYTHONUNBUFFERED=1

[[ -x "$bvs_command" ]] || {
    echo "bvs executable does not exist: $bvs_command" >&2
    exit 1
}
[[ -f "$config_path" ]] || {
    echo "Training config does not exist: $config_path" >&2
    exit 1
}
[[ -d "$BVS_DATA_ROOT" ]] || {
    echo "Dataset root does not exist: $BVS_DATA_ROOT" >&2
    exit 1
}

echo "Job: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Project: $project_root"
echo "Config: $config_path"
echo "Dataset: $BVS_DATA_ROOT"
nvidia-smi

srun "$bvs_command" train \
    --config "$config_path" \
    --c

echo "Finished: $(date --iso-8601=seconds)"
