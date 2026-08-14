#!/usr/bin/env bash
#SBATCH --job-name=bvs-topcow-student-kd
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --chdir=/home/user/xiangyu/Projects/BrainVesselSegmentation
#SBATCH --output=/home/user/xiangyu/Projects/BrainVesselSegmentation/dicc/logs/%x_%j.log
#SBATCH --error=/home/user/xiangyu/Projects/BrainVesselSegmentation/dicc/logs/%x_%j.log

set -euo pipefail

project_root=/home/user/xiangyu/Projects/BrainVesselSegmentation
bvs_command=/home/user/xiangyu/.conda/envs/mu/bin/bvs
python_command=/home/user/xiangyu/.conda/envs/mu/bin/python
student_config="$project_root/configs/experiments/topcow_mra_student_kd.yaml"
teacher_config="$project_root/configs/experiments/topcow_mra_cta_teacher.yaml"
config_resolver="$project_root/scripts/prepare_topcow_student_kd_config.py"
export BVS_DATA_ROOT=/home/user/xiangyu/st/LFModel/raw/Dataset001_BrainVesselSegmentation
export PYTHONUNBUFFERED=1

for executable in "$bvs_command" "$python_command"; do
    [[ -x "$executable" ]] || {
        echo "Required executable does not exist: $executable" >&2
        exit 1
    }
done
for required_file in "$student_config" "$teacher_config" "$config_resolver"; do
    [[ -f "$required_file" ]] || {
        echo "Required file does not exist: $required_file" >&2
        exit 1
    }
done
[[ -d "$BVS_DATA_ROOT" ]] || {
    echo "Dataset root does not exist: $BVS_DATA_ROOT" >&2
    exit 1
}

resolved_config=$(mktemp "$project_root/configs/experiments/.topcow_mra_student_kd.XXXXXX.yaml")
trap 'rm -f "$resolved_config"' EXIT
teacher_checkpoint=$(
    "$python_command" "$config_resolver" \
        --student-config "$student_config" \
        --teacher-config "$teacher_config" \
        --output "$resolved_config"
)

echo "Job: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Project: $project_root"
echo "Config: $student_config"
echo "Dataset: $BVS_DATA_ROOT"
echo "Teacher checkpoint: $teacher_checkpoint"
nvidia-smi

srun "$bvs_command" train \
    --config "$resolved_config" \
    --c

echo "Finished: $(date --iso-8601=seconds)"
