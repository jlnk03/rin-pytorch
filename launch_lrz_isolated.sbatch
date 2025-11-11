#!/bin/bash
#SBATCH --job-name=rin-isolated
#SBATCH --output=/dss/dsshome1/0D/di38teq/Documents/slurm/logs/slurm-isolated-%j.out
#SBATCH --cpus-per-task=8
#SBATCH --clusters=hpda2
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=6-00:00:00

# 1) Source and destination (per-job working copy)
SRC_DIR="/dss/dsshome1/0D/di38teq/Documents/rin-pytorch"
WORKROOT="${SLURM_TMPDIR:-/dss/dsshome1/0D/di38teq/Documents/slurm/$USER/$SLURM_JOB_ID}"
DEST_DIR="$WORKROOT/rin-pytorch"

mkdir -p "$DEST_DIR"

# 2) Copy project (exclude VCS/venv/caches)
rsync -a --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.vscode' \
  "$SRC_DIR/" "$DEST_DIR/"

# 3) Move into isolated copy
cd "$DEST_DIR" || { echo "Failed to cd into $DEST_DIR"; exit 1; }

echo "Starting isolated job in $DEST_DIR"
echo "Job ID: $SLURM_JOB_ID"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

# 4) Run training
# You can override CONFIG and EXTRA_ARGS via: sbatch --export=ALL,CONFIG=...,EXTRA_ARGS="..."
CONFIG_PATH="${CONFIG:-configs/cifar.yaml}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

srun python train.py --config "$CONFIG_PATH" $EXTRA_ARGS

echo "Job finished"


