#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/stud/ljul/Documents/slurm/logs/slurm-%j.out

module load cuda/12.8.1
MAX_JOBS=4 pip install flash-attn --no-build-isolation
