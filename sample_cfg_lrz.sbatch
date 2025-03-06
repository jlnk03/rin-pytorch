#!/bin/bash
#SBATCH --job-name=rin
#SBATCH --output=/dss/dsshome1/0D/di38teq/Documents/slurm/logs/slurm-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --clusters=hpda2
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=0-01:00:00

echo "Starting the job"
echo "Job ID: $SLURM_JOB_ID"
srun python sample_from_config.py --config configs/default_large.yaml --checkpoint /dss/dsstbyfs02/pn52ko/pn52ko-dss-0000/tum/results/imagenet/imagenet_flex_large_no_grad_clip_20250303_171920/last.ckpt --num_samples 100 --output_dir /dss/dsshome1/0D/di38teq/Documents/rin-pytorch/output/imagenet_flex_large
echo "Job finished"
