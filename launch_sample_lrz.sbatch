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
srun python sample.py --checkpoint /dss/dsshome1/0D/di38teq/Documents/rin-pytorch/results/cifar10/20250130_033044/model-latest.pt --num_samples 100 --output_dir /dss/dsshome1/0D/di38teq/Documents/rin-pytorch/output
echo "Job finished"
