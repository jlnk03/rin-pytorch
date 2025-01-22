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
#SBATCH --time=4-00:00:00

echo "Starting the job"
echo "Job ID: $SLURM_JOB_ID"
srun python train_cifar10.py
echo "Job finished"
