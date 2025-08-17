#!/bin/bash
#SBATCH --job-name=rin
#SBATCH --output=/dss/dsshome1/0D/di38teq/Documents/slurm/logs/slurm-%j.out
#SBATCH --cpus-per-task=8
#SBATCH --clusters=hpda2
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00

echo "Starting the job"
echo "Job ID: $SLURM_JOB_ID"
srun python train.py --config configs/cifar.yaml --wandb_resume cvldx4ia --resume_checkpoint /dss/dsstbyfs02/pn52ko/pn52ko-dss-0000/tum/results/cifar/masked/cifar_flex_nested_comp_weights_20250815_181925/model-step=21000.ckpt
echo "Job finished"
