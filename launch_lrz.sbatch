#!/bin/bash
#SBATCH --job-name=rin
#SBATCH --output=/dss/dsshome1/0D/di38teq/Documents/slurm/logs/slurm-%j.out
#SBATCH --cpus-per-task=8
#SBATCH --clusters=hpda2
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --exclude=hpdar04c05s03
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=6-00:00:00

echo "Starting the job"
echo "Job ID: $SLURM_JOB_ID"
srun python train.py --config configs/cifar.yaml --resume_checkpoint /dss/dsstbyfs02/pn52ko/pn52ko-dss-0000/tum/results/cifar/cifar_flex_ps_2_20251109_155144/last.ckpt --wandb_resume wjde77ag
echo "Job finished"
