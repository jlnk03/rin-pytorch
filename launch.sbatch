#!/bin/bash
#SBATCH --job-name=rin
#SBATCH --output=/home/stud/ljul/Documents/slurm/logs/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=48G
#SBATCH --nodelist=node[3,8,9,10,11,12,13,14,15,16,17,18]
#SBATCH --time=4-00:00:00

echo "Starting the job"
echo "Job ID: $SLURM_JOB_ID"
srun python train_cifar10.py
echo "Job finished"
