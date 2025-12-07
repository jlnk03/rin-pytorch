#!/bin/bash
#SBATCH --job-name=rin
#SBATCH --output=/home/stud/ljul/Documents/slurm/logs/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00

echo "Starting the job"
echo "Job ID: $SLURM_JOB_ID"
# srun python train_lightning.py --resume_checkpoint /home/stud/ljul/storage/user/imagenet/rin_imagenet_flex_20251116_152918/last.ckpt
srun python train_lightning.py --config configs/cifar.yaml
echo "Job finished"
