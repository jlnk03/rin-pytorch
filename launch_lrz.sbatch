#!/bin/bash
#SBATCH --job-name=rin
#SBATCH --output=/dss/dsshome1/0D/di38teq/Documents/slurm/logs/slurm-%j.out
#SBATCH --cpus-per-task=8
#SBATCH --clusters=hpda2
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --exclude=hpdar04c05s03,hpdar04c03s01,hpdar04c05s04
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00

echo "Starting the job"
echo "Job ID: $SLURM_JOB_ID"
# srun python train_lightning.py --config configs/64.yaml --resume_checkpoint /dss/dsstbyfs02/pn52ko/pn52ko-dss-0000/tum/results/imagenet/rin_imagenet_cfg_64_lrz_fast_matmul_300k_20251127_161918/last.ckpt
# srun python train_lightning.py --config configs/64.yaml --resume_checkpoint /dss/dsstbyfs02/pn52ko/pn52ko-dss-0000/tum/results/imagenet/rin_imagenet_flex_cfg_64_lrz_300k_20251207_035527/model-step=51000.ckpt --wandb_resume zyxo6kg8
srun python train_lightning.py --config configs/64.yaml
# python rolling_fid.py \
#     --config configs/64.yaml \
#     --checkpoint /dss/dsstbyfs02/pn52ko/pn52ko-dss-0000/tum/results/imagenet/rin_imagenet_flex_cfg_64_lrz_300k_continue_20251207_041226/last.ckpt \
#     --num_samples 50000 \
#     --dataset imagenet \
#     --split validation
echo "Job finished"
