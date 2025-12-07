#!/bin/bash
#SBATCH --job-name=rin
#SBATCH --output=/home/stud/ljul/Documents/slurm/logs/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=48G
#SBATCH --time=4-00:00:00

echo "Starting the job"
echo "Job ID: $SLURM_JOB_ID"
# srun python train_cifar10.py
# srun python train_lightning.py --config configs/cifar.yaml --resume_checkpoint /home/stud/ljul/storage/user/cifar/rin_cifar_flex_cfg_300k_20251130_022139/last.ckpt --wandb_resume 4kx51qx8
# srun python train_lightning.py --config configs/cifar.yaml --resume_checkpoint /home/stud/ljul/storage/user/cifar/rin_cifar_flex_cfg_300k_sigmoid_0.7_0.0_8_20251130_043240/last.ckpt --wandb_resume fwh7d6ii
# srun python train_lightning.py --config configs/cifar.yaml --resume_checkpoint /home/stud/ljul/storage/user/cifar/rin_cifar_flex_cfg_300k_const_0.3_20251130_041341/last.ckpt --wandb_resume p004xym9
# srun python train_lightning.py --config configs/cifar.yaml --resume_checkpoint /home/stud/ljul/storage/user/cifar/rin_cifar_flex_cfg_300k_linear_0.7_0.0_20251130_041540/last.ckpt --wandb_resume ibn2pcuv
# srun python train_lightning.py --config configs/cifar.yaml --resume_checkpoint /home/stud/ljul/storage/user/cifar/rin_cifar_flex_cfg_300k_sigmoid_0.7_0.0_8_20251201_154207/last.ckpt --wandb_resume fwh7d6ii
srun python train_lightning.py --config configs/cifar.yaml
echo "Job finished"
