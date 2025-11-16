import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import yaml

from rin_pytorch.TrainerLightning import RinLightningModule

DEFAULT_IMAGENET_ROOT = "/home/stud/ljul/storage/group/dataset_mirrors/imagenet2012/imagenet2012_download/train"

torch.set_float32_matmul_precision("medium")


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_dataloader(config, data_root: str):
    rin_cfg = config["rin"]
    trainer_cfg = config["trainer"]

    transform = transforms.Compose(
        [
            transforms.Resize((rin_cfg["image_height"], rin_cfg["image_width"])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform)

    return DataLoader(
        dataset,
        batch_size=trainer_cfg["train_batch_size"],
        shuffle=True,
        num_workers=trainer_cfg["num_dl_workers"],
        pin_memory=True,
        persistent_workers=trainer_cfg["num_dl_workers"] > 0,
        drop_last=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Rin Lightning Training Script")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/128.yaml",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to the ImageNet-style folder. Overrides config/default.",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--wandb_resume",
        type=str,
        default=None,
        help="WandB resume flag or run id to resume an existing run.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    data_root = args.data_root or config["trainer"].get("dataset_root", DEFAULT_IMAGENET_ROOT)

    checkpoint_root = Path(config["trainer"]["checkpoint_folder"])
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config["trainer"]["run_name"]
    checkpoint_dir = checkpoint_root / f"{run_name}_{timestamp}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config["trainer"]["checkpoint_folder"] = str(checkpoint_dir)

    train_loader = build_dataloader(config, data_root)

    wandb_logger = (
        WandbLogger(
            project="rin",
            name=run_name,
            log_model=False,
            id=args.wandb_resume,
            resume=True if args.wandb_resume else False,
        )
        if config["trainer"]["log_to_wandb"]
        else None
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=config["trainer"]["checkpoint_folder"],
        filename="model-{step}",
        every_n_train_steps=config["trainer"]["sample_every"],
        save_weights_only=False,
        save_top_k=1,
        save_last=True,
        monitor="loss",
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = torch.cuda.device_count() if accelerator == "gpu" else 1
    devices = max(1, devices)
    accumulate = max(1, config["trainer"].get("gradient_accumulation_steps", 1))
    precision = "bf16" if config["trainer"].get("fp16", False) else "32-true"

    trainer = pl.Trainer(
        max_steps=config["trainer"]["train_num_steps"],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        strategy="ddp_find_unused_parameters_true" if devices > 1 else "auto",
        accumulate_grad_batches=accumulate,
        log_every_n_steps=config["trainer"].get("log_every_n_steps", 50),
    )

    with trainer.init_module():
        model = RinLightningModule(config)

    trainer.fit(model, train_dataloaders=train_loader, ckpt_path=args.resume_checkpoint)


if __name__ == "__main__":
    main()