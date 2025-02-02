import torch
from datetime import datetime
import pytorch_lightning as pl
from lightning.pytorch.strategies import DDPStrategy
from rin_pytorch.TrainerLightning import (
    ImageNetDataModule,
    RinLightningModule
)
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import argparse
import yaml

torch.set_float32_matmul_precision('medium')

def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="Rin Training Script")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to the configuration YAML file.")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to a checkpoint to resume training from.")
    parser.add_argument("--wandb_resume", type=str, default=None,
                        help="Wandb resume flag or run id to resume an existing run.")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Optionally, update the checkpoint folder with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["trainer"]["checkpoint_folder"] = f"{config['trainer']['checkpoint_folder']}{config['trainer']['run_name']}_{timestamp}"
    
    data_module = ImageNetDataModule(config)
    model = RinLightningModule(config)
    
    # Initialize WandB logger if applicable, with resume functionality
    wandb_logger = None
    if config["trainer"]["log_to_wandb"]:
        wandb_logger = WandbLogger(
            project="rin",
            name=config["trainer"]["run_name"],
            log_model=False,
            id=args.wandb_resume,
            resume=True if args.wandb_resume else False
        )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["trainer"]["checkpoint_folder"],
        filename="model-{step}",
        every_n_train_steps=config["trainer"]["sample_every"],
        save_top_k=1,
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(
        max_steps=config["trainer"]["train_num_steps"],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=4,
        precision="bf16" if config["trainer"]["fp16"] else "32",
        gradient_clip_val=config["trainer"]["clip_grad_norm"],
        strategy='ddp_find_unused_parameters_true' if torch.cuda.device_count() > 1 else "auto",
    )
    
    # Pass the resume checkpoint path to trainer.fit() to resume from a given checkpoint
    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_checkpoint)

if __name__ == "__main__":
    main()