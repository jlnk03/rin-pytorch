import torch
from datetime import datetime

import pytorch_lightning as pl
from lightning.pytorch.strategies import DDPStrategy

from rin_pytorch.TrainerLightningCifar import (
    ImageNetDataModule,
    RinLightningModule
)

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

torch.set_float32_matmul_precision('medium')

def main():
    # Create timestamp string
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    config = dict(
        rin=dict(
            num_layers="2,2,2",
            latent_slots=128,
            latent_dim=512,
            latent_mlp_ratio=4,
            latent_num_heads=16,
            tape_dim=256,
            tape_mlp_ratio=2,
            rw_num_heads=8,
            image_height=32,
            image_width=32,
            image_channels=3,
            patch_size=2,
            latent_pos_encoding="learned",
            tape_pos_encoding="learned",
            drop_path=0.1,
            drop_units=0.1,
            drop_att=0.0,
            time_scaling=1000,
            self_cond="latent",
            time_on_latent=True,
            cond_on_latent_n=1,
            cond_tape_writable=False,
            cond_dim=0,
            cond_proj=True,
            cond_decoupled_read=False,
            xattn_enc_ln=False,
            num_classes=1000,
        ),
        diffusion=dict(
            train_schedule="sigmoid@-3,3,0.9",
            inference_schedule="cosine",
            pred_type="eps",
            self_cond="latent",
            loss_type="eps",
            num_classes=1000,
        ),
        trainer=dict(
            num_classes=1000,
            train_num_steps=150_000,
            train_batch_size=64,
            split_batches=True,
            fp16=False,
            amp=False,
            lr_scheduler_name="cosine",
            lr=3e-3,
            lr_warmup_steps=10_000,
            optimizer_name="lamb",
            optimizer_exclude_weight_decay=["bias", "beta", "gamma"],
            optimizer_kwargs=dict(weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8),
            clip_grad_norm=None,
            sample_every=1000,
            num_dl_workers=4,
            ema_decay=0.9999,
            ema_update_every=1,
            sampling_kwargs=dict(iterations=100, method="ddim"),
            checkpoint_folder=f"results/cifar10/{timestamp}",
            run_name=f"rin_cifar10_full",
            log_to_wandb=True,
        ),
    )
    
    data_module = ImageNetDataModule(config)
    
    model = RinLightningModule(config)
    
    # Initialize WandB logger
    wandb_logger = WandbLogger(
        project="rin",
        name=config["trainer"]["run_name"],
        log_model=False
    ) if config["trainer"]["log_to_wandb"] else None
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["trainer"]["checkpoint_folder"],
        filename="model-{step}",
        every_n_train_steps=config["trainer"]["sample_every"],
        save_weights_only=True,
        save_top_k=-1,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Configure PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_steps=config["trainer"]["train_num_steps"],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        num_nodes= 2,
        devices=4,
        precision="bf16" if config["trainer"]["fp16"] else "32",
        gradient_clip_val=config["trainer"]["clip_grad_norm"],
        strategy='ddp_find_unused_parameters_true' if torch.cuda.device_count() > 1 else "auto",
        accumulate_grad_batches=1,
    )
    
    # Start training
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()