import torch
import torchvision
from torch.utils.data import DataLoader, IterableDataset, Dataset
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import webdataset as wds
from huggingface_hub import HfFileSystem, get_token, hf_hub_url
from datasets import load_dataset

from PIL import Image

from rin_pytorch import Rin, RinDiffusionModel
from .utils.optimization_utils import (
    build_parameters_mapping,
    get_optimizer,
    override_config_for_names,
)
from .utils.pos_embedding import create_2d_sin_cos_pos_emb

import wandb

import os
from dotenv import load_dotenv

from diffusers.optimization import get_scheduler as get_lr_scheduler

load_dotenv()

class ImageNetWebDataset(Dataset):
    def __init__(self, split='train', transform=None):
        super().__init__()
        self.transform = transform
        try:
            self.dataset = load_dataset("imagenet-1k", split=split, trust_remote_code=True)
        except Exception as e:
            print(e)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]
        if self.transform:
            image = self.transform(image)

        c, _, _ = image.shape
        if c == 1:
            image = image.repeat(3, 1, 1)

        label = torch.tensor(label, dtype=torch.long)

        return image, label

class ImageNetDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    
    def setup(self, stage=None):
        self.train_dataset = ImageNetWebDataset(
            split='train',
            transform=self.transform
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["trainer"]["train_batch_size"],
            num_workers=self.config["trainer"]["num_dl_workers"],
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )


class RinLightningModule(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        rin_config = config["rin"]
        diffusion_config = config["diffusion"]
        
        self.automatic_optimization = False
        
        self.rin = Rin(**rin_config)
        self.rin.pass_dummy_data(num_classes=config["trainer"]["num_classes"])  # Populate lazy model with weights
        self.diffusion_model = RinDiffusionModel(rin=self.rin, **diffusion_config)
        
        self.rin_ema = Rin(**rin_config)
        self.rin_ema.pass_dummy_data(num_classes=config["trainer"]["num_classes"])
        self.ema_diffusion_model = RinDiffusionModel(rin=self.rin_ema, **diffusion_config)
        self.ema_decay = config["trainer"]["ema_decay"]
        self.ema_update_every = config["trainer"]["ema_update_every"]
        
        self.sample_every = config["trainer"]["sample_every"]
        self.sampling_kwargs = config["trainer"]["sampling_kwargs"]
        self.tape_dim = rin_config["tape_dim"]
        self.num_classes = config["trainer"]["num_classes"]
        self.patch_size = rin_config["patch_size"]
    
    def forward(self, batch_img, batch_class):
        return self.diffusion_model(batch_img, batch_class)
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        
        batch_img, batch_class = batch
        batch_class = batch_class.to(torch.long)
        batch_class = torch.nn.functional.one_hot(batch_class, num_classes=self.num_classes).float()

        opt.zero_grad()
        loss = self(batch_img, batch_class)
        self.manual_backward(loss)
        opt.step()

        # Update learning rate
        sch = self.lr_schedulers()
        sch.step()

        logs = {
            "loss": loss.item(),
            "lr": sch.get_last_lr()[0],
        }

        self.log_dict(logs, on_step=True, prog_bar=True)

        # Update EMA model
        if self.global_step % self.ema_update_every == 0:
            with torch.no_grad():
                for ema_param, param in zip(self.ema_diffusion_model.parameters(), self.diffusion_model.parameters()):
                    if param.requires_grad:
                        ema_param.data.lerp_(param.data, 1 - self.ema_decay)

        # Generate samples
        if self.global_step % self.sample_every == 0:
            self.ema_diffusion_model.eval()
            n = 8
            samples = self.ema_diffusion_model.sample(num_samples=n * n, image_height=256, image_width=256, tape_dim=self.tape_dim, **self.sampling_kwargs)
            grid = torchvision.utils.make_grid(samples, nrow=n, normalize=True, value_range=(0, 1), padding=0)
            self.logger.experiment.log({"samples": [wandb.Image(grid)]}, step=self.global_step)

            del samples
            self.ema_diffusion_model.train()

        return loss
    
    def configure_optimizers(self):
        optimizer = get_optimizer(
            self.hparams["trainer"]["optimizer_name"],
            override_config_for_names(
                self.diffusion_model.parameters(),
                self.hparams["trainer"]["optimizer_exclude_weight_decay"],
                {"weight_decay": 0.0, "disable_layer_adaption": True},
                build_parameters_mapping(self.diffusion_model),
            ),
            lr=self.hparams["trainer"]["lr"],
            **self.hparams["trainer"]["optimizer_kwargs"],
        )
        
        scheduler = get_lr_scheduler(
                self.hparams["trainer"]["lr_scheduler_name"],
                optimizer,
                num_warmup_steps=self.hparams["trainer"]["lr_warmup_steps"],
                num_training_steps=self.hparams["trainer"]["train_num_steps"],
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }