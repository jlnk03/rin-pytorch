import torch
import torchvision
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import webdataset as wds
from huggingface_hub import HfFileSystem, get_token, hf_hub_url
from datasets import load_dataset

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

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from diffusers.optimization import get_scheduler as get_lr_scheduler

load_dotenv()

class FlexibleCIFAR10(Dataset):
    def __init__(self, root_dir, train=True, transform=None, target_class=None, num_samples=None, ensure_vertical=False, ensure_horizontal=False):
        self.root_dir = Path(root_dir)
        self.split = 'train' if train else 'test'
        self.transform = transform
        self.ensure_vertical = ensure_vertical  # Only ensure vertical for overfit mode
        self.ensure_horizontal = ensure_horizontal
        
        # Get all image paths
        self.image_paths = []
        self.labels = []

        for class_idx in range(10):
            class_dir = self.root_dir / self.split / str(class_idx)
            for img_path in class_dir.glob('*.png'):
                self.image_paths.append(img_path)
                self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # Check if image needs rotation
        width, height = image.size
        if self.ensure_vertical:
            if width > height:
                image = image.rotate(90, expand=True)
        elif self.ensure_horizontal and not self.ensure_vertical:
            if height > width:
                image = image.rotate(90, expand=True)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class ImageNetWebDataset(IterableDataset):
    def __init__(self, split='train', transform=None):
        super().__init__()
        self.transform = transform

        try:
            self.dataset = load_dataset("imagenet-1k", split="train", trust_remote_code=True)
        except Exception as e:
            print(e)
        
    def __iter__(self):
        return iter(self.dataset)


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    # C, H, W -> T, D
    c, h, w = x.shape
    nh, nw = h // p, w // p
    x = x.view(c, nh, p, nw, p)
    x = x.permute(1, 3, 2, 4, 0).contiguous()
    x = x.view(nh * nw, p * p * c)
    return x


def pad_to_max_size(batch, patch_size, tape_dim, transform=None):
    # Extract images and labels from the batch
    # images, labels = zip(*batch)

    images = []
    labels = []

    for sample in batch:
        images.append(sample["image"].convert("RGB"))
        labels.append(sample["label"])

    if transform:
        images = [transform(img) for img in images]
    
    # Find the maximum height and width in the batch
    max_height = max(img.shape[1] for img in images)
    max_width = max(img.shape[2] for img in images)

    max_pixels = max_height * max_width
    nmh = max_height // patch_size
    nmw = max_width // patch_size
    
    padded_images = []
    patch_masks = []
    image_masks = []
    pos_embs = []
    for img in images:
        _, h, w = img.shape
        
        # Calculate how many pixels to crop to make dimensions divisible by patch_size
        h_crop = h - (h // patch_size) * patch_size
        w_crop = w - (w // patch_size) * patch_size
        
        # Crop the image if needed
        if h_crop > 0 or w_crop > 0:
            img = img[:, :h-h_crop, :w-w_crop]
            _, h, w = img.shape  # Update dimensions after cropping
        
        nh = h // patch_size
        nw = w // patch_size
        pixel_row = patchify(img, patch_size)
        pixel_row = pixel_row

        pos_emb = create_2d_sin_cos_pos_emb(nh, nw, tape_dim)

        pixel_mask = torch.ones(h * w)
        patch_mask = torch.ones(nh * nw)

        # pad to max pixels
        pixel_row = torch.nn.functional.pad(pixel_row, (0, 0, 0, nmh * nmw - pixel_row.shape[0]), value=0)
        
        pixel_mask = torch.nn.functional.pad(pixel_mask, (0, max_pixels - pixel_mask.shape[0]), value=0)
        patch_mask = torch.nn.functional.pad(patch_mask, (0, nmh * nmw - patch_mask.shape[0]), value=0)
        # Only pad the first dimension (0), leave second dimension unchanged
        pos_emb = torch.nn.functional.pad(pos_emb, (0, 0, 0, nmh * nmw - pos_emb.shape[0]), value=0)
        image_masks.append(pixel_mask)
        patch_masks.append(patch_mask)

        padded_images.append(pixel_row)
        pos_embs.append(pos_emb)

    # Convert labels to a tensor and move to GPU
    labels = torch.tensor(labels)
    padded_images = torch.stack(padded_images)
    patch_masks = torch.stack(patch_masks).bool()
    image_masks = torch.stack(image_masks).bool()
    pos_embs = torch.stack(pos_embs)
    
    return padded_images, patch_masks, image_masks, labels, pos_embs, nmh, nmw


class ImageNetDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
        ])
    
    def setup(self, stage=None):
    #     self.train_dataset = FlexibleCIFAR10(
    #     "datasets/cifar10_flex",
    #     train=True,
    #     transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.RandomHorizontalFlip(),
    #     ])
    # )
        self.train_dataset = ImageNetWebDataset(
            split='train',
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["trainer"]["train_batch_size"],
            num_workers=self.config["trainer"]["num_dl_workers"],
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
            collate_fn=lambda batch: pad_to_max_size(batch, self.config["rin"]["patch_size"], self.config["rin"]["tape_dim"], self.transform)
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
    
    def forward(self, batch_img, batch_mask, image_mask, batch_class, pos_embs, nmh, nmw):
        return self.diffusion_model(batch_img, batch_mask, image_mask, batch_class, pos_embs, nmh, nmw)
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        
        batch_img, batch_mask, image_mask, batch_class, pos_embs, nmh, nmw = batch
        batch_class = torch.nn.functional.one_hot(batch_class, num_classes=self.num_classes).float()

        opt.zero_grad()
        loss = self(batch_img, batch_mask, image_mask, batch_class, pos_embs, nmh, nmw)
        self.manual_backward(loss)
        opt.step()

        # Update learning rate
        sch = self.lr_schedulers()
        sch.step()

        # Log metrics
        # self.log("train_loss", loss, on_step=True, prog_bar=True)
        # self.log("lr", opt.param_groups[0]["lr"], on_step=True, prog_bar=True)

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

            samples_horizontal = self.ema_diffusion_model.sample(num_samples=n * n, image_height=128, image_width=256, tape_dim=self.tape_dim, **self.sampling_kwargs)
            grid_horizontal = torchvision.utils.make_grid(samples_horizontal, nrow=n, normalize=True, value_range=(0, 1), padding=0)
            self.logger.experiment.log({"samples_horizontal": [wandb.Image(grid_horizontal)]}, step=self.global_step)

            samples_vertical = self.ema_diffusion_model.sample(num_samples=n * n, image_height=256, image_width=128, tape_dim=self.tape_dim, **self.sampling_kwargs)
            grid_vertical = torchvision.utils.make_grid(samples_vertical, nrow=n, normalize=True, value_range=(0, 1), padding=0)
            self.logger.experiment.log({"samples_vertical": [wandb.Image(grid_vertical)]}, step=self.global_step)

            del samples
            del samples_horizontal
            del samples_vertical
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
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_model"] = self.ema_diffusion_model.state_dict()
        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        if "ema_model" in checkpoint:
            self.ema_diffusion_model.load_state_dict(checkpoint["ema_model"])