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

from .utils.FlexibleCifar import FlexibleCIFAR10

from PIL import Image

from rin_pytorch import Rin, RinDiffusionModel
from .utils.optimization_utils import (
    build_parameters_mapping,
    get_optimizer,
    override_config_for_names,
)
from .utils.pos_embedding import create_2d_sin_cos_pos_emb
from .utils.logging_utils import log_first_tensor

import wandb

import os
from dotenv import load_dotenv

from diffusers.optimization import get_scheduler as get_lr_scheduler

from datetime import datetime, timedelta
from wandb import AlertLevel

load_dotenv()

class ResizeMaxSide:
    def __init__(self, max_side, interpolation=Image.BILINEAR):
        self.max_side = max_side
        self.interpolation = interpolation

    def __call__(self, img):
        # img is expected to be a PIL Image
        width, height = img.size

        # Compute the scaling factor such that the longest edge equals max_side
        max_dim = max(width, height)
        if max_dim > self.max_side:
            scale = self.max_side / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), self.interpolation)
        return img

class ImageNetWebDataset(IterableDataset):
    def __init__(self, split='train', transform=None):
        super().__init__()
        self.transform = transform

        try:
            self.dataset = load_dataset("imagenet-1k", split="train") 
            # self.dataset = load_dataset("timm/imagenet-1k-wds", split="train")
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


def pad_to_max_size(batch, patch_size, tape_dim, transform=None, second_patch_size=None):
    # Extract images and labels from the batch
    # images, labels = zip(*batch)
    images = []
    labels = []

    first_logged = False
    for example in batch:
        if example["image"].mode == "RGBA":
            print(f"Warning: Image has 4 channels (RGBA), converting to 3")
            example["image"] = example["image"].convert("RGB")
        if transform:
            example["image"] = transform(example["image"])
        c, _, _ = example["image"].shape
        if c > 3:
            print(f"Warning: Image has {c} channels, truncating to 3")
            example["image"] = example["image"][:3]
        if not first_logged:
            log_first_tensor("dataloader.image_transformed", example["image"])
            first_logged = True
        images.append(example["image"])
        labels.append(example["label"])
    
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
    first_logged2 = False
    for img in images:
        c, h, w = img.shape

        # Calculate how many pixels to crop to make dimensions divisible by patch_size
        h_crop = h - (h // patch_size) * patch_size
        w_crop = w - (w // patch_size) * patch_size
        
        # Crop the image if needed
        if h_crop > 0 or w_crop > 0:
            img = img[:, :h-h_crop, :w-w_crop]
            _, h, w = img.shape  # Update dimensions after cropping

        if c == 1:
            img = img.repeat(3, 1, 1)
        if not first_logged2:
            log_first_tensor("dataloader.image_cropped", img)
        
        nh = h // patch_size
        nw = w // patch_size
        pixel_row = patchify(img, patch_size)
        pixel_row = pixel_row
        if not first_logged2:
            log_first_tensor("dataloader.image_patchified", pixel_row)

        pos_emb = create_2d_sin_cos_pos_emb(nh, nw, tape_dim)

        pixel_mask = torch.ones(h * w)
        patch_mask = torch.ones(nh * nw)

        # pad to max pixels
        pixel_row = torch.nn.functional.pad(pixel_row, (0, 0, 0, nmh * nmw - pixel_row.shape[0]), value=0)
        if not first_logged2:
            log_first_tensor("dataloader.image_padded", pixel_row)
        
        pixel_mask = torch.nn.functional.pad(pixel_mask, (0, max_pixels - pixel_mask.shape[0]), value=0)
        patch_mask = torch.nn.functional.pad(patch_mask, (0, nmh * nmw - patch_mask.shape[0]), value=0)
        # Only pad the first dimension (0), leave second dimension unchanged
        pos_emb = torch.nn.functional.pad(pos_emb, (0, 0, 0, nmh * nmw - pos_emb.shape[0]), value=0)
        image_masks.append(pixel_mask)
        patch_masks.append(patch_mask)

        padded_images.append(pixel_row)
        pos_embs.append(pos_emb)
        if not first_logged2:
            first_logged2 = True

    # Convert labels to a tensor and move to GPU
    labels = torch.tensor(labels)
    padded_images = torch.stack(padded_images)
    patch_masks = torch.stack(patch_masks).bool()
    image_masks = torch.stack(image_masks).bool()
    pos_embs = torch.stack(pos_embs)
    
    # Optionally create a second tensor with a different patch size, preserving image order
    if second_patch_size is not None and second_patch_size != patch_size:
        nmh2 = max_height // second_patch_size
        nmw2 = max_width // second_patch_size

        padded_images2 = []
        for img in images:
            c2, h2, w2 = img.shape
            h_crop2 = h2 - (h2 // second_patch_size) * second_patch_size
            w_crop2 = w2 - (w2 // second_patch_size) * second_patch_size
            if h_crop2 > 0 or w_crop2 > 0:
                img2 = img[:, :h2 - h_crop2, :w2 - w_crop2]
                _, h2, w2 = img2.shape
            else:
                img2 = img
            if c2 == 1:
                img2 = img2.repeat(3, 1, 1)
            nh2 = h2 // second_patch_size
            nw2 = w2 // second_patch_size
            pixel_row2 = patchify(img2, second_patch_size)
            pixel_row2 = torch.nn.functional.pad(pixel_row2, (0, 0, 0, nmh2 * nmw2 - pixel_row2.shape[0]), value=0)
            padded_images2.append(pixel_row2)

        padded_images2 = torch.stack(padded_images2)
        return padded_images, patch_masks, image_masks, labels, pos_embs, nmh, nmw, padded_images2

    return padded_images, patch_masks, image_masks, labels, pos_embs, nmh, nmw


def collate_fixed_size(batch, patch_size, tape_dim, transform, image_height, image_width, second_patch_size=None):
    # Produce fixed-size, square, non-padded tensors per config dims
    images = []
    labels = []

    first_logged = False
    for example in batch:
        if example["image"].mode == "RGBA":
            print(f"Warning: Image has 4 channels (RGBA), converting to 3")
            example["image"] = example["image"].convert("RGB")
        if transform:
            example["image"] = transform(example["image"])

        c, h, w = example["image"].shape
        if c > 3:
            print(f"Warning: Image has {c} channels, truncating to 3")
            example["image"] = example["image"][:3]
            c, h, w = example["image"].shape
        if c == 1:
            example["image"] = example["image"].repeat(3, 1, 1)
            c, h, w = example["image"].shape

        if not first_logged:
            log_first_tensor("dataloader_sq.image_transformed", example["image"])
            first_logged = True

        # Ensure final dims are the configured ones and divisible by patch size
        assert h == image_height and w == image_width, f"Transformed image dims {(h, w)} != {(image_height, image_width)}"
        assert h % patch_size == 0 and w % patch_size == 0, "Image dims must be divisible by patch size"

        images.append(example["image"])
        labels.append(example["label"])

    nmh = image_height // patch_size
    nmw = image_width // patch_size

    padded_images = []
    patch_masks = []
    image_masks = []
    pos_embs = []

    first_logged2 = False
    for img in images:
        c, h, w = img.shape
        nh = h // patch_size
        nw = w // patch_size

        pixel_row = patchify(img, patch_size)
        if not first_logged2:
            log_first_tensor("dataloader_sq.image_patchified", pixel_row)

        pos_emb = create_2d_sin_cos_pos_emb(nh, nw, tape_dim)

        pixel_mask = torch.ones(h * w)
        patch_mask = torch.ones(nh * nw)

        image_masks.append(pixel_mask)
        patch_masks.append(patch_mask)
        padded_images.append(pixel_row)
        pos_embs.append(pos_emb)

        if not first_logged2:
            log_first_tensor("dataloader_sq.image_masks", image_masks[-1])
            first_logged2 = True

    labels = torch.tensor(labels)
    padded_images = torch.stack(padded_images)
    patch_masks = torch.stack(patch_masks).bool()
    image_masks = torch.stack(image_masks).bool()
    pos_embs = torch.stack(pos_embs)

    # Optionally create second tensor with a different patch size; images/order preserved
    if second_patch_size is not None and second_patch_size != patch_size:
        assert image_height % second_patch_size == 0 and image_width % second_patch_size == 0, "Configured image dims must be divisible by second_patch_size"
        nmh2 = image_height // second_patch_size
        nmw2 = image_width // second_patch_size

        padded_images2 = []
        first_logged3 = False
        for img in images:
            c2, h2, w2 = img.shape
            if c2 == 1:
                img = img.repeat(3, 1, 1)
            nh2 = h2 // second_patch_size
            nw2 = w2 // second_patch_size
            pixel_row2 = patchify(img, second_patch_size)
            if not first_logged3:
                log_first_tensor("dataloader_sq.image_patchified_second", pixel_row2)
                first_logged3 = True
            padded_images2.append(pixel_row2)
        padded_images2 = torch.stack(padded_images2)
        return padded_images, patch_masks, image_masks, labels, pos_embs, nmh, nmw, padded_images2

    return padded_images, patch_masks, image_masks, labels, pos_embs, nmh, nmw

class ImageNetDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config["run"]["cifar"]:
            if self.config["run"].get("square_images"):
                H, W = self.config["rin"]["image_height"], self.config["rin"]["image_width"]
                self.transform = transforms.Compose([
                    transforms.Resize(H),
                    transforms.CenterCrop((H, W)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
        else:
            if self.config["run"].get("square_images"):
                H, W = self.config["rin"]["image_height"], self.config["rin"]["image_width"]
                self.transform = transforms.Compose([
                    transforms.Resize(H),
                    transforms.CenterCrop((H, W)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((128, 128)) if self.config["run"]["vanilla"] else ResizeMaxSide(128),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
    
    def setup(self, stage=None):
        if self.config["run"]["cifar"]: 
            self.train_dataset = FlexibleCIFAR10(
                "datasets/cifar10_flex",
                train=True,
            )
        else:
            self.train_dataset = ImageNetWebDataset(
                split='train',
                # transform=self.transform
            )
    
    def train_dataloader(self):
        if self.config["run"].get("square_images"):
            collate = lambda batch: collate_fixed_size(
                batch,
                self.config["rin"]["patch_size"],
                self.config["rin"]["tape_dim"],
                self.transform,
                self.config["rin"]["image_height"],
                self.config["rin"]["image_width"],
                self.config["rin"].get("second_patch_size"),
            )
        else:
            collate = lambda batch: pad_to_max_size(
                batch,
                self.config["rin"]["patch_size"],
                self.config["rin"]["tape_dim"],
                self.transform,
                self.config["rin"].get("second_patch_size"),
            )

        return DataLoader(
                self.train_dataset,
                batch_size=self.config["trainer"]["train_batch_size"],
                num_workers=self.config["trainer"]["num_dl_workers"],
                pin_memory=True,
                persistent_workers=True,
                drop_last=True,
                collate_fn=collate,
                shuffle=False,
        )


class RinLightningModule(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        rin_config = config["rin"]
        diffusion_config = config["diffusion"]
        
        self.automatic_optimization = False
        
        self.rin = Rin(**rin_config)
        self.rin.pass_dummy_data(num_classes=rin_config["num_classes"])  # Populate lazy model with weights
        self.diffusion_model = RinDiffusionModel(rin=self.rin, **diffusion_config)
        
        self.rin_ema = Rin(**rin_config)
        self.rin_ema.pass_dummy_data(num_classes=rin_config["num_classes"])
        self.ema_diffusion_model = RinDiffusionModel(rin=self.rin_ema, **diffusion_config)
        self.ema_decay = config["trainer"]["ema_decay"]
        self.ema_update_every = config["trainer"]["ema_update_every"]
        
        self.sample_every = config["trainer"]["sample_every"]
        self.sampling_kwargs = config["trainer"]["sampling_kwargs"]
        self.tape_dim = rin_config["tape_dim"]
        self.num_classes = rin_config["num_classes"]
        self.patch_size = rin_config["patch_size"]
        
        # Add tracking for lowest loss
        self.lowest_loss = float('inf')
        self.run_started = False

        self.image_height = rin_config["image_height"]
        self.image_width = rin_config["image_width"]

    def on_train_start(self):
        try:
            wandb.alert(
                title='Training Started',
                text=f'Training run has begun with config: {self.hparams}',
                level=AlertLevel.INFO
            )
        except Exception as e:
            print(e)
        self.run_started = True

    def on_train_end(self):
        try:
            wandb.alert(
                title='Training Completed',
                text=f'Training run has completed successfully. Final loss: {self.lowest_loss}',
                level=AlertLevel.INFO
            )
        except Exception as e:
            print(e)

    def on_train_epoch_end(self):
        if self.trainer.should_stop:
            try:
                wandb.alert(
                    title='Training Stopped Early',
                    text=f'Training stopped before completion. Final loss: {self.lowest_loss}',
                    level=AlertLevel.WARN
                )
            except Exception as e:
                print(e)

    def forward(self, batch_img, batch_mask, image_mask, batch_class, pos_embs, nmh, nmw):
        return self.diffusion_model(batch_img, batch_mask, image_mask, batch_class, pos_embs, nmh, nmw)
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        # Support optional second padded tensor appended by collate
        if len(batch) == 8:
            batch_img, batch_mask, image_mask, batch_class, pos_embs, nmh, nmw, _batch_img2 = batch
        else:
            batch_img, batch_mask, image_mask, batch_class, pos_embs, nmh, nmw = batch
        # Log the padded/patchified batch input arriving to the trainer
        log_first_tensor("trainer.batch_img_in", batch_img[0])
        batch_class = torch.nn.functional.one_hot(batch_class, num_classes=self.num_classes).float()

        opt.zero_grad()
        loss = self(batch_img, batch_mask, image_mask, batch_class, pos_embs, nmh, nmw)
        self.manual_backward(loss)
        
        if self.hparams["trainer"].get("clip_grad_norm"):
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.hparams["trainer"]["clip_grad_norm"])
        
        opt.step()

        # Track lowest loss and alert if current loss is too high
        # try:
        #     if loss.item() < self.lowest_loss:
        #         self.lowest_loss = loss.item()
        #     elif loss.item() > 2 * self.lowest_loss:  # Alert if loss is more than double the lowest loss
        #         wandb.alert(
        #             title='High Loss Detected',
        #             text=f'Current loss ({loss.item():.4f}) is more than 100% higher than lowest loss ({self.lowest_loss:.4f})',
        #                 level=AlertLevel.WARN,
        #                 wait_duration=timedelta(minutes=5)
        #         )
        # except Exception as e:
        #     print(e)

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
            samples = self.ema_diffusion_model.sample(num_samples=n * n, image_height=self.image_height, image_width=self.image_width, tape_dim=self.tape_dim, **self.sampling_kwargs)
            grid = torchvision.utils.make_grid(samples, nrow=n, normalize=True, value_range=(0, 1), padding=0)
            self.logger.experiment.log({"samples": [wandb.Image(grid)]}, step=self.global_step)

            samples_horizontal = self.ema_diffusion_model.sample(num_samples=n * n, image_height=int(self.image_height * 0.75), image_width=self.image_width, tape_dim=self.tape_dim, **self.sampling_kwargs)
            grid_horizontal = torchvision.utils.make_grid(samples_horizontal, nrow=n, normalize=True, value_range=(0, 1), padding=0)
            self.logger.experiment.log({"samples_horizontal": [wandb.Image(grid_horizontal)]}, step=self.global_step)

            samples_vertical = self.ema_diffusion_model.sample(num_samples=n * n, image_height=self.image_height, image_width=int(self.image_width * 0.75), tape_dim=self.tape_dim, **self.sampling_kwargs)
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

    # def on_exception(self, trainer, pl_module, exception):
    #     wandb.alert(
    #         title='Training Crashed',
    #         text=f'Training run crashed with exception: {str(exception)}',
    #         level=AlertLevel.ERROR
    #     )