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

import random

from .utils.FlexibleCifar import FlexibleCIFAR10

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


def create_random_token_mask(x: torch.Tensor, mask_ratio: float = 0.5) -> torch.Tensor:
    """
    Creates a random boolean mask for the given tensor x such that each sample
    in the batch has exactly 'mask_ratio' fraction of its tokens masked.

    Args:
        x (torch.Tensor): Input tensor of shape [b, t, c].
        mask_ratio (float): Fraction of tokens to mask in each sample.

    Returns:
        A tuple of boolean masks (expanded to [b, t, c] for tokens) where the 
        inverted mask (~mask) can be used to select visible tokens.
    """
    b, t, c = x.shape
    # print(f'mask ratio create_random_token_mask: {mask_ratio}')
    num_mask = int(mask_ratio * t)
    # print(f'num_mask create_random_token_mask: {num_mask}')

    rand_scores = torch.rand(b, t, device=x.device)
    _, indices = torch.topk(rand_scores, k=num_mask, dim=1, largest=False)

    token_mask = torch.zeros((b, t), dtype=torch.bool, device=x.device)
    token_mask.scatter_(1, indices, True)

    mask = token_mask.unsqueeze(-1).expand(b, t, c)

    return mask, token_mask


def create_random_token_mask_single(x: torch.Tensor, mask_ratio: float = 0.5) -> torch.Tensor:
    """
    Creates a random boolean mask for a single image tensor x such that
    'mask_ratio' fraction of its tokens are masked.

    Args:
        x (torch.Tensor): Input tensor of shape [t, c].
        mask_ratio (float): Fraction of tokens to mask.

    Returns:
        A tuple of boolean masks (expanded to [t, c] for tokens) where the 
        inverted mask (~mask) can be used to select visible tokens.
    """
    t, c = x.shape
    num_mask = int(mask_ratio * t)

    rand_scores = torch.rand(t, device=x.device)
    _, indices = torch.topk(rand_scores, k=num_mask, largest=False)

    token_mask = torch.zeros(t, dtype=torch.bool, device=x.device)
    token_mask[indices] = True # Direct indexing for single dimension

    mask = token_mask.unsqueeze(-1).expand(t, c)

    return mask, token_mask


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
            self.dataset = load_dataset(
                "imagenet-1k", split="train", trust_remote_code=True)
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


def pad_to_max_size(batch, patch_size, tape_dim, transform=None, initial_mask_ratio=0.5, final_mask_ratio=0.0):
    images = []
    labels = []
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
        images.append(example["image"])
        labels.append(example["label"])

    max_height = max(img.shape[1] for img in images)
    max_width = max(img.shape[2] for img in images)

    max_pixels = max_height * max_width
    nmh = max_height // patch_size
    nmw = max_width // patch_size

    padded_images = []
    patch_masks = []
    image_masks = []
    pos_embs = []

    max_size = 0
    min_size = 0
    original_token_counts = []  # Keep track of original token counts
    for img in images:
        c, h, w = img.shape

        h_crop = h - (h // patch_size) * patch_size
        w_crop = w - (w // patch_size) * patch_size

        if h_crop > 0 or w_crop > 0:
            img = img[:, :h - h_crop, :w - w_crop]
            _, h, w = img.shape

        if c == 1:
            img = img.repeat(3, 1, 1)

        nh = h // patch_size
        nw = w // patch_size
        pixel_row = patchify(img, patch_size)

        pos_emb = create_2d_sin_cos_pos_emb(nh, nw, tape_dim)

        pixel_mask = torch.ones(h * w)
        patch_mask = torch.ones(nh * nw)       

        mask_ratio = random.uniform(initial_mask_ratio, final_mask_ratio)
        _, token_mask = create_random_token_mask_single(pixel_row, mask_ratio)

        # mask the pixel_row, patch_mask, and pos_emb, and pixel_mask
        pixel_row = pixel_row[token_mask]
        patch_mask = patch_mask[token_mask]
        pos_emb = pos_emb[token_mask]

        original_token_count = pixel_row.shape[0] # Store token count before padding
        original_token_counts.append(original_token_count)

        max_size = max(max_size, pixel_row.shape[0])
        if min_size == 0:
            min_size = pixel_row.shape[0]
        else:
            min_size = min(min_size, pixel_row.shape[0])

        # Pad to max tokens / max patches for consistency across the batch
        pixel_row = torch.nn.functional.pad(
            pixel_row, (0, 0, 0, nmh * nmw - pixel_row.shape[0]), value=0)
        pixel_mask = torch.nn.functional.pad(
            pixel_mask, (0, max_pixels - pixel_mask.shape[0]), value=0)
        patch_mask = torch.nn.functional.pad(
            patch_mask, (0, nmh * nmw - patch_mask.shape[0]), value=0)
        pos_emb = torch.nn.functional.pad(
            pos_emb, (0, 0, 0, nmh * nmw - pos_emb.shape[0]), value=0)

        image_masks.append(pixel_mask)
        patch_masks.append(patch_mask)
        padded_images.append(pixel_row)
        pos_embs.append(pos_emb)

    print(f'max size: {max_size}')
    print(f'min size: {min_size}')
    print('here')
    labels = torch.tensor(labels)
    padded_images = torch.stack(padded_images)
    patch_masks = torch.stack(patch_masks).bool()
    image_masks = torch.stack(image_masks).bool()
    pos_embs = torch.stack(pos_embs)

    # Crop to max token length after masking
    patch_masks = patch_masks[:, :max_size]
    pos_embs = pos_embs[:, :max_size]
    padded_images = padded_images[:, :max_size]

    # Calculate padding percentage
    total_tokens = padded_images.shape[0] * padded_images.shape[1]
    total_original_tokens = sum(original_token_counts)
    num_padding_tokens = total_tokens - total_original_tokens
    padding_percentage = (num_padding_tokens / total_tokens) * 100 if total_tokens > 0 else 0

    min_max_ratio = min_size / max_size

    # _, token_mask = create_random_token_mask(padded_images, mask_ratio=mask_ratio)

    # visible_padded_images = []
    # visible_patch_masks = []
    # visible_pos_embs = []
    # for i, (padded_image, patch_mask, pos_emb) in enumerate(zip(padded_images, patch_masks, pos_embs)):
    #     # print(token_mask[i])
    #     visible_idx = ~token_mask[i]
    #     visible_padded_images.append(padded_image[visible_idx])
    #     visible_patch_masks.append(patch_mask[visible_idx])
    #     visible_pos_embs.append(pos_emb[visible_idx])

    # visible_padded_images = torch.stack(visible_padded_images)
    # visible_patch_masks = torch.stack(visible_patch_masks)
    # visible_pos_embs = torch.stack(visible_pos_embs)

    # print(f'visible_padded_images.shape: {visible_padded_images.shape}')

    # return visible_padded_images, visible_patch_masks, image_masks, labels, visible_pos_embs, nmh, nmw
    return padded_images, patch_masks, image_masks, labels, pos_embs, nmh, nmw, max_size, min_size, padding_percentage, min_max_ratio


class ImageNetDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config["run"]["cifar"]:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(
                    (128, 128)) if self.config["run"]["vanilla"] else ResizeMaxSide(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])

        # Add masking schedule parameters
        self.initial_mask_ratio = 0.5
        self.final_mask_ratio = 0.0
        self.total_steps = config["trainer"]["train_num_steps"]
        self.current_step = 0  # Add this to track steps

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

    def get_current_mask_ratio(self):
        # Random value between initial_mask_ratio and final_mask_ratio

        return random.uniform(self.final_mask_ratio, self.initial_mask_ratio)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["trainer"]["train_batch_size"],
            num_workers=self.config["trainer"]["num_dl_workers"],
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
            collate_fn=lambda batch: pad_to_max_size(
                batch,
                self.config["rin"]["patch_size"],
                self.config["rin"]["tape_dim"],
                transform=self.transform,
                initial_mask_ratio=self.initial_mask_ratio,
                final_mask_ratio=self.final_mask_ratio
            )
        )


class RinLightningModule(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        rin_config = config["rin"]
        diffusion_config = config["diffusion"]

        self.automatic_optimization = False

        self.rin = Rin(**rin_config)
        # Populate lazy model with weights
        self.rin.pass_dummy_data(num_classes=rin_config["num_classes"])
        self.diffusion_model = RinDiffusionModel(
            rin=self.rin, **diffusion_config)

        self.rin_ema = Rin(**rin_config)
        self.rin_ema.pass_dummy_data(num_classes=rin_config["num_classes"])
        self.ema_diffusion_model = RinDiffusionModel(
            rin=self.rin_ema, **diffusion_config)
        self.ema_decay = config["trainer"]["ema_decay"]
        self.ema_update_every = config["trainer"]["ema_update_every"]

        self.sample_every = config["trainer"]["sample_every"]
        self.sampling_kwargs = config["trainer"]["sampling_kwargs"]
        self.tape_dim = rin_config["tape_dim"]
        self.num_classes = rin_config["num_classes"]
        self.patch_size = rin_config["patch_size"]

        # Add masking schedule parameters
        self.initial_mask_ratio = 0.5
        self.final_mask_ratio = 0.0
        self.total_steps = config["trainer"]["train_num_steps"]

    def get_current_mask_ratio(self):
        # Random value between initial_mask_ratio and final_mask_ratio

        # progress = min(self.global_step / self.total_steps, 1.0)
        # return self.initial_mask_ratio + (self.final_mask_ratio - self.initial_mask_ratio) * progress
        return random.uniform(self.final_mask_ratio, self.initial_mask_ratio)

    def forward(self, batch_img, batch_mask, image_mask, batch_class, pos_embs, nmh, nmw, tape_length):
        # print(f'mask_ratio_forward: {mask_ratio}')
        return self.diffusion_model(batch_img, batch_mask, image_mask, batch_class, pos_embs, nmh, nmw, tape_length)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        # Update the step counter in the DataModule
        # self.trainer.datamodule.current_step = self.global_step

        # mask_ratio = self.trainer.datamodule.get_current_mask_ratio()

        # batch_img, batch_mask, image_mask, batch_class, pos_embs, nmh, nmw = batch

        padded_images, patch_masks, image_masks, batch_class, pos_embs, nmh, nmw, max_size, min_size, padding_percentage, min_max_ratio = batch # Unpack padding_percentage

        # _, token_mask = create_random_token_mask(
        #     padded_images, mask_ratio=mask_ratio)

        # visible_padded_images = []
        # visible_patch_masks = []
        # visible_pos_embs = []
        # for i, (padded_image, patch_mask, pos_emb) in enumerate(zip(padded_images, patch_masks, pos_embs)):
        #     # print(token_mask[i])
        #     visible_idx = ~token_mask[i]
        #     visible_padded_images.append(padded_image[visible_idx])
        #     visible_patch_masks.append(patch_mask[visible_idx])
        #     visible_pos_embs.append(pos_emb[visible_idx])

        # visible_padded_images = torch.stack(visible_padded_images)
        # visible_patch_masks = torch.stack(visible_patch_masks)
        # visible_pos_embs = torch.stack(visible_pos_embs)

        # print(f'visible_padded_images.shape: {visible_padded_images.shape}')

        # return visible_padded_images, visible_patch_masks, image_masks, labels, visible_pos_embs, nmh, nmw

        batch_class = torch.nn.functional.one_hot(
            batch_class, num_classes=self.num_classes).float()

        opt.zero_grad()
        # # mask_ratio = self.trainer.datamodule.get_current_mask_ratio()
        # mask_ratio = 1 - (visible_padded_images.shape[1] / (nmh * nmw))

        tape_length = padded_images.shape[1]
        print(f'max size fwd: {max_size}')
        print(f'min size fwd: {min_size}')
        # print(f'mask_ratio_trainer: {mask_ratio}')
        # loss = self(batch_img, batch_mask, image_mask, batch_class, pos_embs, nmh, nmw, mask_ratio)
        loss = self(padded_images, patch_masks, image_masks,
                    batch_class, pos_embs, nmh, nmw, tape_length)
        self.manual_backward(loss)
        opt.step()

        sch = self.lr_schedulers()
        sch.step()

        logs = {
            "loss": loss.item(),
            "lr": sch.get_last_lr()[0],
            "padding_percentage": padding_percentage,
            "min_max_ratio": min_max_ratio,
            # "mask_ratio": self.trainer.datamodule.get_current_mask_ratio(),
            # "mask_ratio": mask_ratio
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
            samples = self.ema_diffusion_model.sample(
                num_samples=n * n, image_height=32, image_width=32, tape_dim=self.tape_dim, **self.sampling_kwargs)
            grid = torchvision.utils.make_grid(
                samples, nrow=n, normalize=True, value_range=(0, 1), padding=0)
            self.logger.experiment.log(
                {"samples": [wandb.Image(grid)]}, step=self.global_step)

            samples_horizontal = self.ema_diffusion_model.sample(
                num_samples=n * n, image_height=24, image_width=32, tape_dim=self.tape_dim, **self.sampling_kwargs)
            grid_horizontal = torchvision.utils.make_grid(
                samples_horizontal, nrow=n, normalize=True, value_range=(0, 1), padding=0)
            self.logger.experiment.log(
                {"samples_horizontal": [wandb.Image(grid_horizontal)]}, step=self.global_step)

            samples_vertical = self.ema_diffusion_model.sample(
                num_samples=n * n, image_height=32, image_width=24, tape_dim=self.tape_dim, **self.sampling_kwargs)
            grid_vertical = torchvision.utils.make_grid(
                samples_vertical, nrow=n, normalize=True, value_range=(0, 1), padding=0)
            self.logger.experiment.log(
                {"samples_vertical": [wandb.Image(grid_vertical)]}, step=self.global_step)

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
