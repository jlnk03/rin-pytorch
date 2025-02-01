from pathlib import Path

import torch
import wandb
from accelerate import Accelerator
from diffusers.optimization import get_scheduler as get_lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from tqdm import tqdm

from .RinDiffusionModel import RinDiffusionModel
from .utils.optimization_utils import (
    build_parameters_mapping,
    get_optimizer,
    override_config_for_names,
)

from .utils.pos_embedding import create_2d_sin_cos_pos_emb

from torch.nn.functional import pad

import torch
from torch.nn.functional import pad, avg_pool2d
from einops import rearrange

import os

from dotenv import load_dotenv

load_dotenv()

def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    # C, H, W -> T, D
    c, h, w = x.shape
    nh, nw = h // p, w // p
    x = x.view(c, nh, p, nw, p)
    x = x.permute(1, 3, 2, 4, 0).contiguous()
    x = x.view(nh * nw, p * p * c)
    return x

def pad_to_max_size(batch, patch_size, tape_dim):
    # Extract images and labels from the batch
    images, labels = zip(*batch)
    
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
        pixel_row = pad(pixel_row, (0, 0, 0, nmh * nmw - pixel_row.shape[0]), value=0)
        
        pixel_mask = pad(pixel_mask, (0, max_pixels - pixel_mask.shape[0]), value=0)
        patch_mask = pad(patch_mask, (0, nmh * nmw - patch_mask.shape[0]), value=0)
        # Only pad the first dimension (0), leave second dimension unchanged
        pos_emb = pad(pos_emb, (0, 0, 0, nmh * nmw - pos_emb.shape[0]), value=0)
        image_masks.append(pixel_mask)
        patch_masks.append(patch_mask)

        padded_images.append(pixel_row)
        pos_embs.append(pos_emb)

    # Convert labels to a tensor
    labels = torch.tensor(labels)
    padded_images = torch.stack(padded_images)
    patch_masks = torch.stack(patch_masks).bool()
    image_masks = torch.stack(image_masks).bool()
    pos_embs = torch.stack(pos_embs)
    
    return padded_images, patch_masks, image_masks, labels, pos_embs, nmh, nmw


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class Trainer:
    def __init__(
        self,
        diffusion_model: RinDiffusionModel,
        ema_diffusion_model: RinDiffusionModel,  # since RinDiffusionModel can't be copied, we need a second one for EMA
        dataset: Dataset,
        num_classes: int,
        train_num_steps: int,
        train_batch_size=256,
        split_batches=True,
        fp16=False,
        amp=False,
        lr_scheduler_name="cosine",
        lr=1e-4,
        lr_warmup_steps=1000,
        optimizer_name="lamb",
        optimizer_exclude_weight_decay=["bias", "beta", "gamma"],
        optimizer_kwargs=dict(weight_decay=1e-2),
        clip_grad_norm=None,
        sample_every=1000,
        num_dl_workers=2,
        ema_decay=0.9999,
        ema_update_every=1,
        sampling_kwargs=dict(iterations=100, method="ddim"),
        checkpoint_folder=os.getenv("CHECKPOINT_PATH"),
        run_name="rin_16",
        log_to_wandb=True,
        patch_size=2,
        tape_dim=256,
    ):
        self.accelerator = Accelerator(split_batches=split_batches, mixed_precision="fp16" if fp16 else "no")
        self.accelerator.native_amp = amp

        self.diffusion_model = diffusion_model

        self.num_classes = num_classes
        self.train_num_steps = train_num_steps
        self.clip_grad_norm = clip_grad_norm
        self.sample_every = sample_every
        self.ema_decay = ema_decay
        self.ema_update_every = ema_update_every
        self.sampling_kwargs = sampling_kwargs
        self.fixed_class = None

        self.patch_size = patch_size
        self.tape_dim = tape_dim

        dl = DataLoader(
            dataset,
            batch_size=train_batch_size,
            # shuffle=True,
            num_workers=num_dl_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
            collate_fn=lambda batch: pad_to_max_size(batch, patch_size, tape_dim)
        )

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        self.optimizer = get_optimizer(
            optimizer_name,
            override_config_for_names(
                self.diffusion_model.parameters(),
                optimizer_exclude_weight_decay,
                {"weight_decay": 0.0, "disable_layer_adaption": True},
                build_parameters_mapping(self.diffusion_model),
            ),
            lr=lr,
            **optimizer_kwargs,
        )

        self.lr_scheduler = get_lr_scheduler(
            lr_scheduler_name,
            self.optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=train_num_steps,
        )

        self.checkpoint_folder = Path(checkpoint_folder)

        if self.accelerator.is_main_process:
            self.ema_diffusion_model = ema_diffusion_model
            self.ema_diffusion_model.requires_grad_(False)

            self.checkpoint_folder.mkdir(exist_ok=True, parents=True)

        self.step = 0

        self.diffusion_model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.diffusion_model, self.optimizer, self.lr_scheduler
        )

        if self.accelerator.is_main_process:
            wandb.init(project="rin", name=run_name, mode="online" if log_to_wandb else "disabled")

    def save(self, milestone, absolute=False):
        if not self.accelerator.is_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.diffusion_model),
            "ema_model": self.ema_diffusion_model.state_dict(),
            "opt": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }

        if absolute:
            checkpoint_file = milestone
        else:
            checkpoint_file = self.checkpoint_folder / f"model-{milestone}.pt"

        torch.save(data, checkpoint_file)

    def load(self, milestone, absolute=False):
        if absolute:
            checkpoint_file = milestone
        else:
            checkpoint_file = self.checkpoint_folder / f"model-{milestone}.pt"

        data = torch.load(checkpoint_file)

        self.step = data["step"]

        diffusion_model = self.accelerator.unwrap_model(self.diffusion_model)
        diffusion_model.load_state_dict(data["model"])

        if self.accelerator.is_main_process:
            self.ema_diffusion_model.load_state_dict(data["ema_model"])

        self.optimizer.load_state_dict(data["opt"])

        self.lr_scheduler.load_state_dict(data["lr_scheduler"])

    def train(self):
        self.diffusion_model.train()

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not self.accelerator.is_main_process,
            desc="Training",
        ) as pbar:
            while self.step < self.train_num_steps:
                batch_img, batch_mask, image_mask, batch_class, pos_embs, nmh, nmw = next(self.dl)
                batch_class = torch.nn.functional.one_hot(batch_class, num_classes=self.num_classes).float()

                self.optimizer.zero_grad()

                loss = self.diffusion_model(batch_img, batch_mask, image_mask, batch_class, pos_embs, nmh, nmw)

                self.accelerator.backward(loss)

                if self.clip_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(self.diffusion_model.parameters(), self.clip_grad_norm)

                self.optimizer.step()

                logs = {
                    "loss": loss.item(),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                }

                pbar.set_postfix(logs)
                if self.accelerator.is_main_process:
                    wandb.log(logs, step=self.step)

                self.step += 1
                self.lr_scheduler.step()
                pbar.update(1)

                if self.accelerator.is_main_process:
                    if self.step % self.ema_update_every == 0:
                        # perform ema update
                        with torch.no_grad():
                            for ema_param, param in zip(
                                self.ema_diffusion_model.parameters(),
                                self.diffusion_model.parameters(),
                            ):
                                if param.requires_grad:
                                    ema_param.data.lerp_(param.data, 1 - self.ema_decay)

                    if self.step % self.sample_every == 0:
                        self.ema_diffusion_model.eval()
                        n = 8
                        if self.fixed_class is not None:
                            self.sampling_kwargs['class_override'] = self.fixed_class
                        samples = self.ema_diffusion_model.sample(num_samples=n * n, tape_dim=self.tape_dim, **self.sampling_kwargs)

                        samples = make_grid(samples, nrow=n, normalize=True, value_range=(0, 1), padding=0)
                        wandb.log({"samples": [wandb.Image(samples)]}, step=self.step)

                        self.save("latest")
                        del samples

