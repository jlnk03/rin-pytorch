import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import webdataset as wds
from huggingface_hub import HfFileSystem, get_token, hf_hub_url
from datasets import load_dataset

from .utils.FlexibleCifar import FlexibleCIFAR10
from .utils.ragged_tensor import (
    ragged_list_to_tensor, 
    ragged_tensor_to_list, 
    get_ragged_lengths,
    get_document_ids
)

from PIL import Image

from rin_pytorch import Rin, RinDiffusionModel
from .utils.optimization_utils import (
    build_parameters_mapping,
    get_optimizer,
    override_config_for_names,
)
from .utils.pos_embedding import create_2d_sin_cos_pos_emb
from .utils.logging_utils import log_first_tensor, log_first_document

import wandb

import os
from dotenv import load_dotenv

from diffusers.optimization import get_scheduler as get_lr_scheduler

from torchmetrics.image.fid import FrechetInceptionDistance

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
    num_mask = int(mask_ratio * t)
    
    rand_scores = torch.rand(b, t, device=x.device)
    _, indices = torch.topk(rand_scores, k=num_mask, dim=1, largest=False)
    
    token_mask = torch.zeros((b, t), dtype=torch.bool, device=x.device)
    token_mask.scatter_(1, indices, True)
    
    mask = token_mask.unsqueeze(-1).expand(b, t, c)

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

class ImageNetWebDataset(Dataset):
    def __init__(self, split='train', transform=None):
        super().__init__()
        self.transform = transform

        try:
            self.dataset = load_dataset("imagenet-1k", split="train", trust_remote_code=True)
        except Exception as e:
            print(e)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    # C, H, W -> T, D
    c, h, w = x.shape
    nh, nw = h // p, w // p
    x = x.view(c, nh, p, nw, p)
    x = x.permute(1, 3, 2, 4, 0).contiguous()
    x = x.view(nh * nw, p * p * c)
    return x


def pad_to_max_size(batch, patch_size, tape_dim, transform=None, return_padded_images_for_fid=False):
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
    
    padded_images = []
    pos_embs = []
    first_logged2 = False
    fid_images_for_batch = []
    for img in images:
        c, h, w = img.shape

        h_crop = h - (h // patch_size) * patch_size
        w_crop = w - (w // patch_size) * patch_size
        
        if h_crop > 0 or w_crop > 0:
            img = img[:, :h - h_crop, :w - w_crop]
            _, h, w = img.shape

        if c == 1:
            img = img.repeat(3, 1, 1)
        
        if not first_logged2:
            log_first_tensor("dataloader.image_cropped", img)

        nh = h // patch_size
        nw = w // patch_size
        pixel_row = patchify(img, patch_size)

        if not first_logged2:
            log_first_tensor("dataloader.image_patchified", pixel_row)
        
        pos_emb = create_2d_sin_cos_pos_emb(nh, nw, tape_dim)

        padded_images.append(pixel_row)
        pos_embs.append(pos_emb)

        if return_padded_images_for_fid:
            fid_images_for_batch.append(img)

        if not first_logged2:
            first_logged2 = True

    labels = torch.tensor(labels)
    padded_images, offsets = ragged_list_to_tensor(padded_images)

    pos_embs, offsets_pos_embs = ragged_list_to_tensor(pos_embs)

    document_ids = get_document_ids(offsets)

    if return_padded_images_for_fid and len(fid_images_for_batch) > 0:
        max_h = max(x.shape[1] for x in fid_images_for_batch)
        max_w = max(x.shape[2] for x in fid_images_for_batch)
        fid_padded = []
        for x in fid_images_for_batch:
            pad_h = max_h - x.shape[1]
            pad_w = max_w - x.shape[2]
            x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)
            fid_padded.append(x)
        fid_batch = torch.stack(fid_padded, dim=0)
        return padded_images, pos_embs, labels, offsets, offsets_pos_embs, document_ids, fid_batch

    return padded_images, pos_embs, labels, offsets, offsets_pos_embs, document_ids


class ImageNetDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.overfit_one_sample = self.config["trainer"].get("overfit_one_sample", False)
        self.overfit_class = self.config["trainer"].get("overfit_class", None)
        if self.config["run"]["cifar"]:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)) if self.config["run"]["vanilla"] else ResizeMaxSide(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        # Validation will use the exact same transform via collate_fn; datasets return PIL
    
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

        # Optionally wrap to overfit on a single sample from a random (or specified) class
        if self.overfit_one_sample:
            def extract_label(sample):
                if isinstance(sample, dict):
                    return sample.get("label")
                # Fallbacks if dataset returns tuples
                if isinstance(sample, tuple) and len(sample) > 1:
                    return sample[1]
                return None

            # Decide target class
            target_class = self.overfit_class
            if target_class is None:
                # Prefer picking a random class from config if available
                try:
                    num_classes = int(self.config["rin"]["num_classes"])
                    target_class = torch.randint(low=0, high=num_classes, size=(1,)).item()
                except Exception:
                    target_class = None

            # Find one sample matching the target class (or just take the first)
            chosen_sample = None
            try:
                dataset_len = len(self.train_dataset)
            except Exception:
                dataset_len = 0

            if dataset_len and dataset_len > 0:
                # If we know the length, scan deterministically
                for idx in range(dataset_len):
                    sample = self.train_dataset[idx]
                    if target_class is None or extract_label(sample) == target_class:
                        chosen_sample = sample
                        break
                if chosen_sample is None:
                    # Fallback to the first sample
                    chosen_sample = self.train_dataset[0]
                    target_class = extract_label(chosen_sample)
            else:
                # Iterable fallback: iterate a few times to find a matching sample
                try:
                    for sample in self.train_dataset:
                        if target_class is None or extract_label(sample) == target_class:
                            chosen_sample = sample
                            break
                except Exception:
                    chosen_sample = None

            if chosen_sample is None:
                raise RuntimeError("Failed to select a sample for overfitting.")

            # Store chosen class for logging/sampling alignment and propagate to config
            self.chosen_overfit_class = extract_label(chosen_sample)
            try:
                # Mutate shared config so the module can see it
                self.config["trainer"]["overfit_class"] = int(self.chosen_overfit_class) if self.chosen_overfit_class is not None else None
            except Exception:
                pass

            class OverfitOneSampleDataset(Dataset):
                def __init__(self, sample, length_hint: int | None = None):
                    self.sample = sample
                    self._length = length_hint if (isinstance(length_hint, int) and length_hint > 0) else 1_000_000

                def __len__(self):
                    return self._length

                def __getitem__(self, idx):
                    return self.sample

            # Replace the training dataset with the constant-sample dataset
            length_hint = None
            try:
                length_hint = len(self.train_dataset)
            except Exception:
                pass
            self.train_dataset = OverfitOneSampleDataset(chosen_sample, length_hint)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["trainer"]["train_batch_size"],
            num_workers=self.config["trainer"]["num_dl_workers"],
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
            collate_fn=lambda batch: pad_to_max_size(batch, self.config["rin"]["patch_size"], self.config["rin"]["tape_dim"], self.transform),
            # shuffle=False
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

        self.image_height = rin_config["image_height"]
        self.image_width = rin_config["image_width"]

        # Overfit mode controls (for sampling/logging)
        self.overfit_one_sample = self.hparams["trainer"].get("overfit_one_sample", False)
        self.overfit_class = self.hparams["trainer"].get("overfit_class", None)
    
    def forward(self, batch_img, pos_embs, batch_class, offsets, offsets_pos_embs, document_ids):
        return self.diffusion_model(batch_img, pos_embs, batch_class, offsets, offsets_pos_embs, document_ids)
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        
        # batch_img, batch_mask, image_mask, batch_class, pos_embs, nmh, nmw = batch
        batch_img, pos_embs, batch_class, offsets, offsets_pos_embs, document_ids = batch
        # Log only the first packed sample using document_ids
        log_first_document("trainer.batch_img_in", batch_img, document_ids)
        # Capture overfit class from first batch if not set yet
        if self.overfit_one_sample and self.overfit_class is None:
            try:
                self.overfit_class = int(batch_class[0].item())
            except Exception:
                pass
        batch_class = torch.nn.functional.one_hot(batch_class, num_classes=self.num_classes).float()

        opt.zero_grad()
        # Optional FLOPs profiling (run once on the first training_step that hits this code)
        if self.hparams["trainer"].get("profile_flops", False) and not getattr(self, "_flops_profiled", False):
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
                torch.cuda.synchronize()

            # Profile forward pass
            with torch.profiler.profile(activities=activities, with_flops=True) as prof_fwd:
                loss = self(batch_img, pos_embs, batch_class, offsets, offsets_pos_embs, document_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            fwd_flops = float(sum(getattr(ev, "flops", 0) for ev in prof_fwd.key_averages()))

            # Profile backward pass
            with torch.profiler.profile(activities=activities, with_flops=True) as prof_bwd:
                self.manual_backward(loss)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            bwd_flops = float(sum(getattr(ev, "flops", 0) for ev in prof_bwd.key_averages()))

            step_total_flops = float(fwd_flops + bwd_flops)

            print(f"FLOPs/forward_total: {fwd_flops}")
            print(f"FLOPs/backward_total: {bwd_flops}")
            print(f"FLOPs/step_total: {step_total_flops}")
            print(f"FLOPs/forward_G: {fwd_flops / 1e9}")
            print(f"FLOPs/backward_G: {bwd_flops / 1e9}")
            print(f"FLOPs/step_G: {step_total_flops / 1e9}")

            # Log as metrics (reliable across logger types)
            self.log("FLOPs/forward_total", torch.tensor(fwd_flops), on_step=True, prog_bar=False, logger=True)
            self.log("FLOPs/backward_total", torch.tensor(bwd_flops), on_step=True, prog_bar=False, logger=True)
            self.log("FLOPs/step_total", torch.tensor(step_total_flops), on_step=True, prog_bar=False, logger=True)
            self.log("FLOPs/forward_G", torch.tensor(fwd_flops / 1e9), on_step=True, prog_bar=False, logger=True)
            self.log("FLOPs/backward_G", torch.tensor(bwd_flops / 1e9), on_step=True, prog_bar=False, logger=True)
            self.log("FLOPs/step_G", torch.tensor(step_total_flops / 1e9), on_step=True, prog_bar=False, logger=True)

            # Best-effort: also push to WandB summary if available
            try:
                exp = None
                if isinstance(self.logger, WandbLogger):
                    exp = self.logger.experiment
                elif hasattr(self.logger, "experiment"):
                    exp = getattr(self.logger, "experiment", None)
                if exp is not None and hasattr(exp, "summary"):
                    exp.summary["FLOPs/forward_total"] = int(fwd_flops)
                    exp.summary["FLOPs/backward_total"] = int(bwd_flops)
                    exp.summary["FLOPs/step_total"] = int(step_total_flops)
                    exp.summary["FLOPs/forward_G"] = float(fwd_flops) / 1e9
                    exp.summary["FLOPs/backward_G"] = float(bwd_flops) / 1e9
                    exp.summary["FLOPs/step_G"] = step_total_flops / 1e9
                else:
                    # Fallback to global wandb if active
                    if hasattr(wandb, "run") and wandb.run is not None:
                        wandb.run.summary["FLOPs/forward_total"] = int(fwd_flops)
                        wandb.run.summary["FLOPs/backward_total"] = int(bwd_flops)
                        wandb.run.summary["FLOPs/step_total"] = int(step_total_flops)
                        wandb.run.summary["FLOPs/forward_G"] = float(fwd_flops) / 1e9
                        wandb.run.summary["FLOPs/backward_G"] = float(bwd_flops) / 1e9
                        wandb.run.summary["FLOPs/step_G"] = step_total_flops / 1e9
            except Exception:
                pass
            finally:
                # Ensure we only profile once
                self._flops_profiled = True
        else:
            loss = self(batch_img, pos_embs, batch_class, offsets, offsets_pos_embs, document_ids)
            self.manual_backward(loss)

        opt.step()

        sch = self.lr_schedulers()
        sch.step()

        # logs = {
        #     "loss": loss.item(),
        #     "lr": sch.get_last_lr()[0],
        # }

        # logs = {
        #     "loss": 1,
        #     "lr": 1,
        # }

        # self.log_dict(logs, on_step=True, prog_bar=True)

                # logs = {
        #     "loss": loss.item(),
        #     "lr": sch.get_last_lr()[0],
        # }

        # Prefer: log tensors to avoid graph breaks; LR is handled by LearningRateMonitor
        # Expose to callbacks (e.g., ModelCheckpoint) but do not send to external loggers
        self.log("loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=False, sync_dist=True)

        # If you still want to log LR yourself, convert to a tensor (optional)
        # lr_tensor = torch.tensor(sch.get_last_lr()[0], device=loss.device)
        # self.log("lr", lr_tensor, on_step=True, prog_bar=False, logger=True)

        # Update EMA model
        if self.global_step % self.ema_update_every == 0:
            with torch.no_grad():
                for ema_param, param in zip(self.ema_diffusion_model.parameters(), self.diffusion_model.parameters()):
                    if param.requires_grad:
                        ema_param.data.lerp_(param.data, 1 - self.ema_decay)

        # Generate samples
        if self.global_step % self.sample_every == 0:
            self.ema_diffusion_model.eval()
            n = 1 if self.overfit_one_sample else 8
            class_override = self.overfit_class
            samples = self.ema_diffusion_model.sample(num_samples=n * n, image_height=self.image_height, image_width=self.image_width, tape_dim=self.tape_dim, class_override=class_override, **self.sampling_kwargs)
            grid = torchvision.utils.make_grid(samples, nrow=n, normalize=True, value_range=(0, 1), padding=0)
            self.logger.experiment.log({"samples": [wandb.Image(grid)]}, step=self.global_step)

            samples_horizontal = self.ema_diffusion_model.sample(num_samples=n * n, image_height=int(self.image_height * 0.75), image_width=self.image_width, tape_dim=self.tape_dim, class_override=class_override, **self.sampling_kwargs)
            grid_horizontal = torchvision.utils.make_grid(samples_horizontal, nrow=n, normalize=True, value_range=(0, 1), padding=0)
            self.logger.experiment.log({"samples_horizontal": [wandb.Image(grid_horizontal)]}, step=self.global_step)

            samples_vertical = self.ema_diffusion_model.sample(num_samples=n * n, image_height=self.image_height, image_width=int(self.image_width * 0.75), tape_dim=self.tape_dim, class_override=class_override, **self.sampling_kwargs)
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