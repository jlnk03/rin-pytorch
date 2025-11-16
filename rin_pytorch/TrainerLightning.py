from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from torchvision.utils import make_grid

from pytorch_lightning import LightningModule

import wandb

from rin_pytorch import Rin, RinDiffusionModel
from rin_pytorch.data import build_training_dataloader, resolve_data_root
from .utils.optimization_utils import (
    build_parameters_mapping,
    get_optimizer,
    override_config_for_names,
)
from diffusers.optimization import get_scheduler as get_lr_scheduler


class RinLightningModule(LightningModule):
    """
    Lean LightningModule that mirrors the functionality of `Trainer`
    from `train_imagenet.py` while keeping EMA updates, sampling,
    custom optimizer parameter groups, and first-step profiling.
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        rin_config = config["rin"]
        diffusion_config = config["diffusion"]
        trainer_config = config["trainer"]
        self._config = config
        self._train_loader: Optional[torch.utils.data.DataLoader] = None

        self.automatic_optimization = False

        self.rin = Rin(**rin_config)
        self.rin.pass_dummy_data(num_classes=rin_config["num_classes"])
        self.diffusion_model = RinDiffusionModel(rin=self.rin, **diffusion_config)

        self.rin_ema = Rin(**rin_config)
        self.rin_ema.pass_dummy_data(num_classes=rin_config["num_classes"])
        self.ema_diffusion_model = RinDiffusionModel(rin=self.rin_ema, **diffusion_config)
        self.ema_diffusion_model.requires_grad_(False)

        self.num_classes = rin_config["num_classes"]
        self.sample_every = trainer_config["sample_every"]
        self.sampling_kwargs = trainer_config["sampling_kwargs"]
        self.ema_decay = trainer_config["ema_decay"]
        self.ema_update_every = trainer_config["ema_update_every"]
        self.clip_grad_norm = trainer_config.get("clip_grad_norm")
        self.train_batch_size = trainer_config["train_batch_size"]
        self.grad_accum_steps = max(1, trainer_config.get("gradient_accumulation_steps", 1))
        self.log_images = trainer_config.get("log_to_wandb", True)

        self._should_profile_first_step = True
        self._grad_accum_counter = 0
        self._trainer_step = 0

    def forward(self, batch_img, batch_class):
        return self.diffusion_model(batch_img, batch_class)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        scheduler = self.lr_schedulers()

        batch_img, batch_class = batch
        batch_class = F.one_hot(batch_class, num_classes=self.num_classes).float()
        self._trainer_step += 1
        should_log_samples = self.log_images and (self._trainer_step % self.sample_every == 0)

        if self._grad_accum_counter == 0:
            opt.zero_grad(set_to_none=True)

        profiler_ctx = nullcontext()
        profiling_active = False
        prof = None
        if self._should_profile_first_step:
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
            profiler_ctx = profile(
                activities=activities,
                record_shapes=True,
                profile_memory=False,
                with_stack=False,
                with_flops=True,
            )
            profiling_active = True

        with profiler_ctx as prof:
            loss = self.forward(batch_img, batch_class)
            loss_to_backward = loss / self.grad_accum_steps
            self.manual_backward(loss_to_backward)

        self._grad_accum_counter += 1

        total_batches = getattr(self.trainer, "num_training_batches", None)
        is_last_batch = (
            isinstance(total_batches, int) and total_batches > 0 and (batch_idx + 1) == total_batches
        )
        should_step = self._grad_accum_counter >= self.grad_accum_steps or is_last_batch

        if should_step:
            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), self.clip_grad_norm)

            opt.step()
            if scheduler is not None:
                scheduler.step()

            opt.zero_grad(set_to_none=True)
            self._grad_accum_counter = 0

        self.log(
            "loss",
            loss.detach(),
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_img.size(0),
        )

        if should_step:
            if scheduler is not None:
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = opt.param_groups[0]["lr"]

            self.log(
                "lr",
                current_lr,
                on_step=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=batch_img.size(0),
            )

        if profiling_active and prof is not None:
            total_flops = sum(event.flops for event in prof.key_averages() if event.flops)
            if total_flops:
                self.log(
                    "profiler/total_flops",
                    float(total_flops),
                    on_step=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                )
                self.log(
                    "profiler/total_tflops",
                    float(total_flops) / 1e12,
                    on_step=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                )
                self.print(f"Profiled FLOPs (first forward/backward): {total_flops / 1e12:.4f} TFLOPs")
            else:
                self.print("Profiler executed but no FLOPs information was collected.")
            self._should_profile_first_step = False

        if should_step:
            current_step = self.global_step + 1
            if current_step % self.ema_update_every == 0:
                self._update_ema()

        if should_log_samples:
            self._log_samples(self._trainer_step)

        return loss

    def train_dataloader(self):
        if self._train_loader is None:
            data_root = resolve_data_root(None, self._config)
            self._train_loader = build_training_dataloader(self._config, data_root)
        return self._train_loader

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
            },
        }

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_model"] = self.ema_diffusion_model.state_dict()
        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        if "ema_model" in checkpoint:
            self.ema_diffusion_model.load_state_dict(checkpoint["ema_model"])

    def _update_ema(self):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_diffusion_model.parameters(), self.diffusion_model.parameters()):
                if param.requires_grad:
                    ema_param.data.lerp_(param.data, 1 - self.ema_decay)

    def _log_samples(self, step):
        logger = getattr(self.logger, "experiment", None)
        if logger is None:
            return

        self.ema_diffusion_model.eval()
        n = 8
        with torch.no_grad():
            samples = self.ema_diffusion_model.sample(num_samples=n * n, **self.sampling_kwargs)

        grid = make_grid(samples, nrow=n, normalize=True, value_range=(0, 1), padding=0)
        logger.log({"samples": [wandb.Image(grid)]}, step=step)

        del samples
        self.ema_diffusion_model.train()