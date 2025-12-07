from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from torchvision.utils import make_grid
from fvcore.nn import FlopCountAnalysis, flop_count_table

from pytorch_lightning import LightningModule

from rin_pytorch import Rin, RinDiffusionModel
from .utils.optimization_utils import (
    build_parameters_mapping,
    get_optimizer,
    override_config_for_names,
)
from .utils.masking_schedule import parse_masking_schedule, apply_token_masking
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

        self.automatic_optimization = False

        self.image_height = rin_config["image_height"]
        self.image_width = rin_config["image_width"]
        self.tape_dim = rin_config["tape_dim"]
        self.patch_size = rin_config["patch_size"]

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

        # Token masking/dropping schedule (format: "schedule@param1,param2,...")
        # Examples: "sigmoid@0.7,0.1", "linear@0.5,0.0", "constant@0.3", "none"
        masking_schedule_str = trainer_config.get("token_masking_schedule", "none")
        masking_warmup = trainer_config.get("token_masking_warmup_steps", 0)
        self.masking_enabled = masking_schedule_str.lower() not in ("none", "disabled", "")
        if self.masking_enabled:
            self.masking_schedule_fn = parse_masking_schedule(
                schedule_str=masking_schedule_str,
                warmup_steps=masking_warmup,
                total_steps=trainer_config["train_num_steps"],
            )
        else:
            self.masking_schedule_fn = None

    def forward(self, batch_tokens, batch_class, batch_mask=None, pos_embs=None):
        return self.diffusion_model(
            batch_tokens,
            batch_class,
            attn_mask=batch_mask,
            tape_pos_emb=pos_embs,
        )

    def _extract_batch(self, batch):
        if isinstance(batch, dict):
            batch_tokens = batch["patches"]
            batch_class = batch["labels"]
            batch_mask = batch.get("patch_mask")
            pos_embs = batch.get("token_pos_embs")
        else:
            batch_tokens, batch_class = batch
            batch_mask = None
            pos_embs = None
        return batch_tokens, batch_class, batch_mask, pos_embs

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        scheduler = self.lr_schedulers()

        batch_tokens, batch_class, batch_mask, pos_embs = self._extract_batch(batch)
        batch_class = F.one_hot(batch_class, num_classes=self.num_classes).float()

        # Apply token masking/dropping based on schedule
        current_mask_ratio = 0.0
        if self.masking_enabled and self.masking_schedule_fn is not None:
            current_mask_ratio = self.masking_schedule_fn(self.global_step)
            if current_mask_ratio > 0.0:
                batch_tokens, batch_mask = apply_token_masking(
                    batch_tokens,
                    mask_ratio=current_mask_ratio,
                    existing_mask=batch_mask,
                )

        # if self._grad_accum_counter == 0:
        opt.zero_grad(set_to_none=True)

        profiler_ctx = nullcontext()
        profiling_active = False
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
            loss = self.forward(batch_tokens, batch_class, batch_mask, pos_embs)
            self.manual_backward(loss)

        self._grad_accum_counter += 1

        total_batches = getattr(self.trainer, "num_training_batches", None)
        is_last_batch = (
            isinstance(total_batches, int)
            and total_batches > 0
            and (batch_idx + 1) == total_batches
        )
        # should_step = self._grad_accum_counter >= self.grad_accum_steps or is_last_batch

        # if should_step:
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), self.clip_grad_norm)

        opt.step()
        if scheduler is not None:
            scheduler.step()

        opt.zero_grad(set_to_none=True)
        # self._grad_accum_counter = 0

        self.log(
            "loss",
            loss.detach(),
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
                batch_size=batch_tokens.size(0),
        )

        # if should_step:
        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else opt.param_groups[0]["lr"]
        self.log(
            "lr",
            current_lr,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=batch_tokens.size(0),
        )

        # Log masking ratio if enabled
        if self.masking_enabled:
            self.log(
                "mask_ratio",
                current_mask_ratio,
                on_step=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=batch_tokens.size(0),
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
                self.print(f"[torch.profiler] FLOPs (first forward/backward): {total_flops / 1e12:.4f} TFLOPs")
            else:
                self.print("[torch.profiler] Executed but no FLOPs information was collected.")

            # fvcore FLOPs analysis for forward pass only
            try:
                self._log_fvcore_flops(batch_tokens, batch_class, batch_mask, pos_embs)
            except Exception as e:
                self.print(f"[fvcore] FLOPs analysis failed: {e}")

            self._should_profile_first_step = False

        # if should_step:
        current_step = self.global_step + 1
        should_log_samples = self.log_images and (current_step % self.sample_every == 0)
        if current_step % self.ema_update_every == 0:
            self._update_ema()

        if should_log_samples:
            self._log_samples(current_step)

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

    def _log_fvcore_flops(self, batch_tokens, batch_class, batch_mask, pos_embs):
        """Compute and log GFLOPs using fvcore's FlopCountAnalysis.
        
        Analyzes the Rin denoiser directly since the full RinDiffusionModel
        contains random sampling and dynamic control flow that breaks tracing.
        
        Reports FLOPs per single sample (batch_size=1) as is standard in papers.
        """
        was_training = self.rin.training
        self.rin.eval()

        with torch.no_grad():
            # Use batch_size=1 for per-sample FLOPs (standard for paper comparisons)
            _, seq_len, token_dim = batch_tokens.shape
            device = batch_tokens.device

            # x: input tokens (1, T, D)
            dummy_x = torch.zeros(1, seq_len, token_dim, device=device)

            # t: timesteps (1,)
            dummy_t = torch.full((1,), 0.5, device=device, dtype=torch.float32)

            # cond: class conditioning (1, num_classes)
            dummy_cond = torch.zeros(1, batch_class.shape[-1], device=device)

            # latent_prev: previous latent (1, latent_slots, latent_dim)
            dummy_latent = torch.zeros(1, *self.rin.latent_shape, device=device)

            # tape_prev: previous tape (1, T, tape_dim)
            dummy_tape = torch.zeros(1, seq_len, self.rin.tape_dim, device=device)

            # For single sample, no masking needed
            dummy_mask = None
            dummy_pos_emb = pos_embs[:1] if pos_embs is not None else None

            # fvcore analyzes the Rin model forward pass
            flops = FlopCountAnalysis(
                self.rin,
                (dummy_x, dummy_t, dummy_cond, dummy_latent, dummy_tape, dummy_mask, dummy_pos_emb),
            )
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)

            total_flops = flops.total()
            gflops = total_flops / 1e9

            self.log(
                "fvcore/gflops_per_sample",
                gflops,
                on_step=True,
                prog_bar=False,
                logger=True,
                sync_dist=False,
            )
            self.print(f"[fvcore] Rin forward pass GFLOPs (per sample): {gflops:.4f}")

            # Print detailed breakdown
            self.print("[fvcore] FLOPs breakdown by operator:")
            self.print(flop_count_table(flops))

        if was_training:
            self.rin.train()

    def _log_samples(self, step):
        log_image = getattr(self.logger, "log_image", None)
        if log_image is None:
            return

        self.ema_diffusion_model.eval()
        n = 8
        num_samples = n * n

        def _snap_to_patch_multiple(value: int, upper_bound: int) -> int:
            value = min(value, upper_bound)
            remainder = value % self.patch_size
            if remainder:
                value -= remainder
            if value <= 0:
                value = self.patch_size
            return value

        with torch.no_grad():
            def _sample_with_overrides(**overrides):
                sampling_kwargs = dict(self.sampling_kwargs)
                sampling_kwargs.update(overrides)
                return self.ema_diffusion_model.sample(**sampling_kwargs)

            samples = _sample_with_overrides(num_samples=num_samples)
            grid = make_grid(samples, nrow=n, normalize=True, value_range=(0, 1), padding=0)
            self._safe_log_image(log_image, "samples", [grid], step)

            horizontal_height = _snap_to_patch_multiple(int(self.image_height * 0.75), self.image_height)
            samples_horizontal = _sample_with_overrides(
                num_samples=num_samples,
                image_height=horizontal_height,
                image_width=self.image_width,
                tape_dim=self.tape_dim,
            )
            grid_horizontal = make_grid(
                samples_horizontal, nrow=n, normalize=True, value_range=(0, 1), padding=0
            )
            self._safe_log_image(log_image, "samples_horizontal", [grid_horizontal], step)

            vertical_width = _snap_to_patch_multiple(int(self.image_width * 0.75), self.image_width)
            samples_vertical = _sample_with_overrides(
                num_samples=num_samples,
                image_height=self.image_height,
                image_width=vertical_width,
                tape_dim=self.tape_dim,
            )
            grid_vertical = make_grid(
                samples_vertical, nrow=n, normalize=True, value_range=(0, 1), padding=0
            )
            self._safe_log_image(log_image, "samples_vertical", [grid_vertical], step)

        del samples
        del samples_horizontal
        del samples_vertical
        self.ema_diffusion_model.train()

    def _safe_log_image(self, log_image_fn, key, images, step):
        try:
            log_image_fn(key=key, images=images, step=step)
        except Exception as exc:  # noqa: BLE001
            self.print(f"[wandb] Failed to log {key}: {exc}")
