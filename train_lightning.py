import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only, rank_zero_warn

import wandb
import yaml

from rin_pytorch.TrainerLightning import RinLightningModule
from rin_pytorch.data import build_training_dataloader, resolve_data_root

torch.set_float32_matmul_precision("medium")


try:
    from wandb.errors import Error as WandbError
except Exception:  # pragma: no cover - wandb is expected to be installed
    class WandbError(Exception):
        pass

try:
    from wandb.sdk.interface.router import MessageRouterClosedError
except Exception:  # pragma: no cover - fallback for wandb internals
    class MessageRouterClosedError(Exception):
        pass

try:
    from wandb.sdk.lib.sock_client import SockClientClosedError
except Exception:  # pragma: no cover - fallback for wandb internals
    class SockClientClosedError(Exception):
        pass

try:
    from wandb.sdk.lib.sock_client import BrokenPipeError
except Exception:  # pragma: no cover - fallback for wandb internals
    class BrokenPipeError(Exception):
        pass


class ResilientWandbLogger(WandbLogger):
    """A WandB logger that keeps training alive when network/logging fails."""

    _RECOVERABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
        WandbError,
        MessageRouterClosedError,
        SockClientClosedError,
        OSError,
        BrokenPipeError,
    )

    def __init__(
        self,
        *args: Any,
        reconnect_cooldown: float = 120.0,
        max_retries: int = 5,
        **kwargs: Any,
    ) -> None:
        self._reconnect_cooldown = reconnect_cooldown
        self._max_retries = max_retries
        self._logging_disabled = False
        self._last_failure_ts = 0.0
        self._failure_count = 0
        super().__init__(*args, **kwargs)

    def _run_with_resilience(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        if self._logging_disabled:
            self._attempt_reconnect()
        if self._logging_disabled:
            return
        try:
            fn(*args, **kwargs)
            self._failure_count = 0
        except self._RECOVERABLE_EXCEPTIONS as exc:
            self._handle_failure(fn, exc)

    def _handle_failure(self, fn: Callable[..., Any], exc: BaseException) -> None:
        self._logging_disabled = True
        self._last_failure_ts = time.monotonic()
        self._failure_count += 1
        rank_zero_warn(
            f"[wandb] Disabling logging after '{getattr(fn, '__name__', fn.__class__.__name__)}' failed with {exc!r}. "
            "Training will continue without W&B logging."
        )

    def _attempt_reconnect(self) -> None:
        if self._failure_count >= self._max_retries:
            return
        if (time.monotonic() - self._last_failure_ts) < self._reconnect_cooldown:
            return
        try:
            if wandb.run is not None:
                wandb.finish()
        except Exception:  # noqa: BLE001 - best effort cleanup
            pass

        self._experiment = None  # type: ignore[assignment]
        self._logging_disabled = False
        try:
            _ = self.experiment
        except self._RECOVERABLE_EXCEPTIONS as exc:
            self._handle_failure(self._attempt_reconnect, exc)
        else:
            rank_zero_info("[wandb] Successfully reconnected to Weights & Biases.")

    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:  # type: ignore[override]
        super_log_metrics = super().log_metrics
        self._run_with_resilience(super_log_metrics, metrics, step)

    def _log_media_impl(self, payload: Mapping[str, Any], step: Optional[int]) -> None:
        experiment = self.experiment
        experiment.log(payload, step=step)

    @rank_zero_only
    def log_media(self, payload: Mapping[str, Any], step: Optional[int] = None) -> None:
        self._run_with_resilience(self._log_media_impl, payload, step)


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Rin Lightning Training Script")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/128.yaml",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to the ImageNet-style folder. Overrides config/default.",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--wandb_resume",
        type=str,
        default=None,
        help="WandB resume flag or run id to resume an existing run.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    trainer_cfg = config["trainer"]

    data_root = resolve_data_root(args.data_root, config)

    checkpoint_root = Path(trainer_cfg["checkpoint_folder"])
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = trainer_cfg["run_name"]
    checkpoint_dir = checkpoint_root / f"{run_name}_{timestamp}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer_cfg["checkpoint_folder"] = str(checkpoint_dir)

    train_loader = build_training_dataloader(config, data_root)

    wandb_logger = None
    if trainer_cfg["log_to_wandb"]:
        wandb_logger = ResilientWandbLogger(
            project="rin",
            name=run_name,
            log_model=False,
            id=args.wandb_resume,
            resume=True if args.wandb_resume else False,
            reconnect_cooldown=trainer_cfg.get("wandb_reconnect_cooldown", 120),
            max_retries=trainer_cfg.get("wandb_max_retries", 5),
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=trainer_cfg["checkpoint_folder"],
        filename="model-{step}",
        every_n_train_steps=trainer_cfg["sample_every"],
        save_weights_only=False,
        save_top_k=1,
        save_last=True,
        monitor="loss",
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = torch.cuda.device_count() if accelerator == "gpu" else 1
    # devices = max(1, devices)
    devices = 4
    precision = "bf16" if trainer_cfg.get("fp16", False) else "32-true"

    trainer = pl.Trainer(
        max_steps=trainer_cfg["train_num_steps"],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator=accelerator,
        devices=devices,
        num_nodes=4,
        precision=precision,
        strategy="ddp_find_unused_parameters_true" if devices > 1 else "auto",
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 50),
    )

    with trainer.init_module():
        model = RinLightningModule(config)

    trainer.fit(model, train_dataloaders=train_loader, ckpt_path=args.resume_checkpoint)


if __name__ == "__main__":
    main()