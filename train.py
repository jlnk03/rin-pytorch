import torch
from datetime import datetime
import pytorch_lightning as pl
from lightning.pytorch.strategies import DDPStrategy
from rin_pytorch.TrainerLightning import (
    ImageNetDataModule,
    RinLightningModule
)
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import argparse
import yaml
from rin_pytorch.utils.logging_utils import set_log_path

torch._dynamo.config.recompile_limit = 16
# torch._dynamo.explain()
# torch._dynamo.config.capture_scalar_outputs = True

from dotenv import load_dotenv
load_dotenv()

import os

os.environ["TORCH_LOGS"] = "recompiles"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"


# pl.seed_everything(42, workers=True)

torch.set_float32_matmul_precision('medium')

def _move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj

def maybe_profile_flops_precompile(model, data_module):
    try:
        # Respect config flag if present
        profile = False
        try:
            profile = bool(model.hparams["trainer"].get("profile_flops", False))
        except Exception:
            pass
        if not profile or getattr(model, "_flops_profiled", False):
            return

        # Ensure dataloaders are ready
        try:
            data_module.setup(stage="fit")
        except Exception:
            try:
                data_module.setup()
            except Exception:
                pass
        train_loader = data_module.train_dataloader()
        first_batch = next(iter(train_loader))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.train()

        # Unpack training-style batch; ignore any extra items
        if isinstance(first_batch, (list, tuple)) and len(first_batch) >= 6:
            batch_img, pos_embs, batch_class, offsets, offsets_pos_embs, document_ids = first_batch[:6]
        elif isinstance(first_batch, dict) and "image" in first_batch:
            # Fallback for unexpected formats
            batch_img = first_batch["image"]
            pos_embs = offsets = offsets_pos_embs = document_ids = torch.empty(0)
            batch_class = torch.zeros(batch_img.size(0), dtype=torch.long)
        else:
            return

        batch_img, pos_embs, batch_class, offsets, offsets_pos_embs, document_ids = _move_to_device(
            (batch_img, pos_embs, batch_class, offsets, offsets_pos_embs, document_ids), device
        )

        import torch.profiler as tprof

        activities = [tprof.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(tprof.ProfilerActivity.CUDA)
            torch.cuda.synchronize()

        # Run in eager (disable Dynamo) so profiler isn't ignored
        import torch._dynamo as dynamo
        with dynamo.disable():
            with tprof.profile(activities=activities, with_flops=True) as prof_fwd:
                one_hot = torch.nn.functional.one_hot(batch_class, num_classes=model.num_classes).float()
                loss = model(batch_img, pos_embs, one_hot, offsets, offsets_pos_embs, document_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            fwd_flops = float(sum(getattr(ev, "flops", 0) for ev in prof_fwd.key_averages()))

            with tprof.profile(activities=activities, with_flops=True) as prof_bwd:
                loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            bwd_flops = float(sum(getattr(ev, "flops", 0) for ev in prof_bwd.key_averages()))

        step_total_flops = float(fwd_flops + bwd_flops)
        print(f"Precompile FLOPs/forward_total: {fwd_flops}")
        print(f"Precompile FLOPs/backward_total: {bwd_flops}")
        print(f"Precompile FLOPs/step_total: {step_total_flops}")

        # Best-effort WandB summary
        try:
            import wandb
            if getattr(wandb, "run", None) is not None:
                wandb.run.summary["FLOPs/forward_total"] = int(fwd_flops)
                wandb.run.summary["FLOPs/backward_total"] = int(bwd_flops)
                wandb.run.summary["FLOPs/step_total"] = int(step_total_flops)
                wandb.run.summary["FLOPs/forward_G"] = float(fwd_flops) / 1e9
                wandb.run.summary["FLOPs/backward_G"] = float(bwd_flops) / 1e9
                wandb.run.summary["FLOPs/step_G"] = step_total_flops / 1e9
        except Exception:
            pass
        finally:
            # Mark as profiled to skip in-training profiling
            setattr(model, "_flops_profiled", True)
            model.zero_grad(set_to_none=True)
    except Exception as e:
        print(f"Precompile FLOPs profiling failed: {e}")

def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="Rin Training Script")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to the configuration YAML file.")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to a checkpoint to resume training from.")
    parser.add_argument("--wandb_resume", type=str, default=None,
                        help="Wandb resume flag or run id to resume an existing run.")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Optionally, update the checkpoint folder with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["trainer"]["checkpoint_folder"] = f"{config['trainer']['checkpoint_folder']}{config['trainer']['run_name']}_{timestamp}"

    # Initialize first-sample trace log file
    # trace_path = f"{config['trainer']['checkpoint_folder']}/first_image_trace.txt"
    trace_path = f"/dss/dsshome1/0D/di38teq/Documents/rin-pytorch/first_image_trace.txt"
    set_log_path(trace_path, disable_logging=True)
    
    data_module = ImageNetDataModule(config)
    
    # Initialize WandB logger
    wandb_logger = WandbLogger(
        project="rin",
        name=config["trainer"]["run_name"],
        log_model=False,
        id=args.wandb_resume,
        resume=True if args.wandb_resume else False
    ) if config["trainer"]["log_to_wandb"] else None
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["trainer"]["checkpoint_folder"],
        filename="model-{step}",
        every_n_train_steps=config["trainer"]["sample_every"],
        save_weights_only=False,
        save_top_k=1,
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Configure PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_steps=config["trainer"]["train_num_steps"],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        num_nodes= 4,
        devices=4,
        precision="bf16" if config["trainer"]["fp16"] else "32",
        # gradient_clip_val=config["trainer"]["clip_grad_norm"],
        strategy='ddp_find_unused_parameters_true' if torch.cuda.device_count() > 1 else "auto",
        accumulate_grad_batches=1,
        log_every_n_steps=config["trainer"]["log_every_n_steps"],
        check_val_every_n_epoch=config["trainer"].get("check_val_every_n_epoch", 1),
        limit_val_batches=config["trainer"].get("limit_val_batches", 100),
        val_check_interval=config["trainer"].get("val_check_interval", None),
    )

    with trainer.init_module():
        print("Initializing model")
        print(torch.cuda.is_available())
        # model = RinLightningModule(config)
        model = RinLightningModule(config)

    # Run one-time FLOPs profiling before compiling (ensures profiler is not ignored)
    maybe_profile_flops_precompile(model, data_module)

    if torch.cuda.is_available():
        print("Compiling model")
        model = torch.compile(model, dynamic=True)
    
    # Start training
    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_checkpoint)

if __name__ == "__main__":
    main()