import argparse
from pathlib import Path
import math
import os
import json
import yaml

import torch
from rin_pytorch import Rin, RinDiffusionModel
from torchvision.utils import save_image
from tqdm import tqdm

def generate_batch(model, batch_size, iterations, method, config, class_label=None):
    with torch.no_grad():
        samples = model.sample(
            num_samples=batch_size,
            iterations=iterations,
            method=method,
            class_override=class_label if class_label is not None else None,
            image_height=config.get("image_height", 32),
            image_width=config.get("image_width", 32),
            tape_dim=config.get("tape_dim", 256)  # Important: tape_dim is required for sampling
        )
    return samples
# Default configuration similar to train.py style
DEFAULT_CONFIG = {
    # Model architecture parameters
    "num_layers": "2,2,2",
    "latent_slots": 128,
    "latent_dim": 512,
    "latent_mlp_ratio": 4,
    "latent_num_heads": 16,
    "tape_dim": 256,
    "tape_mlp_ratio": 2,
    "rw_num_heads": 8,
    "image_height": 32,
    "image_width": 32,
    "image_channels": 3,
    "patch_size": 2,
    "latent_pos_encoding": "learned",
    "tape_pos_encoding": "learned",
    "drop_path": 0.1,
    "drop_units": 0.1,
    "drop_att": 0.0,
    "time_scaling": 1000,
    "self_cond": "latent",
    "time_on_latent": True,
    "cond_on_latent_n": 1,
    "cond_tape_writable": False,
    "cond_dim": 0,
    "cond_proj": True,
    "cond_decoupled_read": False,
    "xattn_enc_ln": False,
    "num_classes": 10,
    
    # Diffusion model parameters
    "train_schedule": "sigmoid@-3,3,0.9",
    "inference_schedule": "cosine",
    "pred_type": "eps",
    "loss_type": "eps",
    
    # Sampling parameters
    "num_samples": 100,
    "batch_size": 64,
    "iterations": 100,
    "method": "ddim",
    "class_label": None,
    "output_dir": "samples"
}

def load_config(config_path=None):
    """Load configuration from YAML file (matching training script) or use defaults"""
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        print(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                yaml_config = yaml.safe_load(f)
                # Flatten the nested YAML structure
                if 'rin' in yaml_config:
                    config.update(yaml_config['rin'])
                if 'diffusion' in yaml_config:
                    config.update(yaml_config['diffusion'])
                if 'trainer' in yaml_config:
                    # Extract sampling-related config from trainer section
                    trainer_config = yaml_config['trainer']
                    if 'sampling_kwargs' in trainer_config:
                        config.update(trainer_config['sampling_kwargs'])
                    # Also get num_classes from trainer if not in rin section
                    if 'num_classes' in trainer_config and 'num_classes' not in config:
                        config['num_classes'] = trainer_config['num_classes']
                if 'sampling' in yaml_config:
                    # Extract sampling-specific config if present
                    config.update(yaml_config['sampling'])
            else:
                # Fallback to JSON for backwards compatibility
                file_config = json.load(f)
                config.update(file_config)
    else:
        print("Using default configuration")
    
    return config

def create_model_from_config(config):
    """Create Rin model and diffusion model from configuration"""
    print(f"Creating model with num_classes: {config['num_classes']}")
    
    # Initialize EMA model only since we only use it for sampling
    ema_model = Rin(
        num_layers=config["num_layers"],
        latent_slots=config["latent_slots"],
        latent_dim=config["latent_dim"],
        latent_mlp_ratio=config["latent_mlp_ratio"],
        latent_num_heads=config["latent_num_heads"],
        tape_dim=config["tape_dim"],
        tape_mlp_ratio=config["tape_mlp_ratio"],
        rw_num_heads=config["rw_num_heads"],
        image_height=config["image_height"],
        image_width=config["image_width"],
        image_channels=config["image_channels"],
        patch_size=config["patch_size"],
        latent_pos_encoding=config["latent_pos_encoding"],
        tape_pos_encoding=config["tape_pos_encoding"],
        drop_path=config["drop_path"],
        drop_units=config["drop_units"],
        drop_att=config["drop_att"],
        time_scaling=config["time_scaling"],
        self_cond=config["self_cond"],
        time_on_latent=config["time_on_latent"],
        cond_on_latent_n=config["cond_on_latent_n"],
        cond_tape_writable=config["cond_tape_writable"],
        cond_dim=config["cond_dim"],
        cond_proj=config["cond_proj"],
        cond_decoupled_read=config["cond_decoupled_read"],
        xattn_enc_ln=config["xattn_enc_ln"],
        num_classes=config["num_classes"],
    )
    
    ema_diffusion_model = RinDiffusionModel(
        rin=ema_model,
        train_schedule=config["train_schedule"],
        inference_schedule=config["inference_schedule"],
        pred_type=config["pred_type"],
        self_cond=config["self_cond"],
        loss_type=config["loss_type"],
        num_classes=config["num_classes"],  # This was missing!
    )
    
    print(f"Created RinDiffusionModel with num_classes: {ema_diffusion_model._num_classes}")
    
    return ema_diffusion_model

def show_memory_stats():
    """Show GPU memory statistics similar to train.py"""
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        current_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{current_memory} GB of memory reserved.")
    else:
        print("CUDA not available - using CPU")

def main():
    parser = argparse.ArgumentParser(description="Rin Sampling Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--config", type=str, default="sample_config.yaml", help="Path to YAML/JSON config file (same as used for training)")
    parser.add_argument("--num_samples", type=int, default=None, help="Total number of images to generate (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for generation (overrides config)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save generated images (overrides config)")
    parser.add_argument("--iterations", type=int, default=None, help="Number of sampling iterations (overrides config)")
    parser.add_argument("--method", type=str, default=None, help="Sampling method (ddim or ddpm) (overrides config)")
    parser.add_argument("--class_label", type=int, default=None, help="Optional: Generate images for specific class (overrides config)")
    parser.add_argument("--num_classes", type=int, default=None, help="Number of classes in the model (overrides config)")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.num_samples is not None:
        config["num_samples"] = args.num_samples
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.iterations is not None:
        config["iterations"] = args.iterations
    if args.method is not None:
        config["method"] = args.method
    if args.class_label is not None:
        config["class_label"] = args.class_label
    if args.num_classes is not None:
        config["num_classes"] = args.num_classes

    # Create output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("Configuration:")
    print(f"  Model: {config['num_layers']} layers, {config['latent_slots']} slots")
    print(f"  Image size: {config['image_height']}x{config['image_width']}x{config['image_channels']}")
    print(f"  Classes: {config['num_classes']}")
    print(f"  Latent dim: {config['latent_dim']}, Tape dim: {config['tape_dim']}")
    print(f"  Sampling: {config['num_samples']} samples, batch size {config['batch_size']}")
    print(f"  Method: {config['method']}, {config['iterations']} iterations")
    print(f"  Output: {config['output_dir']}")
    if config['class_label'] is not None:
        print(f"  Class override: {config['class_label']}")

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Show initial memory stats similar to train.py
    show_memory_stats()
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    # Load checkpoint first to get the actual hyperparameters
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Check what keys are available in the checkpoint
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Check hyperparameters stored in the checkpoint
    if 'hyper_parameters' in checkpoint:
        hyper_params = checkpoint['hyper_parameters']
        print("Checkpoint hyperparameters:")
        if 'rin' in hyper_params and 'num_classes' in hyper_params['rin']:
            print(f"  Checkpoint num_classes (rin): {hyper_params['rin']['num_classes']}")
        if 'diffusion' in hyper_params and 'num_classes' in hyper_params['diffusion']:
            print(f"  Checkpoint num_classes (diffusion): {hyper_params['diffusion']['num_classes']}")
        if 'trainer' in hyper_params and 'num_classes' in hyper_params['trainer']:
            print(f"  Checkpoint num_classes (trainer): {hyper_params['trainer']['num_classes']}")
        print(f"  Config num_classes: {config['num_classes']}")
        
        # Override config with checkpoint hyperparameters to ensure compatibility
        if 'rin' in hyper_params:
            print("Overriding config with checkpoint rin hyperparameters")
            for key, value in hyper_params['rin'].items():
                if key in config:
                    print(f"    {key}: {config[key]} -> {value}")
                config[key] = value
        if 'diffusion' in hyper_params:
            print("Overriding config with checkpoint diffusion hyperparameters")
            for key, value in hyper_params['diffusion'].items():
                if key in config:
                    print(f"    {key}: {config[key]} -> {value}")
                config[key] = value

    print(f"Initializing model with checkpoint hyperparameters...")
    print("Final model configuration:")
    print(f"  Model: {config['num_layers']} layers, {config['latent_slots']} slots")
    print(f"  Image size: {config['image_height']}x{config['image_width']}x{config['image_channels']}")
    print(f"  Classes: {config['num_classes']}")
    print(f"  Latent dim: {config['latent_dim']}, Tape dim: {config['tape_dim']}")
    
    ema_diffusion_model = create_model_from_config(config)
    ema_diffusion_model = ema_diffusion_model.to(device)
    
    # Initialize model with dummy data using the correct num_classes
    print("Initializing model with dummy data...")
    ema_diffusion_model.denoiser.pass_dummy_data(num_classes=config["num_classes"])
    
    # PyTorch Lightning saves model state in 'state_dict' key
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Filter for EMA model parameters only
        ema_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('ema_diffusion_model.'):
                # Remove the 'ema_diffusion_model.' prefix
                new_key = key[len('ema_diffusion_model.'):]
                ema_state_dict[new_key] = value
        
        if ema_state_dict:
            print(f"Loading EMA model with {len(ema_state_dict)} parameters")
            ema_diffusion_model.load_state_dict(ema_state_dict)
        else:
            print("No EMA model found in checkpoint, trying to load main model")
            # Fallback: try to load the main diffusion model
            main_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('diffusion_model.'):
                    new_key = key[len('diffusion_model.'):]
                    main_state_dict[new_key] = value
            if main_state_dict:
                ema_diffusion_model.load_state_dict(main_state_dict)
            else:
                raise KeyError("Could not find model parameters in checkpoint")
    elif 'ema_model' in checkpoint:
        # Legacy format - direct EMA model state dict
        ema_diffusion_model.load_state_dict(checkpoint['ema_model'])
    else:
        # Assume the checkpoint is a direct state dict
        ema_diffusion_model.load_state_dict(checkpoint)
    
    # Use EMA model for sampling
    ema_diffusion_model.eval()

    # Calculate number of batches
    num_batches = math.ceil(config["num_samples"] / config["batch_size"])
    samples_left = config["num_samples"]
    current_index = 0

    print(f"Generating {config['num_samples']} samples in {num_batches} batches...")
    show_memory_stats()
    
    for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
        # Calculate batch size for this iteration
        current_batch_size = min(config["batch_size"], samples_left)
        
        # Generate batch
        samples = generate_batch(
            ema_diffusion_model,
            current_batch_size,
            config["iterations"],
            config["method"],
            config,
            config["class_label"]
        )

        # Save individual images from this batch
        for i, sample in enumerate(samples):
            save_image(
                sample,
                output_dir / f"sample_{current_index + i:05d}.png",
                normalize=True,
                value_range=(0, 1)
            )
        
        # Update counters
        current_index += current_batch_size
        samples_left -= current_batch_size

        # Clear GPU memory
        del samples
        torch.cuda.empty_cache()

    # Show final memory stats similar to train.py
    print("Final memory statistics:")
    show_memory_stats()
    print("Done!")

if __name__ == "__main__":
    main() 