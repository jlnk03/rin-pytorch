#!/usr/bin/env python3
import argparse
import torch
import torchvision
from pathlib import Path
import yaml
from tqdm import tqdm
import math
from torchvision.utils import save_image
import os

from rin_pytorch import Rin, RinDiffusionModel

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def sample_grids(config_path, checkpoint_path, output_dir, 
                grid_size=8, iterations=100, method="ddim", class_label=None, num_grids=1):
    """
    Generate three grids of samples with different aspect ratios:
    1. Square: 128x128
    2. Horizontal: 72x128
    3. Vertical: 128x72
    
    Parameters:
    -----------
    num_grids : int
        Number of grids to generate for each aspect ratio
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create subdirectories for each aspect ratio
    square_dir = output_dir / "square"
    horizontal_dir = output_dir / "horizontal"
    vertical_dir = output_dir / "vertical"
    
    for dir_path in [square_dir, horizontal_dir, vertical_dir]:
        dir_path.mkdir(exist_ok=True, parents=True)
    
    print(f"Initializing model...")
    # Initialize Rin model from config
    rin_config = config["rin"]
    ema_model = Rin(
        num_layers=rin_config["num_layers"],
        latent_slots=rin_config["latent_slots"],
        latent_dim=rin_config["latent_dim"],
        latent_mlp_ratio=rin_config["latent_mlp_ratio"],
        latent_num_heads=rin_config["latent_num_heads"],
        tape_dim=rin_config["tape_dim"],
        tape_mlp_ratio=rin_config["tape_mlp_ratio"],
        rw_num_heads=rin_config["rw_num_heads"],
        image_height=rin_config["image_height"],
        image_width=rin_config["image_width"],
        image_channels=rin_config["image_channels"],
        patch_size=rin_config["patch_size"],
        latent_pos_encoding=rin_config["latent_pos_encoding"],
        tape_pos_encoding=rin_config["tape_pos_encoding"],
        drop_path=rin_config["drop_path"],
        drop_units=rin_config["drop_units"],
        drop_att=rin_config["drop_att"],
        time_scaling=rin_config["time_scaling"],
        self_cond=rin_config["self_cond"],
        time_on_latent=rin_config["time_on_latent"],
        cond_on_latent_n=rin_config["cond_on_latent_n"],
        cond_tape_writable=rin_config["cond_tape_writable"],
        cond_dim=rin_config["cond_dim"],
        cond_proj=rin_config["cond_proj"],
        cond_decoupled_read=rin_config["cond_decoupled_read"],
        xattn_enc_ln=rin_config["xattn_enc_ln"],
        num_classes=rin_config["num_classes"],
    )
    
    # Initialize diffusion model from config
    diffusion_config = config["diffusion"]
    ema_diffusion_model = RinDiffusionModel(
        rin=ema_model,
        train_schedule=diffusion_config["train_schedule"],
        inference_schedule=diffusion_config["inference_schedule"],
        pred_type=diffusion_config["pred_type"],
        self_cond=diffusion_config["self_cond"],
        loss_type=diffusion_config["loss_type"],
    )

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    ema_diffusion_model = ema_diffusion_model.to(device)
    
    # Initialize model with dummy data
    print("Initializing model with dummy data...")
    ema_diffusion_model.denoiser.pass_dummy_data(num_classes=rin_config["num_classes"])

    print(f"Loading checkpoint from {checkpoint_path}...")
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check which key the model state is stored under
    if 'ema_model' in checkpoint:
        ema_diffusion_model.load_state_dict(checkpoint['ema_model'])
    elif 'model' in checkpoint:
        ema_diffusion_model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        # Remove 'ema_diffusion_model.' prefix if it exists
        state_dict = {k.replace('ema_diffusion_model.', ''): v for k, v in checkpoint['state_dict'].items() 
                     if k.startswith('ema_diffusion_model.')}
        ema_diffusion_model.load_state_dict(state_dict)
    else:
        raise ValueError("Could not find model weights in checkpoint")
    
    # Use EMA model for sampling
    ema_diffusion_model.eval()
    
    # Define sampling parameters
    sampling_kwargs = {
        "iterations": iterations,
        "method": method,
        "class_override": class_label
    }
    
    print(f"Generating {num_grids} grids with {grid_size}x{grid_size} samples for each aspect ratio...")
    
    with torch.no_grad():
        # Generate multiple grids for each aspect ratio
        for grid_idx in tqdm(range(num_grids), desc="Generating grids"):
            # Generate square samples (128x128)
            samples = ema_diffusion_model.sample(
                num_samples=grid_size * grid_size, 
                image_height=128, 
                image_width=128, 
                tape_dim=rin_config["tape_dim"], 
                **sampling_kwargs
            )
            grid = torchvision.utils.make_grid(samples, nrow=grid_size, normalize=True, value_range=(0, 1), padding=2)
            save_image(grid, square_dir / f"grid_square_{grid_idx:04d}.png")
            
            # Generate horizontal samples (72x128)
            samples_horizontal = ema_diffusion_model.sample(
                num_samples=grid_size * grid_size, 
                image_height=72, 
                image_width=128, 
                tape_dim=rin_config["tape_dim"], 
                **sampling_kwargs
            )
            grid_horizontal = torchvision.utils.make_grid(samples_horizontal, nrow=grid_size, normalize=True, value_range=(0, 1), padding=2)
            save_image(grid_horizontal, horizontal_dir / f"grid_horizontal_{grid_idx:04d}.png")
            
            # Generate vertical samples (128x72)
            samples_vertical = ema_diffusion_model.sample(
                num_samples=grid_size * grid_size, 
                image_height=128, 
                image_width=72, 
                tape_dim=rin_config["tape_dim"], 
                **sampling_kwargs
            )
            grid_vertical = torchvision.utils.make_grid(samples_vertical, nrow=grid_size, normalize=True, value_range=(0, 1), padding=2)
            save_image(grid_vertical, vertical_dir / f"grid_vertical_{grid_idx:04d}.png")
            
            # Free up memory
            del samples
            del samples_horizontal
            del samples_vertical
            torch.cuda.empty_cache()
    
    print(f"Done! Generated {num_grids} grids for each aspect ratio saved to {output_dir}")
    return output_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--output_dir", type=str, default="grid_samples", help="Directory to save generated grids")
    parser.add_argument("--grid_size", type=int, default=8, help="Size of the grid (grid_size x grid_size)")
    parser.add_argument("--iterations", type=int, default=100, help="Number of sampling iterations")
    parser.add_argument("--method", type=str, default="ddim", help="Sampling method (ddim or ddpm)")
    parser.add_argument("--class_label", type=int, default=None, help="Optional: Generate images for specific class")
    parser.add_argument("--num_grids", type=int, default=1, help="Number of grids to generate for each aspect ratio")
    args = parser.parse_args()
    
    sample_grids(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        grid_size=args.grid_size,
        iterations=args.iterations,
        method=args.method,
        class_label=args.class_label,
        num_grids=args.num_grids,
    )

if __name__ == "__main__":
    main() 