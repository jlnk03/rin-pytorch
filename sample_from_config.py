import argparse
from pathlib import Path
import math
import yaml

import torch
from rin_pytorch import Rin, RinDiffusionModel
from torchvision.utils import save_image, make_grid
from tqdm import tqdm


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def generate_batch(model, batch_size, iterations, method, image_height, image_width, tape_dim, class_label=None):
    with torch.no_grad():
        samples = model.sample(
            num_samples=batch_size,
            iterations=iterations,
            method=method,
            class_override=class_label if class_label is not None else None,
            image_height=image_height,
            image_width=image_width,
            tape_dim=tape_dim,
        )
    return samples


def get_starting_index(output_dir: Path, prefix: str = "sample_", suffix: str = ".png") -> int:
    """Find the next available index based on existing files in the directory."""
    existing_files = list(output_dir.glob(f"{prefix}*{suffix}"))
    if not existing_files:
        return 0
    
    max_index = -1
    for f in existing_files:
        # Extract the number from filename like "sample_00042.png"
        name = f.stem  # "sample_00042"
        try:
            num_str = name.replace(prefix, "")
            num = int(num_str)
            max_index = max(max_index, num)
        except ValueError:
            continue
    
    return max_index + 1


def get_starting_grid_index(output_dir: Path, prefix: str = "grid_", suffix: str = ".png") -> int:
    """Find the next available grid index based on existing files in the directory."""
    existing_files = list(output_dir.glob(f"{prefix}*{suffix}"))
    if not existing_files:
        return 0
    
    max_index = -1
    for f in existing_files:
        name = f.stem  # "grid_042"
        try:
            num_str = name.replace(prefix, "")
            num = int(num_str)
            max_index = max(max_index, num)
        except ValueError:
            continue
    
    return max_index + 1


def sample_from_config(
    config_path,
    checkpoint_path,
    num_samples,
    output_dir,
    batch_size=64,
    iterations=100,
    method="ddim",
    class_label=None,
    image_height=None,
    image_width=None,
    make_grid=False,
    grid_nrow=None,
):
    # Load configuration
    config = load_config(config_path)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find starting index based on existing files
    if make_grid:
        starting_index = get_starting_grid_index(output_dir)
        if starting_index > 0:
            print(f"Found existing grid images, continuing from grid_{starting_index:03d}.png")
    else:
        starting_index = get_starting_index(output_dir)
        if starting_index > 0:
            print(f"Found existing images, continuing from sample_{starting_index:05d}.png")

    # Use either provided dimensions or config dimensions
    if image_height is None:
        image_height = config["rin"]["image_height"]
    if image_width is None:
        image_width = config["rin"]["image_width"]

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

    print(f"num classes: {rin_config['num_classes']}")

    # Initialize diffusion model from config
    diffusion_config = config["diffusion"]
    ema_diffusion_model = RinDiffusionModel(
        rin=ema_model,
        train_schedule=diffusion_config["train_schedule"],
        inference_schedule=diffusion_config["inference_schedule"],
        pred_type=diffusion_config["pred_type"],
        self_cond=diffusion_config["self_cond"],
        loss_type=diffusion_config["loss_type"],
        num_classes=diffusion_config.get("num_classes", rin_config["num_classes"]),
    )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ema_diffusion_model = ema_diffusion_model.to(device)

    # Initialize model with dummy data
    print("Initializing model with dummy data...")
    ema_diffusion_model.denoiser.pass_dummy_data(num_classes=rin_config["num_classes"])

    print(f"Loading checkpoint from {checkpoint_path}...")
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Check which key the model state is stored under
    if "ema_model" in checkpoint:
        ema_diffusion_model.load_state_dict(checkpoint["ema_model"])
    elif "model" in checkpoint:
        ema_diffusion_model.load_state_dict(checkpoint["model"])
    elif "state_dict" in checkpoint:
        # Remove 'ema_diffusion_model.' prefix if it exists
        state_dict = {
            k.replace("ema_diffusion_model.", ""): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("ema_diffusion_model.")
        }
        ema_diffusion_model.load_state_dict(state_dict)
    else:
        raise ValueError("Could not find model weights in checkpoint")

    # Use EMA model for sampling
    ema_diffusion_model.eval()

    # Calculate number of batches
    num_batches = math.ceil(num_samples / batch_size)
    samples_left = num_samples
    current_index = starting_index
    current_grid_index = starting_index if make_grid else 0

    # Generate samples in batches
    print(f"Generating {num_samples} samples in {num_batches} batches...")
    print(f"Image dimensions: {image_height}x{image_width}")

    for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
        # Calculate batch size for this iteration
        current_batch_size = min(batch_size, samples_left)

        # Generate batch
        samples = generate_batch(
            ema_diffusion_model,
            current_batch_size,
            iterations,
            method,
            image_height,
            image_width,
            rin_config["tape_dim"],
            class_label,
        )

        if make_grid:
            # Images-per-row in grid
            nrow = grid_nrow or min(8, current_batch_size)
            save_image(
                samples,
                output_dir / f"grid_{current_grid_index:03d}.png",
                nrow=nrow,
                normalize=False,
                value_range=(0, 1),
                padding=0,
            )
            current_grid_index += 1
        else:
            # Save individual images from this batch
            for i, sample in enumerate(samples):
                save_image(
                    sample,
                    output_dir / f"sample_{current_index + i:05d}.png",
                    normalize=False,
                    value_range=(0, 1),
                )

        # Update counters
        current_index += current_batch_size
        samples_left -= current_batch_size

        # Clear GPU memory
        del samples
        torch.cuda.empty_cache()

    if make_grid:
        print(f"Done! Generated {num_samples} samples in {num_batches} grids saved to {output_dir}")
    else:
        print(f"Done! Generated {num_samples} samples (indices {starting_index}-{current_index - 1}) saved to {output_dir}")

    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--num_samples", type=int, required=True, help="Total number of images to generate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--output_dir", type=str, default="samples", help="Directory to save generated images")
    parser.add_argument("--iterations", type=int, default=100, help="Number of sampling iterations")
    parser.add_argument("--method", type=str, default="ddim", help="Sampling method (ddim or ddpm)")
    parser.add_argument("--class_label", type=int, default=None, help="Optional: Generate images for specific class")
    parser.add_argument("--image_height", type=int, default=None, help="Optional: Override image height from config")
    parser.add_argument("--image_width", type=int, default=None, help="Optional: Override image width from config")

    # New grid options
    parser.add_argument(
        "--grid",
        action="store_true",
        help="If set, save each batch as a grid image instead of individual images",
    )
    parser.add_argument(
        "--grid_nrow",
        type=int,
        default=None,
        help="Number of images per row in the grid (default: min(8, batch_size))",
    )

    args = parser.parse_args()

    sample_from_config(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        iterations=args.iterations,
        method=args.method,
        class_label=args.class_label,
        image_height=args.image_height,
        image_width=args.image_width,
        make_grid=args.grid,
        grid_nrow=args.grid_nrow,
    )


if __name__ == "__main__":
    main()