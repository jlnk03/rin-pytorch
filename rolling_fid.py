"""
Rolling FID calculation: Generate samples and compute FID without storing images.

This script combines sampling and FID computation by feeding generated images
directly into the FID metric, avoiding the need to store thousands of images.
"""

import argparse
from pathlib import Path
import math
import yaml

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from rin_pytorch import Rin, RinDiffusionModel
from rin_pytorch.data import DEFAULT_CIFAR_ROOT, FlexibleCIFAR10, ImageNetWebDataset


# Default cache directories for real image statistics
CIFAR_CACHE_DIR = Path("/dss/dsstbyfs02/pn52ko/pn52ko-dss-0000/tum/results/cifar")
IMAGENET_CACHE_DIR = Path("/dss/dsstbyfs02/pn52ko/pn52ko-dss-0000/tum/results/imagenet")


def get_fid_transform(target_size: int = 299):
    """Transform for FID: resize to 299x299 and convert to uint8 tensor [0, 255]."""
    return transforms.Compose([
        transforms.Resize(target_size, antialias=True),
        transforms.CenterCrop(target_size),
        transforms.Lambda(lambda x: (x * 255).to(torch.uint8)),
    ])


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_cache_path(cache_dir: Path, feature: int, split: str = "train") -> Path:
    """Get the cache file path for real image statistics."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"fid_stats_feature{feature}_{split}.pt"


def save_real_stats(fid: FrechetInceptionDistance, cache_path: Path):
    """Save the real image statistics from the FID metric."""
    stats = {
        "real_features_sum": fid.real_features_sum.cpu(),
        "real_features_cov_sum": fid.real_features_cov_sum.cpu(),
        "real_features_num_samples": fid.real_features_num_samples.cpu(),
    }
    torch.save(stats, cache_path)
    print(f"Saved real image statistics to {cache_path}")


def load_real_stats(fid: FrechetInceptionDistance, cache_path: Path, device: torch.device) -> bool:
    """Load cached real image statistics into the FID metric. Returns True if successful."""
    if not cache_path.exists():
        return False
    
    print(f"Loading cached real image statistics from {cache_path}")
    stats = torch.load(cache_path, map_location=device, weights_only=True)
    fid.real_features_sum = stats["real_features_sum"].to(device)
    fid.real_features_cov_sum = stats["real_features_cov_sum"].to(device)
    fid.real_features_num_samples = stats["real_features_num_samples"].to(device)
    return True


class FlexibleCIFARImageDataset:
    """Wrapper for FlexibleCIFAR10 that returns only images (no labels)."""
    def __init__(self, root_dir: str, train: bool, transform):
        self.dataset = FlexibleCIFAR10(root_dir=root_dir, train=train, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image


class ImageNetFIDDataset:
    """Wrapper around ImageNetWebDataset for FID computation (returns only images)."""
    def __init__(self, split: str = "train", transform=None):
        self.dataset = ImageNetWebDataset(split=split, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image


def compute_real_stats(
    fid: FrechetInceptionDistance,
    dataset_type: str,
    split: str,
    cifar_root: str,
    batch_size: int,
    device: torch.device,
    fid_transform,
):
    """Compute real image statistics and update the FID metric."""
    
    # Create dataset based on type
    if dataset_type == "cifar":
        # For CIFAR, we need PIL transform first, then FID transform
        pil_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(299, antialias=True),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 255).to(torch.uint8)),
        ])
        dataset = FlexibleCIFARImageDataset(
            root_dir=cifar_root,
            train=(split == "train"),
            transform=pil_transform,
        )
    elif dataset_type == "imagenet":
        pil_transform = transforms.Compose([
            transforms.Resize(299, antialias=True),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 255).to(torch.uint8)),
        ])
        dataset = ImageNetFIDDataset(split=split, transform=pil_transform)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Processing {dataset_type} ({split}) for real image statistics...")
    for batch in tqdm(dataloader, desc="Real images"):
        batch = batch.to(device)
        fid.update(batch, real=True)


def load_diffusion_model(config_path: str, checkpoint_path: str, device: torch.device):
    """Load the diffusion model from config and checkpoint."""
    config = load_config(config_path)
    
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
        num_classes=diffusion_config.get("num_classes", rin_config["num_classes"]),
    )

    ema_diffusion_model = ema_diffusion_model.to(device)

    # Initialize model with dummy data
    ema_diffusion_model.denoiser.pass_dummy_data(num_classes=rin_config["num_classes"])

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "ema_model" in checkpoint:
        ema_diffusion_model.load_state_dict(checkpoint["ema_model"])
    elif "model" in checkpoint:
        ema_diffusion_model.load_state_dict(checkpoint["model"])
    elif "state_dict" in checkpoint:
        state_dict = {
            k.replace("ema_diffusion_model.", ""): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("ema_diffusion_model.")
        }
        ema_diffusion_model.load_state_dict(state_dict)
    else:
        raise ValueError("Could not find model weights in checkpoint")

    ema_diffusion_model.eval()
    
    return ema_diffusion_model, config


def generate_and_update_fid(
    model,
    fid: FrechetInceptionDistance,
    config: dict,
    num_samples: int,
    batch_size: int,
    iterations: int,
    method: str,
    class_label: int | None,
    image_height: int | None,
    image_width: int | None,
    device: torch.device,
    fid_transform,
):
    """Generate samples and update FID metric without storing images."""
    
    rin_config = config["rin"]
    
    # Use either provided dimensions or config dimensions
    if image_height is None:
        image_height = rin_config["image_height"]
    if image_width is None:
        image_width = rin_config["image_width"]
    
    num_batches = math.ceil(num_samples / batch_size)
    samples_left = num_samples
    
    print(f"Generating {num_samples} samples in {num_batches} batches...")
    print(f"Image dimensions: {image_height}x{image_width}")
    
    for batch_idx in tqdm(range(num_batches), desc="Generating & computing FID"):
        current_batch_size = min(batch_size, samples_left)
        
        # Generate batch
        with torch.no_grad():
            samples = model.sample(
                num_samples=current_batch_size,
                iterations=iterations,
                method=method,
                class_override=class_label,
                image_height=image_height,
                image_width=image_width,
                tape_dim=rin_config["tape_dim"],
            )
        
        # Transform samples for FID (resize to 299x299, convert to uint8)
        # samples are in [0, 1] range, shape (B, C, H, W)
        samples_fid = fid_transform(samples)
        
        # Update FID with generated samples
        fid.update(samples_fid, real=False)
        
        samples_left -= current_batch_size
        
        # Clear GPU memory
        del samples, samples_fid
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Rolling FID: Generate samples and compute FID without storing images"
    )
    
    # Model arguments
    parser.add_argument("--config", type=str, required=True,
                       help="Path to the configuration YAML file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to the checkpoint file")
    
    # Sampling arguments
    parser.add_argument("--num_samples", type=int, required=True,
                       help="Number of samples to generate for FID computation")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for generation")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of sampling iterations (DDIM/DDPM steps)")
    parser.add_argument("--method", type=str, default="ddim",
                       choices=["ddim", "ddpm"],
                       help="Sampling method")
    parser.add_argument("--class_label", type=int, default=None,
                       help="Generate images for specific class (optional)")
    parser.add_argument("--image_height", type=int, default=None,
                       help="Override image height from config")
    parser.add_argument("--image_width", type=int, default=None,
                       help="Override image width from config")
    
    # Real dataset arguments
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["cifar", "imagenet"],
                       help="Real dataset to compare against")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split (train/test for CIFAR, train/validation for ImageNet)")
    parser.add_argument("--cifar_root", type=str, default=None,
                       help=f"Root directory for CIFAR-10. Defaults to '{DEFAULT_CIFAR_ROOT}'")
    
    # FID arguments
    parser.add_argument("--feature", type=int, default=2048,
                       choices=[64, 192, 768, 2048],
                       help="Feature dimension for FID")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Directory to cache real image statistics")
    parser.add_argument("--recompute_cache", action="store_true",
                       help="Force recomputation of cached real image statistics")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Determine cache directory
    if args.dataset == "cifar":
        cache_dir = Path(args.cache_dir) if args.cache_dir else CIFAR_CACHE_DIR
    else:
        cache_dir = Path(args.cache_dir) if args.cache_dir else IMAGENET_CACHE_DIR
    
    cache_path = get_cache_path(cache_dir, args.feature, args.split)
    
    # Initialize FID metric
    fid = FrechetInceptionDistance(feature=args.feature, normalize=False).to(device)
    
    # Try to load cached real statistics
    cache_loaded = False
    if not args.recompute_cache:
        cache_loaded = load_real_stats(fid, cache_path, device)
    
    # Compute real statistics if not loaded from cache
    if not cache_loaded:
        cifar_root = args.cifar_root or DEFAULT_CIFAR_ROOT
        fid_transform = get_fid_transform()
        compute_real_stats(
            fid=fid,
            dataset_type=args.dataset,
            split=args.split,
            cifar_root=cifar_root,
            batch_size=args.batch_size,
            device=device,
            fid_transform=fid_transform,
        )
        save_real_stats(fid, cache_path)
    
    # Load diffusion model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_diffusion_model(args.config, args.checkpoint, device)
    
    # FID transform for generated samples
    fid_transform = get_fid_transform()
    
    # Generate samples and update FID
    generate_and_update_fid(
        model=model,
        fid=fid,
        config=config,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        iterations=args.iterations,
        method=args.method,
        class_label=args.class_label,
        image_height=args.image_height,
        image_width=args.image_width,
        device=device,
        fid_transform=fid_transform,
    )
    
    # Compute final FID score
    print("Computing final FID score...")
    fid_value = fid.compute()
    
    print(f"\n{'='*50}")
    print(f"FID Score: {fid_value:.2f}")
    print(f"{'='*50}")
    print(f"  - Generated samples: {args.num_samples}")
    print(f"  - Real dataset: {args.dataset} ({args.split})")
    print(f"  - Sampling method: {args.method}")
    print(f"  - Iterations: {args.iterations}")
    if args.class_label is not None:
        print(f"  - Class label: {args.class_label}")
    print(f"{'='*50}")
    
    return fid_value.item()


if __name__ == "__main__":
    main()

