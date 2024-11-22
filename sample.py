import argparse
from pathlib import Path
import math

import torch
from rin_pytorch import Rin, RinDiffusionModel
from torchvision.utils import save_image
from tqdm import tqdm

def generate_batch(model, batch_size, iterations, method, class_label=None):
    with torch.no_grad():
        samples = model.sample(
            num_samples=batch_size,
            iterations=iterations,
            method=method,
            class_override=class_label if class_label is not None else None
        )
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--num_samples", type=int, required=True, help="Total number of images to generate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--output_dir", type=str, default="samples", help="Directory to save generated images")
    parser.add_argument("--iterations", type=int, default=100, help="Number of sampling iterations")
    parser.add_argument("--method", type=str, default="ddim", help="Sampling method (ddim or ddpm)")
    parser.add_argument("--class_label", type=int, default=None, help="Optional: Generate images for specific class")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes in the model")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Initializing model...")
    # Initialize EMA model only since we only use it for sampling
    ema_model = Rin(
        num_layers="2,2,2",
        latent_slots=128,
        latent_dim=512,
        latent_mlp_ratio=4,
        latent_num_heads=16,
        tape_dim=256,
        tape_mlp_ratio=2,
        rw_num_heads=8,
        image_height=32,
        image_width=32,
        image_channels=3,
        patch_size=2,
        latent_pos_encoding="learned",
        tape_pos_encoding="learned",
        drop_path=0.1,
        drop_units=0.1,
        drop_att=0.0,
        time_scaling=1000,
        self_cond="latent",
        time_on_latent=True,
        cond_on_latent_n=1,
        cond_tape_writable=False,
        cond_dim=0,
        cond_proj=True,
        cond_decoupled_read=False,
        xattn_enc_ln=False,
        num_classes=args.num_classes,
    )
    ema_diffusion_model = RinDiffusionModel(
        rin=ema_model,
        train_schedule="sigmoid@-3,3,0.9",
        inference_schedule="cosine",
        pred_type="eps",
        self_cond="latent",
        loss_type="eps",
    )

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    ema_diffusion_model = ema_diffusion_model.to(device)
    
    # Initialize model with dummy data
    print("Initializing model with dummy data...")
    ema_diffusion_model.denoiser.pass_dummy_data(num_classes=args.num_classes)

    print(f"Loading checkpoint from {args.checkpoint}...")
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    ema_diffusion_model.load_state_dict(checkpoint['ema_model'])
    
    # Use EMA model for sampling
    ema_diffusion_model.eval()

    # Calculate number of batches
    num_batches = math.ceil(args.num_samples / args.batch_size)
    samples_left = args.num_samples
    current_index = 0

    # Generate samples in batches
    print(f"Generating {args.num_samples} samples in {num_batches} batches...")
    for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
        # Calculate batch size for this iteration
        current_batch_size = min(args.batch_size, samples_left)
        
        # Generate batch
        samples = generate_batch(
            ema_diffusion_model,
            current_batch_size,
            args.iterations,
            args.method,
            args.class_label
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

    print("Done!")

if __name__ == "__main__":
    main() 