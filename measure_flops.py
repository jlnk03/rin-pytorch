#!/usr/bin/env python
"""
Measure GFLOPs of RIN model forward pass.

Similar to DiT paper Table 1, reports computational cost per forward pass.

Usage:
    python measure_flops.py --config configs/cifar.yaml
    python measure_flops.py --config configs/64.yaml --image_size 64
    python measure_flops.py --config configs/128.yaml --image_size 128
"""

import argparse
import torch
import yaml
from pathlib import Path

# fvcore for accurate FLOP counting
# NOTE: fvcore can cause segfaults with some models, so disabled by default
# Set ENABLE_FVCORE=1 to enable
import os
HAS_FVCORE = False
if os.environ.get('ENABLE_FVCORE', '0') == '1':
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
        HAS_FVCORE = True
    except ImportError:
        print("Warning: fvcore not installed. Install with: pip install fvcore")

# For torch profiler
from torch.profiler import profile, ProfilerActivity


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model_from_config(config: dict, device: str = 'cuda'):
    """Create RIN model from config."""
    from rin_pytorch.Rin import Rin
    
    rin_config = config.get('rin', config.get('model', {}))
    
    # Get image size (handle both image_size and image_height/width)
    image_height = rin_config.get('image_height', rin_config.get('image_size', 32))
    image_width = rin_config.get('image_width', rin_config.get('image_size', 32))
    
    model = Rin(
        image_height=image_height,
        image_width=image_width,
        image_channels=rin_config.get('image_channels', rin_config.get('num_channels', 3)),
        patch_size=rin_config.get('patch_size', 4),
        latent_slots=rin_config.get('latent_slots', 128),
        latent_dim=rin_config.get('latent_dim', 512),
        latent_mlp_ratio=rin_config.get('latent_mlp_ratio', 4),
        latent_num_heads=rin_config.get('latent_num_heads', rin_config.get('latent_heads', 8)),
        tape_dim=rin_config.get('tape_dim', 256),
        tape_mlp_ratio=rin_config.get('tape_mlp_ratio', 4),
        rw_num_heads=rin_config.get('rw_num_heads', rin_config.get('rw_heads', 8)),
        num_layers=rin_config.get('num_layers', '8'),
        num_classes=rin_config.get('num_classes', 10),
        time_on_latent=rin_config.get('time_on_latent', True),
        cond_on_latent_n=rin_config.get('cond_on_latent_n', 1),
        self_cond=rin_config.get('self_cond', 'latent'),
        cond_tape_writable=rin_config.get('cond_tape_writable', False),
        cond_dim=rin_config.get('cond_dim', 0),
        sparse_hierarchy=rin_config.get('sparse_hierarchy', None),
    )
    
    return model.to(device).eval()


def create_dummy_input(config: dict, batch_size: int = 1, device: str = 'cuda'):
    """Create dummy input for the model."""
    rin_config = config.get('rin', config.get('model', {}))
    
    image_height = rin_config.get('image_height', rin_config.get('image_size', 32))
    image_width = rin_config.get('image_width', rin_config.get('image_size', 32))
    image_channels = rin_config.get('image_channels', rin_config.get('num_channels', 3))
    num_classes = rin_config.get('num_classes', 10)
    latent_slots = rin_config.get('latent_slots', 128)
    latent_dim = rin_config.get('latent_dim', 512)
    
    # Create dummy inputs matching Rin.forward signature
    # x: (B, C, H, W) - raw image
    x = torch.randn(batch_size, image_channels, image_height, image_width, device=device)
    t = torch.tensor([0.5] * batch_size, device=device, dtype=torch.float32)  # timestep
    # cond: one-hot encoded class labels (num_classes dimensions) for cond_proj Linear
    cond = torch.zeros(batch_size, num_classes, device=device, dtype=torch.float32)
    cond[:, 0] = 1.0  # Set first class as one-hot
    
    # For self-conditioning (can be None or previous latent)
    latent_prev = None  # Start with None for accurate FLOP counting
    tape_prev = None
    
    return x, t, cond, latent_prev, tape_prev


def measure_flops_fvcore(model, inputs, verbose: bool = True):
    """Measure FLOPs using fvcore."""
    if not HAS_FVCORE:
        print("fvcore not available - skipping")
        return None, None
    
    x, t, cond, latent_prev, tape_prev = inputs
    
    try:
        # fvcore expects a tuple of inputs
        flops = FlopCountAnalysis(model, (x, t, cond, latent_prev, tape_prev))
        
        # Set to unsupported ops to not raise errors
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        
        total_flops = flops.total()
        
        if verbose:
            print("\n" + "=" * 60)
            print("FVCORE FLOP ANALYSIS")
            print("=" * 60)
            try:
                print(flop_count_table(flops, max_depth=3))
            except:
                print(f"Total FLOPs: {total_flops / 1e9:.2f} GFLOPs")
            
        params = parameter_count(model)
        
        return total_flops, params
    except Exception as e:
        print(f"fvcore error: {e}")
        return None, None


def measure_flops_profiler(model, inputs, device: str = 'cuda', verbose: bool = True):
    """Measure FLOPs using torch.profiler."""
    x, t, cond, latent_prev, tape_prev = inputs
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(x, t, cond, latent_prev, tape_prev)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Profile - use CPU only to avoid CUDA profiler segfaults
    # CPU profiler still captures FLOPs for all operations
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        with_flops=True,
    ) as prof:
        with torch.no_grad():
            _ = model(x, t, cond, latent_prev, tape_prev)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    if verbose:
        print("\n" + "=" * 60)
        print("TORCH PROFILER ANALYSIS")
        print("=" * 60)
        print(prof.key_averages().table(sort_by="flops", row_limit=20))
    
    # Sum up FLOPs from profiler
    # Note: torch profiler reports MACs (multiply-accumulate), not FLOPs
    # Each MAC = 1 multiply + 1 add = 2 FLOPs
    # Papers (like DiT) report FLOPs, so we multiply by 2
    total_macs = sum(
        event.flops for event in prof.key_averages() 
        if event.flops is not None and event.flops > 0
    )
    total_flops = total_macs * 2  # Convert MACs to FLOPs
    
    return total_flops


def count_parameters(model):
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_flops(flops: int) -> str:
    """Format FLOPs in human-readable form."""
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    else:
        return f"{flops:.0f} FLOPs"


def main():
    parser = argparse.ArgumentParser(description="Measure model FLOPs")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for measurement')
    parser.add_argument('--image_size', type=int, default=None, help='Override image size')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--verbose', action='store_true', help='Print detailed breakdown')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override image size if specified
    if args.image_size:
        if 'rin' in config:
            config['rin']['image_height'] = args.image_size
            config['rin']['image_width'] = args.image_size
        else:
            config['model']['image_size'] = args.image_size
    
    # Create model
    print(f"Loading model from {args.config}...")
    model = create_model_from_config(config, args.device)
    
    # Create dummy input
    inputs = create_dummy_input(config, args.batch_size, args.device)
    
    # Get config info
    rin_config = config.get('rin', config.get('model', {}))
    image_size = rin_config.get('image_height', rin_config.get('image_size', 32))
    patch_size = rin_config.get('patch_size', 4)
    num_patches = (image_size // patch_size) ** 2
    latent_slots = rin_config.get('latent_slots', 128)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    
    print("\n" + "=" * 60)
    print("MODEL CONFIGURATION")
    print("=" * 60)
    print(f"  Config: {args.config}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Patch size: {patch_size}")
    print(f"  Num patches (tape tokens): {num_patches}")
    print(f"  Latent slots: {latent_slots}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # Measure with fvcore
    fvcore_flops = None
    if HAS_FVCORE:
        try:
            fvcore_flops, _ = measure_flops_fvcore(model, inputs, verbose=args.verbose)
        except Exception as e:
            print(f"\nfvcore measurement failed: {e}")
            fvcore_flops = None
    
    # Measure with torch profiler
    profiler_flops = None
    try:
        profiler_flops = measure_flops_profiler(model, inputs, args.device, verbose=args.verbose)
    except Exception as e:
        print(f"\nTorch profiler measurement failed: {e}")
        profiler_flops = None
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (per forward pass)")
    print("=" * 60)
    print(f"  Model: RIN")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Parameters: {total_params / 1e6:.2f}M")
    
    if fvcore_flops:
        gflops_fvcore = fvcore_flops / 1e9
        print(f"  FLOPs (fvcore): {format_flops(fvcore_flops)} = {gflops_fvcore:.2f} GFLOPs")
        if args.batch_size > 1:
            print(f"  FLOPs per sample (fvcore): {format_flops(fvcore_flops // args.batch_size)}")
    
    if profiler_flops:
        gflops_profiler = profiler_flops / 1e9
        print(f"  FLOPs (profiler): {format_flops(profiler_flops)} = {gflops_profiler:.2f} GFLOPs")
        if args.batch_size > 1:
            print(f"  FLOPs per sample (profiler): {format_flops(profiler_flops // args.batch_size)}")
    
    # Paper-style output
    print("\n" + "=" * 60)
    print("FOR PAPER (DiT-style)")
    print("=" * 60)
    best_flops = fvcore_flops or profiler_flops
    if best_flops:
        gflops = best_flops / 1e9
        if args.batch_size > 1:
            gflops = gflops / args.batch_size
        print(f"  RIN-{image_size}: {total_params / 1e6:.0f}M params, {gflops:.1f} GFLOPs")
    
    # Save to CSV
    csv_file = Path(args.config).stem + "_flops.csv"
    with open(csv_file, 'w') as f:
        f.write("model,image_size,patch_size,num_patches,latent_slots,params_M,gflops_fvcore,gflops_profiler\n")
        fvcore_g = (fvcore_flops / 1e9 / args.batch_size) if fvcore_flops else ""
        profiler_g = (profiler_flops / 1e9 / args.batch_size) if profiler_flops else ""
        f.write(f"RIN,{image_size},{patch_size},{num_patches},{latent_slots},{total_params/1e6:.2f},{fvcore_g},{profiler_g}\n")
    print(f"\nResults saved to: {csv_file}")


if __name__ == "__main__":
    main()

