#!/usr/bin/env python3
"""
Compare attention backends: nn.MHA, xformers, and flex_attention
- Numerical accuracy comparison (attention module + full model)
- Error accumulation tracking across model layers
- CSV export for pgfplots/tikz
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check available backends
BACKENDS = ["mha"]  # Always available

try:
    from xformers.ops import memory_efficient_attention
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask
    BACKENDS.append("xformers")
    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False
    print("⚠ xformers not available")

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    flex_attention_compiled = torch.compile(flex_attention, dynamic=True)
    create_block_mask_compiled = torch.compile(create_block_mask, dynamic=True)
    BACKENDS.append("flex")
    HAS_FLEX = True
except ImportError:
    HAS_FLEX = False
    print("⚠ flex_attention not available")


def create_mha_module(dim, nheads, device):
    """Create standard nn.MultiheadAttention"""
    return nn.MultiheadAttention(dim, nheads, batch_first=True).to(device)


def create_xformers_mha(dim, nheads, device, ref_mha):
    """Create xformers MHA with same weights as reference"""
    from rin_pytorch.modules.MHA_xformers import XformersMultiheadAttention
    
    xf_mha = XformersMultiheadAttention(
        E_q=dim, E_k=dim, E_v=dim, E_total=dim, nheads=nheads
    ).to(device)
    
    # Copy weights
    with torch.no_grad():
        xf_mha.packed_proj.weight.copy_(ref_mha.in_proj_weight)
        xf_mha.packed_proj.bias.copy_(ref_mha.in_proj_bias)
        xf_mha.out_proj.weight.copy_(ref_mha.out_proj.weight)
        xf_mha.out_proj.bias.copy_(ref_mha.out_proj.bias)
    
    return xf_mha


def create_flex_mha(dim, nheads, device, ref_mha):
    """Create flex attention MHA with same weights as reference"""
    from rin_pytorch.modules.MHA import FlexMultiheadAttention
    
    flex_mha = FlexMultiheadAttention(
        E_q=dim, E_k=dim, E_v=dim, E_total=dim, nheads=nheads
    ).to(device)
    
    # Copy weights
    with torch.no_grad():
        flex_mha.packed_proj.weight.copy_(ref_mha.in_proj_weight)
        flex_mha.packed_proj.bias.copy_(ref_mha.in_proj_bias)
        flex_mha.out_proj.weight.copy_(ref_mha.out_proj.weight)
        flex_mha.out_proj.bias.copy_(ref_mha.out_proj.bias)
    
    return flex_mha


def create_masks(batch_size, seq_len, device):
    """Create masks for each backend"""
    doc_ids = torch.arange(batch_size, device=device).repeat_interleave(seq_len)
    offsets = torch.arange(batch_size + 1, device=device) * seq_len
    
    masks = {}
    
    # xformers mask
    if HAS_XFORMERS:
        seqlens = [seq_len] * batch_size
        masks["xformers"] = BlockDiagonalMask.from_seqlens(seqlens)
    
    # flex attention mask
    if HAS_FLEX:
        from rin_pytorch.modules.MHA import create_document_block_mask
        masks["flex"] = create_document_block_mask(doc_ids, offsets=offsets)
    
    return masks, doc_ids, offsets


def compare_outputs(ref_out, test_out):
    """Compare outputs and return statistics"""
    diff = (ref_out - test_out).abs()
    return {
        "max_diff": diff.max().item(),
        "mean_diff": diff.mean().item(),
        "std_diff": diff.std().item(),
    }


def run_attention_comparison(dim=256, nheads=8, batch_sizes=[2, 8, 32, 64, 128, 256], seq_len=256):
    """
    Run attention module comparison across batch sizes.
    Uses packed sequences with document masks (block-diagonal) for xformers and flex.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: dim={dim}, nheads={nheads}, seq_len={seq_len} tokens/doc")
    print(f"Total tokens at max batch: {batch_sizes[-1]} docs × {seq_len} tokens = {batch_sizes[-1] * seq_len}")
    print("=" * 70)
    
    results = {
        "batch_size": batch_sizes,
        "total_tokens": [b * seq_len for b in batch_sizes],
        "mha_time_ms": [],
        "xformers_time_ms": [],
        "flex_time_ms": [],
        "xformers_max_diff": [],
        "xformers_mean_diff": [],
        "flex_max_diff": [],
        "flex_mean_diff": [],
    }
    
    # Compile flex_attention and create_block_mask once
    if HAS_FLEX:
        print("\nCompiling flex_attention and create_block_mask...")
        flex_attn_compiled = torch.compile(flex_attention, dynamic=False)
        create_mask_compiled = torch.compile(create_block_mask, dynamic=False)
        print("  ✓ Compilation ready (will compile on first use)")
    
    for batch_size in batch_sizes:
        total_tokens = batch_size * seq_len
        print(f"\nBatch: {batch_size} docs × {seq_len} tokens = {total_tokens} total tokens")
        
        # Create reference MHA
        torch.manual_seed(42)
        ref_mha = create_mha_module(dim, nheads, device)
        
        # Create input - batched for nn.MHA, packed for others
        torch.manual_seed(123)
        x_batched = torch.randn(batch_size, seq_len, dim, device=device)
        x_packed = x_batched.view(total_tokens, dim)
        
        # Create document masks
        doc_ids = torch.arange(batch_size, device=device).repeat_interleave(seq_len)
        offsets = torch.arange(batch_size + 1, device=device) * seq_len
        
        # xformers BlockDiagonalMask
        xf_mask = None
        if HAS_XFORMERS:
            seqlens = [seq_len] * batch_size
            xf_mask = BlockDiagonalMask.from_seqlens(seqlens)
        
        # flex BlockMask (compiled)
        flex_mask = None
        if HAS_FLEX:
            from rin_pytorch.modules.MHA import create_document_block_mask
            flex_mask = create_document_block_mask(doc_ids, offsets=offsets)
        
        # ===== Run nn.MHA (batched, no packing) =====
        ref_mha.eval()
        with torch.no_grad():
            for _ in range(5):
                _ = ref_mha(x_batched, x_batched, x_batched, need_weights=False)
            torch.cuda.synchronize()
            
            start = time.time()
            for _ in range(20):
                mha_out, _ = ref_mha(x_batched, x_batched, x_batched, need_weights=False)
            torch.cuda.synchronize()
            mha_time = (time.time() - start) / 20 * 1000
            mha_out_flat = mha_out.view(total_tokens, dim)
        
        results["mha_time_ms"].append(mha_time)
        print(f"  nn.MHA (batched):    {mha_time:8.3f} ms")
        
        # ===== Run xformers (packed with BlockDiagonalMask) =====
        if HAS_XFORMERS:
            xf_mha = create_xformers_mha(dim, nheads, device, ref_mha)
            xf_mha.eval()
            with torch.no_grad():
                for _ in range(5):
                    _ = xf_mha(x_packed, x_packed, x_packed, block_mask=xf_mask)
                torch.cuda.synchronize()
                
                start = time.time()
                for _ in range(20):
                    xf_out = xf_mha(x_packed, x_packed, x_packed, block_mask=xf_mask)
                torch.cuda.synchronize()
                xf_time = (time.time() - start) / 20 * 1000
                
                diff = compare_outputs(mha_out_flat, xf_out)
            
            results["xformers_time_ms"].append(xf_time)
            results["xformers_max_diff"].append(diff["max_diff"])
            results["xformers_mean_diff"].append(diff["mean_diff"])
            print(f"  xformers (packed):   {xf_time:8.3f} ms  (max_diff: {diff['max_diff']:.2e})")
        else:
            results["xformers_time_ms"].append(float('nan'))
            results["xformers_max_diff"].append(float('nan'))
            results["xformers_mean_diff"].append(float('nan'))
        
        # ===== Run flex_attention (packed with compiled BlockMask) =====
        if HAS_FLEX:
            flex_mha = create_flex_mha(dim, nheads, device, ref_mha)
            flex_mha.eval()
            with torch.no_grad():
                # Warmup (triggers compilation)
                for _ in range(5):
                    _ = flex_mha(x_packed, x_packed, x_packed, block_mask=flex_mask)
                torch.cuda.synchronize()
                
                start = time.time()
                for _ in range(20):
                    flex_out = flex_mha(x_packed, x_packed, x_packed, block_mask=flex_mask)
                torch.cuda.synchronize()
                flex_time = (time.time() - start) / 20 * 1000
                
                diff = compare_outputs(mha_out_flat, flex_out)
            
            results["flex_time_ms"].append(flex_time)
            results["flex_max_diff"].append(diff["max_diff"])
            results["flex_mean_diff"].append(diff["mean_diff"])
            print(f"  flex (packed+compiled): {flex_time:8.3f} ms  (max_diff: {diff['max_diff']:.2e})")
        else:
            results["flex_time_ms"].append(float('nan'))
            results["flex_max_diff"].append(float('nan'))
            results["flex_mean_diff"].append(float('nan'))
        
        torch.cuda.empty_cache()
    
    return results


def run_full_model_comparison(config_path="configs/cifar.yaml", batch_size=16, device="cuda"):
    """
    Compare full model forward pass:
    - Vanilla model with xformers (packed sequences)
    - Vanilla model with flex_attention (packed sequences)
    
    Both use identical interfaces, so comparison is straightforward.
    """
    import yaml
    
    print("\n" + "=" * 70)
    print("FULL MODEL COMPARISON")
    print("Vanilla (xformers) vs Vanilla (flex_attention)")
    print("=" * 70)
    
    vanilla_path = os.path.dirname(os.path.abspath(__file__))
    
    # Load config
    with open(os.path.join(vanilla_path, config_path)) as f:
        config = yaml.safe_load(f)
    
    rin_config = config["rin"]
    diffusion_config = config["diffusion"]
    
    results = {
        "component": [],
        "xformers_vs_flex_max": [],
        "xformers_vs_flex_mean": [],
    }
    
    # ===== Load Vanilla Models =====
    # Clear cached imports
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("rin_pytorch"):
            del sys.modules[mod_name]
    
    print("\nLoading vanilla model with xformers...")
    os.environ["ATTENTION_BACKEND"] = "xformers"
    from rin_pytorch.Rin import Rin
    from rin_pytorch.RinDiffusionModel import RinDiffusionModel
    from rin_pytorch.utils.data_utils import pack_sequences
    
    torch.manual_seed(42)
    rin_xf = Rin(**rin_config).to(device).eval()
    model_xf = RinDiffusionModel(rin=rin_xf, **diffusion_config).to(device).eval()
    
    # Clear and reload for flex
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("rin_pytorch"):
            del sys.modules[mod_name]
    
    print("Loading vanilla model with flex_attention...")
    os.environ["ATTENTION_BACKEND"] = "flex"
    from rin_pytorch.Rin import Rin as RinFlex
    from rin_pytorch.RinDiffusionModel import RinDiffusionModel as RinDiffusionModelFlex
    from rin_pytorch.utils.data_utils import pack_sequences as pack_sequences_flex
    
    torch.manual_seed(42)
    rin_flex = RinFlex(**rin_config).to(device).eval()
    model_flex = RinDiffusionModelFlex(rin=rin_flex, **diffusion_config).to(device).eval()
    
    # Copy weights from xformers to flex (both have identical structure)
    model_flex.load_state_dict(model_xf.state_dict())
    print("  Weights copied from xformers to flex model")
    
    # ===== Create test inputs =====
    torch.manual_seed(123)
    tape_dim = rin_config.get("tape_dim", 256)
    patch_size = rin_config.get("patch_height", 4)
    image_size = rin_config.get("image_height", 32)
    
    # Create batch as list of (image, label) tuples
    images_list = [torch.randn(3, image_size, image_size) for _ in range(batch_size)]
    labels_list = [i % 10 for i in range(batch_size)]
    batch = list(zip(images_list, labels_list))
    
    t = torch.rand(batch_size, device=device) * 0.5 + 0.25
    
    # One-hot encode labels for cond_proj (expects (batch, num_classes))
    num_classes = rin_config.get("num_classes", 10)
    labels_tensor = torch.tensor(labels_list, device=device)
    cond = torch.nn.functional.one_hot(labels_tensor, num_classes).float()
    
    # Prepare input (packed)
    packed_data = pack_sequences(batch, patch_size, tape_dim)
    packed_patches = packed_data["patches"].to(device)
    packed_pos_embs = packed_data["token_pos_embs"].to(device)
    doc_ids = packed_data["doc_ids"].to(device)
    offsets = packed_data["offsets"].to(device)
    
    print(f"\nConfig: {config_path}")
    print(f"Batch size: {batch_size}, Image size: {image_size}x{image_size}")
    print(f"Packed patches: {packed_patches.shape}")
    
    # ===== Forward pass comparison =====
    print("\n" + "-" * 60)
    print("Running forward passes...")
    print("-" * 60)
    
    with torch.no_grad():
        # xformers model
        out_xf = model_xf(
            packed_patches, cond, t, None, packed_pos_embs,
            doc_ids=doc_ids, offsets=offsets,
        )
        
        # flex model
        out_flex = model_flex(
            packed_patches, cond, t, None, packed_pos_embs,
            doc_ids=doc_ids, offsets=offsets,
        )
        
        # Compare
        diff = compare_outputs(out_xf, out_flex)
        
        results["component"].append("Output")
        results["xformers_vs_flex_max"].append(diff["max_diff"])
        results["xformers_vs_flex_mean"].append(diff["mean_diff"])
        
        print(f"\nNumerical Comparison (xformers vs flex_attention):")
        print(f"  max_diff:  {diff['max_diff']:.2e}")
        print(f"  mean_diff: {diff['mean_diff']:.2e}")
    
    return results, model_xf, model_flex, pack_sequences


def run_error_accumulation_test(model_xf, model_flex, vanilla_pack_fn,
                                config_path="configs/cifar.yaml", 
                                batch_size=4, num_steps=10, device="cuda"):
    """
    Test how numerical errors accumulate over multiple forward passes.
    Compares: Vanilla (xformers) vs Vanilla (flex_attention)
    """
    import yaml
    
    print("\n" + "=" * 70)
    print("ERROR ACCUMULATION TEST - Across t Values")
    print("xformers vs flex_attention")
    print("=" * 70)
    
    vanilla_path = os.path.dirname(os.path.abspath(__file__))
    
    # Load config
    with open(os.path.join(vanilla_path, config_path)) as f:
        config = yaml.safe_load(f)
    
    rin_config = config["rin"]
    
    results = {
        "step": list(range(1, num_steps + 1)),
        "t_value": [],
        "xf_vs_flex_max": [],
        "xf_vs_flex_mean": [],
    }
    
    # Create test input
    torch.manual_seed(123)
    tape_dim = rin_config.get("tape_dim", 256)
    patch_size = rin_config.get("patch_height", 4)
    image_size = rin_config.get("image_height", 32)
    
    # Create batch
    images_list = [torch.randn(3, image_size, image_size) for _ in range(batch_size)]
    labels_list = [i % 10 for i in range(batch_size)]
    batch = list(zip(images_list, labels_list))
    
    # One-hot encode labels
    num_classes = rin_config.get("num_classes", 10)
    labels_tensor = torch.tensor(labels_list, device=device)
    cond = torch.nn.functional.one_hot(labels_tensor, num_classes).float()
    
    packed_data = vanilla_pack_fn(batch, patch_size, tape_dim)
    packed_patches = packed_data["patches"].to(device)
    packed_pos_embs = packed_data["token_pos_embs"].to(device)
    doc_ids = packed_data["doc_ids"].to(device)
    offsets = packed_data["offsets"].to(device)
    
    print(f"\nNum steps: {num_steps}, Batch size: {batch_size}")
    print("-" * 50)
    print(f"{'Step':>4} | {'t value':>8} | {'max_diff':>12} | {'mean_diff':>12}")
    print("-" * 50)
    
    with torch.no_grad():
        for step in range(num_steps):
            t_val = 1.0 - (step / num_steps)
            t = torch.full((batch_size,), t_val, device=device)
            
            # Forward passes
            out_xf = model_xf(packed_patches, cond, t, None, packed_pos_embs, 
                            doc_ids=doc_ids, offsets=offsets)
            out_flex = model_flex(packed_patches, cond, t, None, packed_pos_embs,
                                 doc_ids=doc_ids, offsets=offsets)
            
            diff = compare_outputs(out_xf, out_flex)
            
            results["t_value"].append(t_val)
            results["xf_vs_flex_max"].append(diff["max_diff"])
            results["xf_vs_flex_mean"].append(diff["mean_diff"])
            
            print(f"{step+1:>4} | {t_val:>8.2f} | {diff['max_diff']:>12.2e} | {diff['mean_diff']:>12.2e}")
    
    print("-" * 50)
    
    return results


def save_results_csv(results, prefix="attention_comparison"):
    """Save results to CSV files (full and pgfplot-friendly)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Full results CSV
    full_csv = f"{prefix}_full_{timestamp}.csv"
    with open(full_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = list(results.keys())
        writer.writerow(header)
        
        # Data rows
        num_rows = len(results[header[0]])
        for i in range(num_rows):
            row = [results[key][i] if i < len(results[key]) else "" for key in header]
            writer.writerow(row)
    
    print(f"✓ Full CSV saved: {full_csv}")
    
    # pgfplot-friendly CSV (space-separated, clean headers)
    pgf_csv = f"{prefix}_pgfplot_{timestamp}.csv"
    with open(pgf_csv, 'w') as f:
        # Clean header names for pgfplot
        clean_header = [key.replace("_", "") for key in results.keys()]
        f.write(" ".join(clean_header) + "\n")
        
        num_rows = len(results[list(results.keys())[0]])
        for i in range(num_rows):
            row = []
            for key in results.keys():
                val = results[key][i] if i < len(results[key]) else 0
                if isinstance(val, float):
                    if np.isnan(val) or val == float('inf'):
                        row.append("nan")
                    else:
                        row.append(f"{val:.6e}")
                else:
                    row.append(str(val))
            f.write(" ".join(row) + "\n")
    
    print(f"✓ pgfplot CSV saved: {pgf_csv}")
    
    return full_csv, pgf_csv


def plot_numerical_comparison(attn_results, accum_results, save_path="numerical_comparison.png"):
    """Create visualization focused on numerical stability and performance"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    batch_sizes = attn_results["batch_size"]
    total_tokens = attn_results["total_tokens"]
    
    # Plot 1: Runtime comparison
    ax1 = axes[0]
    
    mha_times = [t for t in attn_results["mha_time_ms"] if t != float('inf') and not np.isnan(t)]
    xf_times = [t for t in attn_results["xformers_time_ms"] if t != float('inf') and not np.isnan(t)]
    flex_times = [t for t in attn_results["flex_time_ms"] if t != float('inf') and not np.isnan(t)]
    
    if mha_times:
        ax1.plot(batch_sizes[:len(mha_times)], mha_times, 'o-', label='nn.MHA (batched)', linewidth=2, markersize=8)
    if xf_times:
        ax1.plot(batch_sizes[:len(xf_times)], xf_times, 's-', label='xformers (packed)', linewidth=2, markersize=8, color='C1')
    if flex_times:
        ax1.plot(batch_sizes[:len(flex_times)], flex_times, '^-', label='flex (packed+compiled)', linewidth=2, markersize=8, color='C2')
    
    ax1.set_xlabel('Documents (batch size)', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('Runtime Comparison\n(256 tokens/doc)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    
    # Plot 2: Numerical accuracy vs batch size
    ax2 = axes[1]
    
    if "xformers_max_diff" in attn_results:
        xf_diff = [d for d in attn_results["xformers_max_diff"] if not np.isnan(d)]
        if xf_diff:
            ax2.plot(batch_sizes[:len(xf_diff)], xf_diff, 's-', label='xformers vs nn.MHA', linewidth=2, markersize=8, color='C1')
    
    if "flex_max_diff" in attn_results:
        flex_diff = [d for d in attn_results["flex_max_diff"] if not np.isnan(d)]
        if flex_diff:
            ax2.plot(batch_sizes[:len(flex_diff)], flex_diff, '^-', label='flex vs nn.MHA', linewidth=2, markersize=8, color='C2')
    
    ax2.set_xlabel('Documents (batch size)', fontsize=12)
    ax2.set_ylabel('Max Diff vs nn.MHA', fontsize=12)
    ax2.set_title('Numerical Accuracy\n(packed vs batched)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    
    # Plot 3: Speedup vs nn.MHA
    ax3 = axes[2]
    
    if mha_times and xf_times:
        xf_speedup = [mha_times[i] / xf_times[i] for i in range(min(len(mha_times), len(xf_times)))]
        ax3.plot(batch_sizes[:len(xf_speedup)], xf_speedup, 's-', label='xformers', linewidth=2, markersize=8, color='C1')
    
    if mha_times and flex_times:
        flex_speedup = [mha_times[i] / flex_times[i] for i in range(min(len(mha_times), len(flex_times)))]
        ax3.plot(batch_sizes[:len(flex_speedup)], flex_speedup, '^-', label='flex_attention', linewidth=2, markersize=8, color='C2')
    
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='nn.MHA baseline')
    ax3.set_xlabel('Documents (batch size)', fontsize=12)
    ax3.set_ylabel('Speedup vs nn.MHA', fontsize=12)
    ax3.set_title('Relative Performance\n(>1 = faster than nn.MHA)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {save_path}")
    plt.close()


def print_summary_table(results):
    """Print a nice summary table"""
    print("\n" + "=" * 100)
    print("ATTENTION MODULE COMPARISON - Packed Sequences with Document Masks")
    print("=" * 100)
    
    header = f"{'Batch':>6} | {'Tokens':>8} | {'nn.MHA (ms)':>12} | {'xformers (ms)':>13} | {'flex (ms)':>11} | {'xf diff':>10} | {'flex diff':>10}"
    print(header)
    print("-" * 100)
    
    for i, batch in enumerate(results["batch_size"]):
        tokens = results["total_tokens"][i]
        mha_t = results["mha_time_ms"][i]
        xf_t = results["xformers_time_ms"][i]
        flex_t = results["flex_time_ms"][i]
        xf_d = results["xformers_max_diff"][i] if i < len(results["xformers_max_diff"]) else float('nan')
        flex_d = results["flex_max_diff"][i] if i < len(results["flex_max_diff"]) else float('nan')
        
        mha_str = f"{mha_t:12.3f}" if mha_t != float('inf') else "OOM"
        xf_str = f"{xf_t:13.3f}" if not np.isnan(xf_t) and xf_t != float('inf') else "N/A"
        flex_str = f"{flex_t:11.3f}" if not np.isnan(flex_t) and flex_t != float('inf') else "N/A"
        xf_d_str = f"{xf_d:.2e}" if not np.isnan(xf_d) else "N/A"
        flex_d_str = f"{flex_d:.2e}" if not np.isnan(flex_d) else "N/A"
        
        print(f"{batch:>6} | {tokens:>8} | {mha_str:>12} | {xf_str:>13} | {flex_str:>11} | {xf_d_str:>10} | {flex_d_str:>10}")
    
    print("=" * 100)
    
    # Speedup summary
    print("\nSpeedup vs nn.MHA:")
    for i, batch in enumerate(results["batch_size"]):
        mha_t = results["mha_time_ms"][i]
        xf_t = results["xformers_time_ms"][i]
        flex_t = results["flex_time_ms"][i]
        
        if not np.isnan(xf_t) and mha_t > 0:
            xf_speedup = mha_t / xf_t
            print(f"  Batch {batch:>3}: xformers {xf_speedup:.2f}x", end="")
        if not np.isnan(flex_t) and mha_t > 0:
            flex_speedup = mha_t / flex_t
            print(f", flex {flex_speedup:.2f}x", end="")
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare attention backends - numerical stability focus")
    parser.add_argument("--dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--nheads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length per sample")
    parser.add_argument("--batch_sizes", type=str, default="2,4,8,16,32,64,128,256",
                        help="Comma-separated batch sizes (documents)")
    parser.add_argument("--output", type=str, default="numerical_comparison.png",
                        help="Output plot filename")
    parser.add_argument("--config", type=str, default="configs/cifar.yaml",
                        help="Config file for full model comparison")
    parser.add_argument("--num_steps", type=int, default=10,
                        help="Number of steps for error accumulation test")
    parser.add_argument("--skip_full_model", action="store_true",
                        help="Skip full model comparison")
    
    args = parser.parse_args()
    
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    
    print("=" * 70)
    print("ATTENTION BACKEND COMPARISON - Numerical Stability Focus")
    print("=" * 70)
    print(f"Available backends: {BACKENDS}")
    
    # 1. Attention module comparison
    print("\n" + "=" * 70)
    print("PART 1: ATTENTION MODULE COMPARISON")
    print("=" * 70)
    
    attn_results = run_attention_comparison(
        dim=args.dim,
        nheads=args.nheads,
        batch_sizes=batch_sizes,
        seq_len=args.seq_len,
    )
    
    print_summary_table(attn_results)
    
    # Save attention results to CSV
    save_results_csv(attn_results, prefix="attn_comparison")
    
    # 2. Full model comparison (optional)
    accum_results = None
    if not args.skip_full_model:
        try:
            # Full model comparison - returns models for reuse
            model_results, model_xf, model_flex, vanilla_pack_fn = run_full_model_comparison(
                config_path=args.config,
                batch_size=16,
            )
            save_results_csv(model_results, prefix="model_comparison")
            
            # Error accumulation test using the same models
            accum_results = run_error_accumulation_test(
                model_xf=model_xf,
                model_flex=model_flex,
                vanilla_pack_fn=vanilla_pack_fn,
                config_path=args.config,
                batch_size=4,
                num_steps=args.num_steps,
            )
            save_results_csv(accum_results, prefix="error_accumulation")
            
        except Exception as e:
            print(f"\n⚠ Full model comparison failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 3. Create visualization
    plot_numerical_comparison(attn_results, accum_results, save_path=args.output)
    
    print("\n" + "=" * 70)
    print("✓ Comparison complete!")
    print("=" * 70)
