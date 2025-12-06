#!/usr/bin/env python3
"""
Benchmark attention backends: Flex Attention vs xformers vs Flash Attention

This script measures:
1. Mask creation time (one-time cost)
2. Attention computation time (after warmup/compilation)
3. Total forward pass time

Usage:
    python benchmark_attention.py [--batch_sizes 4,16,64,128,256] [--seq_len 256] [--warmup 50] [--iters 100]
"""

import torch
import torch.nn as nn
import time
import argparse
import sys

# Check available backends
FLEX_AVAILABLE = True
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask
    # Compile flex_attention and create_block_mask for better performance
    compiled_flex_attention = torch.compile(flex_attention, dynamic=True)
    compiled_create_block_mask = torch.compile(create_block_mask, dynamic=True)
except ImportError:
    FLEX_AVAILABLE = False
    compiled_flex_attention = None
    compiled_create_block_mask = None

try:
    from xformers.ops import memory_efficient_attention
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

try:
    from flash_attn import flash_attn_varlen_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False


def create_flex_mask(doc_ids, device, use_compiled=True):
    """Create flex attention block mask."""
    doc_ids = doc_ids.to(torch.long)
    seq_len = int(doc_ids.shape[0])
    
    def _mask(_: torch.Tensor, __: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        return doc_ids[q_idx] == doc_ids[kv_idx]
    
    mask_fn = compiled_create_block_mask if use_compiled else create_block_mask
    return mask_fn(mask_mod=_mask, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)


def create_xformers_mask(offsets):
    """Create xformers BlockDiagonalMask from offsets."""
    offsets_list = offsets.tolist()
    seqlens = [offsets_list[i+1] - offsets_list[i] for i in range(len(offsets_list) - 1)]
    return BlockDiagonalMask.from_seqlens(seqlens)


def create_flash_cu_seqlens(offsets):
    """Create cu_seqlens for flash attention."""
    cu_seqlens = offsets.to(torch.int32)
    seqlens = offsets[1:] - offsets[:-1]
    max_seqlen = seqlens.max().item()
    return cu_seqlens, max_seqlen


class Timer:
    def __init__(self, device='cuda'):
        self.device = device
        
    def sync(self):
        if self.device == 'cuda':
            torch.cuda.synchronize()
    
    def time_fn(self, fn, warmup=10, iters=100):
        """Time a function with warmup."""
        # Warmup
        for _ in range(warmup):
            fn()
        self.sync()
        
        # Timed iterations
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        self.sync()
        
        return (time.perf_counter() - start) / iters * 1000  # ms


def benchmark_batch_size(batch_size, seq_len, dim, nheads, warmup, iters, device):
    """Benchmark all backends for a given batch size."""
    total_tokens = batch_size * seq_len
    head_dim = dim // nheads
    
    # Create inputs
    torch.manual_seed(42)
    
    # For flex: needs [B, H, S, D] format
    q_flex = torch.randn(1, nheads, total_tokens, head_dim, device=device)
    k_flex = torch.randn(1, nheads, total_tokens, head_dim, device=device)
    v_flex = torch.randn(1, nheads, total_tokens, head_dim, device=device)
    
    # For xformers: needs [B, S, H, D] format
    q_xf = torch.randn(1, total_tokens, nheads, head_dim, device=device)
    k_xf = torch.randn(1, total_tokens, nheads, head_dim, device=device)
    v_xf = torch.randn(1, total_tokens, nheads, head_dim, device=device)
    
    # For flash: needs [total, H, D] format
    q_flash = torch.randn(total_tokens, nheads, head_dim, device=device, dtype=torch.float16)
    k_flash = torch.randn(total_tokens, nheads, head_dim, device=device, dtype=torch.float16)
    v_flash = torch.randn(total_tokens, nheads, head_dim, device=device, dtype=torch.float16)
    
    # Document IDs and offsets
    doc_ids = torch.arange(batch_size, device=device).repeat_interleave(seq_len)
    offsets = torch.arange(batch_size + 1, device=device) * seq_len
    
    timer = Timer(device)
    results = {'batch_size': batch_size, 'total_tokens': total_tokens}
    
    # ==================== FLEX ATTENTION ====================
    if FLEX_AVAILABLE:
        try:
            # Time mask creation with compiled create_block_mask
            def flex_mask_fn():
                return create_flex_mask(doc_ids, device, use_compiled=True)
            
            # Create mask for attention timing (with warmup for compilation)
            print(f"  [Flex] Warming up compiled create_block_mask for batch={batch_size}...")
            flex_mask = flex_mask_fn()  # First call triggers compilation
            results['flex_mask_time'] = timer.time_fn(flex_mask_fn, warmup=5, iters=20)
            
            # Time compiled attention (after warmup/compilation)
            def flex_attn_fn():
                return compiled_flex_attention(q_flex, k_flex, v_flex, block_mask=flex_mask)
            
            print(f"  [Flex] Warming up compiled flex_attention for batch={batch_size}...")
            results['flex_attn_time'] = timer.time_fn(flex_attn_fn, warmup=warmup, iters=iters)
            
            # Time total (mask + attention) - simulating real usage
            def flex_total_fn():
                mask = create_flex_mask(doc_ids, device, use_compiled=True)
                return compiled_flex_attention(q_flex, k_flex, v_flex, block_mask=mask)
            
            results['flex_total_time'] = timer.time_fn(flex_total_fn, warmup=10, iters=20)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                print(f"  [Flex] OOM at batch_size={batch_size}")
                results['flex_mask_time'] = float('inf')  # OOM marker
                results['flex_attn_time'] = float('inf')
                results['flex_total_time'] = float('inf')
                torch.cuda.empty_cache()
            else:
                raise
    else:
        results['flex_mask_time'] = float('nan')
        results['flex_attn_time'] = float('nan')
        results['flex_total_time'] = float('nan')
    
    # ==================== XFORMERS ====================
    if XFORMERS_AVAILABLE:
        # Time mask creation
        def xf_mask_fn():
            return create_xformers_mask(offsets)
        
        xf_mask = xf_mask_fn()
        results['xformers_mask_time'] = timer.time_fn(xf_mask_fn, warmup=warmup, iters=iters)
        
        # Time attention
        def xf_attn_fn():
            return memory_efficient_attention(q_xf, k_xf, v_xf, attn_bias=xf_mask)
        
        results['xformers_attn_time'] = timer.time_fn(xf_attn_fn, warmup=warmup, iters=iters)
        
        # Time total
        def xf_total_fn():
            mask = create_xformers_mask(offsets)
            return memory_efficient_attention(q_xf, k_xf, v_xf, attn_bias=mask)
        
        results['xformers_total_time'] = timer.time_fn(xf_total_fn, warmup=warmup, iters=iters)
    else:
        results['xformers_mask_time'] = float('nan')
        results['xformers_attn_time'] = float('nan')
        results['xformers_total_time'] = float('nan')
    
    # ==================== FLASH ATTENTION ====================
    if FLASH_AVAILABLE:
        cu_seqlens, max_seqlen = create_flash_cu_seqlens(offsets)
        
        # Time cu_seqlens creation (minimal)
        def flash_mask_fn():
            return create_flash_cu_seqlens(offsets)
        
        results['flash_mask_time'] = timer.time_fn(flash_mask_fn, warmup=warmup, iters=iters)
        
        # Time attention
        def flash_attn_fn():
            return flash_attn_varlen_func(
                q_flash, k_flash, v_flash,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
            )
        
        results['flash_attn_time'] = timer.time_fn(flash_attn_fn, warmup=warmup, iters=iters)
        
        # Time total
        def flash_total_fn():
            cu, maxlen = create_flash_cu_seqlens(offsets)
            return flash_attn_varlen_func(
                q_flash, k_flash, v_flash,
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
                max_seqlen_q=maxlen,
                max_seqlen_k=maxlen,
            )
        
        results['flash_total_time'] = timer.time_fn(flash_total_fn, warmup=warmup, iters=iters)
    else:
        results['flash_mask_time'] = float('nan')
        results['flash_attn_time'] = float('nan')
        results['flash_total_time'] = float('nan')
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark attention backends')
    parser.add_argument('--batch_sizes', type=str, default='4,16,64,128,256',
                        help='Comma-separated batch sizes to test')
    parser.add_argument('--seq_len', type=int, default=256,
                        help='Sequence length per document (default: 256 for 16x16 patches)')
    parser.add_argument('--dim', type=int, default=256, help='Model dimension')
    parser.add_argument('--nheads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--warmup', type=int, default=50, help='Warmup iterations')
    parser.add_argument('--iters', type=int, default=100, help='Benchmark iterations')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    args = parser.parse_args()
    
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    
    print("=" * 80)
    print("ATTENTION BACKEND BENCHMARK")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Sequence length per doc: {args.seq_len}")
    print(f"  Model dim: {args.dim}, Heads: {args.nheads}")
    print(f"  Warmup: {args.warmup}, Iterations: {args.iters}")
    print(f"  Device: {args.device}")
    
    print(f"\nAvailable backends:")
    print(f"  Flex Attention: {'✓' if FLEX_AVAILABLE else '✗'}")
    print(f"  xformers:       {'✓' if XFORMERS_AVAILABLE else '✗'}")
    print(f"  Flash Attention: {'✓' if FLASH_AVAILABLE else '✗'}")
    
    if args.device == 'cuda':
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    
    print("\n" + "=" * 80)
    print("MASK CREATION TIME (ms)")
    print("=" * 80)
    print(f"{'Batch':<8} {'Tokens':<10} {'Flex':<12} {'xformers':<12} {'Flash':<12}")
    print("-" * 54)
    
    all_results = []
    
    def fmt_time(t):
        if t == float('inf'):
            return "OOM"
        elif t != t:  # nan check
            return "N/A"
        else:
            return f"{t:.3f}"
    
    for batch_size in batch_sizes:
        results = benchmark_batch_size(
            batch_size, args.seq_len, args.dim, args.nheads,
            args.warmup, args.iters, args.device
        )
        all_results.append(results)
        
        print(f"{batch_size:<8} {results['total_tokens']:<10} "
              f"{fmt_time(results['flex_mask_time']):<12} "
              f"{fmt_time(results['xformers_mask_time']):<12} "
              f"{fmt_time(results['flash_mask_time']):<12}")
    
    print("\n" + "=" * 80)
    print("ATTENTION COMPUTATION TIME (ms) - after warmup/compilation")
    print("=" * 80)
    print(f"{'Batch':<8} {'Tokens':<10} {'Flex':<12} {'xformers':<12} {'Flash':<12}")
    print("-" * 54)
    
    for results in all_results:
        print(f"{results['batch_size']:<8} {results['total_tokens']:<10} "
              f"{fmt_time(results['flex_attn_time']):<12} "
              f"{fmt_time(results['xformers_attn_time']):<12} "
              f"{fmt_time(results['flash_attn_time']):<12}")
    
    print("\n" + "=" * 80)
    print("TOTAL TIME (mask + attention) (ms)")
    print("=" * 80)
    print(f"{'Batch':<8} {'Tokens':<10} {'Flex':<12} {'xformers':<12} {'Flash':<12}")
    print("-" * 54)
    
    for results in all_results:
        print(f"{results['batch_size']:<8} {results['total_tokens']:<10} "
              f"{fmt_time(results['flex_total_time']):<12} "
              f"{fmt_time(results['xformers_total_time']):<12} "
              f"{fmt_time(results['flash_total_time']):<12}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY (at batch_size=256)")
    print("=" * 80)
    
    final = all_results[-1]  # batch_size=256
    
    def fmt_summary(t):
        if t == float('inf'):
            return "OOM"
        elif t != t:
            return "N/A"
        else:
            return f"{t:.2f} ms"
    
    print("\nMask creation:")
    if FLEX_AVAILABLE:
        print(f"  Flex:     {fmt_summary(final['flex_mask_time'])}")
    if XFORMERS_AVAILABLE:
        print(f"  xformers: {fmt_summary(final['xformers_mask_time'])}")
    if FLASH_AVAILABLE:
        print(f"  Flash:    {fmt_summary(final['flash_mask_time'])}")
    
    print("\nAttention (compiled/warmed up):")
    if FLEX_AVAILABLE:
        print(f"  Flex:     {fmt_summary(final['flex_attn_time'])}")
    if XFORMERS_AVAILABLE:
        print(f"  xformers: {fmt_summary(final['xformers_attn_time'])}")
    if FLASH_AVAILABLE:
        print(f"  Flash:    {fmt_summary(final['flash_attn_time'])}")
    
    print("\nTotal (realistic usage):")
    if FLEX_AVAILABLE:
        print(f"  Flex:     {fmt_summary(final['flex_total_time'])}")
    if XFORMERS_AVAILABLE:
        print(f"  xformers: {fmt_summary(final['xformers_total_time'])}")
    if FLASH_AVAILABLE:
        print(f"  Flash:    {fmt_summary(final['flash_total_time'])}")
    
    # Find fastest
    print("\n" + "-" * 40)
    backends = []
    
    def is_valid(t):
        return t == t and t != float('inf')  # not nan and not inf
    
    if FLEX_AVAILABLE and is_valid(final['flex_total_time']):
        backends.append(('Flex', final['flex_total_time']))
    if XFORMERS_AVAILABLE and is_valid(final['xformers_total_time']):
        backends.append(('xformers', final['xformers_total_time']))
    if FLASH_AVAILABLE and is_valid(final['flash_total_time']):
        backends.append(('Flash', final['flash_total_time']))
    
    if backends:
        backends.sort(key=lambda x: x[1])
        fastest = backends[0]
        print(f"\n🏆 FASTEST: {fastest[0]} ({fastest[1]:.2f} ms)")
        
        if len(backends) > 1:
            for name, time in backends[1:]:
                slowdown = time / fastest[1]
                print(f"   {name}: {time:.2f} ms ({slowdown:.1f}x slower)")


if __name__ == '__main__':
    main()

