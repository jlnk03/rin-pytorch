#!/usr/bin/env python3
"""
Compare attention backends with proper document masking:
- nn.MHA with attention_mask
- xformers with BlockDiagonalMask  
- flex_attention with compiled create_block_mask

Batch size 256, seq_len 256 (65536 total tokens)
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Config
DIM = 256
NHEADS = 8
BATCH_SIZE = 256
SEQ_LEN = 256
TOTAL_TOKENS = BATCH_SIZE * SEQ_LEN

print(f"Config: dim={DIM}, nheads={NHEADS}, batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}")
print(f"Total tokens: {TOTAL_TOKENS:,}")
print("=" * 70)

# ============================================================================
# Setup backends
# ============================================================================

# Standard MHA
print("\nSetting up nn.MultiheadAttention...")
torch.manual_seed(42)
std_mha = nn.MultiheadAttention(DIM, NHEADS, batch_first=True).to(device)
std_mha.eval()

# xformers
print("Setting up xformers...")
from xformers.ops import memory_efficient_attention
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from rin_pytorch.modules.MHA_xformers import XformersMultiheadAttention

torch.manual_seed(42)
xf_mha = XformersMultiheadAttention(E_q=DIM, E_k=DIM, E_v=DIM, E_total=DIM, nheads=NHEADS).to(device)
with torch.no_grad():
    xf_mha.packed_proj.weight.copy_(std_mha.in_proj_weight)
    xf_mha.packed_proj.bias.copy_(std_mha.in_proj_bias)
    xf_mha.out_proj.weight.copy_(std_mha.out_proj.weight)
    xf_mha.out_proj.bias.copy_(std_mha.out_proj.bias)
xf_mha.eval()

# Flex attention (compiled)
print("Setting up flex_attention (with torch.compile)...")
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask

# Compile flex_attention and create_block_mask
flex_attention_compiled = torch.compile(flex_attention, dynamic=False)
create_block_mask_compiled = torch.compile(create_block_mask, dynamic=False)

from rin_pytorch.modules.MHA import FlexMultiheadAttention

torch.manual_seed(42)
flex_mha = FlexMultiheadAttention(E_q=DIM, E_k=DIM, E_v=DIM, E_total=DIM, nheads=NHEADS).to(device)
with torch.no_grad():
    flex_mha.packed_proj.weight.copy_(std_mha.in_proj_weight)
    flex_mha.packed_proj.bias.copy_(std_mha.in_proj_bias)
    flex_mha.out_proj.weight.copy_(std_mha.out_proj.weight)
    flex_mha.out_proj.bias.copy_(std_mha.out_proj.bias)
flex_mha.eval()

# ============================================================================
# Create masks
# ============================================================================
print("\nCreating masks...")

# For standard MHA: block diagonal attention mask [batch, seq, seq] or [batch*nheads, seq, seq]
# We create a mask where each sample only attends to itself
def create_block_diagonal_attn_mask(batch_size, seq_len, device):
    """Create attention mask for nn.MHA that mimics document isolation"""
    # For batch_first=True MHA, mask shape is [batch, seq, seq] or use key_padding_mask
    # Actually, attn_mask in nn.MHA is additive and broadcast across batch
    # So we can't easily do per-sample masking with attn_mask
    # Instead we'll just run without mask for fair comparison (nn.MHA can't do block diagonal easily)
    return None

# For xformers: BlockDiagonalMask
seqlens = [SEQ_LEN] * BATCH_SIZE
xf_mask = BlockDiagonalMask.from_seqlens(seqlens)
print(f"  xformers mask: BlockDiagonalMask with {len(seqlens)} blocks of {SEQ_LEN} tokens")

# For flex: compiled create_block_mask
doc_ids = torch.arange(BATCH_SIZE, device=device).repeat_interleave(SEQ_LEN)
offsets = torch.arange(BATCH_SIZE + 1, device=device) * SEQ_LEN

def document_mask_fn(b, h, q_idx, kv_idx):
    """Mask function: only attend within same document"""
    return doc_ids[q_idx] == doc_ids[kv_idx]

print("  Compiling flex attention block mask...")
# Pre-compile the mask
flex_mask = create_block_mask_compiled(
    document_mask_fn,
    B=None,
    H=None, 
    Q_LEN=TOTAL_TOKENS,
    KV_LEN=TOTAL_TOKENS,
    device=device,
)
print(f"  flex mask: BlockMask for {TOTAL_TOKENS} tokens")

# ============================================================================
# Create test data
# ============================================================================
print("\nCreating test data...")
torch.manual_seed(123)

# Batched format for nn.MHA
x_batched = torch.randn(BATCH_SIZE, SEQ_LEN, DIM, device=device)

# Packed format for xformers/flex
x_packed = x_batched.view(TOTAL_TOKENS, DIM)

print(f"  x_batched shape: {x_batched.shape}")
print(f"  x_packed shape: {x_packed.shape}")

# ============================================================================
# Warmup
# ============================================================================
print("\nWarming up...")

# Warmup nn.MHA
for _ in range(3):
    with torch.no_grad():
        _ = std_mha(x_batched, x_batched, x_batched, need_weights=False)
torch.cuda.synchronize()

# Warmup xformers
for _ in range(3):
    with torch.no_grad():
        _ = xf_mha(x_packed, x_packed, x_packed, block_mask=xf_mask)
torch.cuda.synchronize()

# Warmup flex (this triggers compilation)
print("  Warming up flex_attention (compilation may take a moment)...")
for _ in range(3):
    with torch.no_grad():
        _ = flex_mha(x_packed, x_packed, x_packed, block_mask=flex_mask)
torch.cuda.synchronize()

print("  Warmup complete!")

# ============================================================================
# Single batch comparison
# ============================================================================
print("\n" + "=" * 70)
print("SINGLE BATCH COMPARISON")
print("=" * 70)

# nn.MHA
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    mha_out, _ = std_mha(x_batched, x_batched, x_batched, need_weights=False)
torch.cuda.synchronize()
mha_time = (time.time() - start) * 1000
mha_out_flat = mha_out.view(TOTAL_TOKENS, DIM)
print(f"nn.MHA:         {mha_time:8.2f} ms")

# xformers
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    xf_out = xf_mha(x_packed, x_packed, x_packed, block_mask=xf_mask)
torch.cuda.synchronize()
xf_time = (time.time() - start) * 1000
xf_diff = (xf_out - mha_out_flat).abs().max().item()
print(f"xformers:       {xf_time:8.2f} ms  (diff vs MHA: {xf_diff:.2e})")

# flex compiled
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    flex_out = flex_mha(x_packed, x_packed, x_packed, block_mask=flex_mask)
torch.cuda.synchronize()
flex_time = (time.time() - start) * 1000
flex_diff = (flex_out - mha_out_flat).abs().max().item()
print(f"flex (compiled):{flex_time:8.2f} ms  (diff vs MHA: {flex_diff:.2e})")

# ============================================================================
# Multi-batch comparison (10 batches)
# ============================================================================
print("\n" + "=" * 70)
print("10 BATCH COMPARISON (average time)")
print("=" * 70)

NUM_BATCHES = 10

# nn.MHA
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(NUM_BATCHES):
        _ = std_mha(x_batched, x_batched, x_batched, need_weights=False)
torch.cuda.synchronize()
mha_time_avg = (time.time() - start) / NUM_BATCHES * 1000
print(f"nn.MHA:          {mha_time_avg:8.2f} ms/batch")

# xformers
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(NUM_BATCHES):
        _ = xf_mha(x_packed, x_packed, x_packed, block_mask=xf_mask)
torch.cuda.synchronize()
xf_time_avg = (time.time() - start) / NUM_BATCHES * 1000
print(f"xformers:        {xf_time_avg:8.2f} ms/batch")

# flex compiled
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(NUM_BATCHES):
        _ = flex_mha(x_packed, x_packed, x_packed, block_mask=flex_mask)
torch.cuda.synchronize()
flex_time_avg = (time.time() - start) / NUM_BATCHES * 1000
print(f"flex (compiled): {flex_time_avg:8.2f} ms/batch")

# ============================================================================
# Including mask creation time
# ============================================================================
print("\n" + "=" * 70)
print("10 BATCHES WITH MASK CREATION (realistic scenario)")
print("=" * 70)

# xformers with mask creation
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(NUM_BATCHES):
        # Create mask each time (simulates different batch)
        xf_mask_new = BlockDiagonalMask.from_seqlens(seqlens)
        _ = xf_mha(x_packed, x_packed, x_packed, block_mask=xf_mask_new)
torch.cuda.synchronize()
xf_time_with_mask = (time.time() - start) / NUM_BATCHES * 1000
print(f"xformers (w/ mask creation): {xf_time_with_mask:8.2f} ms/batch")

# flex with compiled mask creation
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(NUM_BATCHES):
        # Create mask each time using compiled version
        flex_mask_new = create_block_mask_compiled(
            document_mask_fn,
            B=None, H=None,
            Q_LEN=TOTAL_TOKENS, KV_LEN=TOTAL_TOKENS,
            device=device,
        )
        _ = flex_mha(x_packed, x_packed, x_packed, block_mask=flex_mask_new)
torch.cuda.synchronize()
flex_time_with_mask = (time.time() - start) / NUM_BATCHES * 1000
print(f"flex (w/ compiled mask):     {flex_time_with_mask:8.2f} ms/batch")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
Configuration:
  - Batch size: {BATCH_SIZE}
  - Sequence length: {SEQ_LEN}
  - Total tokens: {TOTAL_TOKENS:,}
  - Embedding dim: {DIM}
  - Num heads: {NHEADS}

Numerical Accuracy (vs nn.MHA):
  - xformers:       {xf_diff:.2e} {'✓ EXACT' if xf_diff < 1e-6 else '✓ CLOSE' if xf_diff < 1e-4 else '⚠ DIFFERS'}
  - flex_attention: {flex_diff:.2e} {'✓ EXACT' if flex_diff < 1e-6 else '✓ CLOSE' if flex_diff < 1e-4 else '⚠ DIFFERS'}

Runtime (single batch):
  - nn.MHA:         {mha_time:8.2f} ms
  - xformers:       {xf_time:8.2f} ms  ({mha_time/xf_time:.2f}x vs MHA)
  - flex (compiled):{flex_time:8.2f} ms  ({mha_time/flex_time:.2f}x vs MHA)

Runtime (10 batches avg):
  - nn.MHA:         {mha_time_avg:8.2f} ms/batch
  - xformers:       {xf_time_avg:8.2f} ms/batch  ({mha_time_avg/xf_time_avg:.2f}x vs MHA)
  - flex (compiled):{flex_time_avg:8.2f} ms/batch  ({mha_time_avg/flex_time_avg:.2f}x vs MHA)

With mask creation (realistic):
  - xformers:       {xf_time_with_mask:8.2f} ms/batch
  - flex (compiled):{flex_time_with_mask:8.2f} ms/batch

Winner: {'xformers' if xf_time_avg < flex_time_avg else 'flex_attention'}
""")

# Best choice recommendation
print("=" * 70)
print("RECOMMENDATION")
print("=" * 70)
if xf_diff < 1e-6 and xf_time_with_mask < flex_time_with_mask:
    print("✓ Use xformers: Exact numerical match + faster with mask creation")
elif flex_time_avg < xf_time_avg:
    print("✓ Use flex_attention: Faster when mask is pre-computed")
else:
    print("✓ Use xformers: Best overall performance")

