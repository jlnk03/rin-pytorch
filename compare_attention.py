"""
Diagnostic script to find where HierarchicalSparseAttention diverges from SparseMultiheadAttention
"""

import torch
import torch.nn as nn
import sys

# We'll inline minimal versions to test specific functions

def test_expand_block_mask():
    """Test if vectorized vs loop indexing gives same results"""
    print("=" * 60)
    print("TEST 1: _expand_block_mask_to_tokens")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    bsz, num_q_blocks, num_k_blocks = 2, 4, 6
    tgt_len, src_len = 14, 22  # Not exact multiples of block_size
    block_size = 4
    
    critical_blocks = torch.rand(bsz, num_q_blocks, num_k_blocks) > 0.5
    
    q_token_to_block = torch.div(
        torch.arange(tgt_len), block_size, rounding_mode="floor"
    ).clamp_max(num_q_blocks - 1)
    
    k_token_to_block = torch.div(
        torch.arange(src_len), block_size, rounding_mode="floor"
    ).clamp_max(num_k_blocks - 1)
    
    # Document 1: Vectorized
    allowed_v1 = critical_blocks[:, q_token_to_block[:, None], k_token_to_block[None, :]]
    mask_v1 = ~allowed_v1
    
    # Document 2: Loop
    mask_v2 = torch.zeros((bsz, tgt_len, src_len), dtype=torch.bool)
    for b in range(bsz):
        allowed = critical_blocks[b][q_token_to_block][:, k_token_to_block]
        mask_v2[b] = ~allowed
    
    match = torch.equal(mask_v1, mask_v2)
    print(f"Masks match: {match}")
    if not match:
        diff = (mask_v1 != mask_v2).sum().item()
        print(f"Number of differences: {diff}")
        print(f"mask_v1 shape: {mask_v1.shape}")
        print(f"mask_v2 shape: {mask_v2.shape}")
    
    return match


def test_mean_pool_blocks():
    """Test if the count calculation differs"""
    print("\n" + "=" * 60)
    print("TEST 2: _mean_pool_blocks count calculation")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Test WITH padding
    bsz, num_heads, seq_len, head_dim = 2, 4, 14, 32
    block_size = 4
    num_blocks = (seq_len + block_size - 1) // block_size  # 4
    pad_len = num_blocks * block_size - seq_len  # 2
    
    x = torch.randn(bsz, num_heads, seq_len, head_dim)
    
    print(f"seq_len={seq_len}, block_size={block_size}, num_blocks={num_blocks}, pad_len={pad_len}")
    
    # Document 1 approach
    x1 = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
    x1 = x1.view(bsz, num_heads, num_blocks, block_size, head_dim)
    sums1 = x1.sum(dim=3)
    
    counts1 = x1.new_ones(bsz, num_heads, num_blocks, block_size, 1)
    counts1[:, :, -1, -pad_len:, :] = 0
    counts1 = counts1.sum(dim=3)
    pooled1 = sums1 / counts1
    
    # Document 2 approach  
    pad_shape = (bsz, num_heads, pad_len, head_dim)
    x2 = torch.cat([x, x.new_zeros(pad_shape)], dim=2)
    x2 = x2.view(bsz, num_heads, num_blocks, block_size, head_dim)
    sums2 = x2.sum(dim=3)
    
    counts2 = x.new_ones(bsz, num_heads, seq_len, 1)
    counts2 = torch.cat([counts2, x.new_zeros(bsz, num_heads, pad_len, 1)], dim=2)
    counts2 = counts2.view(bsz, num_heads, num_blocks, block_size, 1).sum(dim=3)
    pooled2 = sums2 / counts2.clamp_min(1.0)
    
    diff = (pooled1 - pooled2).abs().max().item()
    print(f"With padding - Max difference: {diff:.2e}")
    print(f"counts1: {counts1[0, 0, :, 0].tolist()}")
    print(f"counts2: {counts2[0, 0, :, 0].tolist()}")
    
    # Test WITHOUT padding (where scalar vs tensor matters)
    seq_len_no_pad = 16  # Exact multiple
    num_blocks_no_pad = 4
    pad_len_no_pad = 0
    
    x_np = torch.randn(bsz, num_heads, seq_len_no_pad, head_dim)
    
    print(f"\nseq_len={seq_len_no_pad}, pad_len={pad_len_no_pad}")
    
    # Document 1: returns scalar when no padding
    x1_np = x_np.view(bsz, num_heads, num_blocks_no_pad, block_size, head_dim)
    sums1_np = x1_np.sum(dim=3)
    counts1_np = float(block_size)  # SCALAR!
    pooled1_np = sums1_np / counts1_np
    
    # Document 2: always returns tensor
    x2_np = x_np.view(bsz, num_heads, num_blocks_no_pad, block_size, head_dim)
    sums2_np = x2_np.sum(dim=3)
    counts2_np = x_np.new_ones(bsz, num_heads, seq_len_no_pad, 1)
    counts2_np = counts2_np.view(bsz, num_heads, num_blocks_no_pad, block_size, 1).sum(dim=3)
    pooled2_np = sums2_np / counts2_np.clamp_min(1.0)
    
    diff_np = (pooled1_np - pooled2_np).abs().max().item()
    print(f"Without padding - Max difference: {diff_np:.2e}")
    print(f"counts1 type: {type(counts1_np)}, value: {counts1_np}")
    print(f"counts2 type: {type(counts2_np)}, shape: {counts2_np.shape}, values: {counts2_np[0, 0, :, 0].tolist()}")
    
    return diff < 1e-6 and diff_np < 1e-6


def test_linear_marginal_simple():
    """
    Test if vectorized vs loop linear marginal gives same results
    This is a simplified version focusing on the core computation
    """
    print("\n" + "=" * 60)
    print("TEST 3: compute_linear_marginal_output (simplified)")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    bsz, num_heads, head_dim = 2, 4, 32
    block_size = 4
    num_q_blocks, num_k_blocks = 3, 5
    q_len = num_q_blocks * block_size  # 12
    k_len = num_k_blocks * block_size  # 20
    
    # Random features (simulating phi(Q), phi(K), V)
    q_phi = torch.rand(bsz, num_heads, q_len, head_dim) + 0.1
    k_phi = torch.rand(bsz, num_heads, k_len, head_dim) + 0.1
    vh = torch.randn(bsz, num_heads, k_len, head_dim)
    
    # Random critical blocks mask
    critical_blocks = torch.rand(bsz, num_q_blocks, num_k_blocks) > 0.6
    
    # Global sums
    H_sum = torch.einsum("bhld,bhlm->bhdm", k_phi, vh)
    Z_sum = k_phi.sum(dim=2)
    
    eps = 1e-6
    
    # ============ Document 1: Vectorized ============
    k_phi_blocks = k_phi.view(bsz, num_heads, num_k_blocks, block_size, head_dim)
    vh_blocks = vh.view(bsz, num_heads, num_k_blocks, block_size, head_dim)
    q_phi_blocks = q_phi.view(bsz, num_heads, num_q_blocks, block_size, head_dim)
    
    s_blocks = torch.einsum("bhnld,bhnlm->bhndm", k_phi_blocks, vh_blocks)
    z_blocks = k_phi_blocks.sum(dim=3)
    
    crit_mask_s = critical_blocks[:, None, :, :, None, None].float()
    crit_mask_z = critical_blocks[:, None, :, :, None].float()
    
    s_blocks_expanded = s_blocks[:, :, None, :, :, :]
    z_blocks_expanded = z_blocks[:, :, None, :, :]
    
    s_crit_v1 = (s_blocks_expanded * crit_mask_s).sum(dim=3)
    z_crit_v1 = (z_blocks_expanded * crit_mask_z).sum(dim=3)
    
    H_sum_expanded = H_sum[:, :, None, :, :]
    Z_sum_expanded = Z_sum[:, :, None, :]
    
    s_qi_v1 = H_sum_expanded - s_crit_v1
    z_qi_v1 = Z_sum_expanded - z_crit_v1
    
    num_v1 = torch.einsum("bhnld,bhndm->bhnlm", q_phi_blocks, s_qi_v1)
    den_v1 = torch.einsum("bhnld,bhnd->bhnl", q_phi_blocks, z_qi_v1).unsqueeze(-1)
    ol_v1 = (num_v1 / (den_v1 + eps)).view(bsz, num_heads, q_len, head_dim)
    
    # ============ Document 2: Loop ============
    ol_v2 = torch.zeros_like(q_phi)
    
    for b in range(bsz):
        k_phi_b = k_phi[b]
        vh_b = vh[b]
        q_phi_b = q_phi[b]
        H_sum_b = H_sum[b]
        Z_sum_b = Z_sum[b]
        
        for qi in range(num_q_blocks):
            qs = qi * block_size
            qe = (qi + 1) * block_size
            
            s_crit = torch.zeros_like(H_sum_b)
            z_crit = torch.zeros_like(Z_sum_b)
            
            cb_row = critical_blocks[b, qi]
            crit_k_indices = torch.nonzero(cb_row, as_tuple=False).flatten().tolist()
            
            for kj in crit_k_indices:
                ks = kj * block_size
                ke = (kj + 1) * block_size
                
                k_blk = k_phi_b[:, ks:ke, :]
                v_blk = vh_b[:, ks:ke, :]
                
                s_j = torch.einsum("hld,hlm->hdm", k_blk, v_blk)
                z_j = k_blk.sum(dim=1)
                
                s_crit = s_crit + s_j
                z_crit = z_crit + z_j
            
            s_qi = H_sum_b - s_crit
            z_qi = Z_sum_b - z_crit
            
            q_blk = q_phi_b[:, qs:qe, :]
            num = torch.einsum("hld,hdm->hlm", q_blk, s_qi)
            den = torch.einsum("hld,hd->hl", q_blk, z_qi).unsqueeze(-1)
            
            ol_v2[b, :, qs:qe, :] = num / (den + eps)
    
    # Compare
    diff = (ol_v1 - ol_v2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    
    if max_diff > 1e-5:
        print("\n⚠️  SIGNIFICANT DIFFERENCE DETECTED!")
        # Find where the biggest differences are
        flat_idx = diff.argmax()
        idx = []
        for dim in reversed(diff.shape):
            idx.append(flat_idx % dim)
            flat_idx //= dim
        idx = tuple(reversed(idx))
        print(f"Largest diff at index {idx}")
        print(f"v1 value: {ol_v1[idx].item():.6f}")
        print(f"v2 value: {ol_v2[idx].item():.6f}")
        
        # Check intermediate values for that query block
        b, h, t, d = idx
        qi = t // block_size
        print(f"\nFor batch={b}, head={h}, query_block={qi}:")
        print(f"Critical key blocks: {critical_blocks[b, qi].tolist()}")
    
    return max_diff < 1e-5


def test_pool_tokens():
    """Test _pool_tokens differences"""
    print("\n" + "=" * 60)
    print("TEST 4: _pool_tokens")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    bsz, seq_len, embed_dim = 2, 14, 128
    block_size = 4
    
    x = torch.randn(bsz, seq_len, embed_dim)
    padding_mask = torch.zeros(bsz, seq_len, dtype=torch.bool)
    padding_mask[0, -3:] = True  # Some padding
    
    num_blocks = (seq_len + block_size - 1) // block_size
    pad_len = num_blocks * block_size - seq_len
    
    # Document 1
    x1 = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
    pm1 = torch.nn.functional.pad(padding_mask.float(), (0, pad_len), value=1.0).bool()
    x1 = x1.view(bsz, num_blocks, block_size, embed_dim)
    mask1 = pm1.view(bsz, num_blocks, block_size, 1)
    valid1 = (~mask1).float()
    sums1 = (x1 * valid1).sum(dim=2)
    counts1 = valid1.sum(dim=2)
    pooled1 = sums1 / counts1.clamp_min(1.0)
    block_mask1 = counts1.squeeze(-1) == 0
    
    # Document 2
    x2 = torch.cat([x, x.new_zeros(bsz, pad_len, embed_dim)], dim=1)
    pad_mask = padding_mask.new_ones(bsz, pad_len)
    pm2 = torch.cat([padding_mask, pad_mask], dim=1)
    x2 = x2.view(bsz, num_blocks, block_size, embed_dim)
    mask2 = pm2.view(bsz, num_blocks, block_size, 1)
    valid2 = (~mask2).float()
    sums2 = (x2 * valid2).sum(dim=2)
    counts2 = valid2.sum(dim=2)
    pooled2 = sums2 / counts2.clamp_min(1.0)
    block_mask2 = counts2.squeeze(-1) == 0
    
    pooled_diff = (pooled1 - pooled2).abs().max().item()
    mask_match = torch.equal(block_mask1, block_mask2)
    
    print(f"Pooled max diff: {pooled_diff:.2e}")
    print(f"Block masks match: {mask_match}")
    
    return pooled_diff < 1e-6 and mask_match


if __name__ == "__main__":
    results = []
    
    results.append(("expand_block_mask", test_expand_block_mask()))
    results.append(("mean_pool_blocks", test_mean_pool_blocks()))
    results.append(("pool_tokens", test_pool_tokens()))
    results.append(("linear_marginal", test_linear_marginal_simple()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
    
    if not all(p for _, p in results):
        print("\n⚠️  Some tests failed - these are likely causing the blurry outputs!")
        sys.exit(1)
    else:
        print("\n✓ All basic tests passed - issue may be more subtle")