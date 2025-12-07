"""
Test batch size effects on numerical precision between vectorized and loop implementations
"""

import torch
import torch.nn.functional as F


def compute_linear_marginal_vectorized(q_phi, k_phi, vh, critical_blocks, block_size, eps=1e-6):
    """Doc 1 style - vectorized"""
    bsz, num_heads, q_len, head_dim = q_phi.shape
    _, _, k_len, _ = k_phi.shape
    num_q_blocks = (q_len + block_size - 1) // block_size
    num_k_blocks = (k_len + block_size - 1) // block_size
    
    H_sum = torch.einsum("bhld,bhlm->bhdm", k_phi, vh)
    Z_sum = k_phi.sum(dim=2)
    
    k_phi_blocks = k_phi.view(bsz, num_heads, num_k_blocks, block_size, head_dim)
    vh_blocks = vh.view(bsz, num_heads, num_k_blocks, block_size, head_dim)
    q_phi_blocks = q_phi.view(bsz, num_heads, num_q_blocks, block_size, head_dim)
    
    s_blocks = torch.einsum("bhnld,bhnlm->bhndm", k_phi_blocks, vh_blocks)
    z_blocks = k_phi_blocks.sum(dim=3)
    
    crit_mask_s = critical_blocks[:, None, :, :, None, None].float()
    crit_mask_z = critical_blocks[:, None, :, :, None].float()
    
    s_blocks_expanded = s_blocks[:, :, None, :, :, :]
    z_blocks_expanded = z_blocks[:, :, None, :, :]
    
    s_crit = (s_blocks_expanded * crit_mask_s).sum(dim=3)
    z_crit = (z_blocks_expanded * crit_mask_z).sum(dim=3)
    
    s_qi = H_sum[:, :, None, :, :] - s_crit
    z_qi = Z_sum[:, :, None, :] - z_crit
    
    num = torch.einsum("bhnld,bhndm->bhnlm", q_phi_blocks, s_qi)
    den = torch.einsum("bhnld,bhnd->bhnl", q_phi_blocks, z_qi).unsqueeze(-1)
    
    return (num / (den + eps)).view(bsz, num_heads, q_len, head_dim)


def compute_linear_marginal_loop(q_phi, k_phi, vh, critical_blocks, block_size, eps=1e-6):
    """Doc 2 style - loop"""
    bsz, num_heads, q_len, head_dim = q_phi.shape
    _, _, k_len, _ = k_phi.shape
    num_q_blocks = (q_len + block_size - 1) // block_size
    num_k_blocks = (k_len + block_size - 1) // block_size
    
    H_sum = torch.einsum("bhld,bhlm->bhdm", k_phi, vh)
    Z_sum = k_phi.sum(dim=2)
    
    out = torch.zeros_like(q_phi)
    
    for b in range(bsz):
        for qi in range(num_q_blocks):
            qs, qe = qi * block_size, min((qi + 1) * block_size, q_len)
            
            s_crit = torch.zeros_like(H_sum[b])
            z_crit = torch.zeros_like(Z_sum[b])
            
            for kj in critical_blocks[b, qi].nonzero(as_tuple=False).flatten().tolist():
                ks, ke = kj * block_size, min((kj + 1) * block_size, k_len)
                k_blk = k_phi[b, :, ks:ke]
                v_blk = vh[b, :, ks:ke]
                s_crit = s_crit + torch.einsum("hld,hlm->hdm", k_blk, v_blk)
                z_crit = z_crit + k_blk.sum(dim=1)
            
            s_qi = H_sum[b] - s_crit
            z_qi = Z_sum[b] - z_crit
            
            q_blk = q_phi[b, :, qs:qe]
            num = torch.einsum("hld,hdm->hlm", q_blk, s_qi)
            den = torch.einsum("hld,hd->hl", q_blk, z_qi).unsqueeze(-1)
            out[b, :, qs:qe] = num / (den + eps)
    
    return out


def test_batch_size_effects():
    print("=" * 70)
    print("BATCH SIZE EFFECTS ON NUMERICAL PRECISION")
    print("=" * 70)
    
    num_heads, head_dim = 8, 64
    block_size = 4
    num_q_blocks, num_k_blocks = 16, 16  # Realistic sizes
    q_len = num_q_blocks * block_size
    k_len = num_k_blocks * block_size
    
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    
    print(f"\nConfig: heads={num_heads}, head_dim={head_dim}, q_len={q_len}, k_len={k_len}")
    print(f"Block size: {block_size}, Q blocks: {num_q_blocks}, K blocks: {num_k_blocks}")
    print()
    
    results = []
    
    for bsz in batch_sizes:
        torch.manual_seed(42)  # Same seed for fair comparison
        
        # Generate test data
        q_phi = torch.rand(bsz, num_heads, q_len, head_dim) + 0.1
        k_phi = torch.rand(bsz, num_heads, k_len, head_dim) + 0.1
        vh = torch.randn(bsz, num_heads, k_len, head_dim)
        critical_blocks = torch.rand(bsz, num_q_blocks, num_k_blocks) > 0.7  # 30% critical
        
        # Forward pass
        out_vec = compute_linear_marginal_vectorized(q_phi, k_phi, vh, critical_blocks, block_size)
        out_loop = compute_linear_marginal_loop(q_phi, k_phi, vh, critical_blocks, block_size)
        
        fwd_diff = (out_vec - out_loop).abs()
        max_diff = fwd_diff.max().item()
        mean_diff = fwd_diff.mean().item()
        
        # Gradient check
        q_phi_v = q_phi.clone().requires_grad_(True)
        k_phi_v = k_phi.clone().requires_grad_(True)
        vh_v = vh.clone().requires_grad_(True)
        
        q_phi_l = q_phi.clone().requires_grad_(True)
        k_phi_l = k_phi.clone().requires_grad_(True)
        vh_l = vh.clone().requires_grad_(True)
        
        loss_v = compute_linear_marginal_vectorized(q_phi_v, k_phi_v, vh_v, critical_blocks, block_size).sum()
        loss_v.backward()
        
        loss_l = compute_linear_marginal_loop(q_phi_l, k_phi_l, vh_l, critical_blocks, block_size).sum()
        loss_l.backward()
        
        grad_q_diff = (q_phi_v.grad - q_phi_l.grad).abs().max().item()
        grad_k_diff = (k_phi_v.grad - k_phi_l.grad).abs().max().item()
        grad_v_diff = (vh_v.grad - vh_l.grad).abs().max().item()
        max_grad_diff = max(grad_q_diff, grad_k_diff, grad_v_diff)
        
        # Memory for vectorized intermediate
        # Shape: (B, H, num_q, num_k, D, D) for s_blocks_expanded * mask
        intermediate_size = bsz * num_heads * num_q_blocks * num_k_blocks * head_dim * head_dim
        intermediate_mb = intermediate_size * 4 / (1024 * 1024)  # float32
        
        results.append({
            'bsz': bsz,
            'fwd_max': max_diff,
            'fwd_mean': mean_diff,
            'grad_max': max_grad_diff,
            'intermediate_mb': intermediate_mb,
        })
        
        status = "✓" if max_diff < 1e-5 and max_grad_diff < 1e-4 else "⚠️"
        print(f"{status} Batch {bsz:3d}: fwd_max={max_diff:.2e}, fwd_mean={mean_diff:.2e}, "
              f"grad_max={max_grad_diff:.2e}, intermediate={intermediate_mb:.1f}MB")
    
    # Check for scaling issues
    print("\n" + "-" * 70)
    print("ANALYSIS")
    print("-" * 70)
    
    # Does error grow with batch size?
    fwd_errors = [r['fwd_max'] for r in results]
    grad_errors = [r['grad_max'] for r in results]
    
    if fwd_errors[-1] > fwd_errors[0] * 10:
        print("⚠️  Forward error grows significantly with batch size!")
    else:
        print("✓ Forward error stable across batch sizes")
    
    if grad_errors[-1] > grad_errors[0] * 10:
        print("⚠️  Gradient error grows significantly with batch size!")
    else:
        print("✓ Gradient error stable across batch sizes")
    
    # Memory scaling
    print(f"\nMemory scaling: {results[0]['intermediate_mb']:.1f}MB (B=1) -> "
          f"{results[-1]['intermediate_mb']:.1f}MB (B={batch_sizes[-1]})")
    
    return results


def test_accumulation_effects():
    """
    Test if the vectorized sum accumulates error differently
    """
    print("\n" + "=" * 70)
    print("ACCUMULATION EFFECTS")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    bsz, num_heads, head_dim = 8, 8, 64
    block_size = 4
    
    # Test with increasing number of blocks
    block_counts = [4, 8, 16, 32, 64]
    
    print(f"\nConfig: bsz={bsz}, heads={num_heads}, head_dim={head_dim}, block_size={block_size}")
    print()
    
    for num_k_blocks in block_counts:
        num_q_blocks = num_k_blocks
        q_len = num_q_blocks * block_size
        k_len = num_k_blocks * block_size
        
        q_phi = torch.rand(bsz, num_heads, q_len, head_dim) + 0.1
        k_phi = torch.rand(bsz, num_heads, k_len, head_dim) + 0.1
        vh = torch.randn(bsz, num_heads, k_len, head_dim)
        
        # Dense critical blocks (worst case for accumulation)
        critical_blocks = torch.rand(bsz, num_q_blocks, num_k_blocks) > 0.3  # 70% critical
        
        out_vec = compute_linear_marginal_vectorized(q_phi, k_phi, vh, critical_blocks, block_size)
        out_loop = compute_linear_marginal_loop(q_phi, k_phi, vh, critical_blocks, block_size)
        
        diff = (out_vec - out_loop).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        num_critical = critical_blocks.float().sum().item() / bsz  # Avg per sample
        
        status = "✓" if max_diff < 1e-5 else "⚠️"
        print(f"{status} {num_k_blocks:3d} blocks ({num_critical:.0f} critical avg): "
              f"max={max_diff:.2e}, mean={mean_diff:.2e}")


def test_dtype_effects():
    """
    Test if float16/bfloat16 makes things worse
    """
    print("\n" + "=" * 70)
    print("DTYPE EFFECTS")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    bsz, num_heads, head_dim = 8, 8, 64
    block_size = 4
    num_q_blocks, num_k_blocks = 16, 16
    q_len = num_q_blocks * block_size
    k_len = num_k_blocks * block_size
    
    dtypes = [torch.float32]
    
    # Check if float16/bfloat16 available
    if torch.cuda.is_available():
        dtypes.extend([torch.float16, torch.bfloat16])
    else:
        print("(CUDA not available, testing float32 only)")
    
    for dtype in dtypes:
        device = 'cuda' if dtype != torch.float32 and torch.cuda.is_available() else 'cpu'
        
        q_phi = (torch.rand(bsz, num_heads, q_len, head_dim) + 0.1).to(dtype).to(device)
        k_phi = (torch.rand(bsz, num_heads, k_len, head_dim) + 0.1).to(dtype).to(device)
        vh = torch.randn(bsz, num_heads, k_len, head_dim).to(dtype).to(device)
        critical_blocks = (torch.rand(bsz, num_q_blocks, num_k_blocks) > 0.7).to(device)
        
        out_vec = compute_linear_marginal_vectorized(q_phi, k_phi, vh, critical_blocks, block_size)
        out_loop = compute_linear_marginal_loop(q_phi, k_phi, vh, critical_blocks, block_size)
        
        diff = (out_vec - out_loop).abs()
        max_diff = diff.max().item()
        
        print(f"{dtype}: max_diff={max_diff:.2e}")


def test_specific_pattern():
    """
    Test specific critical block patterns that might cause issues
    """
    print("\n" + "=" * 70)
    print("SPECIFIC PATTERNS")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    bsz, num_heads, head_dim = 4, 8, 64
    block_size = 4
    num_q_blocks, num_k_blocks = 8, 8
    q_len = num_q_blocks * block_size
    k_len = num_k_blocks * block_size
    
    q_phi = torch.rand(bsz, num_heads, q_len, head_dim) + 0.1
    k_phi = torch.rand(bsz, num_heads, k_len, head_dim) + 0.1
    vh = torch.randn(bsz, num_heads, k_len, head_dim)
    
    patterns = {
        "All zeros": torch.zeros(bsz, num_q_blocks, num_k_blocks, dtype=torch.bool),
        "All ones": torch.ones(bsz, num_q_blocks, num_k_blocks, dtype=torch.bool),
        "Diagonal": torch.zeros(bsz, num_q_blocks, num_k_blocks, dtype=torch.bool),
        "Checkerboard": torch.zeros(bsz, num_q_blocks, num_k_blocks, dtype=torch.bool),
        "First row only": torch.zeros(bsz, num_q_blocks, num_k_blocks, dtype=torch.bool),
        "Last column only": torch.zeros(bsz, num_q_blocks, num_k_blocks, dtype=torch.bool),
        "Random sparse (10%)": torch.rand(bsz, num_q_blocks, num_k_blocks) > 0.9,
        "Random dense (90%)": torch.rand(bsz, num_q_blocks, num_k_blocks) > 0.1,
    }
    
    # Set up special patterns
    for b in range(bsz):
        for i in range(min(num_q_blocks, num_k_blocks)):
            patterns["Diagonal"][b, i, i] = True
    
    for b in range(bsz):
        for i in range(num_q_blocks):
            for j in range(num_k_blocks):
                if (i + j) % 2 == 0:
                    patterns["Checkerboard"][b, i, j] = True
    
    patterns["First row only"][:, 0, :] = True
    patterns["Last column only"][:, :, -1] = True
    
    for name, critical_blocks in patterns.items():
        out_vec = compute_linear_marginal_vectorized(q_phi, k_phi, vh, critical_blocks, block_size)
        out_loop = compute_linear_marginal_loop(q_phi, k_phi, vh, critical_blocks, block_size)
        
        diff = (out_vec - out_loop).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        status = "✓" if max_diff < 1e-5 else "⚠️"
        print(f"{status} {name:25s}: max={max_diff:.2e}, mean={mean_diff:.2e}")


if __name__ == "__main__":
    test_batch_size_effects()
    test_accumulation_effects()
    test_dtype_effects()
    test_specific_pattern()
    
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("""
If all tests pass with small differences (< 1e-5), the issue is likely NOT
in the core computation but in:

1. Training dynamics (learning rate, batch size interaction with optimizer)
2. Model architecture differences elsewhere
3. Random seed / initialization differences
4. Data loading order with different batch sizes
5. Gradient clipping or normalization interacting differently

Try:
- Use the SAME random seed for both implementations
- Use the SAME batch size  
- Use the SAME learning rate schedule
- Compare loss curves, not just final outputs
""")