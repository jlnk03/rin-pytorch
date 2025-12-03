"""
Test script to compare FlexMultiheadAttention vs nn.MultiheadAttention
to verify numerical equivalence.
"""

import torch
import torch.nn as nn
from rin_pytorch.modules.FlexMultiheadAttention import FlexMultiheadAttention

def copy_weights_nn_to_flex(nn_mha: nn.MultiheadAttention, flex_mha: FlexMultiheadAttention):
    """Copy weights from nn.MultiheadAttention to FlexMultiheadAttention."""
    with torch.no_grad():
        # Copy input projection weights (handle both same and different embed dims)
        if nn_mha.in_proj_weight is not None:
            flex_mha.in_proj_weight.copy_(nn_mha.in_proj_weight)
            flex_mha.in_proj_bias.copy_(nn_mha.in_proj_bias)
        else:
            # Separate Q, K, V projections
            flex_mha.q_proj_weight.copy_(nn_mha.q_proj_weight)
            flex_mha.k_proj_weight.copy_(nn_mha.k_proj_weight)
            flex_mha.v_proj_weight.copy_(nn_mha.v_proj_weight)
            # For separate projections, biases are still in in_proj_bias for FlexMHA
            # but nn.MHA doesn't have them when _qkv_same_embed_dim is False
            # So we need to handle this case
        
        # Copy output projection weights
        flex_mha.out_proj.weight.copy_(nn_mha.out_proj.weight)
        flex_mha.out_proj.bias.copy_(nn_mha.out_proj.bias)


def test_mha_equivalence(
    embed_dim: int = 256,
    num_heads: int = 8,
    seq_len: int = 64,
    batch_size: int = 2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype = torch.float32,
):
    """Test that FlexMultiheadAttention produces same output as nn.MultiheadAttention."""
    
    print(f"Testing with:")
    print(f"  embed_dim={embed_dim}, num_heads={num_heads}")
    print(f"  head_dim={embed_dim // num_heads}")
    print(f"  seq_len={seq_len}, batch_size={batch_size}")
    print(f"  device={device}, dtype={dtype}")
    print()
    
    # Create both modules
    nn_mha = nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        batch_first=True,
        bias=True,
    ).to(device, dtype)
    
    flex_mha = FlexMultiheadAttention(
        in_features=embed_dim,
        num_heads=num_heads,
        embed_dim=embed_dim,
    ).to(device, dtype)
    
    # Copy weights from nn.MHA to FlexMHA
    copy_weights_nn_to_flex(nn_mha, flex_mha)
    
    # Create random input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
    
    # Forward pass through both (self-attention)
    nn_mha.eval()
    flex_mha.eval()
    
    with torch.no_grad():
        # nn.MultiheadAttention
        nn_output, nn_attn_weights = nn_mha(x, x, x, need_weights=False)
        
        # FlexMultiheadAttention (without block_mask for fair comparison)
        flex_output, _ = flex_mha(x, x, x, block_mask=None, attn_mask=None)
    
    # Compare outputs
    abs_diff = (nn_output - flex_output).abs()
    rel_diff = abs_diff / (nn_output.abs() + 1e-8)
    
    print("=" * 60)
    print("OUTPUT COMPARISON")
    print("=" * 60)
    print(f"nn.MHA output shape:   {nn_output.shape}")
    print(f"FlexMHA output shape:  {flex_output.shape}")
    print()
    print(f"Absolute difference:")
    print(f"  Max:  {abs_diff.max().item():.2e}")
    print(f"  Mean: {abs_diff.mean().item():.2e}")
    print(f"  Std:  {abs_diff.std().item():.2e}")
    print()
    print(f"Relative difference:")
    print(f"  Max:  {rel_diff.max().item():.2e}")
    print(f"  Mean: {rel_diff.mean().item():.2e}")
    print()
    
    # Check if outputs are close
    is_close = torch.allclose(nn_output, flex_output, rtol=1e-4, atol=1e-5)
    print(f"torch.allclose(rtol=1e-4, atol=1e-5): {is_close}")
    
    is_very_close = torch.allclose(nn_output, flex_output, rtol=1e-5, atol=1e-6)
    print(f"torch.allclose(rtol=1e-5, atol=1e-6): {is_very_close}")
    
    # Sample output comparison
    print()
    print("Sample values (first batch, first 5 positions, first 5 dims):")
    print(f"nn.MHA:\n{nn_output[0, :5, :5]}")
    print(f"FlexMHA:\n{flex_output[0, :5, :5]}")
    
    return nn_output, flex_output, is_close


def test_weight_initialization():
    """Verify weight initialization matches nn.MultiheadAttention."""
    print()
    print("=" * 60)
    print("WEIGHT INITIALIZATION COMPARISON")
    print("=" * 60)
    
    embed_dim = 256
    num_heads = 8
    
    # Create multiple instances and check initialization statistics
    nn_in_proj_stds = []
    nn_out_proj_stds = []
    flex_in_proj_stds = []
    flex_out_proj_stds = []
    
    for _ in range(10):
        nn_mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        flex_mha = FlexMultiheadAttention(in_features=embed_dim, num_heads=num_heads)
        
        nn_in_proj_stds.append(nn_mha.in_proj_weight.std().item())
        nn_out_proj_stds.append(nn_mha.out_proj.weight.std().item())
        flex_in_proj_stds.append(flex_mha.in_proj_weight.std().item())
        flex_out_proj_stds.append(flex_mha.out_proj.weight.std().item())
    
    print(f"in_proj_weight std:")
    print(f"  nn.MHA:   {sum(nn_in_proj_stds)/len(nn_in_proj_stds):.6f}")
    print(f"  FlexMHA:  {sum(flex_in_proj_stds)/len(flex_in_proj_stds):.6f}")
    print()
    print(f"out_proj.weight std:")
    print(f"  nn.MHA:   {sum(nn_out_proj_stds)/len(nn_out_proj_stds):.6f}")
    print(f"  FlexMHA:  {sum(flex_out_proj_stds)/len(flex_out_proj_stds):.6f}")
    
    # Expected xavier_uniform_ std for (256, 256) is approximately sqrt(2 / (256 + 256)) = 0.0625
    expected_std = (2 / (embed_dim + embed_dim)) ** 0.5
    print(f"\nExpected xavier_uniform_ std: ~{expected_std:.6f}")


def test_scale_factor():
    """Verify the attention scale factor is correct."""
    print()
    print("=" * 60)
    print("SCALE FACTOR VERIFICATION")
    print("=" * 60)
    
    embed_dim = 256
    num_heads = 8
    head_dim = embed_dim // num_heads
    
    flex_mha = FlexMultiheadAttention(in_features=embed_dim, num_heads=num_heads)
    
    expected_scale = 1.0 / (head_dim ** 0.5)
    actual_scale = 1.0 / (flex_mha.head_dim ** 0.5)
    
    print(f"head_dim: {head_dim}")
    print(f"Expected scale (1/sqrt(head_dim)): {expected_scale:.6f}")
    print(f"Actual scale in FlexMHA:           {actual_scale:.6f}")
    print(f"Match: {abs(expected_scale - actual_scale) < 1e-10}")


def test_cross_attention_different_dims(
    latent_dim: int = 768,
    tape_dim: int = 256,
    num_heads: int = 16,
    latent_seq_len: int = 128,  # latent_slots
    tape_seq_len: int = 256,    # (64/4)^2 = 256 patches
    batch_size: int = 2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype = torch.float32,
):
    """
    Test cross-attention where query dim != key/value dim.
    This matches the config: latent_dim=768, tape_dim=256
    
    In cross-attention:
    - Query comes from latent (768 dim)
    - Key/Value come from tape (256 dim)
    """
    print()
    print("=" * 60)
    print("CROSS-ATTENTION TEST (latent_dim != tape_dim)")
    print("=" * 60)
    print(f"Testing with:")
    print(f"  latent_dim={latent_dim} (query)")
    print(f"  tape_dim={tape_dim} (key/value)")
    print(f"  num_heads={num_heads}")
    print(f"  head_dim={latent_dim // num_heads}")
    print(f"  latent_seq_len={latent_seq_len}, tape_seq_len={tape_seq_len}")
    print(f"  batch_size={batch_size}")
    print(f"  device={device}, dtype={dtype}")
    print()
    
    # For nn.MultiheadAttention with different kdim/vdim
    nn_mha = nn.MultiheadAttention(
        embed_dim=latent_dim,  # query dim
        num_heads=num_heads,
        kdim=tape_dim,         # key dim
        vdim=tape_dim,         # value dim
        batch_first=True,
        bias=True,
    ).to(device, dtype)
    
    # For FlexMultiheadAttention
    flex_mha = FlexMultiheadAttention(
        in_features=latent_dim,     # query input dim
        num_heads=num_heads,
        key_features=tape_dim,       # key input dim
        value_features=tape_dim,     # value input dim
        out_features=latent_dim,     # output dim
        embed_dim=latent_dim,        # internal embed dim
    ).to(device, dtype)
    
    # Copy weights - for different kdim/vdim, nn.MHA uses separate projections
    with torch.no_grad():
        # nn.MHA with kdim != embed_dim uses separate q_proj, k_proj, v_proj
        flex_mha.q_proj_weight.copy_(nn_mha.q_proj_weight)
        flex_mha.k_proj_weight.copy_(nn_mha.k_proj_weight)
        flex_mha.v_proj_weight.copy_(nn_mha.v_proj_weight)
        
        # Copy biases - nn.MHA stores them differently when using separate projections
        # nn.MHA has bias_k and bias_v as None by default, and in_proj_bias contains q bias
        if nn_mha.in_proj_bias is not None:
            # FlexMHA stores biases in in_proj_bias as [q_bias, k_bias, v_bias]
            q_bias = nn_mha.in_proj_bias[:latent_dim]
            # For k and v, the bias might be zeros or from bias_k/bias_v
            flex_mha.in_proj_bias[:latent_dim].copy_(q_bias)
            flex_mha.in_proj_bias[latent_dim:2*latent_dim].zero_()
            flex_mha.in_proj_bias[2*latent_dim:].zero_()
        
        flex_mha.out_proj.weight.copy_(nn_mha.out_proj.weight)
        flex_mha.out_proj.bias.copy_(nn_mha.out_proj.bias)
    
    # Create inputs
    torch.manual_seed(42)
    query = torch.randn(batch_size, latent_seq_len, latent_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, tape_seq_len, tape_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, tape_seq_len, tape_dim, device=device, dtype=dtype)
    
    # Forward pass
    nn_mha.eval()
    flex_mha.eval()
    
    with torch.no_grad():
        nn_output, _ = nn_mha(query, key, value, need_weights=False)
        flex_output, _ = flex_mha(query, key, value, block_mask=None, attn_mask=None)
    
    # Compare
    abs_diff = (nn_output - flex_output).abs()
    rel_diff = abs_diff / (nn_output.abs() + 1e-8)
    
    print("=" * 60)
    print("OUTPUT COMPARISON")
    print("=" * 60)
    print(f"nn.MHA output shape:   {nn_output.shape}")
    print(f"FlexMHA output shape:  {flex_output.shape}")
    print()
    print(f"Absolute difference:")
    print(f"  Max:  {abs_diff.max().item():.2e}")
    print(f"  Mean: {abs_diff.mean().item():.2e}")
    print(f"  Std:  {abs_diff.std().item():.2e}")
    print()
    print(f"Relative difference:")
    print(f"  Max:  {rel_diff.max().item():.2e}")
    print(f"  Mean: {rel_diff.mean().item():.2e}")
    print()
    
    is_close = torch.allclose(nn_output, flex_output, rtol=1e-4, atol=1e-5)
    print(f"torch.allclose(rtol=1e-4, atol=1e-5): {is_close}")
    
    is_very_close = torch.allclose(nn_output, flex_output, rtol=1e-5, atol=1e-6)
    print(f"torch.allclose(rtol=1e-5, atol=1e-6): {is_very_close}")
    
    print()
    print("Sample values (first batch, first 5 positions, first 5 dims):")
    print(f"nn.MHA:\n{nn_output[0, :5, :5]}")
    print(f"FlexMHA:\n{flex_output[0, :5, :5]}")
    
    return nn_output, flex_output, is_close


if __name__ == "__main__":
    print("=" * 60)
    print("FlexMultiheadAttention vs nn.MultiheadAttention Comparison")
    print("=" * 60)
    print()
    
    # Test weight initialization
    test_weight_initialization()
    
    # Test scale factor
    test_scale_factor()
    
    # Test numerical equivalence (self-attention, same dims)
    print()
    print("=" * 60)
    print("SELF-ATTENTION TEST (same embed_dim)")
    print("=" * 60)
    test_mha_equivalence(dtype=torch.float32)
    
    # Test cross-attention with different dimensions (like in the config)
    # latent_dim=768, tape_dim=256, num_heads=16
    _, _, cross_attn_passed = test_cross_attention_different_dims(
        latent_dim=768,
        tape_dim=256,
        num_heads=16,
        latent_seq_len=128,  # latent_slots from config
        tape_seq_len=256,    # (64/4)^2 = 256 patches
    )
    
    # Test with float16 if CUDA available
    if torch.cuda.is_available():
        print()
        print("=" * 60)
        print("NUMERICAL EQUIVALENCE TEST (float16 on CUDA)")
        print("=" * 60)
        test_mha_equivalence(dtype=torch.float16)
        
        print()
        print("=" * 60)
        print("NUMERICAL EQUIVALENCE TEST (bfloat16 on CUDA)")
        print("=" * 60)
        test_mha_equivalence(dtype=torch.bfloat16)
    
    # Test with different configurations (self-attention)
    print()
    print("=" * 60)
    print("TESTING VARIOUS SELF-ATTENTION CONFIGURATIONS")
    print("=" * 60)
    
    configs = [
        {"embed_dim": 64, "num_heads": 4},
        {"embed_dim": 128, "num_heads": 8},
        {"embed_dim": 512, "num_heads": 8},
        {"embed_dim": 512, "num_heads": 16},
    ]
    
    all_passed = True
    for config in configs:
        print(f"\nConfig: {config}")
        _, _, is_close = test_mha_equivalence(**config, seq_len=32, batch_size=1)
        if not is_close:
            all_passed = False
            print("  ❌ FAILED")
        else:
            print("  ✓ PASSED")
    
    # Test cross-attention configurations matching the config file
    print()
    print("=" * 60)
    print("TESTING CROSS-ATTENTION CONFIGURATIONS (from 64.yaml)")
    print("=" * 60)
    
    cross_configs = [
        # From config: latent_dim=768, tape_dim=256, rw_num_heads=16
        {"latent_dim": 768, "tape_dim": 256, "num_heads": 16, "latent_seq_len": 128, "tape_seq_len": 256},
        # Different head counts
        {"latent_dim": 768, "tape_dim": 256, "num_heads": 8, "latent_seq_len": 128, "tape_seq_len": 256},
        # Smaller dims
        {"latent_dim": 256, "tape_dim": 128, "num_heads": 8, "latent_seq_len": 64, "tape_seq_len": 128},
    ]
    
    for config in cross_configs:
        print(f"\nCross-attention config: latent_dim={config['latent_dim']}, tape_dim={config['tape_dim']}, heads={config['num_heads']}")
        _, _, is_close = test_cross_attention_different_dims(**config)
        if not is_close:
            all_passed = False
            print("  ❌ FAILED")
        else:
            print("  ✓ PASSED")
    
    if not cross_attn_passed:
        all_passed = False
    
    print()
    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ❌")
    print("=" * 60)

