# Hierarchical Sparse Attention

This document explains the `EfficientSparseAttention` module and how it implements true hierarchical coarse-to-fine refinement.

## Overview

The `EfficientSparseAttention` class in `rin_pytorch/modules/SparseAttentionXformers.py` implements hierarchical sparse attention with multiplicative sparsity. Unlike naive approaches that simply AND together masks at the same granularity (resulting in additive sparsity), this implementation properly processes attention in a coarse-to-fine manner.

## Key Concept: Multiplicative vs Additive Sparsity

### ❌ Broken Approach (Additive Sparsity)
- All levels operate on the same block granularity (finest level)
- Masks are computed independently and AND-ed together
- Effective sparsity = 1 - min(ratio₁, ratio₂, ..., ratioₙ)
- Example: With ratios 0.5 and 0.25 → only 75% sparse

### ✅ Correct Approach (Multiplicative Sparsity)
- Levels are processed from coarsest to finest
- Each level refines the previous level's selection
- Effective sparsity = 1 - (ratio₁ × ratio₂ × ... × ratioₙ)
- Example: With ratios 0.5 and 0.25 → 87.5% sparse

## Usage

### Basic Usage

```python
from rin_pytorch.modules import EfficientSparseAttention

# Create attention module with 2-level hierarchy
attn = EfficientSparseAttention(
    embed_dim=512,
    num_heads=8,
    hierarchy=[
        {'block_size': 4, 'critical_ratio': 0.5},   # Coarse: keep 50% of 4x4 blocks
        {'block_size': 2, 'critical_ratio': 0.25}   # Fine: keep 25% of 2x2 blocks
    ]
)

# Forward pass
query = torch.randn(batch_size, seq_len, embed_dim)
key = torch.randn(batch_size, seq_len, embed_dim)
value = torch.randn(batch_size, seq_len, embed_dim)

output, attn_weights = attn(query, key, value, need_weights=True)
```

### Configuration

The `hierarchy` parameter accepts a list of dictionaries or tuples, where each item specifies:
- `block_size`: Size of blocks at this hierarchy level
- `critical_ratio`: Fraction of blocks to keep at this level (0.0 to 1.0)

**Important**: The hierarchy levels are automatically sorted by descending `block_size` (coarsest first), so you don't need to worry about ordering.

### Examples

#### Single-level (Baseline)
```python
hierarchy = [
    {'block_size': 4, 'critical_ratio': 0.5}
]
# Keeps 50% of blocks → 50% sparse
```

#### Two-level (High Sparsity)
```python
hierarchy = [
    {'block_size': 4, 'critical_ratio': 0.5},
    {'block_size': 2, 'critical_ratio': 0.25}
]
# Keeps 0.5 × 0.25 = 12.5% of blocks → 87.5% sparse
```

#### Three-level (Very High Sparsity)
```python
hierarchy = [
    {'block_size': 8, 'critical_ratio': 0.5},
    {'block_size': 4, 'critical_ratio': 0.4},
    {'block_size': 2, 'critical_ratio': 0.25}
]
# Keeps 0.5 × 0.4 × 0.25 = 5% of blocks → 95% sparse
```

## How It Works

### Algorithm Overview

1. **Initialization**: All fine-level blocks start as candidates
2. **Coarsest Level**: 
   - Pool Q/K into coarse blocks
   - Compute attention scores at coarse granularity
   - Select top-k coarse blocks
3. **Subsequent Levels** (for each finer level):
   - Pool Q/K at this level's granularity
   - Mask out blocks already eliminated by coarser levels
   - Compute scores only for surviving blocks
   - Select top-k among the survivors
   - Upsample and AND with the accumulating mask
4. **Final Attention**: Use the finest-level mask for sparse attention

### Example: 2-Level Hierarchy

Given a sequence of 64 tokens with block sizes [4, 2]:

**Level 1 (coarse, block_size=4):**
- 64 tokens → 16 blocks of 4 tokens each
- Grid: 16×16 = 256 block pairs
- Keep 50% → 128 block pairs survive

**Level 2 (fine, block_size=2):**
- 64 tokens → 32 blocks of 2 tokens each
- Within the 128 surviving coarse blocks, subdivide to 2×2 fine blocks
- This creates 128 × 4 = 512 fine block pairs
- Keep 25% of these → 128 fine block pairs survive

**Result:**
- Started with 32×32 = 1024 fine block pairs
- Ended with 128 fine block pairs
- 128/1024 = 12.5% kept → **87.5% sparse**

## API Reference

### EfficientSparseAttention

```python
class EfficientSparseAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hierarchy: List[Dict[str, Any]],
        dropout: float = 0.0,
        backend: str = 'xformers'
    )
```

**Parameters:**
- `embed_dim`: Dimension of embeddings
- `num_heads`: Number of attention heads
- `hierarchy`: List of level configurations (see above)
- `dropout`: Dropout probability (default: 0.0)
- `backend`: Attention backend ('xformers', 'truly_sparse', or 'flex')

**Methods:**

#### forward()
```python
def forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor] = None,
    need_weights: bool = False,
    attn_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]
```

Performs hierarchical sparse attention.

#### compute_linear_marginal_output()
```python
def compute_linear_marginal_output(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    critical_mask: Optional[torch.Tensor] = None
) -> torch.Tensor
```

Computes attention output with linear complexity using only critical blocks.

## Performance Considerations

### Memory Savings
- 2-level (87.5% sparse): ~8× memory reduction
- 3-level (95% sparse): ~20× memory reduction

### Computational Savings
- Attention computation scales with the number of critical blocks
- Hierarchical selection overhead is minimal compared to full attention

### Trade-offs
- More hierarchy levels → higher sparsity → more memory savings
- Too aggressive sparsity may impact quality
- Recommended: Start with 2 levels, tune critical_ratio values

## Testing

Run the test suite to verify the implementation:

```bash
python test_sparse_attention.py
```

This will run comprehensive tests demonstrating:
1. Single-level hierarchy (baseline)
2. Multi-level hierarchy with multiplicative sparsity
3. Automatic sorting of hierarchy levels
4. Linear marginal output computation

All tests should pass, confirming correct hierarchical refinement.

## Integration with RIN

To use sparse attention in a RIN transformer layer:

```python
from rin_pytorch.modules import EfficientSparseAttention

class SparseTransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, hierarchy):
        super().__init__()
        self.sparse_attn = EfficientSparseAttention(
            embed_dim=dim,
            num_heads=num_heads,
            hierarchy=hierarchy
        )
        self.mlp = MLP(...)
    
    def forward(self, x):
        attn_out, _ = self.sparse_attn(x, x, x)
        x = x + attn_out
        x = self.mlp(x)
        return x
```

## References

- Original RIN paper: https://arxiv.org/abs/2212.11972
- Sparse Attention Survey: https://arxiv.org/abs/2009.14794
