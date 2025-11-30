import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class SparseMultiheadAttention(nn.Module):
    """
    Sparse attention that:
    1. Pools tokens into blocks and classifies importance via pooled attention
    2. Drops non-critical tokens, runs full attention only on critical tokens
    3. Merges results back: critical tokens get attention output, others get zeros
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        dropout: float = 0.0,
        batch_first: bool = True,
        block_size: int = 4,
        critical_ratio: float = 0.25,
        critical_k: Optional[int] = None,
        head_aggregation: str = "mean",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.block_size = block_size
        self.critical_ratio = critical_ratio
        self.critical_k = critical_k
        self.head_aggregation = head_aggregation
        self._batch_first = batch_first
        
        if self.critical_ratio is not None:
            assert 0.0 < self.critical_ratio <= 1.0
        assert self.head_aggregation in {"mean", "max"}

        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            batch_first=True,  # Always use batch_first internally
            device=device,
            dtype=dtype,
        )

        self._last_critical_mask: Optional[torch.Tensor] = None

    @property
    def batch_first(self) -> bool:
        return self._batch_first

    def _pool_1d(self, x: torch.Tensor) -> torch.Tensor:
        """Mean-pool sequence into non-overlapping blocks. x: (B, L, E)"""
        bsz, seq_len, embed = x.shape
        block = self.block_size
        num_blocks = (seq_len + block - 1) // block
        pad_len = num_blocks * block - seq_len
        if pad_len > 0:
            x = torch.cat([x, x.new_zeros(bsz, pad_len, embed)], dim=1)
        x = x.view(bsz, num_blocks, block, embed)
        return x.mean(dim=2)

    def _classify_important_tokens(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Classify which tokens are important using pooled attention.
        Returns: token_important (B, L) - True for important tokens
        """
        bsz, seq_len, _ = query.shape
        block = self.block_size
        num_blocks = (seq_len + block - 1) // block

        # Pool and compute attention scores
        q_pool = self._pool_1d(query)
        k_pool = self._pool_1d(key)

        # Compute attention weights on pooled tokens
        with torch.no_grad():
            _, attn_w = self.mha(
                q_pool, k_pool, k_pool,
                need_weights=True,
                average_attn_weights=False,
            )  # (B, H, num_blocks, num_blocks)

        # Aggregate across heads
        if self.head_aggregation == "mean":
            scores = attn_w.mean(dim=1)  # (B, num_blocks, num_blocks)
        else:
            scores = attn_w.max(dim=1).values

        # Aggregate across query blocks: importance = how much attention each key block receives
        block_importance = scores.sum(dim=1)  # (B, num_blocks)

        # Select top-k important blocks
        if self.critical_k is not None:
            k = max(1, min(num_blocks, int(self.critical_k)))
        else:
            k = max(1, int(math.ceil(self.critical_ratio * num_blocks)))

        topk_indices = block_importance.topk(k, dim=-1).indices  # (B, k)
        block_important = torch.zeros(bsz, num_blocks, dtype=torch.bool, device=query.device)
        block_important.scatter_(1, topk_indices, True)

        # Expand to token level
        block_idx = torch.div(
            torch.arange(seq_len, device=query.device), block, rounding_mode="floor"
        ).clamp_max(num_blocks - 1)
        token_important = block_important[:, block_idx]  # (B, seq_len)

        # Also exclude padded tokens if key_padding_mask provided
        if key_padding_mask is not None:
            token_important = token_important & ~key_padding_mask

        self._last_critical_mask = block_important
        return token_important

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sparse attention: drops non-critical tokens, runs attention, merges back.
        """
        # Convert to batch_first internally
        if not self._batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        bsz, seq_len, embed = query.shape

        # 1) Classify which tokens are important
        token_important = self._classify_important_tokens(query, key, key_padding_mask)

        # 2) Find max number of important tokens across batch for padding
        num_important = token_important.sum(dim=1)  # (B,)
        max_important = int(num_important.max().item())

        if max_important == 0:
            # Edge case: no important tokens, return zeros
            out = query.new_zeros(bsz, seq_len, embed)
            if not self._batch_first:
                out = out.transpose(0, 1)
            return out, None

        # 3) Gather important tokens into dense tensors, padded to max_important
        q_gathered = query.new_zeros(bsz, max_important, embed)
        k_gathered = key.new_zeros(bsz, max_important, embed)
        v_gathered = value.new_zeros(bsz, max_important, embed)
        gathered_padding = torch.ones(bsz, max_important, dtype=torch.bool, device=query.device)
        
        # Store indices for scattering back
        scatter_indices = []

        for b in range(bsz):
            idx = token_important[b].nonzero(as_tuple=True)[0]
            n = idx.shape[0]
            q_gathered[b, :n] = query[b, idx]
            k_gathered[b, :n] = key[b, idx]
            v_gathered[b, :n] = value[b, idx]
            gathered_padding[b, :n] = False
            scatter_indices.append(idx)

        # 4) Run full attention on gathered tokens
        out_gathered, attn = self.mha(
            query=q_gathered,
            key=k_gathered,
            value=v_gathered,
            key_padding_mask=gathered_padding,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
        )

        # 5) Scatter results back to original positions
        out = query.new_zeros(bsz, seq_len, embed)
        for b in range(bsz):
            idx = scatter_indices[b]
            n = idx.shape[0]
            out[b, idx] = out_gathered[b, :n]

        # Convert back if needed
        if not self._batch_first:
            out = out.transpose(0, 1)

        return out, attn if need_weights else (out, None)

    def get_last_critical_mask(self) -> Optional[torch.Tensor]:
        """Returns the last computed critical block mask (B, num_blocks)."""
        return self._last_critical_mask