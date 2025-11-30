import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class SparseMultiheadAttention(nn.Module):
    """
    Sparse cross-attention that always pools the LARGER sequence:
    - If key > query: pool keys, all queries attend to selected keys
    - If query > key: pool queries, selected queries attend to all keys, scatter back
    - If both small: full attention (no sparsity)
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
        min_sparse_seq: int = 256,  # Skip sparsity if larger sequence <= this
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
        self.min_sparse_seq = min_sparse_seq
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
            batch_first=True,
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

    def _select_important_blocks(
        self,
        to_pool: torch.Tensor,
        other: torch.Tensor,
        pool_is_key: bool,
    ) -> torch.Tensor:
        """
        Select important blocks from the larger sequence.
        Returns: important mask (B, pool_len) - True for important tokens
        """
        bsz, pool_len, _ = to_pool.shape
        _, other_len, _ = other.shape
        block = self.block_size
        num_blocks = (pool_len + block - 1) // block

        # Pool the larger sequence
        pooled = self._pool_1d(to_pool)

        # print(f"[SPARSE] Pooled {'key' if pool_is_key else 'query'}: {to_pool.shape} -> {pooled.shape}")

        # Compute attention scores to determine importance
        with torch.no_grad():
            if pool_is_key:
                # other=query, pooled=key -> which key blocks are important
                _, attn_w = self.mha(other, pooled, pooled, need_weights=True, average_attn_weights=False)
                # attn_w: (B, H, other_len, num_blocks)
            else:
                # pooled=query, other=key -> which query blocks are important
                _, attn_w = self.mha(pooled, other, other, need_weights=True, average_attn_weights=False)
                # attn_w: (B, H, num_blocks, other_len)

        # Aggregate across heads
        if self.head_aggregation == "mean":
            scores = attn_w.mean(dim=1)
        else:
            scores = attn_w.max(dim=1).values

        # Get block importance
        if pool_is_key:
            # Sum across query positions -> importance of each key block
            block_importance = scores.sum(dim=1)  # (B, num_blocks)
        else:
            # Sum across key positions -> importance of each query block
            block_importance = scores.sum(dim=2)  # (B, num_blocks)

        # Select top-k blocks
        if self.critical_k is not None:
            k = max(1, min(num_blocks, int(self.critical_k)))
        else:
            k = max(1, int(math.ceil(self.critical_ratio * num_blocks)))

        # print(f"[SPARSE] Selecting top k={k} blocks out of {num_blocks} (ratio={self.critical_ratio})")

        topk_indices = block_importance.topk(k, dim=-1).indices
        block_important = torch.zeros(bsz, num_blocks, dtype=torch.bool, device=to_pool.device)
        block_important.scatter_(1, topk_indices, True)

        # Expand to token level
        block_idx = torch.div(
            torch.arange(pool_len, device=to_pool.device), block, rounding_mode="floor"
        ).clamp_max(num_blocks - 1)
        token_important = block_important[:, block_idx]

        # print(f"[SPARSE] Selected {token_important.sum(dim=1).tolist()} tokens per batch")

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
        Sparse cross-attention: always pools the larger sequence.
        """
        if not self._batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        bsz, q_len, q_embed = query.shape
        _, k_len, k_embed = key.shape

        larger_len = max(q_len, k_len)

        # If both sequences are small, use full attention
        if larger_len <= self.min_sparse_seq:
            # print(f"[SPARSE] FULL ATTN - max({q_len}, {k_len})={larger_len} <= {self.min_sparse_seq}")
            out, attn = self.mha(query, key, value, key_padding_mask=key_padding_mask,
                                  need_weights=need_weights, average_attn_weights=average_attn_weights)
            if not self._batch_first:
                out = out.transpose(0, 1)
            return out, attn if need_weights else (out, None)

        # print("=" * 60)
        # print(f"[SPARSE] query: {query.shape}, key: {key.shape}")

        if k_len >= q_len:
            # KEY is larger -> pool keys, all queries attend to selected keys
            # print(f"[SPARSE] KEY is larger ({k_len} >= {q_len}) -> pooling keys")
            
            key_important = self._select_important_blocks(key, query, pool_is_key=True)
            max_important = int(key_important.sum(dim=1).max().item())

            if max_important == 0:
                out = query.new_zeros(bsz, q_len, q_embed)
            else:
                # Gather selected keys/values
                k_selected = key.new_zeros(bsz, max_important, k_embed)
                v_selected = value.new_zeros(bsz, max_important, value.shape[-1])
                kv_mask = torch.ones(bsz, max_important, dtype=torch.bool, device=query.device)
                
                for b in range(bsz):
                    idx = key_important[b].nonzero(as_tuple=True)[0]
                    n = idx.shape[0]
                    k_selected[b, :n] = key[b, idx]
                    v_selected[b, :n] = value[b, idx]
                    kv_mask[b, :n] = False

                # print(f"[SPARSE] Attention: query {query.shape} @ k_selected {k_selected.shape}")
                out, attn = self.mha(query, k_selected, v_selected, key_padding_mask=kv_mask,
                                      need_weights=need_weights, average_attn_weights=average_attn_weights)

        else:
            # QUERY is larger -> pool queries, selected queries attend to all keys
            # print(f"[SPARSE] QUERY is larger ({q_len} > {k_len}) -> pooling queries")
            
            query_important = self._select_important_blocks(query, key, pool_is_key=False)
            
            # Gather selected queries
            batch_idx_list, token_idx_list = [], []
            for b in range(bsz):
                idx = query_important[b].nonzero(as_tuple=True)[0]
                batch_idx_list.append(torch.full((idx.shape[0],), b, device=query.device, dtype=torch.long))
                token_idx_list.append(idx)
            
            batch_indices = torch.cat(batch_idx_list)
            token_indices = torch.cat(token_idx_list)
            total_selected = len(batch_indices)

            if total_selected == 0:
                out = query.new_zeros(bsz, q_len, q_embed)
            else:
                q_selected = query[batch_indices, token_indices]  # (total_selected, q_embed)

                # Expand key/value to match selected queries batch-wise
                # We need to run attention per-batch or use a trick
                # Simplest: run as single batch with all selected queries
                # But they need to attend to their own batch's keys
                
                # For efficiency, pad selected queries per batch and run batched attention
                max_selected = int(query_important.sum(dim=1).max().item())
                q_padded = query.new_zeros(bsz, max_selected, q_embed)
                q_mask = torch.ones(bsz, max_selected, dtype=torch.bool, device=query.device)
                
                scatter_indices = []
                for b in range(bsz):
                    idx = query_important[b].nonzero(as_tuple=True)[0]
                    n = idx.shape[0]
                    q_padded[b, :n] = query[b, idx]
                    q_mask[b, :n] = False
                    scatter_indices.append(idx)

                # print(f"[SPARSE] Attention: q_selected {q_padded.shape} @ key {key.shape}")
                
                # Selected queries attend to ALL keys
                out_selected, attn = self.mha(q_padded, key, value, key_padding_mask=key_padding_mask,
                                               need_weights=need_weights, average_attn_weights=average_attn_weights)

                # Scatter back to original positions (non-selected get zeros)
                out = query.new_zeros(bsz, q_len, q_embed)
                for b in range(bsz):
                    idx = scatter_indices[b]
                    n = idx.shape[0]
                    out[b, idx] = out_selected[b, :n]

        # print(f"[SPARSE] Output: {out.shape}")
        # print("=" * 60)

        if not self._batch_first:
            out = out.transpose(0, 1)

        return out, attn if need_weights else (out, None)

    def get_last_critical_mask(self) -> Optional[torch.Tensor]:
        """Returns the last computed critical block mask (B, num_blocks)."""
        return self._last_critical_mask
