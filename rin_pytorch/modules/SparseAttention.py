import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn


class HierarchicalSparseAttention(nn.Module):
    """
    Hierarchical sparse cross-attention:
    - Coarse-to-fine selection through multiple levels
    - Always pools the LARGER sequence
    - Final full attention on selected tokens
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        dropout: float = 0.0,
        batch_first: bool = True,
        sparse_hierarchy: Optional[List[dict]] = None,
        head_aggregation: str = "mean",
        min_sparse_seq: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        
        if sparse_hierarchy is None:
            sparse_hierarchy = [
                {'block_size': 4, 'critical_ratio': 0.5},
                {'block_size': 2, 'critical_ratio': 0.25},
            ]
        self.sparse_hierarchy = sparse_hierarchy
        self.head_aggregation = head_aggregation
        self.min_sparse_seq = min_sparse_seq
        self._batch_first = batch_first

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

    def _pool_1d(self, x: torch.Tensor, block_size: int) -> torch.Tensor:
        """Mean-pool sequence into non-overlapping blocks."""
        bsz, seq_len, embed = x.shape
        num_blocks = (seq_len + block_size - 1) // block_size
        pad_len = num_blocks * block_size - seq_len
        if pad_len > 0:
            x = torch.cat([x, x.new_zeros(bsz, pad_len, embed)], dim=1)
        x = x.view(bsz, num_blocks, block_size, embed)
        return x.mean(dim=2)

    def _select_important_blocks(
        self,
        to_pool: torch.Tensor,
        other: torch.Tensor,
        pool_is_key: bool,
        block_size: int,
        critical_ratio: float,
    ) -> torch.Tensor:
        """
        Select important blocks from the sequence to pool.
        Returns: important mask (B, pool_len) - True for important tokens
        """
        bsz, pool_len, _ = to_pool.shape
        num_blocks = (pool_len + block_size - 1) // block_size

        # Pool the larger sequence
        pooled = self._pool_1d(to_pool, block_size)

        # Compute attention scores to determine importance
        with torch.no_grad():
            if pool_is_key:
                # other=query, pooled=key -> which key blocks are important
                _, attn_w = self.mha(other, pooled, pooled, need_weights=True, average_attn_weights=False)
            else:
                # pooled=query, other=key -> which query blocks are important
                _, attn_w = self.mha(pooled, other, other, need_weights=True, average_attn_weights=False)

        # Aggregate across heads
        if self.head_aggregation == "mean":
            scores = attn_w.mean(dim=1)
        else:
            scores = attn_w.max(dim=1).values

        # Get block importance
        if pool_is_key:
            block_importance = scores.sum(dim=1)  # (B, num_blocks)
        else:
            block_importance = scores.sum(dim=2)  # (B, num_blocks)

        # Select top-k blocks
        k = max(1, int(math.ceil(critical_ratio * num_blocks)))
        topk_indices = block_importance.topk(k, dim=-1).indices
        block_important = torch.zeros(bsz, num_blocks, dtype=torch.bool, device=to_pool.device)
        block_important.scatter_(1, topk_indices, True)

        # Expand to token level
        block_idx = torch.div(
            torch.arange(pool_len, device=to_pool.device), block_size, rounding_mode="floor"
        ).clamp_max(num_blocks - 1)
        token_important = block_important[:, block_idx]

        self._last_critical_mask = block_important
        return token_important

    def _hierarchical_select(
        self,
        to_select: torch.Tensor,
        other: torch.Tensor,
        pool_is_key: bool,
    ) -> torch.Tensor:
        """
        Hierarchical coarse-to-fine selection.
        Returns: final token mask (B, seq_len) - True for selected tokens
        """
        bsz, seq_len, embed = to_select.shape
        
        # Start with all tokens as candidates
        current_mask = torch.ones(bsz, seq_len, dtype=torch.bool, device=to_select.device)
        
        print(f"[HIER] Starting hierarchical selection on {seq_len} tokens")
        
        for level, config in enumerate(self.sparse_hierarchy):
            block_size = config['block_size']
            critical_ratio = config['critical_ratio']
            
            # Get current candidates
            num_candidates = int(current_mask.sum(dim=1).max().item())
            if num_candidates <= self.min_sparse_seq // 2:
                print(f"[HIER] Level {level}: stopping early, only {num_candidates} candidates")
                break
            
            # Gather current candidates for this level
            candidates = to_select.new_zeros(bsz, num_candidates, embed)
            for b in range(bsz):
                idx = current_mask[b].nonzero(as_tuple=True)[0]
                n = min(idx.shape[0], num_candidates)
                candidates[b, :n] = to_select[b, idx[:n]]
            
            # Select important blocks from candidates
            level_mask = self._select_important_blocks(
                candidates, other, pool_is_key, block_size, critical_ratio
            )
            
            # Map back to original indices
            new_mask = torch.zeros_like(current_mask)
            for b in range(bsz):
                orig_idx = current_mask[b].nonzero(as_tuple=True)[0]
                n = min(len(orig_idx), level_mask.shape[1])
                selected_local = level_mask[b, :n].nonzero(as_tuple=True)[0]
                if len(selected_local) > 0:
                    selected_orig = orig_idx[selected_local]
                    new_mask[b, selected_orig] = True
            
            current_mask = new_mask
            num_selected = int(current_mask.sum(dim=1).float().mean().item())
            print(f"[HIER] Level {level}: block_size={block_size}, ratio={critical_ratio}, -> {num_selected} tokens/batch")

        return current_mask

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
        if not self._batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        bsz, q_len, q_embed = query.shape
        _, k_len, k_embed = key.shape

        larger_len = max(q_len, k_len)

        # Full attention if sequences are small
        if larger_len <= self.min_sparse_seq:
            out, attn = self.mha(query, key, value, key_padding_mask=key_padding_mask,
                                  need_weights=need_weights, average_attn_weights=average_attn_weights)
            if not self._batch_first:
                out = out.transpose(0, 1)
            return out, attn if need_weights else (out, None)

        print("=" * 60)
        print(f"[HIER] query: {query.shape}, key: {key.shape}")

        if k_len >= q_len:
            # KEY is larger -> hierarchical selection on keys
            print(f"[HIER] KEY is larger ({k_len} >= {q_len}) -> hierarchical key selection")
            
            key_mask = self._hierarchical_select(key, query, pool_is_key=True)
            max_selected = int(key_mask.sum(dim=1).max().item())

            if max_selected == 0:
                out = query.new_zeros(bsz, q_len, q_embed)
            else:
                # Gather selected keys/values
                k_selected = key.new_zeros(bsz, max_selected, k_embed)
                v_selected = value.new_zeros(bsz, max_selected, value.shape[-1])
                kv_mask = torch.ones(bsz, max_selected, dtype=torch.bool, device=query.device)
                
                for b in range(bsz):
                    idx = key_mask[b].nonzero(as_tuple=True)[0]
                    n = idx.shape[0]
                    k_selected[b, :n] = key[b, idx]
                    v_selected[b, :n] = value[b, idx]
                    kv_mask[b, :n] = False

                print(f"[HIER] Final attention: query {query.shape} @ k_selected {k_selected.shape}")
                out, attn = self.mha(query, k_selected, v_selected, key_padding_mask=kv_mask,
                                      need_weights=need_weights, average_attn_weights=average_attn_weights)

        else:
            # QUERY is larger -> hierarchical selection on queries
            print(f"[HIER] QUERY is larger ({q_len} > {k_len}) -> hierarchical query selection")
            
            query_mask = self._hierarchical_select(query, key, pool_is_key=False)
            max_selected = int(query_mask.sum(dim=1).max().item())

            if max_selected == 0:
                out = query.new_zeros(bsz, q_len, q_embed)
            else:
                # Gather selected queries
                q_selected = query.new_zeros(bsz, max_selected, q_embed)
                q_pad_mask = torch.ones(bsz, max_selected, dtype=torch.bool, device=query.device)
                
                scatter_indices = []
                for b in range(bsz):
                    idx = query_mask[b].nonzero(as_tuple=True)[0]
                    n = idx.shape[0]
                    q_selected[b, :n] = query[b, idx]
                    q_pad_mask[b, :n] = False
                    scatter_indices.append(idx)

                print(f"[HIER] Final attention: q_selected {q_selected.shape} @ key {key.shape}")
                
                # Selected queries attend to ALL keys
                out_selected, attn = self.mha(q_selected, key, value, key_padding_mask=key_padding_mask,
                                               need_weights=need_weights, average_attn_weights=average_attn_weights)

                # Scatter back to original positions
                out = query.new_zeros(bsz, q_len, q_embed)
                for b in range(bsz):
                    idx = scatter_indices[b]
                    n = idx.shape[0]
                    out[b, idx] = out_selected[b, :n]

        print(f"[HIER] Output: {out.shape}")
        print("=" * 60)

        if not self._batch_first:
            out = out.transpose(0, 1)

        return out, attn if need_weights else (out, None)

    def get_last_critical_mask(self) -> Optional[torch.Tensor]:
        return self._last_critical_mask


# Alias for backwards compatibility
SparseMultiheadAttention = HierarchicalSparseAttention
