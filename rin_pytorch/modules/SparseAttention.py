import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseMultiheadAttention(nn.Module):
    """
    Sparse cross-attention that always pools the LARGER sequence:
    - If key > query: pool keys, all queries attend to selected keys
    - If query > key: pool queries, selected queries attend to all keys, scatter back
    - If both small: full attention (no sparsity)
    
    Additionally applies linear attention to a fraction of remaining (non-critical) tokens.
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
        linear_ratio: float = 0.5,  # Fraction of remaining tokens to apply linear attention
        phi_kind: str = "softmax",  # Feature map for linear attention
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Store kdim/vdim (default to embed_dim if not specified)
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        
        self.block_size = block_size
        self.critical_ratio = critical_ratio
        self.critical_k = critical_k
        self.head_aggregation = head_aggregation
        self.min_sparse_seq = min_sparse_seq
        self.linear_ratio = linear_ratio
        self.phi_kind = phi_kind
        self._batch_first = batch_first
        
        if self.critical_ratio is not None:
            assert 0.0 < self.critical_ratio <= 1.0
        assert self.head_aggregation in {"mean", "max"}
        assert 0.0 <= self.linear_ratio <= 1.0, "linear_ratio must be in [0, 1]"

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

        # Projection layers for linear attention when kdim/vdim != embed_dim
        # Similar to how nn.MultiheadAttention handles different dimensions
        self.k_proj_linear: Optional[nn.Linear] = None
        self.v_proj_linear: Optional[nn.Linear] = None
        
        if self.kdim != embed_dim:
            self.k_proj_linear = nn.Linear(self.kdim, embed_dim, bias=False, device=device, dtype=dtype)
        if self.vdim != embed_dim:
            self.v_proj_linear = nn.Linear(self.vdim, embed_dim, bias=False, device=device, dtype=dtype)

        self._last_critical_mask: Optional[torch.Tensor] = None
        self._last_linear_mask: Optional[torch.Tensor] = None

    @property
    def batch_first(self) -> bool:
        return self._batch_first

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """Feature map for linear attention; returns non-negative features."""
        if self.phi_kind == "elu+1":
            return F.elu(x) + 1.0
        if self.phi_kind == "relu":
            return F.relu(x)
        if self.phi_kind == "softmax":
            return F.softmax(x, dim=-1)
        raise ValueError(f"Unknown phi kind: {self.phi_kind}")

    def _reshape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, E) -> (B, H, L, Dh)
        
        Note: E can be embed_dim, kdim, or vdim depending on whether x is Q, K, or V.
        We compute head_dim dynamically from the input tensor.
        """
        bsz, seq_len, embed = x.shape
        assert embed % self.num_heads == 0, f"dim {embed} not divisible by num_heads {self.num_heads}"
        head_dim = embed // self.num_heads
        x = x.view(bsz, seq_len, self.num_heads, head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, L, Dh) -> (B, L, E)
        
        Note: Reconstructs E = H * Dh from the input tensor dimensions.
        """
        bsz, num_heads, seq_len, head_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(bsz, seq_len, num_heads * head_dim)

    def _compute_linear_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        key_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Compute linear attention: O = (phi(Q) @ H) / (phi(Q) @ Z)
        where H = phi(K)^T @ V and Z = sum(phi(K))
        
        Args:
            query: (B, Lq, embed_dim)
            key: (B, Lk, kdim) - will be projected to embed_dim if kdim != embed_dim
            value: (B, Lk, vdim) - will be projected to embed_dim if vdim != embed_dim
            query_mask: (B, Lq) - True for tokens to compute (others ignored)
            key_mask: (B, Lk) - True for tokens to include (padding mask inverted)
        Returns:
            output: (B, Lq, embed_dim) - zeros where query_mask is False
        """
        bsz, q_len, _ = query.shape
        _, k_len, _ = key.shape

        # Project key and value to embed_dim if they have different dimensions
        # Similar to how nn.MultiheadAttention handles kdim/vdim
        if self.k_proj_linear is not None:
            key = self.k_proj_linear(key)  # (B, Lk, kdim) -> (B, Lk, embed_dim)
        if self.v_proj_linear is not None:
            value = self.v_proj_linear(value)  # (B, Lk, vdim) -> (B, Lk, embed_dim)

        # Reshape to heads: (B, H, L, Dh)
        qh = self._reshape_to_heads(query)
        kh = self._reshape_to_heads(key)
        vh = self._reshape_to_heads(value)

        # Apply feature maps
        q_phi = self._phi(qh)  # (B, H, Lq, Dh)
        k_phi = self._phi(kh)  # (B, H, Lk, Dh)

        # Apply key mask if provided (mask out padding)
        if key_mask is not None:
            # key_mask: (B, Lk) -> (B, 1, Lk, 1)
            km = key_mask[:, None, :, None].to(k_phi.dtype)
            k_phi = k_phi * km
            vh = vh * km

        # Compute H = phi(K)^T @ V: (B, H, Dh, Dh)
        H = torch.einsum("bhld,bhlm->bhdm", k_phi, vh)
        # Compute Z = sum(phi(K)): (B, H, Dh)
        Z = k_phi.sum(dim=2)

        # Compute output: (B, H, Lq, Dh)
        num = torch.einsum("bhld,bhdm->bhlm", q_phi, H)
        den = torch.einsum("bhld,bhd->bhl", q_phi, Z).unsqueeze(-1)
        out_heads = num / (den + eps)

        # Merge heads: (B, Lq, embed_dim)
        out = self._merge_heads(out_heads)

        # Zero out positions where query_mask is False
        if query_mask is not None:
            out = out * query_mask[:, :, None].to(out.dtype)

        return out

    def _select_linear_tokens(
        self,
        important_mask: torch.Tensor,
        token_importance: torch.Tensor,
        total_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Select linear_ratio fraction of non-important tokens for linear attention,
        choosing the most important ones among the remaining tokens.
        
        Args:
            important_mask: (B, L) - True for important/critical tokens
            token_importance: (B, L) - Importance score for each token
            total_len: total sequence length
            device: torch device
        Returns:
            linear_mask: (B, L) - True for tokens to apply linear attention
        """
        bsz = important_mask.shape[0]
        linear_mask = torch.zeros(bsz, total_len, dtype=torch.bool, device=device)
        
        # Mask out critical tokens by setting their importance to -inf
        remaining_importance = token_importance.clone()
        remaining_importance[important_mask] = float('-inf')
        
        for b in range(bsz):
            # Count non-important tokens
            n_remaining = (~important_mask[b]).sum().item()
            
            if n_remaining == 0:
                continue
            
            # Select top linear_ratio fraction of remaining tokens by importance
            n_linear = max(1, int(math.ceil(self.linear_ratio * n_remaining)))
            
            # Get top-k most important remaining tokens
            topk_indices = remaining_importance[b].topk(n_linear, dim=-1).indices
            linear_mask[b, topk_indices] = True
        
        self._last_linear_mask = linear_mask
        return linear_mask

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select important blocks from the larger sequence.
        Returns: 
            - important mask (B, pool_len) - True for important tokens
            - token importance (B, pool_len) - Importance score for each token
        """
        bsz, pool_len, _ = to_pool.shape
        _, other_len, _ = other.shape
        block = self.block_size
        num_blocks = (pool_len + block - 1) // block

        # Pool the larger sequence
        pooled = self._pool_1d(to_pool)

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

        topk_indices = block_importance.topk(k, dim=-1).indices
        block_important = torch.zeros(bsz, num_blocks, dtype=torch.bool, device=to_pool.device)
        block_important.scatter_(1, topk_indices, True)

        # Expand block importance to token level
        block_idx = torch.div(
            torch.arange(pool_len, device=to_pool.device), block, rounding_mode="floor"
        ).clamp_max(num_blocks - 1)
        token_important = block_important[:, block_idx]
        
        # Expand actual block importance scores to token level for linear token selection
        # Each token inherits its block's importance score
        token_importance_scores = block_importance[:, block_idx]  # (B, pool_len)
        
        self._last_critical_mask = block_important
        return token_important, token_importance_scores

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
        Combines sparse attention on critical tokens with linear attention on
        a fraction (linear_ratio) of remaining tokens.
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
            out, attn = self.mha(query, key, value, key_padding_mask=key_padding_mask,
                                  need_weights=need_weights, average_attn_weights=average_attn_weights)
            if not self._batch_first:
                out = out.transpose(0, 1)
            return out, attn if need_weights else (out, None)

        attn = None

        if k_len >= q_len:
            # KEY is larger -> pool keys, all queries attend to selected keys
            key_important, key_importance_scores = self._select_important_blocks(key, query, pool_is_key=True)
            max_important = int(key_important.sum(dim=1).max().item())

            if max_important == 0:
                out_sparse = query.new_zeros(bsz, q_len, q_embed)
            else:
                # Gather selected keys/values for sparse attention
                k_selected = key.new_zeros(bsz, max_important, k_embed)
                v_selected = value.new_zeros(bsz, max_important, value.shape[-1])
                kv_mask = torch.ones(bsz, max_important, dtype=torch.bool, device=query.device)
                
                for b in range(bsz):
                    idx = key_important[b].nonzero(as_tuple=True)[0]
                    n = idx.shape[0]
                    k_selected[b, :n] = key[b, idx]
                    v_selected[b, :n] = value[b, idx]
                    kv_mask[b, :n] = False

                out_sparse, attn = self.mha(query, k_selected, v_selected, key_padding_mask=kv_mask,
                                             need_weights=need_weights, average_attn_weights=average_attn_weights)

            # Linear attention on 50% of remaining (non-critical) keys
            # Select the most important ones among the remaining tokens
            out_linear = query.new_zeros(bsz, q_len, q_embed)
            if self.linear_ratio > 0:
                # Select linear_ratio fraction of non-important keys by importance
                linear_key_mask = self._select_linear_tokens(key_important, key_importance_scores, k_len, query.device)
                
                # Check if any linear tokens were selected
                if linear_key_mask.any():
                    # Gather linear keys/values
                    max_linear = int(linear_key_mask.sum(dim=1).max().item())
                    if max_linear > 0:
                        k_linear = key.new_zeros(bsz, max_linear, k_embed)
                        v_linear = value.new_zeros(bsz, max_linear, value.shape[-1])
                        linear_valid_mask = torch.zeros(bsz, max_linear, dtype=torch.bool, device=query.device)
                        
                        for b in range(bsz):
                            idx = linear_key_mask[b].nonzero(as_tuple=True)[0]
                            n = idx.shape[0]
                            if n > 0:
                                k_linear[b, :n] = key[b, idx]
                                v_linear[b, :n] = value[b, idx]
                                linear_valid_mask[b, :n] = True
                        
                        # Compute linear attention: all queries attend to linear keys
                        out_linear = self._compute_linear_attention(
                            query, k_linear, v_linear,
                            query_mask=None,  # All queries participate
                            key_mask=linear_valid_mask,
                        )

            out = out_sparse + out_linear

        else:
            # QUERY is larger -> pool queries, selected queries attend to all keys
            query_important, query_importance_scores = self._select_important_blocks(query, key, pool_is_key=False)
            
            # Sparse attention for important queries
            max_selected = int(query_important.sum(dim=1).max().item())
            
            if max_selected == 0:
                out_sparse = query.new_zeros(bsz, q_len, q_embed)
            else:
                q_padded = query.new_zeros(bsz, max_selected, q_embed)
                q_mask = torch.ones(bsz, max_selected, dtype=torch.bool, device=query.device)
                
                scatter_indices = []
                for b in range(bsz):
                    idx = query_important[b].nonzero(as_tuple=True)[0]
                    n = idx.shape[0]
                    q_padded[b, :n] = query[b, idx]
                    q_mask[b, :n] = False
                    scatter_indices.append(idx)
                
                # Selected queries attend to ALL keys
                out_selected, attn = self.mha(q_padded, key, value, key_padding_mask=key_padding_mask,
                                               need_weights=need_weights, average_attn_weights=average_attn_weights)

                # Scatter back to original positions
                out_sparse = query.new_zeros(bsz, q_len, q_embed)
                for b in range(bsz):
                    idx = scatter_indices[b]
                    n = idx.shape[0]
                    out_sparse[b, idx] = out_selected[b, :n]

            # Linear attention on 50% of remaining (non-critical) queries
            # Select the most important ones among the remaining tokens
            out_linear = query.new_zeros(bsz, q_len, q_embed)
            if self.linear_ratio > 0:
                # Select linear_ratio fraction of non-important queries by importance
                linear_query_mask = self._select_linear_tokens(query_important, query_importance_scores, q_len, query.device)
                
                if linear_query_mask.any():
                    # Prepare key mask (invert padding mask for linear attention)
                    key_valid_mask = None
                    if key_padding_mask is not None:
                        key_valid_mask = ~key_padding_mask  # True where valid
                    else:
                        key_valid_mask = torch.ones(bsz, k_len, dtype=torch.bool, device=query.device)
                    
                    # Compute linear attention for selected queries
                    out_linear = self._compute_linear_attention(
                        query, key, value,
                        query_mask=linear_query_mask,
                        key_mask=key_valid_mask,
                    )

            out = out_sparse + out_linear

        if not self._batch_first:
            out = out.transpose(0, 1)

        return out, attn if need_weights else (out, None)

    def get_last_critical_mask(self) -> Optional[torch.Tensor]:
        """Returns the last computed critical block mask (B, num_blocks)."""
        return self._last_critical_mask

    def get_last_linear_mask(self) -> Optional[torch.Tensor]:
        """Returns the last computed linear attention mask (B, L) - True for tokens with linear attention."""
        return self._last_linear_mask
