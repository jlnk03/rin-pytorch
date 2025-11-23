import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseMultiheadAttention(nn.Module):
    """
    Drop-in wrapper around nn.MultiheadAttention that additionally computes a compressed
    attention scores matrix Pc using mean-pooled queries and keys, computed by running
    nn.MultiheadAttention on the pooled sequences and saving its attention weights.
    
    Equivalently: pool first along the token dimension, then compute attention.
    
    - pool(·) is mean pooling along the token dimension using non-overlapping blocks
    - d is the per-head dimension
    - block size defaults to 8
    
    Notes:
    - Forward returns the same outputs as nn.MultiheadAttention to remain drop-in compatible.
    - The pooled scores Pc can be retrieved via get_last_pooled_attention().
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        dropout: float = 0.0,
        batch_first: bool = False,
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
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.block_size = block_size
        self.critical_ratio = critical_ratio
        self.critical_k = critical_k
        self.head_aggregation = head_aggregation
        if self.critical_ratio is not None:
            assert 0.0 < self.critical_ratio <= 1.0, "critical_ratio must be in (0, 1]"
        assert self.head_aggregation in {"mean", "max"}, "head_aggregation must be 'mean' or 'max'"

        # Underlying attention (drop-in behavior)
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )

        self._last_pooled_attention: Optional[torch.Tensor] = None
        self._last_critical_mask: Optional[torch.Tensor] = None
        self._last_linear_output: Optional[torch.Tensor] = None

    @property
    def batch_first(self) -> bool:
        return self.mha.batch_first

    def _mean_pool_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool along the sequence dimension into non-overlapping blocks.
        x: (B, H, L, D)
        returns: (B, H, L_blocks, D)
        """
        bsz, num_heads, seq_len, head_dim = x.shape
        block = self.block_size
        num_blocks = (seq_len + block - 1) // block
        pad_len = num_blocks * block - seq_len
        if pad_len > 0:
            pad_shape = (bsz, num_heads, pad_len, head_dim)
            x = torch.cat([x, x.new_zeros(pad_shape)], dim=2)
        x = x.view(bsz, num_heads, num_blocks, block, head_dim)
        sums = x.sum(dim=3)

        counts = x.new_ones(bsz, num_heads, seq_len, 1)
        if pad_len > 0:
            counts = torch.cat([counts, x.new_zeros(bsz, num_heads, pad_len, 1)], dim=2)
        counts = counts.view(bsz, num_heads, num_blocks, block, 1).sum(dim=3)
        pooled = sums / counts.clamp_min(1.0)
        return pooled

    def compute_pooled_scores(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Compute Pc by:
        1) mean-pooling query and key (and value=key) along the token dimension
        2) running nn.MultiheadAttention on pooled sequences with dropout disabled
        3) saving per-head attention weights as Pc with shape (B, H, Lq_blocks, Lk_blocks)
        """
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
        bsz, tgt_len, e_q = query.shape
        _, src_len, e_k = key.shape 
        def pool_1d(x: torch.Tensor) -> torch.Tensor:
            block = self.block_size
            num_blocks = (x.shape[1] + block - 1) // block
            pad_len = num_blocks * block - x.shape[1]
            if pad_len > 0:
                x = torch.cat([x, x.new_zeros(x.shape[0], pad_len, x.shape[2])], dim=1)
            x = x.view(x.shape[0], num_blocks, block, x.shape[2])
            sums = x.sum(dim=2)
            counts = x.new_ones(bsz, num_blocks, block, 1).sum(dim=2)
            return sums / counts.clamp_min(1.0)

        q_pool_seq = pool_1d(query)
        k_pool_seq = pool_1d(key)
        v_pool_seq = k_pool_seq

        if not self.batch_first:
            q_pool_seq = q_pool_seq.transpose(0, 1)
            k_pool_seq = k_pool_seq.transpose(0, 1)
            v_pool_seq = v_pool_seq.transpose(0, 1)

        was_training = self.mha.training
        try:
            self.mha.eval()
            with torch.no_grad():
                _, attn_w = self.mha(
                    q_pool_seq,
                    k_pool_seq,
                    v_pool_seq,
                    need_weights=True,
                    average_attn_weights=False,
                )
        finally:
            if was_training:
                self.mha.train()

        # attn_w: (B, H, Lq_blocks, Lk_blocks)
        self._last_pooled_attention = attn_w
        return attn_w

    def classify_pooled_blocks(self, pc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        SLA-style per-row top-k selection of critical blocks.
        - Aggregates heads (mean or max) to get (B, Lq_blocks, Lk_blocks).
        - For each sample and query block row, selects top-k key blocks.
          k = critical_k if provided else ceil(critical_ratio * Lk_blocks).
        Returns:
            critical_mask: bool tensor of shape (B, Lq_blocks, Lk_blocks)
        """
        if pc is None:
            if self._last_pooled_attention is None:
                raise RuntimeError("No pooled attention available. Call compute_pooled_scores first.")
            pc = self._last_pooled_attention  # (B,H,Lq,Lk)

        if pc.dim() != 4:
            raise ValueError("pc must have shape (B, H, Lq_blocks, Lk_blocks)")

        if self.head_aggregation == "mean":
            pc_agg = pc.mean(dim=1)  # (B,Lq,Lk)
        else:
            pc_agg = pc.max(dim=1).values  # (B,Lq,Lk)

        batch_size, num_query_blocks, num_key_blocks = pc_agg.shape
        if self.critical_k is not None:
            k_per_row = max(1, min(num_key_blocks, int(self.critical_k)))
        else:
            k_per_row = max(1, int(math.ceil(self.critical_ratio * num_key_blocks)))

        topk_indices = pc_agg.topk(k_per_row, dim=-1).indices  # (B,Lq,k)
        critical_mask = torch.zeros_like(pc_agg, dtype=torch.bool)
        critical_mask.scatter_(-1, topk_indices, True)
        self._last_critical_mask = critical_mask
        return critical_mask

    def get_last_pooled_attention(self) -> Optional[torch.Tensor]:
        """
        Returns the most recently computed Pc, or None if compute_pooled_scores hasn't been called yet.
        """
        return self._last_pooled_attention

    def get_last_critical_mask(self) -> Optional[torch.Tensor]:
        """
        Returns the last computed critical block mask (B, Lq_blocks, Lk_blocks),
        or None if classify_pooled_blocks hasn't been called yet.
        """
        return self._last_critical_mask

    def _phi(self, x: torch.Tensor, kind: str = "elu+1") -> torch.Tensor:
        """
        Feature map for linear attention; returns non-negative features.
        """
        if kind == "elu+1":
            return F.elu(x) + 1.0
        if kind == "relu":
            return F.relu(x)
        if kind == "softmax":
            return F.softmax(x, dim=-1)
        raise ValueError("Unknown phi kind")

    def _reshape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, E) or (L, B, E) depending on batch_first
        returns: (B, H, L, Dh)
        """
        if not self.batch_first:
            x = x.transpose(0, 1)  # (B,L,E)
        bsz, seq_len, embed = x.shape
        h = self.num_heads
        dh = self.head_dim
        x = x.view(bsz, seq_len, h, dh).permute(0, 2, 1, 3).contiguous()
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, L, Dh) -> (B, L, E) or (L, B, E) depending on batch_first
        """
        bsz, h, seq_len, dh = x.shape
        out = x.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, h * dh)
        if not self.batch_first:
            out = out.transpose(0, 1)
        return out

    def compute_linear_attention_output(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
        phi_kind: str = "softmax",
    ) -> torch.Tensor:
        """
        Compute linear attention output:
          H = phi(K)^T V
          Z = rowsum(phi(K)^T)
          O = (phi(Q) H) / (phi(Q) Z)
        Uses input embeddings split into heads (no extra projections).
        """
        qh = self._reshape_to_heads(query)  # (B,H,L,Dh)
        kh = self._reshape_to_heads(key)
        vh = self._reshape_to_heads(value)

        q_phi = self._phi(qh, phi_kind)
        k_phi = self._phi(kh, phi_kind)

        if key_padding_mask is not None:
            # Expect (B, L). True indicates masked (ignore)
            if key_padding_mask.dim() == 2:
                mask = key_padding_mask[:, None, :, None].to(k_phi.dtype)  # (B,1,L,1)
            elif key_padding_mask.dim() == 3:
                # (B, L, 1) or (B, 1, L)
                if key_padding_mask.shape[-1] == 1:
                    mask = key_padding_mask[:, None, :, :].to(k_phi.dtype)
                else:
                    mask = key_padding_mask[:, None, :, None].to(k_phi.dtype)
            else:
                mask = key_padding_mask.to(k_phi.dtype)
            keep = 1.0 - mask
            k_phi = k_phi * keep
            vh = vh * keep

        # H: (B,H,Dh,Dh), Z: (B,H,Dh)
        H = torch.einsum("bhld,bhlm->bhdm", k_phi, vh)
        Z = k_phi.sum(dim=2)  # sum over sequence -> (B,H,Dh)

        # Numerator: (B,H,L,Dh); Denominator: (B,H,L,1)
        num = torch.einsum("bhld,bhdm->bhlm", q_phi, H)
        den = torch.einsum("bhld,bhd->bhl", q_phi, Z).unsqueeze(-1)
        out = num / (den + eps)

        out_merged = self._merge_heads(out)  # (B,L,E) or (L,B,E)
        self._last_linear_output = out_merged
        return out_merged

    def compute_linear_marginal_output(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        critical_blocks: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
        phi_kind: str = "softmax",
    ) -> torch.Tensor:
        """
        SLA-style marginal linear attention:
          - Precompute global H_sum = sum_j s_j and Z_sum = sum_j z_j over all key blocks
          - For each query block i, subtract critical blocks' s_j/z_j to get s_qi/z_qi
          - For all tokens in query block i: O_l = (phi(Q_block) @ s_qi) / (phi(Q_block) · z_qi)
        Returns tensor shaped like query (B,L,E) or (L,B,E) depending on batch_first.
        """
        # Bring to (B,L,E) for simpler indexing
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        bsz, q_len, _ = query.shape
        _, k_len, _ = key.shape

        # Heads view
        qh = self._reshape_to_heads(query)  # (B,H,L,Dh)
        kh = self._reshape_to_heads(key)
        vh = self._reshape_to_heads(value)

        # Phi features
        q_phi = self._phi(qh, phi_kind)     # (B,H,L,Dh)
        k_phi = self._phi(kh, phi_kind)     # (B,H,L,Dh)

        # Apply key padding mask to K and V
        if key_padding_mask is not None:
            if key_padding_mask.dim() == 2:
                mask = key_padding_mask[:, None, :, None].to(k_phi.dtype)  # (B,1,L,1)
            elif key_padding_mask.dim() == 3:
                if key_padding_mask.shape[-1] == 1:
                    mask = key_padding_mask[:, None, :, :].to(k_phi.dtype)
                else:
                    mask = key_padding_mask[:, None, :, None].to(k_phi.dtype)
            else:
                mask = key_padding_mask.to(k_phi.dtype)
            keep = 1.0 - mask
            k_phi = k_phi * keep
            vh = vh * keep

        # Global sums over all key tokens
        H_sum = torch.einsum("bhld,bhlm->bhdm", k_phi, vh)  # (B,H,Dh,Dh)
        Z_sum = k_phi.sum(dim=2)                            # (B,H,Dh)

        block = self.block_size
        num_q_blocks = (q_len + block - 1) // block
        num_k_blocks = (k_len + block - 1) // block

        # Output per heads
        ol_heads = qh.new_zeros(qh.shape)  # (B,H,L,Dh)

        # Iterate per-sample, per-query block
        for b in range(bsz):
            # For faster slicing
            k_phi_b = k_phi[b]  # (H,L,Dh)
            vh_b = vh[b]        # (H,L,Dh)
            q_phi_b = q_phi[b]  # (H,L,Dh)
            H_sum_b = H_sum[b]  # (H,Dh,Dh)
            Z_sum_b = Z_sum[b]  # (H,Dh)

            for qi in range(num_q_blocks):
                qs = qi * block
                qe = min((qi + 1) * block, q_len)
                if qs >= qe:
                    continue
                # Accumulate critical block contributions to subtract
                s_crit = H_sum_b.new_zeros(H_sum_b.shape)  # (H,Dh,Dh)
                z_crit = Z_sum_b.new_zeros(Z_sum_b.shape)  # (H,Dh)
                # Which key blocks are critical for this query block
                cb_row = critical_blocks[b, qi]  # (Lk_blocks,)
                crit_k_indices = torch.nonzero(cb_row, as_tuple=False).flatten().tolist()
                for kj in crit_k_indices:
                    ks = kj * block
                    ke = min((kj + 1) * block, k_len)
                    if ks >= ke:
                        continue
                    k_blk = k_phi_b[:, ks:ke, :]  # (H,block,Dh)
                    v_blk = vh_b[:, ks:ke, :]     # (H,block,Dh)
                    # s_j and z_j for this key block
                    s_j = torch.einsum("hld,hlm->hdm", k_blk, v_blk)  # (H,Dh,Dh)
                    z_j = k_blk.sum(dim=1)                           # (H,Dh)
                    s_crit += s_j
                    z_crit += z_j
                # Marginal sums for this query block
                s_qi = H_sum_b - s_crit  # (H,Dh,Dh)
                z_qi = Z_sum_b - z_crit  # (H,Dh)
                # Compute linear outputs for all tokens in this query block at once
                q_blk = q_phi_b[:, qs:qe, :]  # (H,BQL,Dh)
                num = torch.einsum("hld,hdm->hlm", q_blk, s_qi)  # (H,BQL,Dh)
                den = torch.einsum("hld,hd->hl", q_blk, z_qi).unsqueeze(-1)  # (H,BQL,1)
                ol_heads[b, :, qs:qe, :] = num / (den + eps)

        # Merge heads back
        ol = self._merge_heads(ol_heads)  # (B,L,E) or (L,B,E)
        return ol

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

        # 1) Compute pooled attention and classify critical blocks per query block
        critical_blocks = None
        try:
            self.compute_pooled_scores(query, key)
            critical_blocks = self.classify_pooled_blocks(self._last_pooled_attention)
        except Exception:
            self._last_pooled_attention = None
            critical_blocks = None

        # 2) Expand block-level critical mask to token-level mask (B, Lq, Lk)
        block_mask_tokens: Optional[torch.Tensor] = None
        if critical_blocks is not None:
            if self.batch_first:
                bsz, tgt_len, _ = query.shape
                _, src_len, _ = key.shape
            else:
                tgt_len, bsz, _ = query.shape
                src_len, _, _ = key.shape
            block = self.block_size
            num_q_blocks = (tgt_len + block - 1) // block
            num_k_blocks = (src_len + block - 1) // block
            q_block_idx = torch.div(torch.arange(tgt_len, device=query.device), block, rounding_mode="floor").clamp_max(num_q_blocks - 1)
            k_block_idx = torch.div(torch.arange(src_len, device=key.device), block, rounding_mode="floor").clamp_max(num_k_blocks - 1)
            block_mask_tokens = torch.zeros((bsz, tgt_len, src_len), dtype=torch.bool, device=query.device)
            for b in range(bsz):
                cb = critical_blocks[b]
                allowed = cb[q_block_idx][:, k_block_idx]
                block_mask_tokens[b] = ~allowed  # True means mask (disallow)

        # 3) Combine incoming attn_mask with block mask and expand across heads for MHA
        final_attn_mask = attn_mask
        if block_mask_tokens is not None:
            repeat_mask = block_mask_tokens.repeat_interleave(self.num_heads, dim=0)
            if final_attn_mask is None:
                final_attn_mask = repeat_mask
            else:
                # Normalize existing to bool mask shape (B*H, Lq, Lk)
                if final_attn_mask.dtype != torch.bool:
                    final_attn_mask = final_attn_mask != 0
                if final_attn_mask.dim() == 2:
                    final_attn_mask = final_attn_mask.unsqueeze(0).expand(repeat_mask.shape[0], -1, -1)
                elif final_attn_mask.dim() == 3:
                    if final_attn_mask.shape[0] == block_mask_tokens.shape[0]:
                        final_attn_mask = final_attn_mask.repeat_interleave(self.num_heads, dim=0)
                final_attn_mask = final_attn_mask | repeat_mask

        # 4) Sparse exact output via MHA with block mask
        o_s, attn = self.mha(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=final_attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        # 5) Linear marginal output excluding critical blocks and combine
        o_l = None
        if critical_blocks is not None:
            try:
                o_l = self.compute_linear_marginal_output(
                    query=query,
                    key=key,
                    value=value,
                    critical_blocks=critical_blocks,
                    key_padding_mask=key_padding_mask,
                    phi_kind="softmax",
                )
            except Exception:
                o_l = None
        out = o_s if o_l is None else o_s + o_l
        return out, attn if need_weights else (out, None)