import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
from torch._dynamo import mark_dynamic

from typing import Optional

torch._dynamo.config.recompile_limit = 32

flex_attention = torch.compile(flex_attention, dynamic=True)

create_block_mask = torch.compile(create_block_mask, dynamic=True)


def create_document_block_mask(
    document_ids: torch.Tensor | None,
    offsets: torch.Tensor | None = None,  # For API consistency with xformers/flash (not used here)
) -> BlockMask | None:
    if document_ids is None or document_ids.numel() == 0:
        return None

    doc_ids = document_ids.to(torch.long)
    seq_len = int(doc_ids.shape[0])

    def _mask(_: torch.Tensor, __: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        return doc_ids[q_idx] == doc_ids[kv_idx]

    return create_block_mask(mask_mod=_mask, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=doc_ids.device)


def create_cross_document_block_mask(
    latent_document_ids: torch.Tensor | None,
    document_ids: torch.Tensor | None,
    q_offsets: torch.Tensor | None = None,  # For API consistency with xformers/flash (not used here)
    kv_offsets: torch.Tensor | None = None,  # For API consistency with xformers/flash (not used here)
) -> BlockMask | None:
    if (
        latent_document_ids is None
        or latent_document_ids.numel() == 0
        or document_ids is None
        or document_ids.numel() == 0
    ):
        return None

    latent_ids = latent_document_ids.to(torch.long)
    doc_ids = document_ids.to(torch.long)
    q_len = int(latent_ids.shape[0])
    kv_len = int(doc_ids.shape[0])

    def _mask(_: torch.Tensor, __: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        return latent_ids[q_idx] == doc_ids[kv_idx]

    return create_block_mask(mask_mod=_mask, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=doc_ids.device)



class FlexMultiheadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        self.bias = bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_mask: Optional[BlockMask] = None,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (``N``, ``L_q``, ``E_qk``)
            key (torch.Tensor): key of shape (``N``, ``L_kv``, ``E_qk``)
            value (torch.Tensor): value of shape (``N``, ``L_kv``, ``E_v``)
            attn_mask (torch.Tensor, optional): attention mask of shape (``N``, ``L_q``, ``L_kv``) to pass to SDPA. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # Handle packed sequences (no batch dim) by adding a batch dim of 1
        squeezed = query.dim() == 2
        if squeezed:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
        
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        # attn_output = F.scaled_dot_product_attention(
        #     query, key, value, dropout_p=self.dropout, is_causal=is_causal
        # )
        attn_output = flex_attention(
            query, key, value, block_mask=block_mask
        )
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)
        
        # Remove batch dim if input was packed
        if squeezed:
            attn_output = attn_output.squeeze(0)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output