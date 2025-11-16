from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

torch._dynamo.config.recompile_limit = 32

flex_attention = torch.compile(flex_attention, dynamic=True)

create_block_mask = torch.compile(create_block_mask, dynamic=True)


def create_document_block_mask(
    document_ids: torch.Tensor | None,
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
    def __init__(
        self,
        in_features: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        out_features: Optional[int] = None,
        key_features: Optional[int] = None,
        value_features: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        embed_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.key_features = key_features or in_features
        self.value_features = value_features or in_features
        self.out_features = out_features or in_features
        self.num_heads = num_heads
        self.embed_dim = embed_dim or in_features

        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})")

        inferred_head_dim = self.embed_dim // self.num_heads
        if head_dim is not None and head_dim != inferred_head_dim:
            raise ValueError(
                f"head_dim ({head_dim}) must equal embed_dim // num_heads ({inferred_head_dim})",
            )
        self.head_dim = inferred_head_dim

        # Remove grouped-query attention for simplicity
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if self.num_kv_heads != self.num_heads:
            raise ValueError("Grouped-query attention is not supported in the Flex attention backend.")

        self._qkv_same_embed_dim = (
            self.in_features == self.key_features == self.value_features and self.num_kv_heads == self.num_heads
        )

        if self._qkv_same_embed_dim:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * self.embed_dim, self.in_features))
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)
        else:
            self.q_proj_weight = nn.Parameter(torch.empty(self.num_heads * self.head_dim, self.in_features))
            self.k_proj_weight = nn.Parameter(torch.empty(self.num_kv_heads * self.head_dim, self.key_features))
            self.v_proj_weight = nn.Parameter(torch.empty(self.num_kv_heads * self.head_dim, self.value_features))
            self.register_parameter("in_proj_weight", None)

        self.in_proj_bias = nn.Parameter(torch.zeros(3 * self.embed_dim))
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.out_features, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        block_mask: Optional[BlockMask] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        squeeze_batch = False
        if query.ndim == 2:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            squeeze_batch = True

        batch_size, l_query, _ = query.shape

        if self._qkv_same_embed_dim:
            w_q, w_k, w_v = self.in_proj_weight.split(self.embed_dim, dim=0)
            b_q, b_k, b_v = self.in_proj_bias.split(self.embed_dim, dim=0)
            query_states = nn.functional.linear(query, w_q, b_q)
            key_states = nn.functional.linear(key, w_k, b_k)
            value_states = nn.functional.linear(value, w_v, b_v)
        else:  # pragma: no cover - currently unused but kept for parity
            b_q, b_k, b_v = self.in_proj_bias.split(self.embed_dim, dim=0)
            query_states = nn.functional.linear(query, self.q_proj_weight, b_q)
            key_states = nn.functional.linear(key, self.k_proj_weight, b_k)
            value_states = nn.functional.linear(value, self.v_proj_weight, b_v)

        query_states = query_states.view(batch_size, l_query, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, key_states.shape[1], self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, value_states.shape[1], self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )

        score_mod = None
        if attn_mask is not None:
            head_dim_in_mask = attn_mask.shape[1]
            head_max = torch.tensor(head_dim_in_mask - 1, device=attn_mask.device)

            def _score_mod(score, batch, head, q_idx, k_idx):
                return score + attn_mask[batch, torch.minimum(head_max, head), q_idx, k_idx]

            score_mod = _score_mod

        attn_output = flex_attention(
            query_states,
            key_states,
            value_states,
            block_mask=block_mask,
            score_mod=score_mod,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, l_query, self.num_heads * self.head_dim)
        attn_output = self.out_proj(attn_output)

        if squeeze_batch:
            attn_output = attn_output.squeeze(0)

        return attn_output, None

