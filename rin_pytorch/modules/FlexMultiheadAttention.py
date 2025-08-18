import math
import torch
from typing import Optional, Tuple

from torch.nn.attention.flex_attention import flex_attention, BlockMask


flex_compiled = torch.compile(
    flex_attention
)


def _expand_kv_heads(hidden_states: torch.Tensor, repeats_per_kv_head: int) -> torch.Tensor:
    """
    Convert hidden_states from
      [batch, num_kv_heads, seqlen, head_dim] -> [batch, num_attention_heads, seqlen, head_dim]
    using a memory-efficient expand/reshape instead of repeat_interleave.
    """
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    if repeats_per_kv_head == 1:
        return hidden_states
    hidden_states = (
        hidden_states[:, :, None, :, :]
        .expand(batch, num_kv_heads, repeats_per_kv_head, seq_len, head_dim)
        .reshape(batch, num_kv_heads * repeats_per_kv_head, seq_len, head_dim)
    )
    return hidden_states


class FlexMultiheadAttention(torch.nn.Module):
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
        # Native-style API: embed_dim defines q/k/v proj size; derive head_dim from it
        self.embed_dim = embed_dim or in_features
        if head_dim is not None and self.embed_dim % self.num_heads == 0 and head_dim != self.embed_dim // self.num_heads:
            raise ValueError(
                f"head_dim ({head_dim}) must equal embed_dim // num_heads ({self.embed_dim // self.num_heads})"
            )
        self.head_dim = head_dim or (self.embed_dim // self.num_heads)
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )

        # Remove GQA to minimize differences from native nn.MultiheadAttention
        # Always use num_kv_heads == num_heads
        self.num_kv_heads = num_heads

        # Use native-style in-projection weights.
        # When q/k/v input dims are the same and kv heads == q heads, we can keep a single stacked weight.
        self._qkv_same_embed_dim = (
            self.in_features == self.key_features == self.value_features and self.num_kv_heads == self.num_heads
        )

        if self._qkv_same_embed_dim:
            # Single stacked weight [3*embed_dim, embed_dim_in]
            self.in_proj_weight = torch.nn.Parameter(
                torch.empty(3 * self.embed_dim, self.in_features)
            )
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)
        else:
            # Separate weights to support different input dims and/or GQA
            self.q_proj_weight = torch.nn.Parameter(
                torch.empty(self.num_heads * self.head_dim, self.in_features)
            )
            self.k_proj_weight = torch.nn.Parameter(
                torch.empty(self.num_kv_heads * self.head_dim, self.key_features)
            )
            self.v_proj_weight = torch.nn.Parameter(
                torch.empty(self.num_kv_heads * self.head_dim, self.value_features)
            )
            self.register_parameter("in_proj_weight", None)

        # Add native-style input projection bias (shared across q/k/v)
        self.in_proj_bias = torch.nn.Parameter(torch.zeros(3 * self.embed_dim))

        # Output projection (match native MHA naming and include bias)
        self.out_proj = torch.nn.Linear(self.num_heads * self.head_dim, self.out_features, bias=True)

        # Initialize weights similar to Linear defaults
        if self._qkv_same_embed_dim:
            torch.nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            torch.nn.init.xavier_uniform_(self.q_proj_weight)
            torch.nn.init.xavier_uniform_(self.k_proj_weight)
            torch.nn.init.xavier_uniform_(self.v_proj_weight)

        # Initialize biases like native MHA (_reset_parameters)
        torch.nn.init.constant_(self.in_proj_bias, 0.0)
        if self.out_proj.bias is not None:
            torch.nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        block_mask: Optional[BlockMask] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """
        Args:
            query: [batch, L, in_features] or [L, in_features]
            key:   [batch, S, key_features] or [S, key_features]
            value: [batch, S, value_features] or [S, value_features]
            block_mask: Optional FlexAttention BlockMask to restrict attention
            attn_mask:  Optional additive mask broadcastable as [batch, heads|1, L, S]
                        with zeros for allowed positions and -inf for masked positions

        Returns:
            attn_output: [batch, L, out_features] or [L, out_features] (matches input rank)
            None: attention weights are not returned
        """
        squeeze_batch = False
        if query.ndim == 2:
            # Promote to batch=1 for convenience
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            squeeze_batch = True

        batch_size, l_query, _ = query.shape

        # Project inputs (native-style)
        if self._qkv_same_embed_dim:
            # Use single stacked weight like nn.MultiheadAttention
            # q: [B, L, embed_dim]
            # k,v: [B, S, embed_dim]
            w_q, w_k, w_v = self.in_proj_weight.split(self.embed_dim, dim=0)
            b_q, b_k, b_v = self.in_proj_bias.split(self.embed_dim, dim=0)
            query_states = torch.nn.functional.linear(query, w_q, b_q)
            key_states = torch.nn.functional.linear(key, w_k, b_k)
            value_states = torch.nn.functional.linear(value, w_v, b_v)
            # key/value currently have embed_dim heads; adjust for GQA not applicable here as heads equal
            kv_proj_heads = self.num_heads
        else:
            # Separate weights to allow kdim/vdim and grouped kv heads
            b_q, b_k, b_v = self.in_proj_bias.split(self.embed_dim, dim=0)
            query_states = torch.nn.functional.linear(query, self.q_proj_weight, b_q)  # [B, L, H*D]
            key_states = torch.nn.functional.linear(key, self.k_proj_weight, b_k)      # [B, S, H_kv*D]
            value_states = torch.nn.functional.linear(value, self.v_proj_weight, b_v)  # [B, S, H_kv*D]
            kv_proj_heads = self.num_kv_heads

        # Reshape to [B, Heads, Len, HeadDim]
        query_states = query_states.view(batch_size, l_query, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, key_states.shape[1], kv_proj_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, value_states.shape[1], kv_proj_heads, self.head_dim).transpose(1, 2)

        # Expand kv heads if grouped-query attention
        # With GQA removed, num_kv_heads == num_heads and expansion is unnecessary

        # Build optional score modifier for additive masks
        score_mod = None
        if attn_mask is not None:
            # attn_mask expected as additive mask broadcastable to [B, H|1, L, S]
            # Clamp head index to size-1 when mask has a singleton head dimension.
            # Use a tensor for head_max to keep it in the graph.
            head_dim_in_mask = attn_mask.shape[1]
            head_max = torch.tensor(head_dim_in_mask - 1, device=attn_mask.device)

            def _score_mod(score, batch, head, q_idx, k_idx):
                return score + attn_mask[batch, torch.minimum(head_max, head), q_idx, k_idx]

            score_mod = _score_mod

        # Call FlexAttention
        attn_output = flex_compiled(
            query_states,
            key_states,
            value_states,
            block_mask=block_mask,
            score_mod=score_mod,
        )  # [B, H, L, D]

        # Merge heads and project out
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, l_query, self.num_heads * self.head_dim)
        attn_output = self.out_proj(attn_output)  # [B, L, out_features]

        if squeeze_batch:
            attn_output = attn_output.squeeze(0)

        return attn_output, None

