import math
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.nn.attention.flex_attention import flex_attention  # type: ignore


class FlexMultiheadAttention(nn.Module):
    """A drop-in replacement for ``nn.MultiheadAttention`` that routes the
    actual attention computation through the ``flex_attention`` operator.

    Only the features required by the current code-base are implemented:

    * ``batch_first=True`` behaviour (all call-sites rely on this)
    * ``need_weights`` is ignored – we always return ``None`` for the
      attention weights
    * ``attn_mask`` can be a boolean tensor of shape ``(B * num_heads, Q, K)``
      (exactly what the existing call-sites supply).  Elements that are
      *True* are **masked**, just like in ``nn.MultiheadAttention``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        batch_first: bool = True,
        bias: bool = True,
        kdim: int | None = None,
        vdim: int | None = None,
    ) -> None:
        super().__init__()
        if not batch_first:
            raise ValueError("FlexMultiheadAttention currently requires batch_first=True")

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Projections.  We keep them separate for clarity – weight tying /
        # packing can be investigated later if needed.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Pre-compute scale to avoid computing sqrt in every forward.
        self.scale = 1.0 / math.sqrt(self.head_dim)

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, None]:
        if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
            raise ValueError("Expected 3-D inputs (batch, seq, embed)")
        B, Q, _ = query.shape
        _, K, _ = key.shape

        # 1. Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 2. Reshape for multi-head: (B, seq, H, D) -> (B, H, seq, D)
        q = q.view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Build score_mod if a mask is supplied
        score_mod = None
        if attn_mask is not None:
            # attn_mask shape: (B * H, Q, K) with *True* entries to be masked
            if attn_mask.dtype == torch.bool:
                mask_bool = attn_mask
            else:
                # Treat non-zero as masked, matching MultiheadAttention behaviour
                mask_bool = attn_mask.to(torch.bool)
            mask_bool = mask_bool.view(B, self.num_heads, Q, K)

            # We capture the mask inside a closure so that flex_attention can
            # consult it element-wise without materialising the full score
            def _make_score_mod(mask: torch.Tensor):
                def _score_mod(score, b, h, q_idx, kv_idx):
                    masked = mask[b, h, q_idx, kv_idx]
                    return torch.where(masked, torch.tensor(float('-inf'), device=score.device, dtype=score.dtype), score)
                return _score_mod

            score_mod = _make_score_mod(mask_bool)

        # 4. Flex attention (returns (B, H, Q, D))
        attn_output = flex_attention(q, k, v, score_mod=score_mod, scale=self.scale)

        # 5. Merge heads and project out
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Q, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        return attn_output, None  # We never return weights