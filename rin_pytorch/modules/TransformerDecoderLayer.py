import torch
import torch.nn as nn
import os

from .DropPath import DropPath
from .MLP import MLP

# Select attention backend based on environment variable
# ATTENTION_BACKEND: "flex" (default), "xformers", or "flash"
ATTENTION_BACKEND = os.environ.get("ATTENTION_BACKEND", "flex").lower()
ATTENTION_BACKEND = "flex"
if ATTENTION_BACKEND == "flash":
    from .MHA_flash import FlashMultiheadAttention as MultiheadAttention
    from .MHA_flash import create_cross_document_block_mask, create_document_block_mask
    _USE_FLASH = True
    _BACKEND_NAME = "Flash Attention"
elif ATTENTION_BACKEND == "xformers":
    from .MHA_xformers import XformersMultiheadAttention as MultiheadAttention
    from .MHA_xformers import create_cross_document_block_mask, create_document_block_mask
    _USE_FLASH = False
    _BACKEND_NAME = "xformers"
else:
    from .MHA import FlexMultiheadAttention as MultiheadAttention
    from .MHA import create_cross_document_block_mask, create_document_block_mask
    _USE_FLASH = False
    _BACKEND_NAME = "Flex Attention"

print(f"[TransformerDecoderLayer] Using attention backend: {_BACKEND_NAME}")


class TransformerDecoderLayer(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: int,
        num_heads: int,
        drop_path=0.1,
        drop_units=0.1,
        drop_att=0.0,
        dim_x_att=None,
        self_attention=True,
        cross_attention=True,
        use_mlp=True,
        use_enc_ln=False,
        use_ffn_ln=False,
        ln_scale_shift=True,
    ):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.use_mlp = use_mlp
        self._use_flash = _USE_FLASH

        if self_attention:
            self.self_ln = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=ln_scale_shift)
            self.self_mha = MultiheadAttention(
                E_q=dim,
                E_k=dim,
                E_v=dim,
                E_total=dim,
                nheads=num_heads,
            )

        if cross_attention:
            self.cross_ln = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=ln_scale_shift)
            if use_enc_ln:
                self.enc_ln = nn.LayerNorm(
                    dim_x_att if dim_x_att is not None else dim,
                    eps=1e-6,
                    elementwise_affine=ln_scale_shift,
                )
            else:
                self.enc_ln = nn.Identity()

            dim_x_att = dim if dim_x_att is None else dim_x_att
            self.cross_mha = MultiheadAttention(
                E_q=dim,
                E_k=dim_x_att,
                E_v=dim_x_att,
                E_total=dim,
                nheads=num_heads,
            )

        if use_mlp:
            self.mlp = MLP(
                num_layers=1,
                dim=dim,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path,
                drop_units=drop_units,
                use_ffn_ln=use_ffn_ln,
                ln_scale_shift=ln_scale_shift,
            )
        self.dropp = DropPath(drop_path)

    def forward(
        self,
        x: torch.Tensor,
        enc: torch.Tensor,
        cross_attn_mask=None,
        self_attn_mask=None,
        # Legacy args (kept for compatibility but ignored if masks provided)
        query_document_ids: torch.Tensor | None = None,
        key_document_ids: torch.Tensor | None = None,
        query_offsets: torch.Tensor | None = None,
        key_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Query tensor
            enc: Key/value tensor for cross-attention
            cross_attn_mask: Pre-created cross-attention mask (preferred for performance)
            self_attn_mask: Pre-created self-attention mask (preferred for performance)
        """
        if self.self_attention:
            x_ln = self.self_ln(x)
            
            if self._use_flash:
                # Flash attention - mask is a dict with cu_seqlens
                if self_attn_mask is None:
                    self_attn_mask = create_document_block_mask(query_document_ids, offsets=query_offsets)
                x_res = self.self_mha(
                    x_ln, x_ln, x_ln,
                    cu_seqlens_q=self_attn_mask["cu_seqlens"],
                    cu_seqlens_k=self_attn_mask["cu_seqlens"],
                    max_seqlen_q=self_attn_mask["max_seqlen"],
                    max_seqlen_k=self_attn_mask["max_seqlen"],
                )
            else:
                # Flex/xformers - use pre-created mask or create if not provided
                if self_attn_mask is None:
                    self_attn_mask = create_document_block_mask(query_document_ids, offsets=query_offsets)
                x_res = self.self_mha(x_ln, x_ln, x_ln, block_mask=self_attn_mask)
            
            x = x + self.dropp(x_res)

        if self.cross_attention:
            x_ln = self.cross_ln(x)
            enc = self.enc_ln(enc)
            if enc.ndim == 3:
                enc = enc.squeeze(0)
            
            if self._use_flash:
                # Flash attention - mask is a dict with cu_seqlens
                if cross_attn_mask is None:
                    cross_attn_mask = create_cross_document_block_mask(
                        query_document_ids, key_document_ids,
                        q_offsets=query_offsets, kv_offsets=key_offsets
                    )
                x_res = self.cross_mha(
                    x_ln, enc, enc,
                    cu_seqlens_q=cross_attn_mask["cu_seqlens_q"],
                    cu_seqlens_k=cross_attn_mask["cu_seqlens_k"],
                    max_seqlen_q=cross_attn_mask["max_seqlen_q"],
                    max_seqlen_k=cross_attn_mask["max_seqlen_k"],
                )
            else:
                # Flex/xformers - use pre-created mask or create if not provided
                if cross_attn_mask is None:
                    cross_attn_mask = create_cross_document_block_mask(
                        query_document_ids, key_document_ids,
                        q_offsets=query_offsets, kv_offsets=key_offsets
                    )
                x_res = self.cross_mha(x_ln, enc, enc, block_mask=cross_attn_mask)
            
            x = x + self.dropp(x_res)

        if self.use_mlp:
            x = self.mlp(x)
        return x
