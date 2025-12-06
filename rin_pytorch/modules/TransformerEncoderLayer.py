import torch
import torch.nn as nn
import os

from .DropPath import DropPath
from .MLP import MLP

# Select attention backend based on environment variable
# ATTENTION_BACKEND: "flex" (default), "xformers", or "flash"
ATTENTION_BACKEND = os.environ.get("ATTENTION_BACKEND", "flex").lower()
ATTENTION_BACKEND = "flex"
_LOGGED_BACKEND = False

if ATTENTION_BACKEND == "flash":
    from .MHA_flash import FlashMultiheadAttention as MultiheadAttention
    from .MHA_flash import create_document_block_mask
    _USE_FLASH = True
    _BACKEND_NAME = "Flash Attention"
elif ATTENTION_BACKEND == "xformers":
    from .MHA_xformers import XformersMultiheadAttention as MultiheadAttention
    from .MHA_xformers import create_document_block_mask
    _USE_FLASH = False
    _BACKEND_NAME = "xformers"
else:
    from .MHA import FlexMultiheadAttention as MultiheadAttention
    from .MHA import create_document_block_mask
    _USE_FLASH = False
    _BACKEND_NAME = "Flex Attention"

print(f"[TransformerEncoderLayer] Using attention backend: {_BACKEND_NAME}")


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: int,
        num_heads: int,
        drop_path=0.1,
        drop_units=0.1,
        drop_att=0.0,
        self_attention=True,
        use_ffn_ln=False,
        ln_scale_shift=True,
    ):
        super().__init__()

        self.self_attention = self_attention
        self._use_flash = _USE_FLASH
        
        if self_attention:
            self.mha_ln = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=ln_scale_shift)
            self.mha = MultiheadAttention(
                E_q=dim,
                E_k=dim,
                E_v=dim,
                E_total=dim,
                nheads=num_heads,
            )

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

    def forward(self, inputs: tuple[torch.Tensor, any]) -> tuple[torch.Tensor, any]:
        """
        Args:
            inputs: tuple of (x, self_attn_mask) where self_attn_mask is pre-created
        """
        x, self_attn_mask = inputs
        if self.self_attention:
            x_ln = self.mha_ln(x)
            
            if self._use_flash:
                # Flash attention - mask_info is a dict with cu_seqlens
                x_residual = self.mha(
                    x_ln, x_ln, x_ln,
                    cu_seqlens_q=self_attn_mask["cu_seqlens"],
                    cu_seqlens_k=self_attn_mask["cu_seqlens"],
                    max_seqlen_q=self_attn_mask["max_seqlen"],
                    max_seqlen_k=self_attn_mask["max_seqlen"],
                )
            else:
                # Flex/xformers - mask is BlockMask or BlockDiagonalMask
                x_residual = self.mha(x_ln, x_ln, x_ln, block_mask=self_attn_mask)
            
            x = x + self.dropp(x_residual)
        x = self.mlp(x)
        return x, self_attn_mask
