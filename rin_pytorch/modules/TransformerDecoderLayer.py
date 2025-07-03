import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from .DropPath import DropPath
from .MLP import MLP

# Compile flex attention for better performance
# flex_attention = torch.compile(flex_attention, dynamic=False)


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
        self.num_heads = num_heads
        
        if self_attention:
            self.self_ln = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=ln_scale_shift)
            self.self_mha = nn.MultiheadAttention(
                dim, 
                num_heads, 
                dropout=drop_att,
                batch_first=True
            )
            
        if cross_attention:
            self.cross_ln = nn.LayerNorm(
                dim,
                eps=1e-6,
                elementwise_affine=ln_scale_shift,
            )
            if use_enc_ln:
                self.enc_ln = nn.LayerNorm(
                    dim_x_att if dim_x_att is not None else dim,
                    eps=1e-6,
                    elementwise_affine=ln_scale_shift,
                )
            else:
                self.enc_ln = nn.Identity()
                
            # When using FlexAttention we need query/key/value to share the same embed dim.
            # If the provided dim_x_att differs from query dim, create a projection so that
            # key/value are mapped to query space.
            self.kv_proj = None
            dim_x_att = dim if dim_x_att is None else dim_x_att
            if dim_x_att != dim:
                self.kv_proj = nn.Linear(dim_x_att, dim, bias=False)
            # NOTE: we no longer rely on torch.nn.MultiheadAttention here because we call
            # flex_attention directly in forward, so we don't instantiate cross_mha.
            
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
        masks: torch.Tensor | None = None,
        mode: str | None = None,
        block_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.self_attention:
            x_ln = self.self_ln(x)
            x_res, _ = self.self_mha(x_ln, x_ln, x_ln, need_weights=False)
            x = x + self.dropp(x_res)
            
        if self.cross_attention:
            # print(f'x: {x.shape}, enc: {enc.shape}')
            x_ln = self.cross_ln(x)
            enc = self.enc_ln(enc)
            if self.kv_proj is not None:
                enc = self.kv_proj(enc)
            # Unsqueeze dims based on input dimensionality
            if x_ln.dim() == 2:
                x_ln = x_ln.unsqueeze(0).unsqueeze(0)
            elif x_ln.dim() == 3:
                x_ln = x_ln.unsqueeze(0)
            if enc.dim() == 2:
                enc = enc.unsqueeze(0).unsqueeze(0)
            elif enc.dim() == 3:
                enc = enc.unsqueeze(0)
            # Reshape mask to (batch_size, latent_len, image_len)

            # x_res, _ = self.cross_mha(query=x_ln, key=enc, value=enc, need_weights=False, attn_mask=masks)

            # print(f'x_ln: {x_ln.shape}, enc: {enc.shape}')
            x_res = flex_attention(x_ln, enc, enc, block_mask=block_masks)
            x_res = x_res.squeeze(0)
            x = x + self.dropp(x_res)
            
        if self.use_mlp:
            x = self.mlp(x)
        return x
