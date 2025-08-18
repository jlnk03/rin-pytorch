import torch
import torch.nn as nn

from .DropPath import DropPath
from .MLP import MLP

from .FlexMultiheadAttention import FlexMultiheadAttention


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
            self.self_mha = FlexMultiheadAttention(
                in_features=dim,
                num_heads=num_heads,
                out_features=dim,
                key_features=dim,
                value_features=dim,
                num_kv_heads=num_heads,
                embed_dim=dim,
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
                
            dim_x_att = dim if dim_x_att is None else dim_x_att
            self.cross_mha = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                kdim=dim_x_att,
                vdim=dim_x_att,
                dropout=drop_att,
                batch_first=True,
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
        masks: torch.Tensor | None = None,
        mode: str | None = None,
    ) -> torch.Tensor:
        if self.self_attention:
            x_ln = self.self_ln(x)
            x_res, _ = self.self_mha(x_ln, x_ln, x_ln, need_weights=False)
            x = x + self.dropp(x_res)
            
        if self.cross_attention:
            # print(mode)
            # print(f'x: {x.shape}, enc: {enc.shape}')
            x_ln = self.cross_ln(x)
            enc = self.enc_ln(enc)
            # print(f'x_ln: {x_ln.shape}, enc: {enc.shape}')
            # apply masks from var image sizes to cross attention only and not self attention
            # x_res, _ = self.cross_mha(query=x_ln, key=enc, value=enc, need_weights=False, key_padding_mask=masks)
            # Reshape mask to (batch_size, latent_len, image_len)
            if masks is not None:
                # print(f'masks: {masks.shape}')
                # Get latent length from query tensor x_ln
                if mode == "read":
                    # Expand mask to include latent dimension
                    latent_len = x_ln.shape[1]
                    masks = masks.unsqueeze(1).expand(-1, latent_len, -1)
                    # print(f'masks inserted: {masks.shape}')
                    # print(f'masks sum latents: {masks.sum(dim=1)}')
                else:
                    latent_len = enc.shape[1]
                    masks = masks.unsqueeze(2).expand(-1, -1, latent_len)
                masks = masks.repeat_interleave(self.num_heads, dim=0)
                # print(f'masks repeated: {masks.shape}')
                # Invert mask since PyTorch attention masks use True to indicate positions to mask
                masks = ~masks.bool()

            x_res, _ = self.cross_mha(query=x_ln, key=enc, value=enc, attn_mask=masks)
            x = x + self.dropp(x_res)
            
        if self.use_mlp:
            x = self.mlp(x)
        return x
