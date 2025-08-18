import torch
import torch.nn as nn

from .DropPath import DropPath
from .MLP import MLP

from .FlexMultiheadAttention import FlexMultiheadAttention


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
        if self_attention:
            self.mha_ln = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=ln_scale_shift)
            self.mha = FlexMultiheadAttention(
                in_features=dim,
                num_heads=num_heads,
                out_features=dim,
                key_features=dim,
                value_features=dim,
                num_kv_heads=num_heads,
                embed_dim=dim,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.self_attention:
            x_ln = self.mha_ln(x)
            x_residual, _ = self.mha(x_ln, x_ln, x_ln)
            x = x + self.dropp(x_residual)
        x = self.mlp(x)
        return x
