import torch
import torch.nn as nn


class FeedForwardLayer(torch.nn.Module):
    def __init__(
        self,
        dim_att: int,
        dim_mlp: int,
        drop_units=0.1,
        use_ln=False,
        ln_scale_shift=False,
    ):
        super().__init__()
        self.dense1 = nn.Linear(dim_att, dim_mlp)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=drop_units)
        self.dense2 = nn.Linear(dim_mlp, dim_att)
        if use_ln:
            self.ln = nn.LayerNorm(
                dim_mlp,
                eps=1e-6,
                elementwise_affine=ln_scale_shift,
            )
        else:
            self.ln = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense1(x)
        x = self.gelu(x)
        x = self.ln(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x
