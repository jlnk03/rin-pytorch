import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..utils.pos_embedding import positional_encoding


class ScalarEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        scaling: float | torch.Tensor,
        expansion=4,
    ):
        super().__init__()
        self.scalar_encoding = lambda x: positional_encoding(x * scaling, dim)
        
        def variance_scaling_init(m):
            if isinstance(m, nn.Linear):
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
                fan_avg = (fan_in + fan_out) / 2
                limit = math.sqrt(3.0 / fan_avg)  # Keras VarianceScaling formula
                nn.init.uniform_(m.weight, -limit, limit)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.dense_0 = nn.Linear(dim, dim * expansion)
        self.dense_1 = nn.Linear(dim * expansion, dim * expansion)
        
        # Apply Keras-equivalent initialization
        variance_scaling_init(self.dense_0)
        variance_scaling_init(self.dense_1)

    def forward(
        self,
        x: torch.Tensor,
        last_swish=True,
        normalize=False,
    ) -> torch.Tensor:
        assert x.ndim == 1
        x = self.scalar_encoding(x)[0]
        if normalize:
            x_mean = torch.mean(x, -1, keepdim=True)
            x_std = torch.std(x, -1, correction=0, keepdim=True)
            x = (x - x_mean) / x_std
        x = F.silu(self.dense_0(x))
        x = self.dense_1(x)
        return F.silu(x) if last_swish else x
