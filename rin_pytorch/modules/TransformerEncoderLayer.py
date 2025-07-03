import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from .DropPath import DropPath
from .MLP import MLP


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
        group_size: int | None = None,
    ):
        super().__init__()

        self.self_attention = self_attention
        self.group_size = group_size
        self.num_heads = num_heads
        if self_attention:
            self.mha_ln = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=ln_scale_shift)
            self._blockmask_cache = None  # stores BlockMask object, not a buffer
            self._cached_len = None

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

            block_mask = None
            if self.group_size is not None:
                seq_len = x_ln.size(1)
                if self._cached_len != seq_len or self._blockmask_cache is None:
                    def group_mask_mod(b, h, q_idx, kv_idx):
                        return (q_idx // self.group_size) == (kv_idx // self.group_size)

                    block_mask = create_block_mask(
                        group_mask_mod,
                        1,
                        self.num_heads,
                        seq_len,
                        seq_len,
                        device=x_ln.device,
                    )
                    self._blockmask_cache = block_mask
                    self._cached_len = seq_len
                else:
                    block_mask = self._blockmask_cache

            # flex_attention expects (B, H, Q, D_head). Use H=1 for now.
            q = x_ln.unsqueeze(1)  # (B,1,Q,D)
            print(f'q: {q.shape}')
            x_residual = flex_attention(q, q, q, block_mask=block_mask)
            x_residual = x_residual.squeeze(1)  # back to (B,Q,D)
            x = x + self.dropp(x_residual)
        x = self.mlp(x)
        return x
