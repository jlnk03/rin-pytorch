import torch
from torch import nn

from .TransformerEncoderLayer import TransformerEncoderLayer


class TransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
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

        self.enc_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    dim=dim,
                    mlp_ratio=mlp_ratio,
                    num_heads=num_heads,
                    drop_path=drop_path,
                    drop_units=drop_units,
                    drop_att=drop_att,
                    self_attention=self_attention,
                    use_ffn_ln=use_ffn_ln,
                    ln_scale_shift=ln_scale_shift,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask=None,
        # Legacy args (ignored if self_attn_mask provided)
        latent_document_ids: torch.Tensor | None = None,
        latent_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor
            self_attn_mask: Pre-created attention mask (preferred for performance)
        """
        for layer in self.enc_layers:
            x, self_attn_mask = layer((x, self_attn_mask))
        return x
