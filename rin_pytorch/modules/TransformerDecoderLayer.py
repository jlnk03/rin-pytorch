import torch
import torch.nn as nn
from functools import lru_cache
from torch.nn.attention.flex_attention import create_block_mask, BlockMask

from .DropPath import DropPath
from .MLP import MLP
from .FlexMultiheadAttention import FlexMultiheadAttention

# create_block_mask = torch.compile(create_block_mask, dynamic=True)


# @lru_cache
# def create_block_mask_cached(score_mod, B, H, M, N, device):
#     block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
#     return block_mask


def create_document_block_mask(
    latent_document_ids: torch.Tensor,
    device: torch.device,
) -> BlockMask:
    """
    Create a block mask for self-attention document masking using FlexAttention.
    
    Args:
        latent_document_ids: Document IDs for each token, shape (seq_len,)
        
    Returns:
        BlockMask for FlexAttention that restricts attention to same-document tokens
    """
    seq_len = latent_document_ids.shape[0]
    
    def _document_masking(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        # Same document check: tokens can only attend to other tokens from same document
        return latent_document_ids[q_idx] == latent_document_ids[kv_idx]
    
    # Create the block mask for FlexAttention
    return create_block_mask(
        mask_mod=_document_masking,
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
    )


def create_cross_document_block_mask(
    latent_document_ids: torch.Tensor,
    document_ids: torch.Tensor,
    device: torch.device,
) -> BlockMask:
    """
    Create a block mask for cross-attention document masking using FlexAttention.
    
    Args:
        latent_document_ids: Document IDs for query tokens (latent), shape (seq_len_q,)
        document_ids: Document IDs for key/value tokens (input), shape (seq_len_kv,)
        
    Returns:
        BlockMask for FlexAttention that restricts cross-attention to same-document tokens
    """
    seq_len_q = latent_document_ids.shape[0]
    seq_len_kv = document_ids.shape[0]
    
    def _cross_document_masking(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        # Cross document check: latent tokens can only attend to input tokens from same document
        return latent_document_ids[q_idx] == document_ids[kv_idx]
    
    # Create the block mask for FlexAttention
    return create_block_mask(
        mask_mod=_cross_document_masking,
        B=None,
        H=None,
        Q_LEN=seq_len_q,
        KV_LEN=seq_len_kv,
        device=device,
    )


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
            
            # Cross-attention Flex MHA supports different key/value features
            self.cross_mha = FlexMultiheadAttention(
                in_features=dim,
                num_heads=num_heads,
                out_features=dim,
                key_features=dim_x_att,
                value_features=dim_x_att,
                embed_dim=dim,
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
        # masks: torch.Tensor | None = None,
        document_ids, latent_document_ids,
        mode: str | None = None,
    ) -> torch.Tensor:
        # Validate x and latent_document_ids before creating any block masks
        assert (
            x.shape[0] == latent_document_ids.shape[0]
        ), f"x and latent_document_ids must have same seq length. Got x={x.shape[0]}, latent_document_ids={latent_document_ids.shape[0]}"

        assert (
            x.device == latent_document_ids.device
        ), f"x and latent_document_ids must be on the same device. Got x={x.device}, latent_document_ids={latent_document_ids.device}"
            
        if self.cross_attention:
            # print(mode)
            # print(f'x: {x.shape}, enc: {enc.shape}')
            # print(f'type enc: {type(enc)}')
            x_ln = self.cross_ln(x)
            enc = self.enc_ln(enc)

            # Ensure enc is [L, D] not [1, L, D]
            if enc.ndim == 3:
                enc = enc.squeeze(0)

            # Validate enc and document_ids before creating cross-attention block mask
            assert (
                enc.shape[0] == document_ids.shape[0]
            ), f"enc and document_ids must have same seq length. Got enc={enc.shape[0]}, document_ids={document_ids.shape[0]}"

            assert (
                enc.device == document_ids.device
            ), f"enc and document_ids must be on the same device. Got enc={enc.device}, document_ids={document_ids.device}"

            # Create cross-document block mask
            block_mask = create_cross_document_block_mask(latent_document_ids, document_ids, device=x.device)

            x_res, _ = self.cross_mha(query=x_ln, key=enc, value=enc, block_mask=block_mask, attn_mask=None)
            x = x + self.dropp(x_res)
            
        if self.use_mlp:
            x = self.mlp(x)
        return x
