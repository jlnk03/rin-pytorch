import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask
from functools import lru_cache

from torch import compile

from .DropPath import DropPath
from .MLP import MLP

flex_attention = compile(flex_attention)
create_block_mask = compile(create_block_mask)

# @lru_cache
# def create_block_mask_cached(score_mod, B, H, M, N, device):
#     block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
#     return block_mask


def create_document_block_mask(
    latent_document_ids: torch.Tensor,
    device: torch.device,
) -> BlockMask:
    """
    Create a block mask for document masking using FlexAttention.
    
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
            self.mha = nn.MultiheadAttention(
                dim,
                num_heads,
                dropout=drop_att,
                batch_first=True
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

    def forward(self, inputs) -> tuple:
        x, latent_document_ids = inputs
        if self.self_attention:
            x_ln = self.mha_ln(x)
            
            # Use FlexAttention with document masking
            # Create document block mask
            block_mask = create_document_block_mask(latent_document_ids, device=x.device)
            
            # Reshape for FlexAttention: (seq_len, embed_dim) -> (1, seq_len, num_heads, head_dim)
            batch_size = 1  # We have flattened batches
            seq_len = x_ln.shape[0]
            embed_dim = x_ln.shape[-1]
            head_dim = embed_dim // self.mha.num_heads
            
            # Reshape query, key, value for FlexAttention: (seq_len, embed_dim) -> (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
            q = x_ln.view(batch_size, seq_len, self.mha.num_heads, head_dim).transpose(1, 2)
            k = x_ln.view(batch_size, seq_len, self.mha.num_heads, head_dim).transpose(1, 2)
            v = x_ln.view(batch_size, seq_len, self.mha.num_heads, head_dim).transpose(1, 2)
            
            # Apply FlexAttention with document masking
            x_residual = flex_attention(q, k, v, block_mask=block_mask)
            
            # Reshape back to original format: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim) -> (seq_len, embed_dim)
            x_residual = x_residual.transpose(1, 2).contiguous().view(seq_len, embed_dim)
            x = x + self.dropp(x_residual)
        x = self.mlp(x)
        return (x, latent_document_ids)
