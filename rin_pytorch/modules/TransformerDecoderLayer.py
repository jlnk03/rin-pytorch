import torch
import torch.nn as nn
from functools import lru_cache
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask

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
                
            dim_x_att = dim if dim_x_att is None else dim_x_att
            
            # Add projection layer for FlexAttention to match dimensions
            if dim != dim_x_att:
                self.query_proj = nn.Linear(dim, dim_x_att, bias=False)

                self.output_proj = nn.Linear(dim_x_att, dim, bias=False)
            else:
                self.query_proj = nn.Identity()
            
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
        # masks: torch.Tensor | None = None,
        document_ids, latent_document_ids,
        mode: str | None = None,
    ) -> torch.Tensor:
        if self.self_attention:
            x_ln = self.self_ln(x)
            
            # Create document block mask for self-attention
            block_mask = create_document_block_mask(latent_document_ids, device=x.device)
            
            # Reshape for FlexAttention: (seq_len, embed_dim) -> (1, seq_len, num_heads, head_dim)
            batch_size = 1  # We have flattened batches
            seq_len = x_ln.shape[0]
            embed_dim = x_ln.shape[-1]
            head_dim = embed_dim // self.num_heads
            
            # Reshape query, key, value for FlexAttention: (seq_len, embed_dim) -> (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
            q = x_ln.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
            k = x_ln.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
            v = x_ln.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
            
            # Apply FlexAttention with document masking
            x_res = flex_attention(q, k, v, block_mask=block_mask)
            
            # Reshape back to original format: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim) -> (seq_len, embed_dim)
            x_res = x_res.transpose(1, 2).contiguous().view(seq_len, embed_dim)
            x = x + self.dropp(x_res)
            
        if self.cross_attention:
            # print(mode)
            # print(f'x: {x.shape}, enc: {enc.shape}')
            # print(f'type enc: {type(enc)}')
            x_ln = self.cross_ln(x)
            enc = self.enc_ln(enc)
            
            # Project query to match key/value dimension for FlexAttention
            x_ln_proj = self.query_proj(x_ln)
            # print(f'x_ln_proj: {x_ln_proj.shape}, enc: {enc.shape}')
            # print(f'latent_document_ids: {latent_document_ids}, document_ids: {document_ids}')

            # Use FlexAttention with document masking
            # Handle enc shape (could be [seq_len, dim] or [1, seq_len, dim])
            if enc.ndim == 3:
                enc = enc.squeeze(0)  # Remove batch dimension
            
            # Create cross-document block mask
            block_mask = create_cross_document_block_mask(latent_document_ids, document_ids, device=x.device)
            
            batch_size = 1  # We have flattened batches
            seq_len_q = x_ln_proj.shape[0]
            seq_len_kv = enc.shape[0]
            embed_dim = x_ln_proj.shape[-1]  # Now both should have same dim
            head_dim = embed_dim // self.num_heads

            # print(f'embed_dim: {embed_dim}, head_dim: {head_dim}, num heads: {self.num_heads}')
            
            # Reshape query, key, value for FlexAttention: (seq_len, embed_dim) -> (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
            q = x_ln_proj.view(batch_size, seq_len_q, self.num_heads, head_dim).transpose(1, 2)
            k = enc.view(batch_size, seq_len_kv, self.num_heads, head_dim).transpose(1, 2)
            v = enc.view(batch_size, seq_len_kv, self.num_heads, head_dim).transpose(1, 2)

            # print(f'q: {q.shape}, k: {k.shape}, v: {v.shape}')

            # Apply FlexAttention with document masking
            x_res = flex_attention(q, k, v, block_mask=block_mask)
            
            # Reshape back to original format: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim) -> (seq_len, embed_dim)
            x_res = x_res.transpose(1, 2).contiguous().view(seq_len_q, embed_dim)
            
            # Project back to original dimension
            x_res = self.output_proj(x_res)
            # print(f'x_res: {x_res.shape}')

            x = x + self.dropp(x_res)
            
        if self.use_mlp:
            x = self.mlp(x)
        return x
