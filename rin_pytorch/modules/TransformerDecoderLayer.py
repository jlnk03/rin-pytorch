import torch
import torch.nn as nn
from functools import lru_cache
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from torch import compile

from .DropPath import DropPath
from .MLP import MLP

flex_attention = compile(flex_attention)
create_block_mask = compile(create_block_mask)


# @lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask


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
            x_res, _ = self.self_mha(x_ln, x_ln, x_ln, need_weights=False)
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

            # Create document mask for cross attention: latent tokens can only attend to input tokens from same document
            def cross_document_mask(b, h, q_idx, kv_idx):
                return latent_document_ids[q_idx] == document_ids[kv_idx]
            
            # Use FlexAttention with document masking
            # Handle enc shape (could be [seq_len, dim] or [1, seq_len, dim])
            if enc.ndim == 3:
                enc = enc.squeeze(0)  # Remove batch dimension
            
            batch_size = 1  # We have flattened batches
            seq_len_q = x_ln_proj.shape[0]
            seq_len_kv = enc.shape[0]
            embed_dim = x_ln_proj.shape[-1]  # Now both should have same dim
            head_dim = embed_dim // self.num_heads

            # print(f'embed_dim: {embed_dim}, head_dim: {head_dim}, num heads: {self.num_heads}')
            
            # Reshape query, key, value for FlexAttention
            q = x_ln_proj.view(batch_size, self.num_heads, seq_len_q, head_dim)
            k = enc.view(batch_size, self.num_heads, seq_len_kv, head_dim)
            v = enc.view(batch_size, self.num_heads, seq_len_kv, head_dim)

            # print(f'q: {q.shape}, k: {k.shape}, v: {v.shape}')

            block_mask = create_block_mask_cached(cross_document_mask, batch_size, self.num_heads, seq_len_q, seq_len_kv, device=q.device)
            
            # Apply FlexAttention with document masking
            x_res = flex_attention(q, k, v, block_mask=block_mask)
            
            # Reshape back to original format
            x_res = x_res.view(seq_len_q, embed_dim)
            
            # Project back to original dimension
            x_res = self.output_proj(x_res)
            # print(f'x_res: {x_res.shape}')

            x = x + self.dropp(x_res)
            
        if self.use_mlp:
            x = self.mlp(x)
        return x
