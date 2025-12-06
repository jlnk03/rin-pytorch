"""
xformers-based MultiheadAttention with BlockDiagonalMask support for packed sequences.

This is a drop-in replacement for FlexMultiheadAttention that uses xformers
memory_efficient_attention with BlockDiagonalMask for document-level masking.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

try:
    from xformers.ops import memory_efficient_attention
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    print("Warning: xformers not available. Install with: pip install xformers")

# Compile is disabled by default because xformers backward doesn't work well with torch.compile
# (RuntimeError: Unhandled FakeTensor Device Propagation for aten._efficient_attention_backward)
_COMPILE_ATTENTION = os.environ.get("COMPILE_XFORMERS", "0").lower() in ("1", "true", "yes")
_LOGGED_COMPILE = False


def create_block_diagonal_mask_from_seqlens(seqlens: List[int]) -> BlockDiagonalMask:
    """
    Create a BlockDiagonalMask from sequence lengths.
    
    Args:
        seqlens: List of sequence lengths for each document in the batch
        
    Returns:
        BlockDiagonalMask for use with memory_efficient_attention
    """
    if not XFORMERS_AVAILABLE:
        raise RuntimeError("xformers is required for BlockDiagonalMask")
    return BlockDiagonalMask.from_seqlens(seqlens)


def create_block_diagonal_mask_from_offsets(offsets: torch.Tensor) -> BlockDiagonalMask:
    """
    Create a BlockDiagonalMask from cumulative offsets.
    
    Args:
        offsets: Tensor of shape [num_docs + 1] with cumulative token counts
                 e.g., [0, 256, 512, 768] for 3 docs of 256 tokens each
                 
    Returns:
        BlockDiagonalMask for use with memory_efficient_attention
    """
    if not XFORMERS_AVAILABLE:
        raise RuntimeError("xformers is required for BlockDiagonalMask")
    
    # Convert offsets to sequence lengths
    offsets_list = offsets.tolist()
    seqlens = [offsets_list[i+1] - offsets_list[i] for i in range(len(offsets_list) - 1)]
    return BlockDiagonalMask.from_seqlens(seqlens)


def create_cross_block_diagonal_mask(
    q_seqlens: List[int],
    kv_seqlens: List[int],
) -> BlockDiagonalMask:
    """
    Create a BlockDiagonalMask for cross-attention where Q and KV have different lengths.
    
    Args:
        q_seqlens: List of query sequence lengths per document
        kv_seqlens: List of key/value sequence lengths per document
        
    Returns:
        BlockDiagonalMask for cross-attention
    """
    if not XFORMERS_AVAILABLE:
        raise RuntimeError("xformers is required for BlockDiagonalMask")
    # xformers uses q_seqlen (singular) and kv_seqlen (singular) as param names
    return BlockDiagonalMask.from_seqlens(q_seqlen=q_seqlens, kv_seqlen=kv_seqlens)


class XformersMultiheadAttention(nn.Module):
    """
    Multi-head attention using xformers memory_efficient_attention.
    
    Supports packed sequences with BlockDiagonalMask for document-level isolation.
    Interface compatible with FlexMultiheadAttention.
    
    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to projections. Default: True
    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        if not XFORMERS_AVAILABLE:
            raise RuntimeError("xformers is required. Install with: pip install xformers")
            
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.nheads = nheads
        self.dropout = dropout
        self.E_total = E_total
        self.E_head = E_total // nheads
        self.bias = bias
        
        assert E_total % nheads == 0, f"E_total ({E_total}) must be divisible by nheads ({nheads})"
        
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        
        if self._qkv_same_embed_dim:
            # Packed projection for self-attention
            self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        else:
            # Separate projections for cross-attention
            self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)
        
        self.out_proj = nn.Linear(E_total, E_q, bias=bias, **factory_kwargs)
        
        # Compile the inner attention computation if enabled
        global _LOGGED_COMPILE
        if _COMPILE_ATTENTION:
            if not _LOGGED_COMPILE:
                print("[XformersMultiheadAttention] Compiling attention with torch.compile()")
                _LOGGED_COMPILE = True
            self._compiled_attention = torch.compile(
                self._attention_impl, 
                mode="reduce-overhead",
                dynamic=True,
            )
        else:
            self._compiled_attention = self._attention_impl

    def _attention_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor, 
        value: torch.Tensor,
        attn_bias,
        is_self_attn: bool,
    ) -> torch.Tensor:
        """Inner attention computation - compiled for speed."""
        # Step 1: Apply input projections
        if is_self_attn and self._qkv_same_embed_dim:
            # Self-attention: single packed projection
            result = self.packed_proj(query)
            query, key, value = torch.chunk(result, 3, dim=-1)
        elif self._qkv_same_embed_dim:
            # Cross-attention with same dims: split packed weights
            q_weight, k_weight, v_weight = torch.chunk(self.packed_proj.weight, 3, dim=0)
            if self.bias:
                q_bias, k_bias, v_bias = torch.chunk(self.packed_proj.bias, 3, dim=0)
            else:
                q_bias, k_bias, v_bias = None, None, None
            query = F.linear(query, q_weight, q_bias)
            key = F.linear(key, k_weight, k_bias)
            value = F.linear(value, v_weight, v_bias)
        else:
            # Cross-attention with different dims
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)
        
        # Step 2: Reshape for xformers [B, M, H, K]
        is_packed = query.dim() == 2
        
        if is_packed:
            query = query.unsqueeze(0).view(1, -1, self.nheads, self.E_head)
            key = key.unsqueeze(0).view(1, -1, self.nheads, self.E_head)
            value = value.unsqueeze(0).view(1, -1, self.nheads, self.E_head)
        else:
            B, L_q = query.shape[:2]
            L_kv = key.shape[1]
            query = query.view(B, L_q, self.nheads, self.E_head)
            key = key.view(B, L_kv, self.nheads, self.E_head)
            value = value.view(B, L_kv, self.nheads, self.E_head)
        
        # Step 3: Run attention
        dropout_p = self.dropout if self.training else 0.0
        attn_output = memory_efficient_attention(
            query, key, value,
            attn_bias=attn_bias,
            p=dropout_p,
        )
        
        # Step 4: Reshape output
        if is_packed:
            attn_output = attn_output.squeeze(0).view(-1, self.E_total)
        else:
            attn_output = attn_output.view(B, L_q, self.E_total)
        
        # Step 5: Output projection
        return self.out_proj(attn_output)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[BlockDiagonalMask] = None,
        # For compatibility with flex_attention interface
        block_mask=None,
        seqlens: Optional[List[int]] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with xformers memory_efficient_attention.
        
        Mask creation is done outside torch.compile (uses tolist()).
        Attention computation is compiled for speed.
        """
        # Use block_mask if attn_bias not provided (for flex_attention compatibility)
        if attn_bias is None and block_mask is not None:
            attn_bias = block_mask
        
        # Create attn_bias from seqlens or offsets if not provided directly
        # This part cannot be compiled due to tolist() calls
        if attn_bias is None and seqlens is not None:
            attn_bias = create_block_diagonal_mask_from_seqlens(seqlens)
        elif attn_bias is None and offsets is not None:
            attn_bias = create_block_diagonal_mask_from_offsets(offsets)
        
        # Call the (potentially compiled) attention computation
        is_self_attn = query is key and key is value
        return self._compiled_attention(query, key, value, attn_bias, is_self_attn)


# Utility functions that mirror the flex_attention interface
def create_document_block_mask(
    document_ids: torch.Tensor | None,
    offsets: torch.Tensor | None = None,
) -> BlockDiagonalMask | None:
    """
    Create BlockDiagonalMask for self-attention from document_ids or offsets.
    
    This is a compatibility wrapper for the flex_attention interface.
    For best performance, use offsets directly.
    """
    if not XFORMERS_AVAILABLE:
        return None
        
    if offsets is not None:
        return create_block_diagonal_mask_from_offsets(offsets)
    
    if document_ids is None or document_ids.numel() == 0:
        return None
    
    # Convert document_ids to sequence lengths
    # doc_ids: [0,0,0,1,1,1,2,2,2] -> seqlens: [3,3,3]
    unique_ids, counts = torch.unique_consecutive(document_ids, return_counts=True)
    seqlens = counts.tolist()
    
    return BlockDiagonalMask.from_seqlens(seqlens)


def create_cross_document_block_mask(
    latent_document_ids: torch.Tensor | None,
    document_ids: torch.Tensor | None,
    q_offsets: torch.Tensor | None = None,
    kv_offsets: torch.Tensor | None = None,
) -> BlockDiagonalMask | None:
    """
    Create BlockDiagonalMask for cross-attention.
    
    Args:
        latent_document_ids: Document IDs for query (latent)
        document_ids: Document IDs for key/value (tape)
        q_offsets: Optional cumulative offsets for query
        kv_offsets: Optional cumulative offsets for key/value
    """
    if not XFORMERS_AVAILABLE:
        return None
    
    # Use offsets if available (more efficient)
    if q_offsets is not None and kv_offsets is not None:
        q_offsets_list = q_offsets.tolist()
        kv_offsets_list = kv_offsets.tolist()
        q_seqlens = [q_offsets_list[i+1] - q_offsets_list[i] for i in range(len(q_offsets_list) - 1)]
        kv_seqlens = [kv_offsets_list[i+1] - kv_offsets_list[i] for i in range(len(kv_offsets_list) - 1)]
        return BlockDiagonalMask.from_seqlens(q_seqlen=q_seqlens, kv_seqlen=kv_seqlens)
    
    if latent_document_ids is None or document_ids is None:
        return None
    
    # Convert document_ids to sequence lengths
    _, q_counts = torch.unique_consecutive(latent_document_ids, return_counts=True)
    _, kv_counts = torch.unique_consecutive(document_ids, return_counts=True)
    
    q_seqlens = q_counts.tolist()
    kv_seqlens = kv_counts.tolist()
    
    return BlockDiagonalMask.from_seqlens(q_seqlen=q_seqlens, kv_seqlen=kv_seqlens)

