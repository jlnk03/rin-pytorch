"""
Flash Attention 2 based MultiheadAttention with native variable-length (varlen) support.

This is the fastest implementation for packed sequences - it uses Flash Attention's
native varlen API which doesn't materialize any masks.

Requires: pip install flash-attn --no-build-isolation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    print("Warning: flash-attn not available. Install with: pip install flash-attn --no-build-isolation")


class FlashMultiheadAttention(nn.Module):
    """
    Multi-head attention using Flash Attention 2's varlen API.
    
    This is optimized for packed sequences where each document can have
    different lengths. Uses cumulative sequence lengths (cu_seqlens) for
    efficient document-level isolation without materializing masks.
    
    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads
        nheads (int): Number of heads
        dropout (float): Dropout probability (only applied during training)
        bias (bool): Whether to add bias to projections
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
        if not FLASH_AVAILABLE:
            raise RuntimeError(
                "flash-attn is required. Install with: pip install flash-attn --no-build-isolation"
            )
            
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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        # For compatibility with other interfaces
        block_mask=None,
        attn_bias=None,
        offsets: Optional[torch.Tensor] = None,
        q_offsets: Optional[torch.Tensor] = None,
        kv_offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with Flash Attention varlen.
        
        Args:
            query: Query tensor [total_q_tokens, E_q] (packed format)
            key: Key tensor [total_kv_tokens, E_k] (packed format)  
            value: Value tensor [total_kv_tokens, E_v] (packed format)
            cu_seqlens_q: Cumulative query sequence lengths [num_docs + 1], int32
            cu_seqlens_k: Cumulative key sequence lengths [num_docs + 1], int32
            max_seqlen_q: Maximum query sequence length (for efficiency)
            max_seqlen_k: Maximum key sequence length (for efficiency)
            offsets: Alternative to cu_seqlens for self-attention (same for Q and KV)
            q_offsets: Alternative to cu_seqlens_q for cross-attention
            kv_offsets: Alternative to cu_seqlens_k for cross-attention
            
        Returns:
            Output tensor [total_q_tokens, E_q]
        """
        # Handle offset arguments (convert to cu_seqlens format)
        if cu_seqlens_q is None:
            if q_offsets is not None:
                cu_seqlens_q = q_offsets.to(torch.int32)
            elif offsets is not None:
                cu_seqlens_q = offsets.to(torch.int32)
        
        if cu_seqlens_k is None:
            if kv_offsets is not None:
                cu_seqlens_k = kv_offsets.to(torch.int32)
            elif offsets is not None:
                cu_seqlens_k = offsets.to(torch.int32)
        
        # Calculate max_seqlen if not provided
        if cu_seqlens_q is not None and max_seqlen_q is None:
            seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            max_seqlen_q = seqlens_q.max().item()
        
        if cu_seqlens_k is not None and max_seqlen_k is None:
            seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            max_seqlen_k = seqlens_k.max().item()
        
        # Step 1: Apply input projections
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                # Self-attention: single packed projection
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
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
        
        # Step 2: Reshape for Flash Attention
        # flash_attn_varlen expects: [total_tokens, nheads, head_dim]
        total_q = query.shape[0]
        total_kv = key.shape[0]
        
        query = query.view(total_q, self.nheads, self.E_head)
        key = key.view(total_kv, self.nheads, self.E_head)
        value = value.view(total_kv, self.nheads, self.E_head)
        
        # Step 3: Run Flash Attention varlen
        dropout_p = self.dropout if self.training else 0.0
        
        if cu_seqlens_q is not None and cu_seqlens_k is not None:
            # Variable length attention (document-isolated)
            attn_output = flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=dropout_p,
                causal=False,
            )
        else:
            # Fallback: treat as single sequence (no document isolation)
            # This shouldn't happen in normal use
            from flash_attn import flash_attn_func
            query = query.unsqueeze(0)  # [1, total, nheads, head_dim]
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            attn_output = flash_attn_func(query, key, value, dropout_p=dropout_p, causal=False)
            attn_output = attn_output.squeeze(0)
        
        # Step 4: Reshape output
        # [total_q, nheads, head_dim] -> [total_q, E_total]
        attn_output = attn_output.view(total_q, self.E_total)
        
        # Step 5: Output projection
        attn_output = self.out_proj(attn_output)
        
        return attn_output


# Utility functions for creating cu_seqlens from different formats

def offsets_to_cu_seqlens(offsets: torch.Tensor) -> torch.Tensor:
    """Convert offsets tensor to cu_seqlens format (int32)."""
    return offsets.to(torch.int32)


def doc_ids_to_cu_seqlens(doc_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
    """
    Convert document IDs to cumulative sequence lengths.
    
    Args:
        doc_ids: [total_tokens] tensor where each element is the document ID
        
    Returns:
        cu_seqlens: [num_docs + 1] cumulative lengths
        max_seqlen: maximum sequence length
    """
    # Count tokens per document
    num_docs = doc_ids.max().item() + 1
    
    # Use unique_consecutive to get counts (assumes doc_ids are sorted!)
    _, counts = torch.unique_consecutive(doc_ids, return_counts=True)
    
    # Create cumulative sum
    cu_seqlens = torch.zeros(num_docs + 1, dtype=torch.int32, device=doc_ids.device)
    cu_seqlens[1:] = counts.cumsum(0)
    
    max_seqlen = counts.max().item()
    
    return cu_seqlens, max_seqlen


def create_document_cu_seqlens(
    document_ids: torch.Tensor | None,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, int | None]:
    """
    Create cu_seqlens for Flash Attention from document_ids or offsets.
    
    Returns:
        cu_seqlens: Cumulative sequence lengths [num_docs + 1]
        max_seqlen: Maximum sequence length
    """
    if offsets is not None:
        cu_seqlens = offsets.to(torch.int32)
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = seqlens.max().item()
        return cu_seqlens, max_seqlen
    
    if document_ids is None or document_ids.numel() == 0:
        return None, None
    
    return doc_ids_to_cu_seqlens(document_ids)


def create_cross_document_cu_seqlens(
    latent_document_ids: torch.Tensor | None,
    document_ids: torch.Tensor | None,
    q_offsets: torch.Tensor | None = None,
    kv_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int | None, int | None]:
    """
    Create cu_seqlens for cross-attention.
    
    Returns:
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
    """
    # Query (latent) cu_seqlens
    if q_offsets is not None:
        cu_seqlens_q = q_offsets.to(torch.int32)
        seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        max_seqlen_q = seqlens_q.max().item()
    elif latent_document_ids is not None:
        cu_seqlens_q, max_seqlen_q = doc_ids_to_cu_seqlens(latent_document_ids)
    else:
        cu_seqlens_q, max_seqlen_q = None, None
    
    # Key/Value (tape) cu_seqlens
    if kv_offsets is not None:
        cu_seqlens_k = kv_offsets.to(torch.int32)
        seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        max_seqlen_k = seqlens_k.max().item()
    elif document_ids is not None:
        cu_seqlens_k, max_seqlen_k = doc_ids_to_cu_seqlens(document_ids)
    else:
        cu_seqlens_k, max_seqlen_k = None, None
    
    return cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k


# Compatibility wrapper that matches the flex_attention / xformers interface
def create_document_block_mask(
    document_ids: torch.Tensor | None,
    offsets: torch.Tensor | None = None,
):
    """
    Compatibility wrapper - returns cu_seqlens info instead of a mask.
    
    For Flash Attention, we don't create a mask. Instead, we return
    the cu_seqlens and max_seqlen that flash_attn_varlen needs.
    
    Returns a dict with cu_seqlens and max_seqlen.
    """
    cu_seqlens, max_seqlen = create_document_cu_seqlens(document_ids, offsets)
    return {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen}


def create_cross_document_block_mask(
    latent_document_ids: torch.Tensor | None,
    document_ids: torch.Tensor | None,
    q_offsets: torch.Tensor | None = None,
    kv_offsets: torch.Tensor | None = None,
):
    """
    Compatibility wrapper for cross-attention.
    
    Returns a dict with cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k.
    """
    cu_q, cu_k, max_q, max_k = create_cross_document_cu_seqlens(
        latent_document_ids, document_ids, q_offsets, kv_offsets
    )
    return {
        "cu_seqlens_q": cu_q,
        "cu_seqlens_k": cu_k,
        "max_seqlen_q": max_q,
        "max_seqlen_k": max_k,
    }

