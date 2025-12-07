import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

print(f'has matplotlib: {HAS_MATPLOTLIB}')
@dataclass(frozen=True)
class HierarchyLevel:
    block_size: int
    critical_ratio: Optional[float] = None
    critical_k: Optional[int] = None

    def resolve_k(self, num_key_blocks: int) -> int:
        if num_key_blocks <= 0:
            return 0
        if self.critical_k is not None:
            return max(1, min(num_key_blocks, int(self.critical_k)))
        ratio = 1.0 if self.critical_ratio is None else float(self.critical_ratio)
        return max(1, int(math.ceil(ratio * num_key_blocks)))


class HierarchicalSparseAttention(nn.Module):
    """
    Drop-in wrapper around nn.MultiheadAttention that additionally computes a compressed
    attention scores matrix Pc using mean-pooled queries and keys, computed by running
    nn.MultiheadAttention on the pooled sequences and saving its attention weights.

    Equivalently: pool first along the token dimension, then compute attention.

    - pool(·) is mean pooling along the token dimension using non-overlapping blocks
    - d is the per-head dimension
    - block size defaults to 8

    Notes:
    - Forward returns the same outputs as nn.MultiheadAttention to remain drop-in compatible.
    - The pooled scores Pc can be retrieved via get_last_pooled_attention().
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        dropout: float = 0.0,
        batch_first: bool = False,
        block_size: int = 4,
        critical_ratio: float = 0.10,
        critical_k: Optional[int] = None,
        head_aggregation: str = "mean",
        hierarchy: Optional[Sequence[Union[HierarchyLevel, dict]]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.hierarchy_levels = self._normalize_hierarchy(
            hierarchy=hierarchy,
            default_block_size=block_size,
            default_ratio=critical_ratio,
            default_k=critical_k,
        )
        print(f"hierarchy_levels: {self.hierarchy_levels}")
        self.block_size = self.hierarchy_levels[-1].block_size  # finest resolution
        self.critical_ratio = self.hierarchy_levels[-1].critical_ratio
        self.critical_k = self.hierarchy_levels[-1].critical_k
        self.coarsest_block_size = self.hierarchy_levels[0].block_size
        self._hierarchy_depth = len(self.hierarchy_levels)
        self.head_aggregation = head_aggregation
        if self.critical_ratio is not None:
            assert 0.0 < self.critical_ratio <= 1.0, "critical_ratio must be in (0, 1]"
        assert self.head_aggregation in {"mean", "max"}, "head_aggregation must be 'mean' or 'max'"

        # Underlying attention (drop-in behavior)
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )

        self._last_pooled_attention: Optional[torch.Tensor] = None
        self._last_critical_mask: Optional[torch.Tensor] = None
        self._last_linear_output: Optional[torch.Tensor] = None
        self._last_pooled_key_padding_mask: Optional[torch.Tensor] = None
        self._last_pooled_query_padding_mask: Optional[torch.Tensor] = None
        self._last_hierarchy_summary: Optional[List[dict]] = None
        
        # For attention visualization
        self._last_full_attention: Optional[torch.Tensor] = None
        self._capture_attention: bool = False

    @property
    def batch_first(self) -> bool:
        return self.mha.batch_first

    def _normalize_hierarchy(
        self,
        hierarchy: Optional[Sequence[Union[HierarchyLevel, dict]]],
        default_block_size: int,
        default_ratio: Optional[float],
        default_k: Optional[int],
    ) -> List[HierarchyLevel]:
        def _coerce_level(
            cfg: Union[HierarchyLevel, dict],
            fallback_ratio: Optional[float],
            fallback_k: Optional[int],
        ) -> HierarchyLevel:
            if isinstance(cfg, HierarchyLevel):
                block_size = int(cfg.block_size)
                ratio = cfg.critical_ratio
                crit_k = cfg.critical_k
            elif isinstance(cfg, dict):
                if "block_size" not in cfg:
                    raise ValueError("Each hierarchy level must define 'block_size'.")
                block_size = int(cfg["block_size"])
                ratio = cfg.get("critical_ratio", cfg.get("ratio", fallback_ratio))
                crit_k = cfg.get("critical_k", fallback_k)
            else:
                raise TypeError("Hierarchy entries must be dicts or HierarchyLevel instances.")
            if block_size <= 0:
                raise ValueError("block_size must be positive.")
            if ratio is not None:
                if not isinstance(ratio, (int, float)):
                    raise TypeError("critical_ratio must be a number.")
                if not (0.0 < float(ratio) <= 1.0):
                    raise ValueError("critical_ratio must be in (0, 1].")
                ratio = float(ratio)
            if crit_k is not None:
                crit_k = int(crit_k)
                if crit_k <= 0:
                    raise ValueError("critical_k must be positive.")
            if ratio is None and crit_k is None:
                raise ValueError("Each hierarchy level must define critical_ratio or critical_k.")
            return HierarchyLevel(block_size=block_size, critical_ratio=ratio, critical_k=crit_k)

        if hierarchy is None:
            levels = [
                HierarchyLevel(
                    block_size=default_block_size,
                    critical_ratio=default_ratio,
                    critical_k=default_k,
                )
            ]
        else:
            if isinstance(hierarchy, (HierarchyLevel, dict)):
                hierarchy = [hierarchy]  # type: ignore[list-item]
            if not isinstance(hierarchy, Iterable):
                raise TypeError("hierarchy must be an iterable of levels.")
            fallback_ratio = default_ratio
            fallback_k = default_k
            levels = [
                _coerce_level(cfg, fallback_ratio=fallback_ratio, fallback_k=fallback_k)
                for cfg in hierarchy
            ]

        if not levels:
            raise ValueError("Hierarchy must contain at least one level.")

        for i in range(len(levels) - 1):
            parent = levels[i]
            child = levels[i + 1]
            if parent.block_size < child.block_size:
                raise ValueError("Hierarchy block_size must be non-increasing (coarse-to-fine).")
            if parent.block_size % child.block_size != 0:
                raise ValueError(
                    f"block_size {parent.block_size} must be divisible by next level {child.block_size}."
                )
            if (
                parent.critical_ratio is not None
                and child.critical_ratio is not None
                and child.critical_ratio > parent.critical_ratio
            ):
                raise ValueError("critical_ratio must be non-increasing across hierarchy levels.")
        return levels

    def _mean_pool_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool along the sequence dimension into non-overlapping blocks.
        x: (B, H, L, D)
        returns: (B, H, L_blocks, D)
        """
        bsz, num_heads, seq_len, head_dim = x.shape
        block = self.block_size
        num_blocks = (seq_len + block - 1) // block
        pad_len = num_blocks * block - seq_len
        if pad_len > 0:
            # Use F.pad instead of torch.cat for better performance
            x = F.pad(x, (0, 0, 0, pad_len))
        x = x.view(bsz, num_heads, num_blocks, block, head_dim)
        sums = x.sum(dim=3)

        # Optimize count calculation
        if pad_len > 0:
            counts = x.new_ones(bsz, num_heads, num_blocks, block, 1)
            counts[:, :, -1, -pad_len:, :] = 0
            counts = counts.sum(dim=3)
        else:
            counts = float(block)
        pooled = sums / counts
        return pooled

    def _normalize_padding_mask(
        self,
        mask: Optional[torch.Tensor],
        batch: int,
        length: int,
        name: str = "padding_mask",
    ) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        mask_2d = mask
        if mask_2d.dim() == 3:
            if mask_2d.shape[-1] == 1:
                mask_2d = mask_2d.squeeze(-1)
            elif mask_2d.shape[1] == 1:
                mask_2d = mask_2d.squeeze(1)
        if mask_2d.dim() != 2:
            raise ValueError(f"{name} must be broadcastable to (B, L)")
        if mask_2d.shape[0] != batch or mask_2d.shape[1] != length:
            raise ValueError(f"{name} must have shape (B, L)")
        return mask_2d.to(torch.bool)

    def _pool_tokens(
        self,
        x: torch.Tensor,
        block_size: int,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            x: (B, L, E) tensor (batch-first)
            block_size: pooling window
            padding_mask: optional bool mask (B, L) where True indicates padded tokens
        Returns:
            pooled: (B, L_blocks, E)
            block_padding_mask: Optional[(B, L_blocks)] True when entire block is padded
            token_to_block: (L,) index mapping tokens to their block id
        """
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        bsz, seq_len, embed_dim = x.shape
        num_blocks = (seq_len + block_size - 1) // block_size
        pad_len = num_blocks * block_size - seq_len
        if pad_len > 0:
            # Use F.pad instead of torch.cat for better performance
            x = F.pad(x, (0, 0, 0, pad_len))
            if padding_mask is not None:
                padding_mask = F.pad(padding_mask.float(), (0, pad_len), value=1.0).bool()
        x = x.view(bsz, num_blocks, block_size, embed_dim)
        if padding_mask is None:
            pooled = x.mean(dim=2)
            block_padding_mask = None
        else:
            mask = padding_mask.view(bsz, num_blocks, block_size, 1)
            valid = (~mask).to(x.dtype)
            sums = (x * valid).sum(dim=2)
            counts = valid.sum(dim=2)
            pooled = sums / counts.clamp_min(1.0)
            block_padding_mask = counts.squeeze(-1) == 0

        token_to_block = torch.div(
            torch.arange(seq_len, device=x.device),
            block_size,
            rounding_mode="floor",
        ).clamp_max(num_blocks - 1)
        return pooled, block_padding_mask, token_to_block

    def _run_block_attention(
        self,
        pooled_query: torch.Tensor,
        pooled_key: torch.Tensor,
        pooled_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Runs nn.MultiheadAttention on pooled sequences with dropout disabled.
        pooled_* expected in batch-first form (B, L, E).
        """
        if not self.batch_first:
            q_pool = pooled_query.transpose(0, 1)
            k_pool = pooled_key.transpose(0, 1)
            v_pool = pooled_value.transpose(0, 1)
        else:
            q_pool, k_pool, v_pool = pooled_query, pooled_key, pooled_value

        was_training = self.mha.training
        try:
            self.mha.eval()
            with torch.no_grad():
                _, attn_w = self.mha(
                    q_pool,
                    k_pool,
                    v_pool,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,
                    average_attn_weights=False,
                )
        finally:
            if was_training:
                self.mha.train()
        return attn_w

    @staticmethod
    def _child_parent_index(
        child_block_size: int,
        parent_block_size: int,
        child_num_blocks: int,
        parent_num_blocks: int,
        device: torch.device,
    ) -> torch.Tensor:
        starts = torch.arange(child_num_blocks, device=device) * child_block_size
        parent_idx = torch.div(
            starts,
            parent_block_size,
            rounding_mode="floor",
        )
        return parent_idx.clamp_max(max(parent_num_blocks - 1, 0))

    def _expand_block_mask_to_tokens(
        self,
        critical_blocks: torch.Tensor,
        q_token_to_block: torch.Tensor,
        k_token_to_block: torch.Tensor,
        tgt_len: int,
        src_len: int,
    ) -> torch.Tensor:
        """
        critical_blocks: (B, Lq_blocks, Lk_blocks) bool True=allow
        returns bool mask (B, Lq, Lk) where True=disallow

        OPTIMIZED: Vectorized version without batch loop.
        """
        # Vectorized indexing across all batches at once - no loop needed
        # critical_blocks is (B, num_q_blocks, num_k_blocks)
        # Index with broadcasting: (B, tgt_len, src_len)
        allowed = critical_blocks[:, q_token_to_block[:, None], k_token_to_block[None, :]]
        block_mask_tokens = ~allowed  # True means mask (disallow)
        return block_mask_tokens

    def compute_pooled_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single-level pooled attention using the finest block size (for backward compatibility).
        """
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
        bsz, tgt_len, _ = query.shape
        _, src_len, _ = key.shape
        key_mask_2d = self._normalize_padding_mask(key_padding_mask, bsz, src_len, "key_padding_mask")
        query_mask_2d = self._normalize_padding_mask(query_padding_mask, bsz, tgt_len, "query_padding_mask")

        block_size = self.block_size
        q_pool_seq, q_pool_mask, _ = self._pool_tokens(query, block_size, query_mask_2d)
        k_pool_seq, k_pool_mask, _ = self._pool_tokens(key, block_size, key_mask_2d)
        v_pool_seq = k_pool_seq

        attn_w = self._run_block_attention(
            pooled_query=q_pool_seq,
            pooled_key=k_pool_seq,
            pooled_value=v_pool_seq,
            key_padding_mask=k_pool_mask,
        )

        self._last_pooled_key_padding_mask = k_pool_mask
        self._last_pooled_query_padding_mask = q_pool_mask
        self._last_pooled_attention = attn_w
        self._last_hierarchy_summary = None
        return attn_w

    def classify_pooled_blocks(
        self,
        pc: Optional[torch.Tensor] = None,
        *,
        level_cfg: Optional[HierarchyLevel] = None,
        allowed_mask: Optional[torch.Tensor] = None,
        key_block_mask: Optional[torch.Tensor] = None,
        query_block_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        SLA-style per-row top-k selection of critical blocks.
        - Aggregates heads (mean or max) to get (B, Lq_blocks, Lk_blocks).
        - For each sample and query block row, selects top-k key blocks.
          k = critical_k if provided else ceil(critical_ratio * Lk_blocks).
        Returns:
            critical_mask: bool tensor of shape (B, Lq_blocks, Lk_blocks)
        """
        cfg = level_cfg or self.hierarchy_levels[-1]
        if pc is None:
            if self._last_pooled_attention is None:
                raise RuntimeError("No pooled attention available. Call compute_pooled_scores first.")
            pc = self._last_pooled_attention  # (B,H,Lq,Lk)
        if key_block_mask is None:
            key_block_mask = self._last_pooled_key_padding_mask
        if query_block_mask is None:
            query_block_mask = self._last_pooled_query_padding_mask

        if pc.dim() != 4:
            raise ValueError("pc must have shape (B, H, Lq_blocks, Lk_blocks)")

        if self.head_aggregation == "mean":
            pc_agg = pc.mean(dim=1)  # (B,Lq,Lk)
        else:
            pc_agg = pc.max(dim=1).values  # (B,Lq,Lk)

        batch_size, num_query_blocks, num_key_blocks = pc_agg.shape
        key_block_mask = self._last_pooled_key_padding_mask
        query_block_mask = self._last_pooled_query_padding_mask
        if pc_agg.dtype.is_floating_point:
            neg_inf = torch.finfo(pc_agg.dtype).min
        else:
            neg_inf = -1e9
        if key_block_mask is not None:
            expanded = key_block_mask[:, None, :].expand(batch_size, num_query_blocks, num_key_blocks)
            pc_agg = pc_agg.masked_fill(expanded, neg_inf)
        if query_block_mask is not None:
            expanded = query_block_mask[:, :, None].expand(batch_size, num_query_blocks, num_key_blocks)
            pc_agg = pc_agg.masked_fill(expanded, neg_inf)

        if num_key_blocks == 0:
            critical_mask = torch.zeros_like(pc_agg, dtype=torch.bool)
            self._last_critical_mask = critical_mask
            return critical_mask

        k_per_row = cfg.resolve_k(num_key_blocks)

        topk_indices = pc_agg.topk(k_per_row, dim=-1).indices  # (B,Lq,k)
        critical_mask = torch.zeros_like(pc_agg, dtype=torch.bool)
        critical_mask.scatter_(-1, topk_indices, True)
        if allowed_mask is not None:
            if allowed_mask.shape != critical_mask.shape:
                raise ValueError("allowed_mask must match the pooled attention shape.")
            pc_agg = pc_agg.masked_fill(~allowed_mask, neg_inf)
        if key_block_mask is not None:
            expanded = key_block_mask[:, None, :].expand(batch_size, num_query_blocks, num_key_blocks)
            critical_mask = critical_mask & (~expanded)
        if query_block_mask is not None:
            expanded = query_block_mask[:, :, None].expand(batch_size, num_query_blocks, num_key_blocks)
            critical_mask = critical_mask & (~expanded)
        if allowed_mask is not None:
            critical_mask = critical_mask & allowed_mask
        self._last_critical_mask = critical_mask
        return critical_mask

    def compute_hierarchical_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_padding_mask: Optional[torch.Tensor] = None,
    ) -> Optional[dict]:
        """
        Runs the multi-scale coarse-to-fine selection pipeline.
        Returns a dict with the final critical mask, block size, and token/block mappings.
        """
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
        bsz, tgt_len, _ = query.shape
        _, src_len, _ = key.shape
        key_mask_2d = self._normalize_padding_mask(key_padding_mask, bsz, src_len, "key_padding_mask")
        query_mask_2d = self._normalize_padding_mask(query_padding_mask, bsz, tgt_len, "query_padding_mask")

        parent_mask: Optional[torch.Tensor] = None
        parent_meta: Optional[dict] = None
        summaries: List[dict] = []

        final_info: Optional[dict] = None
        for level_idx, level_cfg in enumerate(self.hierarchy_levels):
            q_pool_seq, q_pool_mask, q_assign = self._pool_tokens(query, level_cfg.block_size, query_mask_2d)
            k_pool_seq, k_pool_mask, k_assign = self._pool_tokens(key, level_cfg.block_size, key_mask_2d)
            v_pool_seq = k_pool_seq

            attn_w = self._run_block_attention(
                pooled_query=q_pool_seq,
                pooled_key=k_pool_seq,
                pooled_value=v_pool_seq,
                key_padding_mask=k_pool_mask,
            )

            allowed_mask = None
            if parent_mask is not None and parent_meta is not None:
                q_parent_idx = self._child_parent_index(
                    child_block_size=level_cfg.block_size,
                    parent_block_size=parent_meta["block_size"],
                    child_num_blocks=q_pool_seq.shape[1],
                    parent_num_blocks=parent_meta["num_q_blocks"],
                    device=query.device,
                )
                k_parent_idx = self._child_parent_index(
                    child_block_size=level_cfg.block_size,
                    parent_block_size=parent_meta["block_size"],
                    child_num_blocks=k_pool_seq.shape[1],
                    parent_num_blocks=parent_meta["num_k_blocks"],
                    device=key.device,
                )
                allowed_mask = parent_mask[:, q_parent_idx][:, :, k_parent_idx]

            critical_blocks = self.classify_pooled_blocks(
                attn_w,
                level_cfg=level_cfg,
                allowed_mask=allowed_mask,
                key_block_mask=k_pool_mask,
                query_block_mask=q_pool_mask,
            )

            selection_rate = 0.0
            if critical_blocks.numel() > 0:
                selection_rate = float(critical_blocks.float().mean().detach().cpu())
            summaries.append(
                {
                    "level": level_idx,
                    "block_size": level_cfg.block_size,
                    "num_query_blocks": q_pool_seq.shape[1],
                    "num_key_blocks": k_pool_seq.shape[1],
                    "selection_rate": selection_rate,
                }
            )

            parent_mask = critical_blocks
            parent_meta = {
                "block_size": level_cfg.block_size,
                "num_q_blocks": q_pool_seq.shape[1],
                "num_k_blocks": k_pool_seq.shape[1],
            }

            if level_idx == self._hierarchy_depth - 1:
                self._last_pooled_attention = attn_w
                self._last_pooled_key_padding_mask = k_pool_mask
                self._last_pooled_query_padding_mask = q_pool_mask
                final_info = {
                    "critical_blocks": critical_blocks,
                    "block_size": level_cfg.block_size,
                    "q_assign": q_assign,
                    "k_assign": k_assign,
                }

        self._last_critical_mask = parent_mask
        self._last_hierarchy_summary = summaries
        return final_info

    def get_last_pooled_attention(self) -> Optional[torch.Tensor]:
        """
        Returns the most recently computed Pc, or None if compute_pooled_scores hasn't been called yet.
        """
        return self._last_pooled_attention

    def get_last_critical_mask(self) -> Optional[torch.Tensor]:
        """
        Returns the last computed critical block mask (B, Lq_blocks, Lk_blocks),
        or None if classify_pooled_blocks hasn't been called yet.
        """
        return self._last_critical_mask

    def get_last_hierarchy_summary(self) -> Optional[List[dict]]:
        """
        Returns a list of dicts (one per hierarchy level) containing diagnostics such as
        block size, number of blocks, and average selection rate.
        """
        return self._last_hierarchy_summary

    # ==================== Attention Visualization ====================
    
    def enable_attention_capture(self, enable: bool = True) -> None:
        """Enable or disable capturing full attention weights for visualization."""
        self._capture_attention = enable
    
    def get_last_full_attention(self) -> Optional[torch.Tensor]:
        """
        Returns the last captured full attention weights.
        Shape: (B, H, L_q, L_k) where L_q is query length and L_k is key length.
        """
        return self._last_full_attention
    
    
    def visualize_read_attention(
        self,
        attn_weights: Optional[torch.Tensor] = None,
        image_size: int = 64,
        patch_size: int = 4,
        save_path: Optional[str] = None,
        sample_idx: int = 0,
        average_over_latents: bool = True,
        average_over_heads: bool = True,
        latent_indices: Optional[List[int]] = None,
        cmap: str = 'gray',
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Visualize read attention (latents attending to image patches) as spatial heatmaps.
        
        This recreates the visualization from the RIN paper where attention is averaged
        across latents to show which image regions receive the most attention.
        
        Args:
            attn_weights: (B, H, L_q, L_k) attention weights. If None, uses last captured.
            image_size: Original image size (e.g., 64 for 64x64 images)
            patch_size: Patch size used in patchification (e.g., 4)
            save_path: If provided, saves the visualization to this path
            sample_idx: Which sample in the batch to visualize
            average_over_latents: If True, average attention across all latent queries
            average_over_heads: If True, average attention across all heads
            latent_indices: If provided and average_over_latents=False, visualize these latents
            cmap: Colormap for visualization (default: 'gray' for B&W)
            normalize: Whether to normalize attention to [0, 1]
            
        Returns:
            attn_map: (H_img, W_img) or (num_latents, H_img, W_img) attention heatmap
        """
        if attn_weights is None:
            attn_weights = self._last_full_attention
        
        if attn_weights is None:
            raise RuntimeError(
                "No attention weights available. Either pass attn_weights or "
                "call enable_attention_capture(True) before forward pass."
            )
        
        # Get single sample: (H, L_q, L_k)
        attn = attn_weights[sample_idx]
        
        # Average over heads if requested
        if average_over_heads:
            attn = attn.mean(dim=0)  # (L_q, L_k)
        else:
            attn = attn[0]  # Just use first head: (L_q, L_k)
        
        num_patches = (image_size // patch_size) ** 2
        grid_size = image_size // patch_size
        
        # L_k should correspond to image patches
        if attn.shape[-1] != num_patches:
            # Might have padding or different structure, try to handle
            k_len = attn.shape[-1]
            if k_len > num_patches:
                attn = attn[..., :num_patches]  # Truncate padding
            else:
                # Pad to expected size
                pad_size = num_patches - k_len
                attn = F.pad(attn, (0, pad_size))
        
        if average_over_latents:
            # Average across all query (latent) positions: (L_k,) -> (grid, grid)
            attn_spatial = attn.mean(dim=0)  # (L_k,)
            attn_map = attn_spatial.view(grid_size, grid_size)
        else:
            # Keep per-latent attention maps
            if latent_indices is not None:
                attn = attn[latent_indices]  # Select specific latents
            # (num_latents, L_k) -> (num_latents, grid, grid)
            attn_map = attn.view(-1, grid_size, grid_size)
        
        # Move to CPU for visualization
        attn_map = attn_map.detach().cpu()
        
        # Normalize to [0, 1]
        if normalize:
            attn_min = attn_map.min()
            attn_max = attn_map.max()
            attn_map = (attn_map - attn_min) / (attn_max - attn_min + 1e-8)
        
        # Upsample to full image resolution
        if attn_map.dim() == 2:
            attn_map_up = F.interpolate(
                attn_map[None, None],  # (1, 1, grid, grid)
                size=(image_size, image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze()  # (H, W)
        else:
            attn_map_up = F.interpolate(
                attn_map[:, None],  # (N, 1, grid, grid)
                size=(image_size, image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # (N, H, W)
        
        # Save if path provided
        if save_path is not None and HAS_MATPLOTLIB:
            self._save_attention_image(attn_map_up, save_path, cmap)
        
        return attn_map_up
    
    def _save_attention_image(
        self,
        attn_map: torch.Tensor,
        save_path: str,
        cmap: str = 'gray',
    ) -> None:
        """Save attention map(s) as image(s)."""
        if not HAS_MATPLOTLIB:
            print("matplotlib not available, skipping save")
            return
        
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        if attn_map.dim() == 2:
            # Single attention map
            plt.figure(figsize=(4, 4))
            plt.imshow(attn_map.numpy(), cmap=cmap)
            plt.axis('off')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close()
        else:
            # Multiple attention maps (one per latent)
            base, ext = os.path.splitext(save_path)
            for i, amap in enumerate(attn_map):
                plt.figure(figsize=(4, 4))
                plt.imshow(amap.numpy(), cmap=cmap)
                plt.axis('off')
                plt.savefig(f"{base}_latent{i}{ext}", bbox_inches='tight', pad_inches=0, dpi=100)
                plt.close()
    
    def save_attention_grid(
        self,
        attn_weights: Optional[torch.Tensor] = None,
        image_size: int = 64,
        patch_size: int = 4,
        save_dir: str = "./attention_vis",
        prefix: str = "attn",
        sample_idx: int = 0,
        num_latent_samples: int = 5,
        step: Optional[int] = None,
        block_idx: Optional[int] = None,
    ) -> List[str]:
        """
        Save multiple attention visualizations for analysis.
        
        Creates:
        - One averaged attention map (across all latents)
        - Individual attention maps for first `num_latent_samples` latents
        
        Args:
            attn_weights: Attention weights or None to use last captured
            image_size: Image resolution
            patch_size: Patch size
            save_dir: Directory to save images
            prefix: Filename prefix
            sample_idx: Batch sample index
            num_latent_samples: Number of individual latent attention maps to save
            step: Diffusion step (for filename)
            block_idx: Block index (for filename)
            
        Returns:
            List of saved file paths
        """
        os.makedirs(save_dir, exist_ok=True)
        saved_paths = []
        
        step_str = f"_step{step}" if step is not None else ""
        block_str = f"_block{block_idx}" if block_idx is not None else ""
        
        # Save averaged attention
        avg_path = os.path.join(save_dir, f"{prefix}{step_str}{block_str}_avg.png")
        self.visualize_read_attention(
            attn_weights=attn_weights,
            image_size=image_size,
            patch_size=patch_size,
            save_path=avg_path,
            sample_idx=sample_idx,
            average_over_latents=True,
        )
        saved_paths.append(avg_path)
        
        # Save individual latent attention maps
        if attn_weights is None:
            attn_weights = self._last_full_attention
        
        if attn_weights is not None:
            num_latents = attn_weights.shape[2]
            indices = list(range(min(num_latent_samples, num_latents)))
            
            for idx in indices:
                lat_path = os.path.join(save_dir, f"{prefix}{step_str}{block_str}_latent{idx}.png")
                self.visualize_read_attention(
                    attn_weights=attn_weights,
                    image_size=image_size,
                    patch_size=patch_size,
                    save_path=lat_path,
                    sample_idx=sample_idx,
                    average_over_latents=False,
                    latent_indices=[idx],
                )
                saved_paths.append(lat_path)
        
        return saved_paths

    def _phi(self, x: torch.Tensor, kind: str = "elu+1") -> torch.Tensor:
        """
        Feature map for linear attention; returns non-negative features.
        """
        if kind == "elu+1":
            return F.elu(x) + 1.0
        if kind == "relu":
            return F.relu(x)
        if kind == "softmax":
            return F.softmax(x, dim=-1)
        raise ValueError("Unknown phi kind")

    def _reshape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, E) or (L, B, E) depending on batch_first
        returns: (B, H, L, Dh)
        """
        if not self.batch_first:
            x = x.transpose(0, 1)  # (B,L,E)
        bsz, seq_len, embed = x.shape
        h = self.num_heads
        dh = self.head_dim
        x = x.view(bsz, seq_len, h, dh).permute(0, 2, 1, 3).contiguous()
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, L, Dh) -> (B, L, E) or (L, B, E) depending on batch_first
        """
        bsz, h, seq_len, dh = x.shape
        out = x.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, h * dh)
        if not self.batch_first:
            out = out.transpose(0, 1)
        return out

    def compute_linear_attention_output(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
        phi_kind: str = "softmax",
    ) -> torch.Tensor:
        """
        Compute linear attention output:
          H = phi(K)^T V
          Z = rowsum(phi(K)^T)
          O = (phi(Q) H) / (phi(Q) Z)
        Uses input embeddings split into heads (no extra projections).
        """
        qh = self._reshape_to_heads(query)  # (B,H,L,Dh)
        kh = self._reshape_to_heads(key)
        vh = self._reshape_to_heads(value)

        q_phi = self._phi(qh, phi_kind)
        k_phi = self._phi(kh, phi_kind)

        if key_padding_mask is not None:
            # Expect (B, L). True indicates masked (ignore)
            if key_padding_mask.dim() == 2:
                mask = key_padding_mask[:, None, :, None].to(k_phi.dtype)  # (B,1,L,1)
            elif key_padding_mask.dim() == 3:
                # (B, L, 1) or (B, 1, L)
                if key_padding_mask.shape[-1] == 1:
                    mask = key_padding_mask[:, None, :, :].to(k_phi.dtype)
                else:
                    mask = key_padding_mask[:, None, :, None].to(k_phi.dtype)
            else:
                mask = key_padding_mask.to(k_phi.dtype)
            keep = 1.0 - mask
            k_phi = k_phi * keep
            vh = vh * keep

        # H: (B,H,Dh,Dh), Z: (B,H,Dh)
        H = torch.einsum("bhld,bhlm->bhdm", k_phi, vh)
        Z = k_phi.sum(dim=2)  # sum over sequence -> (B,H,Dh)

        # Numerator: (B,H,L,Dh); Denominator: (B,H,L,1)
        num = torch.einsum("bhld,bhdm->bhlm", q_phi, H)
        den = torch.einsum("bhld,bhd->bhl", q_phi, Z).unsqueeze(-1)
        out = num / (den + eps)

        out_merged = self._merge_heads(out)  # (B,L,E) or (L,B,E)
        self._last_linear_output = out_merged
        return out_merged

    def compute_linear_marginal_output(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        critical_blocks: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
        phi_kind: str = "softmax",
        block_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        SLA-style marginal linear attention (vectorized):
          - Precompute global H_sum = sum_j s_j and Z_sum = sum_j z_j over all key blocks
          - For each query block i, subtract critical blocks' s_j/z_j to get s_qi/z_qi
          - For all tokens in query block i: O_l = (phi(Q_block) @ s_qi) / (phi(Q_block) · z_qi)
        Returns tensor shaped like query (B,L,E) or (L,B,E) depending on batch_first.
        """
        # Bring to (B,L,E) for simpler indexing
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        bsz, q_len, _ = query.shape
        _, k_len, _ = key.shape

        # Heads view
        qh = self._reshape_to_heads(query)  # (B,H,L,Dh)
        kh = self._reshape_to_heads(key)
        vh = self._reshape_to_heads(value)

        # Phi features
        q_phi = self._phi(qh, phi_kind)     # (B,H,L,Dh)
        k_phi = self._phi(kh, phi_kind)     # (B,H,L,Dh)

        # Apply key padding mask to K and V
        if key_padding_mask is not None:
            if key_padding_mask.dim() == 2:
                mask = key_padding_mask[:, None, :, None].to(k_phi.dtype)  # (B,1,L,1)
            elif key_padding_mask.dim() == 3:
                if key_padding_mask.shape[-1] == 1:
                    mask = key_padding_mask[:, None, :, :].to(k_phi.dtype)
                else:
                    mask = key_padding_mask[:, None, :, None].to(k_phi.dtype)
            else:
                mask = key_padding_mask.to(k_phi.dtype)
            keep = 1.0 - mask
            k_phi = k_phi * keep
            vh = vh * keep

        # Global sums over all key tokens
        H_sum = torch.einsum("bhld,bhlm->bhdm", k_phi, vh)  # (B,H,Dh,Dh)
        Z_sum = k_phi.sum(dim=2)                            # (B,H,Dh)

        block = self.block_size if block_size is None else block_size
        num_q_blocks = critical_blocks.shape[1]
        num_k_blocks = critical_blocks.shape[2]

        # Pad sequences to match block boundaries
        pad_q_len = num_q_blocks * block - q_len
        pad_k_len = num_k_blocks * block - k_len
        if pad_q_len > 0:
            q_phi = F.pad(q_phi, (0, 0, 0, pad_q_len))
        if pad_k_len > 0:
            k_phi = F.pad(k_phi, (0, 0, 0, pad_k_len))
            vh = F.pad(vh, (0, 0, 0, pad_k_len))

        # Reshape into blocks: (B, H, num_blocks, block, Dh)
        k_phi_blocks = k_phi.view(bsz, self.num_heads, num_k_blocks, block, self.head_dim)
        vh_blocks = vh.view(bsz, self.num_heads, num_k_blocks, block, self.head_dim)
        q_phi_blocks = q_phi.view(bsz, self.num_heads, num_q_blocks, block, self.head_dim)

        # Precompute per-block sums: (B, H, num_k_blocks, Dh, Dh) and (B, H, num_k_blocks, Dh)
        # s_j for each key block j
        s_blocks = torch.einsum("bhnld,bhnlm->bhndm", k_phi_blocks, vh_blocks)  # (B,H,num_k_blocks,Dh,Dh)
        # z_j for each key block j
        z_blocks = k_phi_blocks.sum(dim=3)  # (B,H,num_k_blocks,Dh)

        # For each query block, compute critical contribution by gathering and summing
        # critical_blocks: (B, num_q_blocks, num_k_blocks) - bool mask
        # Expand for broadcasting: (B, 1, num_q_blocks, num_k_blocks, 1, 1) for s_blocks
        crit_mask_s = critical_blocks[:, None, :, :, None, None].float()  # (B,1,num_q,num_k,1,1)
        crit_mask_z = critical_blocks[:, None, :, :, None].float()        # (B,1,num_q,num_k,1)

        # Broadcast s_blocks and z_blocks: add query block dimension
        s_blocks_expanded = s_blocks[:, :, None, :, :, :]  # (B,H,1,num_k,Dh,Dh)
        z_blocks_expanded = z_blocks[:, :, None, :, :]     # (B,H,1,num_k,Dh)

        # Compute critical contributions per query block by summing over key blocks
        s_crit = (s_blocks_expanded * crit_mask_s).sum(dim=3)  # (B,H,num_q,Dh,Dh)
        z_crit = (z_blocks_expanded * crit_mask_z).sum(dim=3)  # (B,H,num_q,Dh)

        # Marginal sums for each query block: H_sum and Z_sum - critical contributions
        # Expand H_sum and Z_sum to have query block dimension
        H_sum_expanded = H_sum[:, :, None, :, :]  # (B,H,1,Dh,Dh)
        Z_sum_expanded = Z_sum[:, :, None, :]     # (B,H,1,Dh)

        s_qi = H_sum_expanded - s_crit  # (B,H,num_q,Dh,Dh)
        z_qi = Z_sum_expanded - z_crit  # (B,H,num_q,Dh)

        # Compute linear attention output for each query block
        # q_phi_blocks: (B,H,num_q,block,Dh)
        # s_qi: (B,H,num_q,Dh,Dh)
        # z_qi: (B,H,num_q,Dh)
        num = torch.einsum("bhnld,bhndm->bhnlm", q_phi_blocks, s_qi)  # (B,H,num_q,block,Dh)
        den = torch.einsum("bhnld,bhnd->bhnl", q_phi_blocks, z_qi).unsqueeze(-1)  # (B,H,num_q,block,1)
        ol_heads = num / (den + eps)  # (B,H,num_q,block,Dh)

        # Reshape back to (B,H,L,Dh)
        ol_heads = ol_heads.view(bsz, self.num_heads, num_q_blocks * block, self.head_dim)
        # Remove padding
        if pad_q_len > 0:
            ol_heads = ol_heads[:, :, :q_len, :]

        # Merge heads back
        ol = self._merge_heads(ol_heads)  # (B,L,E) or (L,B,E)
        return ol

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if self.batch_first:
            bsz, tgt_len, _ = query.shape
            _, src_len, _ = key.shape
        else:
            tgt_len, bsz, _ = query.shape
            src_len, _, _ = key.shape

        # 1) Compute hierarchical pooled attention and critical block masks
        critical_blocks = None
        q_block_assign = None
        k_block_assign = None
        final_block_size = self.block_size
        try:
            hierarchy_info = self.compute_hierarchical_scores(
                query,
                key,
                key_padding_mask=key_padding_mask,
                query_padding_mask=None,
            )
            if hierarchy_info is not None:
                critical_blocks = hierarchy_info["critical_blocks"]
                q_block_assign = hierarchy_info["q_assign"]
                k_block_assign = hierarchy_info["k_assign"]
                final_block_size = hierarchy_info["block_size"]
        except Exception:
            self._last_pooled_attention = None
            self._last_pooled_key_padding_mask = None
            self._last_pooled_query_padding_mask = None
            critical_blocks = None
            self._last_hierarchy_summary = None

        # 2) Expand block-level critical mask to token-level mask (B, Lq, Lk)
        block_mask_tokens: Optional[torch.Tensor] = None
        if critical_blocks is not None and q_block_assign is not None and k_block_assign is not None:
            block_mask_tokens = self._expand_block_mask_to_tokens(
                critical_blocks=critical_blocks,
                q_token_to_block=q_block_assign,
                k_token_to_block=k_block_assign,
                tgt_len=tgt_len,
                src_len=src_len,
            )

        # 3) Build attention mask from block mask and key padding mask, expand across heads
        combined_mask: Optional[torch.Tensor] = block_mask_tokens

        if key_padding_mask is not None:
            key_mask = key_padding_mask
            if key_mask.dim() == 3:
                if key_mask.shape[-1] == 1:
                    key_mask = key_mask.squeeze(-1)
                elif key_mask.shape[1] == 1:
                    key_mask = key_mask.squeeze(1)
            if key_mask.dim() != 2:
                raise ValueError("key_padding_mask must be broadcastable to (B, Lk)")
            key_mask = key_mask.to(torch.bool)
            if key_mask.shape[0] != bsz or key_mask.shape[1] != src_len:
                raise ValueError("key_padding_mask shape must match key (B, Lk)")
            key_mask_expanded = key_mask[:, None, :].expand(-1, tgt_len, -1)
            combined_mask = key_mask_expanded if combined_mask is None else (combined_mask | key_mask_expanded)

        final_attn_mask: Optional[torch.Tensor] = None
        if combined_mask is not None:
            if combined_mask.dtype != torch.bool:
                combined_mask = combined_mask != 0
            final_attn_mask = combined_mask.repeat_interleave(self.num_heads, dim=0)

        # 4) Sparse exact output via MHA with block mask
        # Request weights if capture is enabled (for visualization)
        request_weights = need_weights or self._capture_attention
        o_s, attn = self.mha(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            need_weights=request_weights,
            attn_mask=final_attn_mask,
            average_attn_weights=False,  # Get per-head weights for visualization
            is_causal=is_causal,
        )
        
        # 5) Store attention weights for visualization if enabled
        if self._capture_attention and attn is not None:
            self._last_full_attention = attn.detach()  # (B, H, L_q, L_k)
        
        # Average weights if originally requested
        if need_weights and average_attn_weights and attn is not None:
            attn = attn.mean(dim=1)  # (B, L_q, L_k)
        # 6) Linear marginal output excluding critical blocks and combine
        o_l = None
        if critical_blocks is not None:
            try:
                o_l = self.compute_linear_marginal_output(
                    query=query,
                    key=key,
                    value=value,
                    critical_blocks=critical_blocks,
                    key_padding_mask=key_padding_mask,
                    phi_kind="softmax",
                    block_size=final_block_size,
                )
            except Exception:
                o_l = None
        out = o_s if o_l is None else o_s + o_l
        return out, attn if need_weights else (out, None)


class AttentionVisualizer:
    """
    Utility class to capture and visualize attention from all HierarchicalSparseAttention
    layers in a model during the diffusion reverse process.
    
    Usage:
        visualizer = AttentionVisualizer(model, image_size=64, patch_size=4)
        visualizer.enable()
        
        # Run your diffusion sampling loop
        for step in range(num_steps):
            output = model(...)
            if step in [5, 20, 60, 120, 200, 500]:
                visualizer.save_all_attention(
                    save_dir="./attention_vis",
                    step=step,
                )
        
        visualizer.disable()
    """
    
    def __init__(
        self,
        model: nn.Module,
        image_size: int = 64,
        patch_size: int = 4,
        layer_names: Optional[List[str]] = None,
    ):
        """
        Args:
            model: The model containing HierarchicalSparseAttention layers
            image_size: Image resolution for visualization
            patch_size: Patch size used in the model
            layer_names: Optional list of layer names to capture (captures all if None)
        """
        self.model = model
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_names = layer_names
        self.attention_layers: List[Tuple[str, HierarchicalSparseAttention]] = []
        
        # Find all HierarchicalSparseAttention layers
        self._find_attention_layers()
    
    def _find_attention_layers(self) -> None:
        """Find all HierarchicalSparseAttention layers in the model."""
        self.attention_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, HierarchicalSparseAttention):
                if self.layer_names is None or name in self.layer_names:
                    self.attention_layers.append((name, module))
        
        if not self.attention_layers:
            print("Warning: No HierarchicalSparseAttention layers found in model")
    
    def enable(self) -> None:
        """Enable attention capture on all tracked layers."""
        for name, layer in self.attention_layers:
            layer.enable_attention_capture(True)
    
    def disable(self) -> None:
        """Disable attention capture on all tracked layers."""
        for name, layer in self.attention_layers:
            layer.enable_attention_capture(False)
    
    def get_all_attention(self) -> dict:
        """
        Get attention weights from all tracked layers.
        
        Returns:
            Dict mapping layer names to attention tensors
        """
        return {
            name: layer.get_last_full_attention()
            for name, layer in self.attention_layers
        }
    
    def save_all_attention(
        self,
        save_dir: str = "./attention_vis",
        step: Optional[int] = None,
        sample_idx: int = 0,
        num_latent_samples: int = 5,
    ) -> List[str]:
        """
        Save attention visualizations from all tracked layers.
        
        Args:
            save_dir: Directory to save images
            step: Diffusion step number (for filenames)
            sample_idx: Batch sample index
            num_latent_samples: Number of individual latent maps to save
            
        Returns:
            List of all saved file paths
        """
        all_paths = []
        
        for block_idx, (name, layer) in enumerate(self.attention_layers):
            attn = layer.get_last_full_attention()
            if attn is not None:
                # Clean layer name for filename
                clean_name = name.replace('.', '_').replace('/', '_')
                paths = layer.save_attention_grid(
                    attn_weights=attn,
                    image_size=self.image_size,
                    patch_size=self.patch_size,
                    save_dir=save_dir,
                    prefix=f"read_{clean_name}",
                    sample_idx=sample_idx,
                    num_latent_samples=num_latent_samples,
                    step=step,
                    block_idx=block_idx,
                )
                all_paths.extend(paths)
        
        return all_paths
    
    def visualize_single_step(
        self,
        step: int,
        save_dir: str = "./attention_vis",
        sample_idx: int = 0,
        create_grid: bool = True,
    ) -> Optional[torch.Tensor]:
        """
        Create a single visualization combining all blocks at one step.
        
        Args:
            step: Diffusion step number
            save_dir: Directory to save images
            sample_idx: Batch sample index
            create_grid: If True and matplotlib available, create a combined grid image
            
        Returns:
            Stacked attention maps tensor (num_blocks, H, W)
        """
        attention_maps = []
        
        for name, layer in self.attention_layers:
            attn = layer.get_last_full_attention()
            if attn is not None:
                attn_map = layer.visualize_read_attention(
                    attn_weights=attn,
                    image_size=self.image_size,
                    patch_size=self.patch_size,
                    sample_idx=sample_idx,
                    average_over_latents=True,
                )
                attention_maps.append(attn_map)
        
        if not attention_maps:
            return None
        
        stacked = torch.stack(attention_maps, dim=0)  # (num_blocks, H, W)
        
        if create_grid and HAS_MATPLOTLIB:
            os.makedirs(save_dir, exist_ok=True)
            num_blocks = len(attention_maps)
            fig, axes = plt.subplots(1, num_blocks, figsize=(4 * num_blocks, 4))
            if num_blocks == 1:
                axes = [axes]
            
            for i, (ax, amap) in enumerate(zip(axes, attention_maps)):
                ax.imshow(amap.numpy(), cmap='gray')
                ax.axis('off')
                ax.set_title(f'Block {i}')
            
            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, f"attention_grid_step{step}.png"),
                bbox_inches='tight',
                dpi=150
            )
            plt.close()
        
        return stacked
