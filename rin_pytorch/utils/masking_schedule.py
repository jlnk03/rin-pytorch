"""
Token masking/dropping schedules for training.

Provides constant, linear, sigmoid, and cosine masking rate schedules.
Uses string format: "schedule_name@param1,param2,..."

Format examples:
    - "constant@0.5"            -> constant 50% masking
    - "linear@0.7,0.1"          -> linear decay from 70% to 10%
    - "sigmoid@0.7,0.1"         -> sigmoid decay 70%->10% (default steepness=10)
    - "sigmoid@0.7,0.1,5"       -> sigmoid with steepness=5 (smoother)
    - "sigmoid@0.7,0.1,20"      -> sigmoid with steepness=20 (sharper)
    - "cosine@0.6,0.1"          -> cosine decay from 60% to 10%
    - "none" or "disabled"      -> no masking

Sigmoid formula:
    mask_ratio = min + (max - min) / (1 + exp(steepness * (t - 0.5)))
"""

import math
from typing import Callable


def parse_masking_schedule(
    schedule_str: str,
    warmup_steps: int = 0,
    total_steps: int = 1,
) -> Callable[[int], float]:
    """
    Parse a masking schedule string and return a schedule function.

    Args:
        schedule_str: Schedule string in format "name@param1,param2,..."
            - "constant@<max_ratio>"
            - "linear@<max_ratio>,<min_ratio>"
            - "sigmoid@<max_ratio>,<min_ratio>[,<steepness>]"  (steepness: higher=sharper, default=10)
            - "cosine@<max_ratio>,<min_ratio>"
            - "none" or "disabled"
        warmup_steps: Number of warmup steps (linear increase from min to max)
        total_steps: Total training steps

    Returns:
        A function that takes the current step and returns the mask ratio.
    """
    schedule_str = schedule_str.strip().lower()

    if schedule_str in ("none", "disabled", ""):
        return _constant_schedule(0.0)

    if "@" in schedule_str:
        name, params_str = schedule_str.split("@", 1)
        params = [float(p.strip()) for p in params_str.split(",")]
    else:
        name = schedule_str
        params = []

    if name == "constant":
        max_ratio = params[0] if params else 0.5
        return _constant_schedule(max_ratio)

    elif name == "linear":
        max_ratio = params[0] if len(params) > 0 else 0.5
        min_ratio = params[1] if len(params) > 1 else 0.0
        return _linear_schedule(max_ratio, min_ratio, warmup_steps, total_steps)

    elif name == "sigmoid":
        max_ratio = params[0] if len(params) > 0 else 0.5
        min_ratio = params[1] if len(params) > 1 else 0.0
        steepness = params[2] if len(params) > 2 else 10.0
        return _sigmoid_schedule(max_ratio, min_ratio, warmup_steps, total_steps, steepness)

    elif name == "cosine":
        max_ratio = params[0] if len(params) > 0 else 0.5
        min_ratio = params[1] if len(params) > 1 else 0.0
        return _cosine_schedule(max_ratio, min_ratio, warmup_steps, total_steps)

    else:
        raise ValueError(
            f"Unknown masking schedule: {schedule_str}. "
            "Use: constant@<r>, linear@<max>,<min>, sigmoid@<max>,<min>[,<steepness>], "
            "cosine@<max>,<min>, or none"
        )


def get_masking_schedule(
    schedule_name: str,
    max_mask_ratio: float = 0.5,
    min_mask_ratio: float = 0.0,
    warmup_steps: int = 0,
    total_steps: int = 1,
) -> Callable[[int], float]:
    """
    Legacy function for backwards compatibility.
    Returns a masking schedule function that maps step -> mask_ratio.

    Args:
        schedule_name: One of "constant", "linear", "sigmoid", "cosine", "none"
        max_mask_ratio: Maximum masking ratio (0.0 to 1.0)
        min_mask_ratio: Minimum masking ratio (0.0 to 1.0)
        warmup_steps: Number of warmup steps (linear increase from min to max)
        total_steps: Total training steps

    Returns:
        A function that takes the current step and returns the mask ratio.
    """
    schedule_name = schedule_name.lower()

    if schedule_name == "constant":
        return _constant_schedule(max_mask_ratio)
    elif schedule_name == "linear":
        return _linear_schedule(max_mask_ratio, min_mask_ratio, warmup_steps, total_steps)
    elif schedule_name == "sigmoid":
        return _sigmoid_schedule(max_mask_ratio, min_mask_ratio, warmup_steps, total_steps)
    elif schedule_name == "cosine":
        return _cosine_schedule(max_mask_ratio, min_mask_ratio, warmup_steps, total_steps)
    elif schedule_name in ("none", "disabled"):
        return _constant_schedule(0.0)
    else:
        raise ValueError(f"Unknown masking schedule: {schedule_name}. Choose from: constant, linear, sigmoid, cosine, none")


def _constant_schedule(mask_ratio: float) -> Callable[[int], float]:
    """Constant masking ratio throughout training."""
    def schedule(step: int) -> float:
        return mask_ratio
    return schedule


def _linear_schedule(
    max_mask_ratio: float,
    min_mask_ratio: float,
    warmup_steps: int,
    total_steps: int,
) -> Callable[[int], float]:
    """
    Linear schedule:
    - Warmup phase: linearly increase from min to max
    - Main phase: linearly decrease from max to min
    """
    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            # Linear warmup from min to max
            progress = step / warmup_steps
            return min_mask_ratio + (max_mask_ratio - min_mask_ratio) * progress
        else:
            # Linear decay from max to min
            remaining_steps = total_steps - warmup_steps
            if remaining_steps <= 0:
                return max_mask_ratio
            progress = (step - warmup_steps) / remaining_steps
            progress = min(1.0, progress)
            return max_mask_ratio - (max_mask_ratio - min_mask_ratio) * progress
    return schedule


def _sigmoid_schedule(
    max_mask_ratio: float,
    min_mask_ratio: float,
    warmup_steps: int,
    total_steps: int,
    steepness: float = 10.0,
) -> Callable[[int], float]:
    """
    Sigmoid schedule:
    - Warmup phase: linearly increase from min to max
    - Main phase: sigmoid decay from max to min
    
    Formula:
        mask_ratio = min + (max - min) / (1 + exp(steepness * (t - 0.5)))
    
    Args:
        steepness: Controls transition sharpness (default 10.0)
                   Higher = sharper transition, lower = smoother
    """
    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            # Linear warmup from min to max
            progress = step / warmup_steps
            return min_mask_ratio + (max_mask_ratio - min_mask_ratio) * progress
        else:
            # Sigmoid decay from max to min
            remaining_steps = total_steps - warmup_steps
            if remaining_steps <= 0:
                return max_mask_ratio
            t = (step - warmup_steps) / remaining_steps
            t = min(1.0, t)
            
            # Simple centered sigmoid: min + (max - min) / (1 + exp(steepness * (t - 0.5)))
            return min_mask_ratio + (max_mask_ratio - min_mask_ratio) / (1.0 + math.exp(steepness * (t - 0.5)))
    return schedule


def _cosine_schedule(
    max_mask_ratio: float,
    min_mask_ratio: float,
    warmup_steps: int,
    total_steps: int,
) -> Callable[[int], float]:
    """
    Cosine schedule:
    - Warmup phase: linearly increase from min to max
    - Main phase: cosine decay from max to min
    """
    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            # Linear warmup from min to max
            progress = step / warmup_steps
            return min_mask_ratio + (max_mask_ratio - min_mask_ratio) * progress
        else:
            # Cosine decay from max to min
            remaining_steps = total_steps - warmup_steps
            if remaining_steps <= 0:
                return max_mask_ratio
            progress = (step - warmup_steps) / remaining_steps
            progress = min(1.0, progress)
            # Cosine decay: starts at max, ends at min
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_mask_ratio + (max_mask_ratio - min_mask_ratio) * cosine_decay
    return schedule


def apply_token_masking(
    tokens: "torch.Tensor",
    mask_ratio: float,
    existing_mask: "torch.Tensor | None" = None,
    return_indices: bool = False,
) -> "tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """
    Apply random token masking/dropping to input tokens.

    Args:
        tokens: Input tokens of shape (batch, seq_len, dim)
        mask_ratio: Ratio of tokens to mask (0.0 to 1.0)
        existing_mask: Optional existing padding mask (True = masked/padded)
        return_indices: If True, return indices of kept tokens

    Returns:
        - masked_tokens: Tokens with masked entries zeroed out
        - combined_mask: Combined mask (True = masked, either by padding or random drop)
        - kept_indices (optional): Indices of kept tokens per batch item
    """
    import torch

    batch_size, seq_len, dim = tokens.shape
    device = tokens.device

    if mask_ratio <= 0.0:
        if existing_mask is None:
            combined_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        else:
            combined_mask = existing_mask
        if return_indices:
            kept_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            return tokens, combined_mask, kept_indices
        return tokens, combined_mask

    # Determine which tokens are valid (not already padded)
    if existing_mask is not None:
        valid_mask = ~existing_mask  # True = valid token
    else:
        valid_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    # Count valid tokens per batch item
    valid_counts = valid_mask.sum(dim=1)

    # Generate random mask for valid tokens
    random_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    for i in range(batch_size):
        n_valid = valid_counts[i].item()
        if n_valid > 0:
            n_mask = int(n_valid * mask_ratio)
            if n_mask > 0:
                valid_indices = valid_mask[i].nonzero(as_tuple=True)[0]
                perm = torch.randperm(n_valid, device=device)[:n_mask]
                mask_indices = valid_indices[perm]
                random_mask[i, mask_indices] = True

    # Combine with existing mask
    if existing_mask is not None:
        combined_mask = existing_mask | random_mask
    else:
        combined_mask = random_mask

    # DON'T zero out masked tokens - just return the mask
    # The mask will be used in attention (to ignore these tokens) and loss (to not penalize them)
    # Zeroing them out would make the model learn to associate certain values with "black"
    masked_tokens = tokens

    if return_indices:
        kept_mask = ~combined_mask
        max_kept = kept_mask.sum(dim=1).max().item()
        kept_indices = torch.zeros(batch_size, max_kept, dtype=torch.long, device=device)
        for i in range(batch_size):
            indices = kept_mask[i].nonzero(as_tuple=True)[0]
            kept_indices[i, :len(indices)] = indices
        return masked_tokens, combined_mask, kept_indices

    return masked_tokens, combined_mask
