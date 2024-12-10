import torch

def create_padding_mask(
    batch_size: int,
    seq_len: int,
    max_len: int,
    device: torch.device
) -> torch.Tensor:
    """Creates padding mask for tokens beyond seq_len
    
    Args:
        batch_size: Batch size
        seq_len: Number of valid tokens
        max_len: Maximum sequence length 
        device: Device to create tensor on
        
    Returns:
        Tensor of shape (batch_size, max_len) where False indicates valid tokens
        and True indicates padding tokens
        
    Example:
        >>> mask = create_padding_mask(2, 3, 5, 'cuda')
        >>> print(mask)
        tensor([[False, False, False,  True,  True],
                [False, False, False,  True,  True]], device='cuda')
    """
    # Create mask of shape [batch_size, max_len] initialized with True (padding)
    mask = torch.ones((batch_size, max_len), device=device, dtype=torch.bool)
    
    # Set False for valid sequence positions
    mask[:, :seq_len] = False
    
    return mask