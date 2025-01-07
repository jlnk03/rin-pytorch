import torch
import torch.nn.functional as F

def downsample_mask(mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Downsamples the mask to match the spatial dimensions of the tape.
    
    Args:
        mask: Binary mask of shape (batch_size, height, width).
        patch_size: Size of the patch used in tape calculation.
        
    Returns:
        Downsampled binary mask of shape (batch_size, height // patch_size, width // patch_size).
    """
    # Ensure mask is float for pooling, then back to binary
    mask = mask.float()
    pooled_mask = F.avg_pool2d(mask, kernel_size=patch_size, stride=patch_size)
    # Convert pooled mask to binary (1 if any value > 0, else 0)
    downsampled_mask = (pooled_mask > 0).int()
    return downsampled_mask