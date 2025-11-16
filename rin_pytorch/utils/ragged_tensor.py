import torch
from typing import List, Tuple, Optional


def ragged_list_to_tensor(
    tensors: List[torch.Tensor],
    dim: int = 0,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a ragged list of tensors to a single concatenated tensor with offsets.
    
    A ragged tensor is represented as:
    1. A concatenated tensor containing all the data
    2. An offsets tensor indicating where each sample starts in the concatenated tensor
    
    Args:
        tensors: List of tensors to concatenate. All tensors must have the same dtype,
                device, and number of dimensions.
        dim: Dimension along which to concatenate (default: 0)
        dtype: Optional dtype to cast the result to
        device: Optional device to move the result to
        
    Returns:
        Tuple of (concatenated_tensor, offsets):
        - concatenated_tensor: Single tensor with all input tensors concatenated along dim
        - offsets: Int64 tensor of length (len(tensors) + 1) where offsets[i] indicates
                  the start position of tensors[i] in the concatenated tensor.
                  The last element is the total length.
                  
    Example:
        >>> t1 = torch.tensor([1, 2])
        >>> t2 = torch.tensor([3, 4, 5]) 
        >>> t3 = torch.tensor([6])
        >>> values, offsets = ragged_list_to_tensor([t1, t2, t3])
        >>> print(values)  # tensor([1, 2, 3, 4, 5, 6])
        >>> print(offsets) # tensor([0, 2, 5, 6])
        
        # To reconstruct t2: values[offsets[1]:offsets[2]] = tensor([3, 4, 5])
    """
    if len(tensors) == 0:
        raise ValueError("Cannot construct a ragged tensor from an empty tensor list")
    
    # Validate input tensors
    if not all(t.dtype == tensors[0].dtype for t in tensors):
        raise ValueError("All tensors must have the same dtype")
    
    if not all(t.device == tensors[0].device for t in tensors):
        raise ValueError("All tensors must be on the same device") 
        
    if not all(t.dim() == tensors[0].dim() for t in tensors):
        raise ValueError("All tensors must have the same number of dimensions")
        
    if tensors[0].dim() == 0:
        raise ValueError("Cannot construct a ragged tensor from zero-dimensional tensors")
        
    # Validate dimension
    if dim < 0:
        dim = tensors[0].dim() + dim
    if dim >= tensors[0].dim():
        raise ValueError(f"Dimension {dim} is out of range for tensors with {tensors[0].dim()} dimensions")
    
    # Concatenate all tensors along the specified dimension
    concatenated = torch.cat(tensors, dim=dim)
    
    # Apply dtype/device conversions if specified
    to_kwargs = {}
    if device is not None:
        to_kwargs["device"] = device
    if dtype is not None:
        to_kwargs["dtype"] = dtype
    if to_kwargs:
        concatenated = concatenated.to(**to_kwargs)
    
    # Calculate offsets - cumulative sizes along the ragged dimension
    sizes = [t.shape[dim] for t in tensors]
    
    # Create offsets tensor: [0, size[0], size[0]+size[1], ..., total_size]
    offsets = torch.cat([
        torch.zeros(1, dtype=torch.int64, device=concatenated.device),
        torch.tensor(sizes, dtype=torch.int64, device=concatenated.device).cumsum(dim=0)
    ])
    
    return concatenated, offsets


def ragged_tensor_to_list(
    values: torch.Tensor, 
    offsets: torch.Tensor,
    dim: int = 0
) -> List[torch.Tensor]:
    """
    Converts a ragged tensor (values + offsets) back to a list of tensors.
    
    Args:
        values: Concatenated tensor containing all the data
        offsets: Int64 tensor indicating start positions of each sample
        dim: Dimension along which the tensor was originally concatenated (default: 0)
        
    Returns:
        List of tensors reconstructed from the ragged representation
        
    Example:
        >>> values = torch.tensor([1, 2, 3, 4, 5, 6])
        >>> offsets = torch.tensor([0, 2, 5, 6])
        >>> tensors = ragged_tensor_to_list(values, offsets)
        >>> # Returns [tensor([1, 2]), tensor([3, 4, 5]), tensor([6])]
    """
    if len(offsets) < 2:
        raise ValueError("Offsets tensor must have at least 2 elements")
        
    if dim < 0:
        dim = values.dim() + dim
    if dim >= values.dim():
        raise ValueError(f"Dimension {dim} is out of range for tensor with {values.dim()} dimensions")
    
    tensors = []
    for i in range(len(offsets) - 1):
        start_idx = offsets[i].item()
        end_idx = offsets[i + 1].item()
        
        # Create slice objects for all dimensions
        slices = [slice(None)] * values.dim()
        slices[dim] = slice(start_idx, end_idx)
        
        tensor_slice = values[tuple(slices)]
        tensors.append(tensor_slice)
    
    return tensors


def get_ragged_lengths(offsets: torch.Tensor) -> torch.Tensor:
    """
    Compute the lengths of each sample from offsets.
    
    Args:
        offsets: Offsets tensor from ragged representation
        
    Returns:
        Tensor containing the length of each sample
        
    Example:
        >>> offsets = torch.tensor([0, 2, 5, 6])
        >>> lengths = get_ragged_lengths(offsets)
        >>> print(lengths)  # tensor([2, 3, 1])
    """
    return offsets[1:] - offsets[:-1]


def get_ragged_max_length(offsets: torch.Tensor) -> int:
    """Get the maximum sequence length from offsets."""
    lengths = get_ragged_lengths(offsets)
    return int(torch.max(lengths).item())


def get_ragged_min_length(offsets: torch.Tensor) -> int:
    """Get the minimum sequence length from offsets."""
    lengths = get_ragged_lengths(offsets)
    return int(torch.min(lengths).item())


def get_document_ids(offsets: torch.Tensor) -> torch.Tensor:
    """
    Calculate the document ID that each token belongs to in the flattened representation.
    
    This is useful for attention masking where tokens should only attend to other
    tokens within the same document/sequence.
    
    Args:
        offsets: Offsets tensor from ragged representation of shape [num_docs + 1]
        
    Returns:
        Document IDs tensor of shape [total_tokens] where each element indicates
        which document that token belongs to.
        
    Example:
        >>> offsets = torch.tensor([0, 3, 5, 11])  # sequences of lengths [3, 2, 6]
        >>> doc_ids = get_document_ids(offsets)
        >>> print(doc_ids)  # tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2])
        
        This means:
        - Tokens 0, 1, 2 belong to document 0
        - Tokens 3, 4 belong to document 1  
        - Tokens 5, 6, 7, 8, 9, 10 belong to document 2
    """
    if len(offsets) < 2:
        raise ValueError("Offsets tensor must have at least 2 elements")
    
    # Get the lengths of each sequence
    lengths = get_ragged_lengths(offsets)
    
    # Create document IDs by repeating each document index by its length
    # torch.arange creates [0, 1, 2, ...] for each document
    # repeat_interleave repeats each ID by the corresponding sequence length
    document_ids = torch.repeat_interleave(
        torch.arange(len(lengths), dtype=torch.int64, device=offsets.device),
        lengths
    )
    
    return document_ids 


def ragged_to_padded_tensor(
    values: torch.Tensor,
    offsets: torch.Tensor,
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a ragged tensor (values + offsets) into a padded dense tensor and mask.

    Args:
        values: Concatenated tensor containing all elements stacked along the first dimension.
        offsets: Offsets tensor describing where each sequence begins/ends.
        pad_value: Value used to fill padded positions (default: 0.0).

    Returns:
        padded: Tensor of shape [batch, max_seq_len, *feature_dims]
        mask: Boolean tensor of shape [batch, max_seq_len] where True marks padding.
    """
    if values.dim() < 1:
        raise ValueError("values must have at least one dimension.")
    if offsets.numel() < 2:
        raise ValueError("offsets must contain at least two entries.")

    lengths = get_ragged_lengths(offsets)
    batch = int(lengths.numel())
    if batch == 0:
        raise ValueError("Cannot pad ragged tensor with zero sequences.")

    max_len = int(lengths.max().item())
    padded_shape = (batch, max_len, *values.shape[1:])
    padded = values.new_full(padded_shape, pad_value)
    mask = torch.ones((batch, max_len), dtype=torch.bool, device=values.device)

    for idx in range(batch):
        start = int(offsets[idx].item())
        end = int(offsets[idx + 1].item())
        length = end - start
        if length <= 0:
            continue
        padded[idx, :length] = values[start:end]
        mask[idx, :length] = False

    return padded, mask