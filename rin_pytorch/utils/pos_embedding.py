import torch


def positional_encoding(
    coords: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Generate sinusoidal positional encodings for arbitrary coordinate values.
    
    Args:
        coords: Coordinate values, shape (batch_size,) or (seqlen,)
        dim: Output embedding dimension
    
    Returns:
        Positional encodings of shape (len(coords), dim)
    """
    # coords: (N,) -> (N, 1) for broadcasting
    coords_expanded = coords.unsqueeze(-1)  # (N, 1)
    dim_indices = torch.arange(dim, device=coords.device).unsqueeze(0)  # (1, dim)
    
    angle_rates = 1 / torch.pow(10000.0, 2 * (dim_indices // 2) / dim)
    angle_rads = coords_expanded.float() * angle_rates.float()  # (N, dim)

    # apply sin to even indices; 2i
    angle_rads_sin = torch.sin(angle_rads[..., 0::2])
    # apply cos to odd indices; 2i+1
    angle_rads_cos = torch.cos(angle_rads[..., 1::2])

    pos_encoding = torch.cat([angle_rads_sin, angle_rads_cos], -1)
    return pos_encoding.float()  # (N, dim)


def get_1d_position_codes(
    seqlen: int,
    out_dim: int,
    normalization_max=6.2831852,
    reference_len: int | None = None,
) -> torch.Tensor:
    """Generate 1D sinusoidal position codes.
    
    Args:
        seqlen: Length of the sequence
        out_dim: Output dimension for position embeddings
        normalization_max: Max value for coordinate normalization
        reference_len: If provided, normalize by this length instead of seqlen.
                       Use this for consistent embeddings across variable sizes.
    
    Returns:
        Position embeddings of shape (seqlen, out_dim)
    """
    coords = torch.arange(seqlen, dtype=torch.float32)
    if normalization_max is not None:
        # Use reference_len if provided, otherwise use seqlen
        norm_len = (reference_len - 1) if reference_len is not None else (seqlen - 1)
        norm_len = max(norm_len, 1)  # Avoid division by zero
        coords = coords / norm_len * normalization_max
    
    # coords: (seqlen,) -> (seqlen, 1) for broadcasting
    coords_expanded = coords.unsqueeze(-1)  # (seqlen, 1)
    dim_indices = torch.arange(out_dim, device=coords.device).unsqueeze(0)  # (1, out_dim)
    
    angle_rates = 1 / torch.pow(10000.0, 2 * (dim_indices // 2) / out_dim)
    angle_rads = coords_expanded.float() * angle_rates.float()  # (seqlen, out_dim)

    # apply sin to even indices in the array; 2i
    angle_rads_sin = torch.sin(angle_rads[..., 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads_cos = torch.cos(angle_rads[..., 1::2])

    pos_encoding = torch.cat([angle_rads_sin, angle_rads_cos], -1)
    return pos_encoding.float()  # (seqlen, out_dim)


def get_2d_position_codes(
    height: int,
    width: int,
    out_dim: int,
    normalization_max=6.2831852,
    reference_height: int | None = None,
    reference_width: int | None = None,
) -> torch.Tensor:
    """Generate 2D sinusoidal position codes.
    
    The first half of the embedding dimensions encode the y (height) position,
    and the second half encode the x (width) position.
    
    Args:
        height: Number of rows
        width: Number of columns
        out_dim: Output dimension (must be even)
        normalization_max: Max value for coordinate normalization
        reference_height, reference_width: If provided, normalize by these instead.
                                           Use for consistent embeddings across variable sizes.
    
    Returns:
        Position embeddings of shape (height, width, out_dim)
    """
    # get_1d_position_codes returns (seqlen, dim)
    y_coords = get_1d_position_codes(height, out_dim // 2, normalization_max, reference_len=reference_height)
    y_coords = y_coords.unsqueeze(1)  # (height, 1, out_dim//2)
    y_coords = torch.cat([y_coords, torch.zeros_like(y_coords)], -1)  # (height, 1, out_dim)

    x_coords = get_1d_position_codes(width, out_dim // 2, normalization_max, reference_len=reference_width)
    x_coords = x_coords.unsqueeze(0)  # (1, width, out_dim//2)
    x_coords = torch.cat([torch.zeros_like(x_coords), x_coords], -1)  # (1, width, out_dim)

    return y_coords + x_coords  # (height, width, out_dim)


def create_2d_sin_cos_pos_emb(
    n_rows: int,
    n_cols: int,
    dim: int,
    normalization_max=6.2831852,
    reference_rows: int | None = None,
    reference_cols: int | None = None,
) -> torch.Tensor:
    """Create 2D sinusoidal positional embeddings for patch sequences.
    
    Args:
        n_rows: Number of patch rows
        n_cols: Number of patch columns
        dim: Embedding dimension
        normalization_max: Max value for coordinate normalization
        reference_rows, reference_cols: If provided, normalize positions by these values
                                        instead of n_rows/n_cols. This ensures consistent
                                        embeddings across different image sizes.
    
    Returns:
        Position embeddings of shape (n_rows * n_cols, dim)
    """
    if n_rows == 1 or n_cols == 1:
        ref_len = None
        if reference_rows is not None and reference_cols is not None:
            ref_len = reference_rows * reference_cols
        sin_cos = get_1d_position_codes(n_rows * n_cols, dim, normalization_max=normalization_max, reference_len=ref_len)
    else:
        sin_cos = get_2d_position_codes(
            n_rows, n_cols, dim, normalization_max=normalization_max,
            reference_height=reference_rows, reference_width=reference_cols
        )
    vis_pos_emb = sin_cos.view(n_rows * n_cols, dim)

    return vis_pos_emb
