import torch
from einops import rearrange


def get_angles(
    pos: torch.Tensor,
    i: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    angle_rates = 1 / torch.pow(10000.0, 2 * (i // 2) / dim)
    return pos.float() * angle_rates.float()


def positional_encoding(
    coords: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    angle_rads = get_angles(
        rearrange(coords, "b -> b 1"),
        rearrange(torch.arange(dim, device=coords.device), "d -> 1 1 d"),
        dim,
    )

    # apply sin to even indices in the array; 2i
    angle_rads1 = torch.sin(angle_rads[..., 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads2 = torch.cos(angle_rads[..., 1::2])

    pos_encoding = torch.cat([angle_rads1, angle_rads2], -1)

    return pos_encoding.float()


def get_1d_position_codes(
    seqlen: int,
    out_dim: int,
    normalization_max=6.2831852,
    reference_len: int | None = None,
) -> torch.Tensor:
    """
    Generate 1D sinusoidal position codes.
    
    Args:
        seqlen: Number of positions to generate
        out_dim: Output dimension
        normalization_max: Max value for position normalization (default 2*pi)
        reference_len: If provided, normalize by this length instead of seqlen.
                       Use this for consistent embeddings across variable sizes.
    """
    coords = torch.arange(seqlen, dtype=torch.float32)
    if normalization_max is not None:
        # Use reference_len if provided, otherwise use seqlen
        norm_len = (reference_len - 1) if reference_len is not None else (seqlen - 1)
        norm_len = max(norm_len, 1)  # Avoid division by zero
        coords = coords / norm_len * normalization_max
    coords = positional_encoding(coords, out_dim)
    return coords


def get_2d_position_codes(
    height: int,
    width: int,
    out_dim: int,
    normalization_max=6.2831852,
    reference_height: int | None = None,
    reference_width: int | None = None,
) -> torch.Tensor:
    """
    Generate 2D sinusoidal position codes.
    
    Args:
        height, width: Grid dimensions
        out_dim: Output dimension
        normalization_max: Max value for position normalization
        reference_height, reference_width: If provided, normalize by these instead.
                                           Use for consistent embeddings across variable sizes.
    """
    y_coords = get_1d_position_codes(height, out_dim // 2, normalization_max, reference_len=reference_height)
    y_coords = y_coords.unsqueeze(2)
    y_coords = torch.cat([y_coords, torch.zeros_like(y_coords)], -1)

    x_coords = get_1d_position_codes(width, out_dim // 2, normalization_max, reference_len=reference_width)
    x_coords = x_coords.unsqueeze(1)
    x_coords = torch.cat([torch.zeros_like(x_coords), x_coords], -1)

    return y_coords + x_coords


def create_2d_sin_cos_pos_emb(
    n_rows: int,
    n_cols: int,
    dim: int,
    normalization_max=6.2831852,
    reference_rows: int | None = None,
    reference_cols: int | None = None,
) -> torch.Tensor:
    """
    Create 2D sinusoidal position embeddings.
    
    Args:
        n_rows, n_cols: Grid dimensions
        dim: Embedding dimension
        normalization_max: Max value for position normalization
        reference_rows, reference_cols: If provided, normalize positions by these values
                                        instead of n_rows/n_cols. This ensures consistent
                                        embeddings across different image sizes.
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
