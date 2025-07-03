import argparse
import numpy as np
import matplotlib.pyplot as plt


def build_block_mask(seq_len: int, group_size: int = 128) -> np.ndarray:
    """Build a boolean block mask where queries can only attend to keys
    within their own group of size `group_size`.

    Args:
        seq_len: Total sequence length (typically 128 * num_images).
        group_size: Size of each block that forms an independent attention group.

    Returns:
        A (seq_len, seq_len) boolean numpy array where True indicates that the
        attention between the corresponding query/key pair is allowed.
    """
    # Create a grid of indices.
    idx = np.arange(seq_len, dtype=np.int32)
    q_idx = idx[:, None]  # Shape (seq_len, 1)
    k_idx = idx[None, :]  # Shape (1, seq_len)

    # Queries can only attend to keys within the same block.
    mask = (q_idx // group_size) == (k_idx // group_size)
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize block attention mask.")
    parser.add_argument(
        "--num_images",
        type=int,
        required=True,
        help="Number of images (sequence length will be 128 * num_images)",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="Block size that defines each independent attention group (default: 128)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional path to save the figure instead of showing it interactively.",
    )

    args = parser.parse_args()

    seq_len = args.group_size * args.num_images
    mask = build_block_mask(seq_len, args.group_size)

    # Visualize.
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap="Greys", interpolation="none")
    plt.title(f"Block Attention Mask (seq_len={seq_len}, group_size={args.group_size})")
    plt.xlabel("Key index")
    plt.ylabel("Query index")
    plt.tight_layout()

    if args.save_path is None:
        plt.show()
    else:
        plt.savefig(args.save_path, dpi=300)
        print(f"Figure saved to {args.save_path}")


if __name__ == "__main__":
    main() 