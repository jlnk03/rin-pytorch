import torch
import torch.nn.functional as F
from PIL import Image

from .pos_embedding import create_2d_sin_cos_pos_emb


class ResizeMaxSide:
    def __init__(self, max_side: int, interpolation=Image.BILINEAR):
        self.max_side = max_side
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        longest = max(width, height)
        if longest <= self.max_side:
            return img

        scale = self.max_side / float(longest)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        return img.resize((new_width, new_height), self.interpolation)


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    c, h, w = x.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError("Image height and width must be divisible by patch size after padding.")

    nh, nw = h // patch_size, w // patch_size
    patches = x.view(c, nh, patch_size, nw, patch_size)
    patches = patches.permute(1, 3, 2, 4, 0).contiguous()
    patches = patches.view(nh * nw, patch_size * patch_size * c)
    return patches


def unpatchify(patches: torch.Tensor, patch_size: int, channels: int, height: int, width: int) -> torch.Tensor:
    """
    Inverse of `patchify`. Converts flattened patch tokens back to BCHW images.
    """
    bsz, num_patches, patch_dim = patches.shape
    expected_dim = patch_size * patch_size * channels
    if patch_dim != expected_dim:
        raise ValueError(f"Patch dimension mismatch. Expected {expected_dim}, got {patch_dim}.")

    nh = height // patch_size
    nw = width // patch_size
    if nh * nw != num_patches:
        raise ValueError("Target height/width and patch count are inconsistent.")

    patches = patches.view(bsz, nh, nw, patch_size, patch_size, channels)
    patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
    images = patches.view(bsz, channels, height, width)
    return images


def pack_sequences(batch, patch_size: int, tape_dim: int, transform=None):
    """
    Pack a batch of variable-length token sequences into a single concatenated sequence.
    
    Returns a dict with:
        - patches: (total_tokens, patch_dim) concatenated tokens from all images
        - token_pos_embs: (total_tokens, tape_dim) concatenated positional embeddings
        - doc_ids: (total_tokens,) document ID for each token (0, 0, 0, 1, 1, 1, 1, ...)
        - offsets: (batch_size + 1,) cumulative offsets where each document starts
        - labels: (batch_size,) class labels
    """
    if not batch:
        raise ValueError("Empty batch encountered in pack_sequences")

    labels = []
    token_sequences = []
    token_pos_sequences = []
    token_counts = []

    for example in batch:
        if isinstance(example, dict):
            image = example.get("image")
            label = example.get("label", example.get("class", 0))
        else:
            image, label = example

        if not isinstance(image, torch.Tensor):
            if transform is None:
                raise ValueError("pack_sequences requires tensor images or a transform to convert them.")
            image = transform(image)

        if image.ndim != 3:
            raise ValueError("Each image must be a CHW tensor.")

        if image.size(0) > 3:
            image = image[:3]
        elif image.size(0) == 1:
            image = image.repeat(3, 1, 1)

        c, h, w = image.shape
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h))

        labels.append(int(label))
        nh = image.shape[1] // patch_size
        nw = image.shape[2] // patch_size
        tokens = patchify(image, patch_size)
        pos = create_2d_sin_cos_pos_emb(nh, nw, tape_dim).view(-1, tape_dim)
        token_sequences.append(tokens)
        token_pos_sequences.append(pos)
        token_counts.append(tokens.size(0))

    # Compute offsets (cumulative token counts, starting with 0)
    offsets = torch.zeros(len(token_counts) + 1, dtype=torch.long)
    offsets[1:] = torch.cumsum(torch.tensor(token_counts, dtype=torch.long), dim=0)

    # Create document IDs for each token
    doc_ids = torch.cat([
        torch.full((count,), i, dtype=torch.long) 
        for i, count in enumerate(token_counts)
    ])

    # Concatenate all tokens and positional embeddings
    all_tokens = torch.cat(token_sequences, dim=0)
    all_pos_embs = torch.cat(token_pos_sequences, dim=0)

    batch_dict = {
        "patches": all_tokens,
        "token_pos_embs": all_pos_embs,
        "doc_ids": doc_ids,
        "offsets": offsets,
        "labels": torch.tensor(labels, dtype=torch.long),
    }

    return batch_dict


# Backwards compatibility alias
pad_to_max_size = pack_sequences

