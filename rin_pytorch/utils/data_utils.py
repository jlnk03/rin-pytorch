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


def pad_to_max_size(batch, patch_size: int, tape_dim: int, transform=None):
    if not batch:
        raise ValueError("Empty batch encountered in pad_to_max_size")

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
                raise ValueError("pad_to_max_size requires tensor images or a transform to convert them.")
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

    max_seq_len = max(token_counts)

    padded_tokens_seq = []
    token_masks = []
    token_pos_embs = []

    for tokens, pos_seq, token_count in zip(token_sequences, token_pos_sequences, token_counts):
        seq_pad = max_seq_len - token_count
        seq_tokens = tokens if seq_pad == 0 else F.pad(tokens, (0, 0, 0, seq_pad))
        seq_pos = pos_seq if seq_pad == 0 else F.pad(pos_seq, (0, 0, 0, seq_pad))

        token_mask = torch.ones(max_seq_len, dtype=torch.bool)
        token_mask[:token_count] = False

        padded_tokens_seq.append(seq_tokens)
        token_masks.append(token_mask)
        token_pos_embs.append(seq_pos)

    batch_dict = {
        "patches": torch.stack(padded_tokens_seq),
        "patch_mask": torch.stack(token_masks),
        "token_pos_embs": torch.stack(token_pos_embs),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

    return batch_dict

