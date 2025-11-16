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


def pad_to_max_size(batch, patch_size: int, tape_dim: int, transform=None):
    if not batch:
        raise ValueError("Empty batch encountered in pad_to_max_size")

    processed_images = []
    valid_sizes = []
    rounded_sizes = []
    labels = []

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

        processed_images.append(image)
        valid_sizes.append((h, w))
        rounded_sizes.append((image.shape[1], image.shape[2]))
        labels.append(int(label))

    max_height = max(h for h, _ in rounded_sizes)
    max_width = max(w for _, w in rounded_sizes)
    max_nh = max_height // patch_size
    max_nw = max_width // patch_size
    max_tokens = max_nh * max_nw
    max_pixels = max_height * max_width

    padded_tokens = []
    patch_masks = []
    image_masks = []
    pos_embs = []
    padded_images = []

    for image, (valid_h, valid_w), (rounded_h, rounded_w) in zip(processed_images, valid_sizes, rounded_sizes):
        nh = rounded_h // patch_size
        nw = rounded_w // patch_size
        patches = patchify(image, patch_size)
        token_count = nh * nw
        pad_tokens = max_tokens - token_count
        if pad_tokens:
            patches = F.pad(patches, (0, 0, 0, pad_tokens))

        patch_mask = torch.zeros(max_tokens, dtype=torch.bool)
        patch_mask[:token_count] = True

        pos_emb = create_2d_sin_cos_pos_emb(nh, nw, tape_dim)
        if pad_tokens:
            pos_emb = F.pad(pos_emb, (0, 0, 0, pad_tokens))

        pad_h = max_height - rounded_h
        pad_w = max_width - rounded_w
        padded_image = image if (pad_h == 0 and pad_w == 0) else F.pad(image, (0, pad_w, 0, pad_h))

        pixel_mask = torch.zeros(max_pixels, dtype=torch.bool)
        pixel_mask[: valid_h * valid_w] = True

        padded_tokens.append(patches)
        patch_masks.append(patch_mask)
        image_masks.append(pixel_mask)
        pos_embs.append(pos_emb)
        padded_images.append(padded_image)

    batch_dict = {
        "patches": torch.stack(padded_tokens),
        "patch_mask": torch.stack(patch_masks),
        "image_mask": torch.stack(image_masks),
        "labels": torch.tensor(labels, dtype=torch.long),
        "pos_embs": torch.stack(pos_embs),
        "nmh": max_nh,
        "nmw": max_nw,
        "images": torch.stack(padded_images),
    }

    return batch_dict

