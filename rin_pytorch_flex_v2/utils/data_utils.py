import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any
from PIL import Image
import numpy as np

def get_2d_sincos_pos_embed(h, w, embed_dim):
    """
    Create 2D positional embeddings using sine and cosine functions.
    
    Args:
        h, w: Height and width of the image
        embed_dim: Embedding dimension (must be divisible by 4)
        
    Returns:
        Positional embedding of shape [h*w, embed_dim]
    """
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')
    
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4'
    pos_dim = embed_dim // 4
    
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (10000**omega)
    
    out_h = torch.einsum('m,d->md', grid_h.reshape(-1), omega)
    out_w = torch.einsum('m,d->md', grid_w.reshape(-1), omega)
    
    pos_emb = torch.cat([torch.sin(out_h), torch.cos(out_h), 
                        torch.sin(out_w), torch.cos(out_w)], dim=1)
    
    return pos_emb

def collate_variable_size_images(batch: List[Tuple[Image.Image, int]], embed_dim: int = 1024) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable size images with positional embeddings and attention masks.
    
    Args:
        batch: List of tuples (image, label) where images can have different sizes
        embed_dim: Dimension of positional embeddings
        
    Returns:
        Dictionary containing:
            - images: Tensor of shape [batch_size, channels, max_height, max_width]
            - labels: Tensor of shape [batch_size]
            - original_sizes: List of original (height, width) tuples
            - pos_embeddings: List of positional embeddings for each image
            - attention_masks: Tensor of shape [batch_size, max_height * max_width]
    """
    # Separate images and labels
    images, labels = zip(*batch)
    
    # Convert PIL images to tensors
    images = [torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0 for img in images]
    
    # Get dimensions
    channels = images[0].size(0)
    max_height = max(img.size(1) for img in images)
    max_width = max(img.size(2) for img in images)
    
    # Store original sizes
    original_sizes = [(img.size(1), img.size(2)) for img in images]
    
    # Calculate positional embeddings and attention masks for each image
    pos_embeddings = []
    attention_masks = []
    padded_images = []
    
    for img in images:
        h, w = img.size(1), img.size(2)
        
        # Calculate positional embedding for original size
        pos_emb = get_2d_sincos_pos_embed(h, w, embed_dim)
        pos_embeddings.append(pos_emb)
        
        # Create attention mask (1 for valid pixels, 0 for padding)
        mask = torch.ones(h * w, dtype=torch.bool)
        
        # Pad attention mask to max size
        pad_length = max_height * max_width - h * w
        if pad_length > 0:
            mask = F.pad(mask, (0, pad_length), mode='constant', value=False)
        attention_masks.append(mask)
        
        # Pad image
        pad_h = max_height - h
        pad_w = max_width - w
        padded_img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
        padded_images.append(padded_img)
    
    # Stack into batches
    images_tensor = torch.stack(padded_images)
    labels_tensor = torch.tensor(labels)
    attention_masks = torch.stack(attention_masks)
    
    return {
        'images': images_tensor,
        'labels': labels_tensor,
        'original_sizes': original_sizes,
        'pos_embeddings': pos_embeddings,  # List of variable-size tensors
        'attention_masks': attention_masks
    } 