import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any
from PIL import Image

def collate_variable_size_images(batch: List[Tuple[Image.Image, int]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable size images.
    
    Args:
        batch: List of tuples (image, label) where images can have different sizes
        
    Returns:
        Dictionary containing:
            - images: Tensor of shape [batch_size, channels, max_height, max_width]
            - labels: Tensor of shape [batch_size]
            - original_sizes: List of original (height, width) tuples
    """
    # Separate images and labels
    images, labels = zip(*batch)
    
    # Convert PIL images to tensors (assuming images are already normalized)
    images = [torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0 for img in images]
    
    # Get dimensions
    channels = images[0].size(0)
    max_height = max(img.size(1) for img in images)
    max_width = max(img.size(2) for img in images)
    
    # Store original sizes for potential use later
    original_sizes = [(img.size(1), img.size(2)) for img in images]
    
    # Pad images to max size
    padded_images = []
    for img in images:
        h, w = img.size(1), img.size(2)
        pad_h = max_height - h
        pad_w = max_width - w
        
        # Pad with zeros
        padded_img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
        padded_images.append(padded_img)
    
    # Stack into batch
    images_tensor = torch.stack(padded_images)
    labels_tensor = torch.tensor(labels)
    
    return {
        'images': images_tensor,
        'labels': labels_tensor,
        'original_sizes': original_sizes
    } 