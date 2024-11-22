import os
import torch
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import random
from pathlib import Path
import argparse

def get_imagenet_aspect_ratios():
    """
    Returns a list of typical ImageNet aspect ratios.
    These numbers are approximated from ImageNet statistics.
    Most images fall between 0.67 (2:3) and 1.5 (3:2)
    """
    # Create a distribution that favors common aspect ratios
    ratios = []
    
    # Most images are close to square (1:1)
    ratios.extend(np.random.normal(1.0, 0.1, size=50))
    
    # Common portrait ratios (2:3, 3:4, 4:5)
    ratios.extend(np.random.normal(0.67, 0.05, size=15))
    ratios.extend(np.random.normal(0.75, 0.05, size=10))
    ratios.extend(np.random.normal(0.8, 0.05, size=5))
    
    # Common landscape ratios (3:2, 4:3, 16:9)
    ratios.extend(np.random.normal(1.5, 0.05, size=15))
    ratios.extend(np.random.normal(1.33, 0.05, size=10))
    ratios.extend(np.random.normal(1.78, 0.05, size=5))
    
    # Clip to reasonable bounds
    ratios = np.clip(ratios, 0.5, 2.0)
    
    return ratios

def resize_with_aspect_ratio(img, target_ratio, min_size=32, patch_size=8):
    """Resize image to have the target aspect ratio while maintaining area"""
    width, height = img.size
    
    # Calculate current area
    current_area = width * height
    
    # Calculate new dimensions maintaining approximately the same area
    new_width = int(np.sqrt(current_area * target_ratio))
    new_height = int(new_width / target_ratio)
    
    # Ensure dimensions are at least min_size
    if new_width < min_size:
        new_width = min_size
        new_height = int(new_width / target_ratio)
    if new_height < min_size:
        new_height = min_size
        new_width = int(new_height * target_ratio)
        
    # Make dimensions divisible by patch_size
    new_width = ((new_width + patch_size - 1) // patch_size) * patch_size
    new_height = ((new_height + patch_size - 1) // patch_size) * patch_size
    
    # Resize image
    return img.resize((new_width, new_height), Image.LANCZOS)

def create_flex_dataset(args):
    # Create output directories
    output_base = Path(args.output_dir)
    for split in ['train', 'test']:
        for i in range(10):  # CIFAR has 10 classes
            (output_base / split / str(i)).mkdir(parents=True, exist_ok=True)
    
    # Load CIFAR-10
    transform = torchvision.transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Process each split
    dataset_stats = {'train': {}, 'test': {}}
    
    for split, dataset in [('train', trainset), ('test', testset)]:
        print(f"Processing {split} split...")
        
        aspect_ratios = get_imagenet_aspect_ratios()
        dataset_stats[split]['aspect_ratios'] = aspect_ratios.tolist()
        dataset_stats[split]['image_sizes'] = []
        
        for idx in tqdm(range(len(dataset))):
            img, label = dataset[idx]
            
            # Convert to PIL
            img = torchvision.transforms.ToPILImage()(img)
            
            # Get random aspect ratio
            target_ratio = random.choice(aspect_ratios)
            
            # Resize image
            img_resized = resize_with_aspect_ratio(
                img, 
                target_ratio,
                min_size=args.min_size,
                patch_size=args.patch_size
            )
            
            # Save image
            output_path = output_base / split / str(label) / f"{idx:05d}.png"
            img_resized.save(output_path)
            
            # Record statistics
            dataset_stats[split]['image_sizes'].append({
                'width': img_resized.size[0],
                'height': img_resized.size[1],
                'aspect_ratio': img_resized.size[0] / img_resized.size[1]
            })
    
    # Save dataset statistics
    with open(output_base / 'dataset_stats.json', 'w') as f:
        json.dump(dataset_stats, f, indent=2)
    
    # Print summary statistics
    print("\nDataset Statistics:")
    for split in ['train', 'test']:
        ratios = [s['aspect_ratio'] for s in dataset_stats[split]['image_sizes']]
        widths = [s['width'] for s in dataset_stats[split]['image_sizes']]
        heights = [s['height'] for s in dataset_stats[split]['image_sizes']]
        
        print(f"\n{split.capitalize()} Split:")
        print(f"Aspect Ratios - Mean: {np.mean(ratios):.2f}, Std: {np.std(ratios):.2f}")
        print(f"Widths - Mean: {np.mean(widths):.1f}, Std: {np.std(widths):.1f}")
        print(f"Heights - Mean: {np.mean(heights):.1f}, Std: {np.std(heights):.1f}")

def parse_args():
    parser = argparse.ArgumentParser(description='Create CIFAR-Flex dataset with varying aspect ratios')
    parser.add_argument('--output-dir', type=str, default='cifar10_flex',
                        help='Output directory for the dataset')
    parser.add_argument('--min-size', type=int, default=32,
                        help='Minimum size for any dimension')
    parser.add_argument('--patch-size', type=int, default=8,
                        help='Patch size for ensuring divisibility')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    create_flex_dataset(args) 