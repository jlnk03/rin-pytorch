import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
from tqdm import tqdm
import random
from PIL import Image
import argparse
from torch.utils.data._utils.collate import default_collate
import math

from rin_flex_pytorch import Rin, RinDiffusionModel

def parse_args():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument("--num-layers", type=str, default="4,4,4")
    parser.add_argument("--latent-slots", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=1024)
    parser.add_argument("--latent-mlp-ratio", type=int, default=4)
    parser.add_argument("--latent-num-heads", type=int, default=16)
    parser.add_argument("--tape-dim", type=int, default=1024)
    parser.add_argument("--tape-mlp-ratio", type=int, default=4)
    parser.add_argument("--rw-num-heads", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--latent-pos-encoding", type=str, default="rotary")
    parser.add_argument("--tape-pos-encoding", type=str, default="rotary")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=10000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    
    # Data parameters
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--min-size", type=int, default=32)
    parser.add_argument("--max-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    
    # Add aspect ratio related arguments
    parser.add_argument("--min-aspect-ratio", type=float, default=0.5,
                       help="Minimum aspect ratio (width/height)")
    parser.add_argument("--max-aspect-ratio", type=float, default=2.0,
                       help="Maximum aspect ratio (width/height)")
    
    return parser.parse_args()

class FlexibleResizeDataset:
    def __init__(self, root_dir, min_size=32, max_size=256, patch_size=8, max_seq_len=256, transform=None):
        self.dataset = ImageFolder(root_dir)
        self.min_size = min_size
        self.max_size = max_size
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.base_transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        orig_w, orig_h = img.size
        
        # Keep aspect ratio while ensuring minimum dimension is at least min_size
        # and maximum dimension is at most max_size
        ratio = orig_w / orig_h
        
        if ratio > 1:  # wider than tall
            w = min(self.max_size, max(self.min_size, orig_w))
            h = int(w / ratio)
            if h < self.min_size:
                h = self.min_size
                w = int(h * ratio)
        else:  # taller than wide
            h = min(self.max_size, max(self.min_size, orig_h))
            w = int(h * ratio)
            if w < self.min_size:
                w = self.min_size
                h = int(w / ratio)
        
        # Ensure the number of patches doesn't exceed max_seq_len
        while (h // self.patch_size) * (w // self.patch_size) > self.max_seq_len:
            if w > h:
                w = w - self.patch_size
            else:
                h = h - self.patch_size
        
        # Make dimensions divisible by patch_size
        w = (w // self.patch_size) * self.patch_size
        h = (h // self.patch_size) * self.patch_size
        
        # Create dynamic transform
        transform = transforms.Compose([
            transforms.Resize((h, w), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        if self.base_transform is not None:
            transform = transforms.Compose([self.base_transform, transform])
            
        img = transform(img)
        
        # Calculate number of patches
        n_patches = (h // self.patch_size) * (w // self.patch_size)
        
        return {
            'image': img,
            'label': label,
            'height': h,
            'width': w,
            'n_patches': n_patches
        }

def flexible_collate_fn(batch):
    """
    Custom collate function to handle varying image sizes.
    Pads images to the maximum size in the batch.
    """
    # Find max height and width in the batch
    max_h = max(item['height'] for item in batch)
    max_w = max(item['width'] for item in batch)
    
    # Pad images to max size
    for item in batch:
        if item['height'] < max_h or item['width'] < max_w:
            padded_image = torch.zeros(3, max_h, max_w)
            padded_image[:, :item['height'], :item['width']] = item['image']
            item['image'] = padded_image
    
    # Collate the batch
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    heights = torch.tensor([item['height'] for item in batch])
    widths = torch.tensor([item['width'] for item in batch])
    n_patches = torch.tensor([item['n_patches'] for item in batch])
    
    return {
        'image': images,
        'label': labels,
        'height': heights,
        'width': widths,
        'n_patches': n_patches
    }

def create_model(args, num_classes):
    model = Rin(
        num_layers=args.num_layers,
        latent_slots=args.latent_slots,
        latent_dim=args.latent_dim,
        latent_mlp_ratio=args.latent_mlp_ratio,
        latent_num_heads=args.latent_num_heads,
        tape_dim=args.tape_dim,
        tape_mlp_ratio=args.tape_mlp_ratio,
        rw_num_heads=args.rw_num_heads,
        image_height=args.max_size,  # This is now the maximum height
        image_width=args.max_size,   # This is now the maximum width
        image_channels=3,
        patch_size=args.patch_size,
        max_seq_len=args.max_seq_len,
        latent_pos_encoding=args.latent_pos_encoding,
        tape_pos_encoding=args.tape_pos_encoding,
        num_classes=num_classes
    )
    
    diffusion_model = RinDiffusionModel(
        rin=model,
        train_schedule="cosine",
        inference_schedule="cosine",
        pred_type="eps",
        conditional="class"
    )
    
    return diffusion_model

def get_lr_schedule(optimizer, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create dataset
    dataset = FlexibleResizeDataset(
        args.data_dir,
        min_size=args.min_size,
        max_size=args.max_size,
        patch_size=args.patch_size,
        max_seq_len=args.max_seq_len
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=flexible_collate_fn
    )
    
    # Create model
    model = create_model(args, num_classes=len(dataset.dataset.classes))
    model = model.to(args.device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate schedule
    scheduler = get_lr_schedule(optimizer, args.warmup_steps)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(args.device)
            labels = batch['label'].to(args.device)
            
            # Convert labels to one-hot
            labels_onehot = torch.zeros(labels.size(0), len(dataset.dataset.classes), device=args.device)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            
            optimizer.zero_grad()
            
            loss = model(images, labels=labels_onehot)
            loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "img_size": f"{images.shape[-2]}x{images.shape[-1]}"
            })
            
            # Save checkpoint
            if global_step % args.save_every == 0:
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint_{global_step}.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                }, checkpoint_path)
            
            global_step += 1
        
        # Save epoch checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        }, checkpoint_path)

if __name__ == "__main__":
    args = parse_args()
    train(args) 