import torchvision
import argparse
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from rin_flex_pytorch import Rin, RinDiffusionModel, Trainer

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
    
    # Data parameters
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--min-size", type=int, default=32)
    parser.add_argument("--max-size", type=int, default=256)
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=150_000)
    parser.add_argument("--output-dir", type=str, default="results/flex")
    parser.add_argument("--run-name", type=str, default="rin_flex")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

class FlexibleResizeDataset(Dataset):
    def __init__(self, root_dir, min_size=32, max_size=256, patch_size=8, max_seq_len=256):
        self.dataset = torchvision.datasets.ImageFolder(root_dir)
        self.min_size = min_size
        self.max_size = max_size
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        orig_w, orig_h = img.size
        
        # Keep aspect ratio while ensuring minimum dimension is at least min_size
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
        
        transform = transforms.Compose([
            transforms.Resize((h, w), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        img = transform(img)
        return img, label

def flexible_collate_fn(batch):
    """Custom collate function to handle varying image sizes"""
    # Find max dimensions in the batch
    max_h = max(img.shape[1] for img, _ in batch)
    max_w = max(img.shape[2] for img, _ in batch)
    
    # Pad images to max dimensions
    padded_images = []
    labels = []
    for img, label in batch:
        # Calculate padding
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        
        # Pad image
        padded_img = torch.nn.functional.pad(
            img, 
            (0, pad_w, 0, pad_h),  # Pad last two dimensions (height, width)
            mode='constant', 
            value=0
        )
        padded_images.append(padded_img)
        labels.append(label)
    
    # Stack into batches
    images = torch.stack(padded_images)
    labels = torch.tensor(labels)
    
    return images, labels

def main():
    args = parse_args()
    
    # Create dataset
    dataset = FlexibleResizeDataset(
        args.data_dir,
        min_size=args.min_size,
        max_size=args.max_size,
        patch_size=args.patch_size,
        max_seq_len=args.max_seq_len
    )
    
    # Create base model
    rin = Rin(
        num_layers=args.num_layers,
        latent_slots=args.latent_slots,
        latent_dim=args.latent_dim,
        latent_mlp_ratio=args.latent_mlp_ratio,
        latent_num_heads=args.latent_num_heads,
        tape_dim=args.tape_dim,
        tape_mlp_ratio=args.tape_mlp_ratio,
        rw_num_heads=args.rw_num_heads,
        image_height=args.max_size,
        image_width=args.max_size,
        image_channels=3,
        patch_size=args.patch_size,
        max_seq_len=args.max_seq_len,
        latent_pos_encoding="learned",
        tape_pos_encoding="learned",
        num_classes=len(dataset.dataset.classes)
    ).cuda()
    
    # Create EMA model with same config
    rin_ema = Rin(
        num_layers=args.num_layers,
        latent_slots=args.latent_slots,
        latent_dim=args.latent_dim,
        latent_mlp_ratio=args.latent_mlp_ratio,
        latent_num_heads=args.latent_num_heads,
        tape_dim=args.tape_dim,
        tape_mlp_ratio=args.tape_mlp_ratio,
        rw_num_heads=args.rw_num_heads,
        image_height=args.max_size,
        image_width=args.max_size,
        image_channels=3,
        patch_size=args.patch_size,
        max_seq_len=args.max_seq_len,
        latent_pos_encoding="learned",
        tape_pos_encoding="learned",
        num_classes=len(dataset.dataset.classes)
    ).cuda()
    
    # Create diffusion models
    diffusion_model = RinDiffusionModel(
        rin=rin,
        train_schedule="cosine",
        inference_schedule="cosine",
        pred_type="eps",
        conditional="class"
    )
    
    ema_diffusion_model = RinDiffusionModel(
        rin=rin_ema,
        train_schedule="cosine",
        inference_schedule="cosine",
        pred_type="eps",
        conditional="class"
    )
    
    # Create trainer
    trainer = Trainer(
        diffusion_model=diffusion_model,
        ema_diffusion_model=ema_diffusion_model,
        dataset=dataset,
        num_classes=len(dataset.dataset.classes),
        train_num_steps=args.num_steps,
        train_batch_size=args.batch_size,
        checkpoint_folder=args.output_dir,
        run_name=args.run_name,
        num_dl_workers=4,
        collate_fn=flexible_collate_fn
    )
    
    # Train
    trainer.train()

if __name__ == "__main__":
    main() 