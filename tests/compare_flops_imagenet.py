#!/usr/bin/env python3
"""
Compare FLOPs: Padded (fixed resize) vs Packed (ResizeMaxSide)
Using REAL ImageNet data.

This script compares:
- rin-pytorch (nn.MHA) with fixed resize to max_side x max_side
- rin-pytorch-vanilla (xformers) with ResizeMaxSide (variable-length sequences)

Averaged over 10 real batches per configuration.
"""

import torch
import torch.nn.functional as F
import numpy as np
import yaml
import sys
import os
import gc
import csv
from pathlib import Path
from datetime import datetime
from torch.profiler import profile, ProfilerActivity
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ImageNet root
IMAGENET_ROOT = '/home/stud/ljul/storage/group/dataset_mirrors/imagenet2012/imagenet2012_download/train'

# Check if ImageNet exists
if not Path(IMAGENET_ROOT).exists():
    print(f'ERROR: ImageNet not found at {IMAGENET_ROOT}')
    exit(1)

print('=' * 80)
print('FLOPS COMPARISON: ImageNet with ResizeMaxSide')
print('=' * 80)
print(f'Device: {device}')
print(f'ImageNet root: {IMAGENET_ROOT}')

# Import ResizeMaxSide
sys.path.insert(0, '/home/stud/ljul/Documents/rin-pytorch-vanilla')
from rin_pytorch.utils.data_utils import ResizeMaxSide, pack_sequences

# Results storage
all_results = []

# Test configs
CONFIGS = [
    ('CIFAR 32x32', '/home/stud/ljul/Documents/rin-pytorch-vanilla/configs/cifar.yaml', 'cifar'),
    ('ImageNet 64x64', '/home/stud/ljul/Documents/rin-pytorch-vanilla/configs/64.yaml', 'imagenet'),
    ('ImageNet 128x128', '/home/stud/ljul/Documents/rin-pytorch-vanilla/configs/128.yaml', 'imagenet'),
]

NUM_BATCHES = 10
BATCH_SIZE_CIFAR = 64
BATCH_SIZE_IMAGENET = 32

for config_name, config_path, dataset_type in CONFIGS:
    print()
    print('=' * 80)
    print(f'CONFIG: {config_name}')
    print('=' * 80)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    rin_cfg = config['rin']
    
    max_side = max(rin_cfg['image_height'], rin_cfg['image_width'])
    patch_size = rin_cfg['patch_size']
    max_tokens = (max_side // patch_size) ** 2
    
    print(f'Max image size: {max_side}x{max_side}')
    print(f'Patch size: {patch_size}')
    print(f'Max tokens per image: {max_tokens}')
    print()
    
    NUM_CLASSES = rin_cfg['num_classes']
    BATCH_SIZE = BATCH_SIZE_CIFAR if dataset_type == 'cifar' else BATCH_SIZE_IMAGENET
    
    # ===== Setup datasets =====
    if dataset_type == 'cifar':
        # CIFAR-10 (uniform 32x32)
        transform_fixed = transforms.Compose([transforms.ToTensor()])
        transform_variable = transform_fixed  # No variable length for CIFAR
        
        CIFAR_ROOT = '/home/stud/ljul/Documents/rin-pytorch-vanilla/datasets/cifar10_flex'
        if Path(CIFAR_ROOT).exists():
            class FlexibleCIFAR10(torch.utils.data.Dataset):
                def __init__(self, root_dir, transform=None):
                    self.root_dir = Path(root_dir)
                    self.transform = transform
                    self.image_paths = []
                    self.labels = []
                    for class_idx in range(10):
                        class_dir = self.root_dir / 'train' / str(class_idx)
                        if not class_dir.exists():
                            class_dir = self.root_dir / str(class_idx)
                        if class_dir.exists():
                            for img_path in class_dir.glob('*.png'):
                                self.image_paths.append(img_path)
                                self.labels.append(class_idx)
                
                def __len__(self):
                    return len(self.image_paths)
                
                def __getitem__(self, idx):
                    image = Image.open(self.image_paths[idx]).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                    return image, self.labels[idx]
            
            dataset_fixed = FlexibleCIFAR10(CIFAR_ROOT, transform=transform_fixed)
            dataset_variable = dataset_fixed
        else:
            dataset_fixed = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_fixed)
            dataset_variable = dataset_fixed
    else:
        # ImageNet with fixed vs variable resize
        transform_fixed = transforms.Compose([
            transforms.Resize((max_side, max_side)),
            transforms.ToTensor(),
        ])
        transform_variable = transforms.Compose([
            ResizeMaxSide(max_side),
            transforms.ToTensor(),
        ])
        dataset_fixed = torchvision.datasets.ImageFolder(root=IMAGENET_ROOT, transform=transform_fixed)
        dataset_variable = torchvision.datasets.ImageFolder(root=IMAGENET_ROOT, transform=transform_variable)
    
    print(f'Dataset size: {len(dataset_fixed)} images')
    print(f'Batch size: {BATCH_SIZE}')
    
    # ===== PADDED MODEL (rin-pytorch with fixed resize) =====
    print()
    print('Loading PADDED model (rin-pytorch with nn.MHA, fixed resize)...')
    
    # Clear modules
    for key in list(sys.modules.keys()):
        if 'rin_pytorch' in key:
            del sys.modules[key]
    
    sys.path.insert(0, '/home/stud/ljul/Documents/rin-pytorch')
    from rin_pytorch.Rin import Rin as PaddedRin
    from rin_pytorch.utils.data_utils import pad_to_max_size as padded_pad
    
    def padded_collate(batch):
        return padded_pad(batch, patch_size=rin_cfg['patch_size'], tape_dim=rin_cfg['tape_dim'])
    
    torch.manual_seed(42)
    padded_model = PaddedRin(**rin_cfg).to(device).eval()
    padded_params = sum(p.numel() for p in padded_model.parameters()) / 1e6
    print(f'  Parameters: {padded_params:.2f}M')
    
    padded_loader = DataLoader(dataset_fixed, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, 
                               collate_fn=padded_collate, drop_last=True)
    
    # Measure FLOPs
    print(f'  Measuring FLOPs across {NUM_BATCHES} batches...')
    padded_flops_list = []
    padded_tokens_list = []
    padded_iter = iter(padded_loader)
    
    for batch_idx in range(NUM_BATCHES):
        try:
            batch = next(padded_iter)
        except StopIteration:
            padded_iter = iter(padded_loader)
            batch = next(padded_iter)
        
        tokens = batch['patches'].to(device)
        pos = batch['token_pos_embs'].to(device)
        labels = F.one_hot(batch['labels'], NUM_CLASSES).float().to(device)
        
        padded_tokens_list.append(tokens.shape[0] * tokens.shape[1])
        
        if batch_idx == 0:
            for _ in range(3):
                with torch.no_grad():
                    _ = padded_model(tokens, 0.5, cond=labels, tape_pos_emb=pos)
            torch.cuda.synchronize()
        
        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
            with torch.no_grad():
                _ = padded_model(tokens, 0.5, cond=labels, tape_pos_emb=pos)
        torch.cuda.synchronize()
        
        macs = sum(e.flops for e in prof.key_averages() if e.flops)
        flops_per_sample = (macs * 2) / BATCH_SIZE
        padded_flops_list.append(flops_per_sample)
        
        del tokens, pos, labels, batch
        torch.cuda.empty_cache()
    
    padded_mean_flops = np.mean(padded_flops_list)
    padded_std_flops = np.std(padded_flops_list)
    avg_padded_tokens = np.mean(padded_tokens_list) / BATCH_SIZE
    print(f'  Avg tokens/image: {avg_padded_tokens:.0f}')
    print(f'  Per-sample FLOPs: {padded_mean_flops/1e9:.4f} ± {padded_std_flops/1e9:.4f} GFLOPs')
    
    del padded_model, padded_loader
    for key in list(sys.modules.keys()):
        if 'rin_pytorch' in key:
            del sys.modules[key]
    torch.cuda.empty_cache()
    gc.collect()
    
    # ===== PACKED MODEL (rin-pytorch-vanilla with ResizeMaxSide) =====
    print()
    print('Loading PACKED model (rin-pytorch-vanilla with xformers, ResizeMaxSide)...')
    
    os.environ['ATTENTION_BACKEND'] = 'xformers'
    sys.path.insert(0, '/home/stud/ljul/Documents/rin-pytorch-vanilla')
    from rin_pytorch.Rin import Rin as PackedRin
    from rin_pytorch.utils.data_utils import pack_sequences
    
    def packed_collate(batch):
        return pack_sequences(batch, patch_size=rin_cfg['patch_size'], tape_dim=rin_cfg['tape_dim'])
    
    torch.manual_seed(42)
    packed_model = PackedRin(**rin_cfg).to(device).eval()
    print(f'  Parameters: {sum(p.numel() for p in packed_model.parameters())/1e6:.2f}M')
    
    packed_loader = DataLoader(dataset_variable, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, 
                               collate_fn=packed_collate, drop_last=True)
    
    # Measure FLOPs
    print(f'  Measuring FLOPs across {NUM_BATCHES} batches...')
    packed_flops_list = []
    packed_tokens_list = []
    packed_iter = iter(packed_loader)
    
    for batch_idx in range(NUM_BATCHES):
        try:
            batch = next(packed_iter)
        except StopIteration:
            packed_iter = iter(packed_loader)
            batch = next(packed_iter)
        
        tokens = batch['patches'].to(device)
        pos = batch['token_pos_embs'].to(device)
        doc_ids = batch['doc_ids'].to(device)
        offsets = batch['offsets'].to(device)
        labels = F.one_hot(batch['labels'], NUM_CLASSES).float().to(device)
        num_docs = doc_ids.max().item() + 1
        
        packed_tokens_list.append(tokens.shape[0])
        
        if batch_idx == 0:
            for _ in range(3):
                with torch.no_grad():
                    _ = packed_model(tokens, 0.5, cond=labels, tape_pos_emb=pos, doc_ids=doc_ids, offsets=offsets)
            torch.cuda.synchronize()
        
        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
            with torch.no_grad():
                _ = packed_model(tokens, 0.5, cond=labels, tape_pos_emb=pos, doc_ids=doc_ids, offsets=offsets)
        torch.cuda.synchronize()
        
        macs = sum(e.flops for e in prof.key_averages() if e.flops)
        flops_per_sample = (macs * 2) / num_docs
        packed_flops_list.append(flops_per_sample)
        
        del tokens, pos, doc_ids, offsets, labels, batch
        torch.cuda.empty_cache()
    
    packed_mean_flops = np.mean(packed_flops_list)
    packed_std_flops = np.std(packed_flops_list)
    avg_packed_tokens = np.mean(packed_tokens_list) / BATCH_SIZE
    print(f'  Avg tokens/image: {avg_packed_tokens:.1f}')
    print(f'  Per-sample FLOPs: {packed_mean_flops/1e9:.4f} ± {packed_std_flops/1e9:.4f} GFLOPs')
    
    # Summary for this config
    token_savings = (avg_padded_tokens - avg_packed_tokens) / avg_padded_tokens * 100 if avg_padded_tokens > 0 else 0
    flops_diff_pct = (padded_mean_flops - packed_mean_flops) / padded_mean_flops * 100
    
    print()
    print(f'--- {config_name} Summary ---')
    print(f'  Fixed resize (padded): {avg_padded_tokens:.0f} tokens/img, {padded_mean_flops/1e9:.2f} GFLOPs')
    print(f'  ResizeMaxSide (packed): {avg_packed_tokens:.1f} tokens/img, {packed_mean_flops/1e9:.2f} GFLOPs')
    print(f'  Token savings: {token_savings:.1f}%')
    print(f'  FLOPs difference: {flops_diff_pct:.1f}%')
    
    all_results.append({
        'config': config_name,
        'max_side': max_side,
        'patch_size': patch_size,
        'params_M': padded_params,
        'padded_tokens': avg_padded_tokens,
        'packed_tokens': avg_packed_tokens,
        'token_savings_pct': token_savings,
        'padded_gflops': padded_mean_flops / 1e9,
        'padded_gflops_std': padded_std_flops / 1e9,
        'packed_gflops': packed_mean_flops / 1e9,
        'packed_gflops_std': packed_std_flops / 1e9,
        'flops_diff_pct': flops_diff_pct,
    })
    
    del packed_model, packed_loader
    for key in list(sys.modules.keys()):
        if 'rin_pytorch' in key:
            del sys.modules[key]
    torch.cuda.empty_cache()
    gc.collect()

# ===== Final Summary =====
print()
print('=' * 80)
print('FINAL SUMMARY')
print('=' * 80)
print()
print(f'{"Config":<20} {"Params":>8} {"Padded":>12} {"Packed":>12} {"Token Δ":>10} {"FLOPs Δ":>10}')
print(f'{"":20} {"(M)":>8} {"(GFLOPs)":>12} {"(GFLOPs)":>12} {"(%)":>10} {"(%)":>10}')
print('-' * 80)
for r in all_results:
    print(f'{r["config"]:<20} {r["params_M"]:>8.1f} {r["padded_gflops"]:>12.2f} {r["packed_gflops"]:>12.2f} {r["token_savings_pct"]:>10.1f} {r["flops_diff_pct"]:>10.1f}')
print()

# Save to CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f'flops_comparison_{timestamp}.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
    writer.writeheader()
    writer.writerows(all_results)
print(f'Results saved to: {csv_path}')

print()
print('Notes:')
print('  - Padded: Fixed resize to max_side x max_side, processes max_tokens always')
print('  - Packed: ResizeMaxSide keeps aspect ratio, variable tokens per image')
print('  - FLOPs difference reflects both token count difference and profiler artifacts')
print('  - For paper reporting, use padded FLOPs as theoretical compute baseline')

