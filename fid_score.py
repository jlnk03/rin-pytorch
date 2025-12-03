import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import linalg
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import inception_v3
from tqdm import tqdm

from rin_pytorch.data import DEFAULT_CIFAR_ROOT, DEFAULT_IMAGENET_ROOT, FlexibleCIFAR10
import torchvision

import os
import pickle

save_path = "statistics/mu_sigma_dataset1.pkl"


FID_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(299, antialias=True),  # Inception V3 input size
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

class ImageFolderDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = Path(path)
        self.files = sorted(list(self.path.glob('*.png')) + list(self.path.glob('*.jpg')))
        self.transform = transform or FID_TRANSFORM

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = self.files[idx]
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)


class FlexibleCIFARImageDataset(Dataset):
    def __init__(self, root_dir: str, train: bool, transform):
        self.dataset = FlexibleCIFAR10(root_dir=root_dir, train=train, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image


class ImageNetImageDataset(Dataset):
    def __init__(self, root_dir: str, transform):
        self.dataset = torchvision.datasets.ImageFolder(root=root_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image

def get_inception_model():
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = nn.Identity()  # Remove final FC layer
    model.eval()
    return model

def calculate_activation_statistics(dataloader, model, device):
    acts = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing images"):
            batch = batch.to(device)
            pred = model(batch)
            acts.append(pred.cpu().numpy())
    
    acts = np.concatenate(acts, axis=0)
    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    
    return mu, sigma



def save_statistics(mu, sigma, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump({'mu': mu, 'sigma': sigma}, f)

def load_statistics(save_path):
    with open(save_path, 'rb') as f:
        stats = pickle.load(f)
    return stats['mu'], stats['sigma']

def calculate_fid(mu1, sigma1, mu2, sigma2):
    """Calculate Frechet Distance between two multivariate Gaussians."""


    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return (diff.dot(diff) + 
            np.trace(sigma1) + 
            np.trace(sigma2) - 
            2 * tr_covmean)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path1", type=str, default=None,
                       help="Path to first image folder. Ignored when --flexible_cifar is set.")
    parser.add_argument("--path2", type=str, required=True,
                       help="Path to second image folder")
    parser.add_argument("--flexible_cifar", action="store_true",
                       help="Use Flexible CIFAR-10 (train split by default) for the first dataset.")
    parser.add_argument("--cifar_split", choices=["train", "test"], default="train",
                       help="Flexible CIFAR-10 split to use when --flexible_cifar is set.")
    parser.add_argument("--cifar_root", type=str, default=None,
                       help=f"Root directory for Flexible CIFAR-10. Defaults to '{DEFAULT_CIFAR_ROOT}'.")
    parser.add_argument("--imagenet", action="store_true",
                       help="Use ImageNet for the first dataset.")
    parser.add_argument("--imagenet_root", type=str, default=None,
                       help=f"Root directory for ImageNet. Defaults to '{DEFAULT_IMAGENET_ROOT}'.")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for processing")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load datasets
    if args.flexible_cifar:
        cifar_root = args.cifar_root or DEFAULT_CIFAR_ROOT
        dataset1 = FlexibleCIFARImageDataset(
            root_dir=cifar_root,
            train=args.cifar_split == "train",
            transform=FID_TRANSFORM,
        )
    elif args.imagenet:
        imagenet_root = args.imagenet_root or DEFAULT_IMAGENET_ROOT
        dataset1 = ImageNetImageDataset(
            root_dir=imagenet_root,
            transform=FID_TRANSFORM,
        )
    else:
        if args.path1 is None:
            parser.error("--path1 is required when --flexible_cifar or --imagenet is not set.")
        dataset1 = ImageFolderDataset(args.path1, transform=FID_TRANSFORM)

    dataset2 = ImageFolderDataset(args.path2, transform=FID_TRANSFORM)
    
    dataloader1 = DataLoader(dataset1, batch_size=args.batch_size, 
                            shuffle=False, num_workers=4)
    dataloader2 = DataLoader(dataset2, batch_size=args.batch_size, 
                            shuffle=False, num_workers=4)

    # Load and prepare Inception model
    model = get_inception_model().to(device)

    print("Calculating statistics for first dataset...")
    if os.path.exists(save_path):
        mu1, sigma1 = load_statistics(save_path)
    else:
        mu1, sigma1 = calculate_activation_statistics(dataloader1, model, device)
        save_statistics(mu1, sigma1, save_path)
    
    print("Calculating statistics for second dataset...")
    mu2, sigma2 = calculate_activation_statistics(dataloader2, model, device)

    print("Calculating FID score...")
    fid_value = calculate_fid(mu1, sigma1, mu2, sigma2)
    
    print(f"FID Score: {fid_value:.2f}")

if __name__ == "__main__":
    main() 