import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from rin_pytorch.data import DEFAULT_CIFAR_ROOT, FlexibleCIFAR10, ImageNetWebDataset


# Default cache directories
# IMAGENET_CACHE_DIR = Path("/home/stud/ljul/storage/user/imagenet")
# CIFAR_CACHE_DIR = Path("/home/stud/ljul/storage/user/cifar")
CIFAR_CACHE_DIR = Path("/dss/dsstbyfs02/pn52ko/pn52ko-dss-0000/tum/results/cifar")
IMAGENET_CACHE_DIR = Path("/dss/dsstbyfs02/pn52ko/pn52ko-dss-0000/tum/results/imagenet")

# Transform for FID: resize to 299x299 and convert to uint8 tensor [0, 255]
FID_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(299, antialias=True),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).to(torch.uint8)),
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


class ImageNetFIDDataset(Dataset):
    """Wrapper around ImageNetWebDataset for FID computation (returns only images)."""
    def __init__(self, split: str = "train", transform=None):
        self.dataset = ImageNetWebDataset(split=split, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image


def get_cache_path(cache_dir: Path, feature: int, split: str = "train") -> Path:
    """Get the cache file path for real image statistics."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"fid_stats_feature{feature}_{split}.pt"


def save_real_stats(fid: FrechetInceptionDistance, cache_path: Path):
    """Save the real image statistics from the FID metric."""
    stats = {
        "real_features_sum": fid.real_features_sum.cpu(),
        "real_features_cov_sum": fid.real_features_cov_sum.cpu(),
        "real_features_num_samples": fid.real_features_num_samples.cpu(),
    }
    torch.save(stats, cache_path)
    print(f"Saved real image statistics to {cache_path}")


def load_real_stats(fid: FrechetInceptionDistance, cache_path: Path, device: torch.device) -> bool:
    """Load cached real image statistics into the FID metric. Returns True if successful."""
    if not cache_path.exists():
        return False
    
    print(f"Loading cached real image statistics from {cache_path}")
    stats = torch.load(cache_path, map_location=device, weights_only=True)
    fid.real_features_sum = stats["real_features_sum"].to(device)
    fid.real_features_cov_sum = stats["real_features_cov_sum"].to(device)
    fid.real_features_num_samples = stats["real_features_num_samples"].to(device)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path1", type=str, default=None,
                       help="Path to first image folder. Ignored when --flexible_cifar or --imagenet is set.")
    parser.add_argument("--path2", type=str, required=True,
                       help="Path to second image folder")
    parser.add_argument("--flexible_cifar", action="store_true",
                       help="Use Flexible CIFAR-10 (train split by default) for the first dataset.")
    parser.add_argument("--cifar_split", choices=["train", "test"], default="train",
                       help="Flexible CIFAR-10 split to use when --flexible_cifar is set.")
    parser.add_argument("--cifar_root", type=str, default=None,
                       help=f"Root directory for Flexible CIFAR-10. Defaults to '{DEFAULT_CIFAR_ROOT}'.")
    parser.add_argument("--imagenet", action="store_true",
                       help="Use ImageNet-1k from HuggingFace for the first dataset.")
    parser.add_argument("--imagenet_split", choices=["train", "validation"], default="train",
                       help="ImageNet split to use when --imagenet is set.")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for processing")
    parser.add_argument("--feature", type=int, default=2048,
                       help="Feature dimension for FID (64, 192, 768, or 2048)")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Directory to cache real image statistics. Auto-set for --flexible_cifar and --imagenet.")
    parser.add_argument("--recompute_cache", action="store_true",
                       help="Force recomputation of cached real image statistics.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine cache directory and path
    cache_path = None
    if args.flexible_cifar:
        cache_dir = Path(args.cache_dir) if args.cache_dir else CIFAR_CACHE_DIR
        cache_path = get_cache_path(cache_dir, args.feature, args.cifar_split)
    elif args.imagenet:
        cache_dir = Path(args.cache_dir) if args.cache_dir else IMAGENET_CACHE_DIR
        cache_path = get_cache_path(cache_dir, args.feature, args.imagenet_split)
    elif args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_path = get_cache_path(cache_dir, args.feature, "custom")

    # Initialize FID metric
    fid = FrechetInceptionDistance(feature=args.feature, normalize=False).to(device)

    # Try to load cached real statistics
    cache_loaded = False
    if cache_path and not args.recompute_cache:
        cache_loaded = load_real_stats(fid, cache_path, device)

    # Compute real statistics if not loaded from cache
    if not cache_loaded:
        if args.flexible_cifar:
            cifar_root = args.cifar_root or DEFAULT_CIFAR_ROOT
            dataset1 = FlexibleCIFARImageDataset(
                root_dir=cifar_root,
                train=args.cifar_split == "train",
                transform=FID_TRANSFORM,
            )
        elif args.imagenet:
            dataset1 = ImageNetFIDDataset(
                split=args.imagenet_split,
                transform=FID_TRANSFORM,
            )
        else:
            if args.path1 is None:
                parser.error("--path1 is required when --flexible_cifar or --imagenet is not set.")
            dataset1 = ImageFolderDataset(args.path1, transform=FID_TRANSFORM)

        dataloader1 = DataLoader(dataset1, batch_size=args.batch_size, 
                                shuffle=False, num_workers=4)

        print("Processing first dataset (real)...")
        for batch in tqdm(dataloader1, desc="Real images"):
            batch = batch.to(device)
            fid.update(batch, real=True)
        
        # Save to cache if cache_path is set
        if cache_path:
            save_real_stats(fid, cache_path)
    
    # Process generated images
    dataset2 = ImageFolderDataset(args.path2, transform=FID_TRANSFORM)
    dataloader2 = DataLoader(dataset2, batch_size=args.batch_size, 
                            shuffle=False, num_workers=4)

    print("Processing second dataset (generated)...")
    for batch in tqdm(dataloader2, desc="Generated images"):
        batch = batch.to(device)
        fid.update(batch, real=False)

    print("Calculating FID score...")
    fid_value = fid.compute()
    
    print(f"FID Score: {fid_value:.2f}")


if __name__ == "__main__":
    main()
