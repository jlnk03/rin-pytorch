import argparse
from pathlib import Path
import gc
import torch
from torch import nn
from torchvision.models import inception_v3
import numpy as np
from scipy import linalg
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ImageFolderDataset(Dataset):
    def __init__(self, path):
        self.path = Path(path)
        self.files = sorted(list(self.path.glob('*.png')) + list(self.path.glob('*.jpg')))
        self.transform = transforms.Compose([
            transforms.Resize(299, antialias=True),  # Inception V3 input size
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = self.files[idx]
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)

def get_inception_model():
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = nn.Identity()  # Remove final FC layer
    model.eval()
    return model

def calculate_activation_statistics(dataloader, model, device):
    """
    Calculate activation statistics using incremental (online) computation.
    This avoids storing all activations in memory.
    """
    # Get feature dimension from first batch
    n_samples = 0
    sum_acts = None
    sum_sq_acts = None
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing images"):
            batch = batch.to(device)
            pred = model(batch).cpu().numpy()
            
            batch_size = pred.shape[0]
            
            if sum_acts is None:
                feat_dim = pred.shape[1]
                sum_acts = np.zeros(feat_dim, dtype=np.float64)
                sum_sq_acts = np.zeros((feat_dim, feat_dim), dtype=np.float64)
            
            # Incremental sum for mean
            sum_acts += pred.sum(axis=0)
            # Incremental sum for covariance (outer product)
            sum_sq_acts += pred.T @ pred
            n_samples += batch_size
    
    # Compute mean
    mu = sum_acts / n_samples
    
    # Compute covariance: E[X^T X] - mu^T mu
    # cov = (sum_sq_acts / n_samples) - np.outer(mu, mu)
    # With Bessel's correction (n-1 denominator):
    sigma = (sum_sq_acts - n_samples * np.outer(mu, mu)) / (n_samples - 1)
    
    return mu, sigma

def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Frechet Distance between two multivariate Gaussians."""
    diff = mu1 - mu2
    
    # Add small epsilon to diagonal for numerical stability
    sigma1 = sigma1 + eps * np.eye(sigma1.shape[0])
    sigma2 = sigma2 + eps * np.eye(sigma2.shape[0])
    
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Handle numerical instability
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            print("Warning: Imaginary component in sqrtm result")
        covmean = covmean.real
    
    # Check for NaN/Inf
    if not np.isfinite(covmean).all():
        print("Warning: Non-finite values in covmean, using trace approximation")
        tr_covmean = np.sqrt(np.trace(sigma1) * np.trace(sigma2))
    else:
        tr_covmean = np.trace(covmean)
    
    return (diff.dot(diff) + 
            np.trace(sigma1) + 
            np.trace(sigma2) - 
            2 * tr_covmean)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path1", type=str, required=True, 
                       help="Path to first image folder")
    parser.add_argument("--path2", type=str, required=True, 
                       help="Path to second image folder")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for processing")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load datasets
    dataset1 = ImageFolderDataset(args.path1)
    dataset2 = ImageFolderDataset(args.path2)
    
    dataloader1 = DataLoader(dataset1, batch_size=args.batch_size, 
                            shuffle=False, num_workers=4, pin_memory=True)
    dataloader2 = DataLoader(dataset2, batch_size=args.batch_size, 
                            shuffle=False, num_workers=4, pin_memory=True)

    # Load and prepare Inception model
    model = get_inception_model().to(device)

    print("Calculating statistics for first dataset...")
    mu1, sigma1 = calculate_activation_statistics(dataloader1, model, device)
    
    # Clean up to free memory
    del dataloader1
    gc.collect()
    torch.cuda.empty_cache()
    
    print("Calculating statistics for second dataset...")
    mu2, sigma2 = calculate_activation_statistics(dataloader2, model, device)
    
    # Clean up
    del dataloader2, model
    gc.collect()
    torch.cuda.empty_cache()

    print("Calculating FID score...")
    fid_value = calculate_fid(mu1, sigma1, mu2, sigma2)
    
    print(f"FID Score: {fid_value:.2f}")

if __name__ == "__main__":
    main() 