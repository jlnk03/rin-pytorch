import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance


class ImageFolderDataset(Dataset):
    def __init__(self, root: str, max_images: int | None = None):
        self.root = Path(root)
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        files: List[Path] = []
        for p in self.root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)
        files.sort()
        if isinstance(max_images, int) and max_images > 0:
            files = files[:max_images]
        self.files = files
        self.tf = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.tf(img)


def main():
    parser = argparse.ArgumentParser(description="Compute FID between two folders using TorchMetrics")
    parser.add_argument("--real", required=True, help="Path to real images folder")
    parser.add_argument("--fake", required=True, help="Path to generated images folder")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    real_ds = ImageFolderDataset(args.real, max_images=args.max_images)
    fake_ds = ImageFolderDataset(args.fake, max_images=args.max_images)

    real_dl = DataLoader(real_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    fake_dl = DataLoader(fake_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    metric = FrechetInceptionDistance(feature=2048).to(args.device)

    with torch.no_grad():
        for x in real_dl:
            x = x.to(args.device).clamp(0, 1)
            metric.update(x, real=True)
        for x in fake_dl:
            x = x.to(args.device).clamp(0, 1)
            metric.update(x, real=False)

    fid = metric.compute()
    print(f"FID: {float(fid):.4f}")


if __name__ == "__main__":
    main()


