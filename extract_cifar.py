import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from torchvision.utils import save_image
from tqdm import tqdm

def main():
    # Create output directory
    output_dir = Path("cifar10_images")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load CIFAR-10 dataset
    # Using ToTensor() to convert to [0,1] range
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.CIFAR10(
        root='./datasets', 
        train=True,
        download=True,
        transform=transform
    )

    # We'll use the first 50,000 images (which is the entire training set)
    print("Extracting and saving CIFAR-10 images...")
    
    # Create DataLoader for efficient processing
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=100,  # Process 100 images at a time
        shuffle=False,
        num_workers=4
    )

    # Save images
    image_idx = 0
    for images, _ in tqdm(dataloader, desc="Saving images"):
        for image in images:
            # Save each image
            save_image(
                image,
                output_dir / f"image_{image_idx:05d}.png",
                normalize=False  # Already in [0,1] range
            )
            image_idx += 1

    print(f"Done! Saved {image_idx} images to {output_dir}")

if __name__ == "__main__":
    main() 