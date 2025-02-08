from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class FlexibleCIFAR10(Dataset):
    def __init__(self, root_dir, train=True, transform=None, target_class=None, num_samples=None, ensure_vertical=False, ensure_horizontal=False):
        self.root_dir = Path(root_dir)
        self.split = 'train' if train else 'test'
        self.transform = transform
        self.ensure_vertical = ensure_vertical  # Only ensure vertical for overfit mode
        self.ensure_horizontal = ensure_horizontal
        
        # Get all image paths
        self.image_paths = []
        self.labels = []

        for class_idx in range(10):
            class_dir = self.root_dir / self.split / str(class_idx)
            for img_path in class_dir.glob('*.png'):
                self.image_paths.append(img_path)
                self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # Check if image needs rotation
        width, height = image.size
        if self.ensure_vertical:
            if width > height:
                image = image.rotate(90, expand=True)
        elif self.ensure_horizontal and not self.ensure_vertical:
            if height > width:
                image = image.rotate(90, expand=True)
        
        if self.transform:
            image = self.transform(image)
            
        return {"image": image, "label": label}