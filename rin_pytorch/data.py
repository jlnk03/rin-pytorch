from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms


DEFAULT_IMAGENET_ROOT = "/home/stud/ljul/storage/group/dataset_mirrors/imagenet2012/imagenet2012_download/train"
DEFAULT_CIFAR_ROOT = "datasets/cifar10_flex"


class FlexibleCIFAR10(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        train: bool = True,
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.split = "train" if train else "test"
        self.transform = transform

        potential_split = self.root_dir / self.split
        self.split_root = potential_split if potential_split.exists() else self.root_dir

        self.image_paths = []
        self.labels = []
        for class_idx in range(10):
            class_dir = self.split_root / str(class_idx)
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.glob("*.png")):
                self.image_paths.append(img_path)
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def resolve_data_root(args_root: Optional[str], config: Dict[str, Any]) -> str:
    if args_root:
        return args_root

    trainer_cfg = config.get("trainer", {})
    run_cfg = config.get("run", {})

    dataset_root = trainer_cfg.get("dataset_root")
    if dataset_root:
        return dataset_root

    if run_cfg.get("cifar", False):
        return DEFAULT_CIFAR_ROOT

    return DEFAULT_IMAGENET_ROOT


def build_train_transform(rin_cfg: Dict[str, Any]):
    image_hw = (rin_cfg["image_height"], rin_cfg["image_width"])
    return transforms.Compose(
        [
            transforms.Resize(image_hw),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )


def build_training_dataset(config: Dict[str, Any], data_root: str):
    rin_cfg = config["rin"]
    run_cfg = config.get("run", {})
    transform = build_train_transform(rin_cfg)

    if run_cfg.get("cifar", False):
        return FlexibleCIFAR10(root_dir=data_root, train=True, transform=transform)

    return torchvision.datasets.ImageFolder(root=data_root, transform=transform)


def build_training_dataloader(config: Dict[str, Any], data_root: str) -> DataLoader:
    trainer_cfg = config["trainer"]
    dataset = build_training_dataset(config, data_root)

    return DataLoader(
        dataset,
        batch_size=trainer_cfg["train_batch_size"],
        shuffle=True,
        num_workers=trainer_cfg["num_dl_workers"],
        pin_memory=True,
        persistent_workers=trainer_cfg["num_dl_workers"] > 0,
        drop_last=True,
    )

