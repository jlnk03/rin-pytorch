import torchvision
from torchvision import transforms
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from datetime import datetime

from rin_pytorch import Rin, RinDiffusionModel, Trainer

# Create timestamp string
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

config = dict(
    rin=dict(
        num_layers="2,2,2",
        latent_slots=128,
        latent_dim=512,
        latent_mlp_ratio=4,
        latent_num_heads=16,
        tape_dim=256,
        tape_mlp_ratio=2,
        rw_num_heads=8,
        image_height=32,
        image_width=32,
        image_channels=3,
        patch_size=2,
        latent_pos_encoding="learned",
        tape_pos_encoding="learned",
        drop_path=0.1,
        drop_units=0.1,
        drop_att=0.0,
        time_scaling=1000,
        self_cond="latent",
        time_on_latent=True,
        cond_on_latent_n=1,
        cond_tape_writable=False,
        cond_dim=0,
        cond_proj=True,
        cond_decoupled_read=False,
        xattn_enc_ln=False,
        num_classes=10,
    ),
    diffusion=dict(
        train_schedule="sigmoid@-3,3,0.9",
        inference_schedule="cosine",
        pred_type="eps",
        self_cond="latent",
        loss_type="eps",
    ),
    trainer=dict(
        num_classes=10,
        train_num_steps=150_000,
        train_batch_size=256,
        split_batches=True,
        fp16=False,
        amp=False,
        lr_scheduler_name="cosine",
        lr=3e-3,
        lr_warmup_steps=10_000,
        optimizer_name="lamb",
        optimizer_exclude_weight_decay=["bias", "beta", "gamma"],
        optimizer_kwargs=dict(weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8),
        clip_grad_norm=None,
        sample_every=1000,
        num_dl_workers=4,
        ema_decay=0.9999,
        ema_update_every=1,
        sampling_kwargs=dict(iterations=100, method="ddim"),
        checkpoint_folder=f"results/cifar10/{timestamp}",
        run_name=f"rin_std_no_mask",
        log_to_wandb=True,
    ),
    # Add overfit configuration
    overfit=dict(
        enabled=True,
        target_class=0,
        num_samples=10,
    ),
)


rin = Rin(**config["rin"]).cuda()
rin.pass_dummy_data(num_classes=10)  # populate lazy model with weights
diffusion_model = RinDiffusionModel(rin=rin, **config["diffusion"])

rin_ema = Rin(**config["rin"]).cuda()
rin_ema.pass_dummy_data(num_classes=10)
ema_diffusion_model = RinDiffusionModel(rin=rin_ema, **config["diffusion"])


class FlexibleCIFAR10(Dataset):
    def __init__(self, root_dir, train=True, transform=None, target_class=None, num_samples=None):
        self.root_dir = Path(root_dir)
        self.split = 'train' if train else 'test'
        self.transform = transform
        
        # Get all image paths
        self.image_paths = []
        self.labels = []
        
        # If target_class is specified, only load that class
        class_range = [target_class] if target_class is not None else range(10)
        
        for class_idx in class_range:
            class_dir = self.root_dir / self.split / str(class_idx)
            paths = list(class_dir.glob('*.png'))
            
            # If num_samples is specified, only take that many samples
            if num_samples is not None and target_class is not None:
                paths = paths[:num_samples]
                
            for img_path in paths:
                self.image_paths.append(img_path)
                self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Create dataset with overfit settings if enabled
if config["overfit"]["enabled"]:
    dataset = FlexibleCIFAR10(
        "datasets/cifar10_flex",
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
        ]),
        target_class=config["overfit"]["target_class"],
        num_samples=config["overfit"]["num_samples"]
    )
    # full_dataset = torchvision.datasets.CIFAR10(
    #         root="datasets",
    #         train=True,
    #         download=True,
    #         transform=transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.RandomHorizontalFlip(),
    #         ])
    #     )
    
    # Filter for target class and take only specified number of samples
    # target_indices = [i for i, (_, label) in enumerate(full_dataset) if label == config["overfit"]["target_class"]][:config["overfit"]["num_samples"]]
    # dataset = torch.utils.data.Subset(full_dataset, target_indices)


    config["trainer"].update({
        "train_batch_size": min(config["trainer"]["train_batch_size"], config["overfit"]["num_samples"]),
        "train_num_steps": 50000,
        "sample_every": 100,
        "run_name": f"rin_overfit_new_mask_write_mask_loss_mask_class{config['overfit']['target_class']}"
    })
else:
    # dataset = FlexibleCIFAR10(
    #     "datasets/cifar10_flex",
    #     train=True,
    #     transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.RandomHorizontalFlip(),
    #     ])
    # )

    dataset = torchvision.datasets.CIFAR10(
        root="datasets",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
        ])
    )


trainer = Trainer(
    diffusion_model,
    ema_diffusion_model,
    dataset,
    patch_size=config["rin"]["patch_size"],
    **config["trainer"],
)

if __name__ == "__main__":
    # If in overfit mode, set the sampling class
    if config["overfit"]["enabled"]:
        trainer.fixed_class = config["overfit"]["target_class"]
    
    trainer.train()
