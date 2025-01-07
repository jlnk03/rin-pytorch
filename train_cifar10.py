import torchvision
from torchvision import transforms
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

from rin_pytorch import Rin, RinDiffusionModel, Trainer

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
        checkpoint_folder="results/cifar10",
        run_name="rin_flex",
        log_to_wandb=True,
    ),
)


rin = Rin(**config["rin"]).cuda()
rin.pass_dummy_data(num_classes=10)  # populate lazy model with weights
diffusion_model = RinDiffusionModel(rin=rin, **config["diffusion"])

rin_ema = Rin(**config["rin"]).cuda()
rin_ema.pass_dummy_data(num_classes=10)
ema_diffusion_model = RinDiffusionModel(rin=rin_ema, **config["diffusion"])


class FlexibleCIFAR10(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = Path(root_dir)
        self.split = 'train' if train else 'test'
        self.transform = transform
        
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
        # Scale image by 2x
        w, h = image.size
        # image = image.resize((int(w*1.5), int(h*1.5)), Image.Resampling.LANCZOS)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

dataset = FlexibleCIFAR10(
    "datasets/cifar10_flex",
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
    ])
)


# dataset = torchvision.datasets.CIFAR10(
#     root="datasets",
#     train=True,
#     download=True,
#     transform=transforms.Compose([
#         transforms.ToTensor(),
#         transforms.RandomHorizontalFlip(),
#     ])
# )


trainer = Trainer(
    diffusion_model,
    ema_diffusion_model,
    dataset,
    patch_size=config["rin"]["patch_size"],
    **config["trainer"],
)


if __name__ == "__main__":

    trainer.train()
