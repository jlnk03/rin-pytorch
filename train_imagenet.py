import torchvision
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from rin_pytorch import Rin, RinDiffusionModel, Trainer

imagenet_root = "/home/stud/ljul/storage/group/dataset_mirrors/imagenet2012/imagenet2012_download/train"


class FlexibleCIFAR10(Dataset):
    def __init__(
        self,
        root_dir,
        train=True,
        transform=None,
        ensure_vertical=False,
        ensure_horizontal=False,
    ):
        self.root_dir = Path(root_dir)
        self.split = "train" if train else "test"
        self.transform = transform
        self.ensure_vertical = ensure_vertical
        self.ensure_horizontal = ensure_horizontal

        self.image_paths = []
        self.labels = []

        for class_idx in range(10):
            class_dir = self.root_dir / self.split / str(class_idx)
            for img_path in class_dir.glob("*.png"):
                self.image_paths.append(img_path)
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        width, height = image.size
        if self.ensure_vertical and width > height:
            image = image.rotate(90, expand=True)
        elif self.ensure_horizontal and height > width:
            image = image.rotate(90, expand=True)

        if self.transform:
            image = self.transform(image)

        return image, label

config = dict(
    rin=dict(
        num_layers="4,4,4,4,4,4",
        latent_slots=128,
        latent_dim=512,
        latent_mlp_ratio=4,
        latent_num_heads=16,
        tape_dim=512,
        tape_mlp_ratio=4,
        rw_num_heads=16,
        image_height=128,
        image_width=128,
        image_channels=3,
        patch_size=4,
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
        num_classes=1000,
    ),
    diffusion=dict(
        train_schedule="sigmoid@-3,3,0.9",
        inference_schedule="cosine",
        pred_type="eps",
        self_cond="latent",
        loss_type="eps",
        num_classes=1000,
    ),
    trainer=dict(
        num_classes=1000,
        train_num_steps=220_000,
        train_batch_size=128,
        split_batches=True,
        fp16=False,
        amp=False,
        lr_scheduler_name="cosine",
        lr=0.002,
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
        checkpoint_folder="/home/stud/ljul/storage/user/imagenet/",
        run_name="rin",
        log_to_wandb=True,
        gradient_accumulation_steps=8,
    ),
)


rin = Rin(**config["rin"]).cuda()
rin.pass_dummy_data(num_classes=1000)  # populate lazy model with weights
diffusion_model = RinDiffusionModel(rin=rin, **config["diffusion"])

rin_ema = Rin(**config["rin"]).cuda()
rin_ema.pass_dummy_data(num_classes=1000)
ema_diffusion_model = RinDiffusionModel(rin=rin_ema, **config["diffusion"])


# dataset = FlexibleCIFAR10(
#     "datasets/cifar10_flex",
#     train=True,
#     transform=torchvision.transforms.Compose(
#         [
#             torchvision.transforms.Resize((32, 32)),
#             torchvision.transforms.RandomHorizontalFlip(),
#             torchvision.transforms.ToTensor(),
#         ]
#     ),
# )

dataset = torchvision.datasets.ImageFolder(
    root=imagenet_root,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ]
    ),
)


trainer = Trainer(
    diffusion_model,
    ema_diffusion_model,
    dataset,
    **config["trainer"],
)


if __name__ == "__main__":

    trainer.train()
