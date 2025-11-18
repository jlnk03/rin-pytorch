import torch
ckpt = torch.load("/home/stud/ljul/storage/user/imagenet/rin_imagenet_flex_20251118_000310/last.ckpt", map_location="cpu")
print("global_step:", ckpt.get("global_step"))
print("epoch:", ckpt.get("epoch"))