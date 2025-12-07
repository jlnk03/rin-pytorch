import torch
ckpt = torch.load("/home/stud/ljul/storage/user/imagenet/rin_imagenet_flex_64_20251130_074829/last.ckpt", map_location="cpu")
print("global_step:", ckpt.get("global_step"))
print("epoch:", ckpt.get("epoch"))