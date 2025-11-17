import torch
ckpt = torch.load("/home/stud/ljul/storage/user/cifar/rin_cifar_nested_20251116_212152/last.ckpt", map_location="cpu")
print("global_step:", ckpt.get("global_step"))
print("epoch:", ckpt.get("epoch"))