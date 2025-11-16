import torch
from rin_pytorch.Rin import Rin
from rin_pytorch.RinDiffusionModel import RinDiffusionModel
from rin_pytorch.utils.data_utils import pack_batch_to_ragged

rin = Rin(
    num_layers="1",
    latent_slots=4,
    latent_dim=32,
    latent_mlp_ratio=2,
    latent_num_heads=2,
    tape_dim=32,
    tape_mlp_ratio=2,
    rw_num_heads=2,
    image_height=16,
    image_width=16,
    image_channels=3,
    patch_size=4,
    num_classes=10,
)

model = RinDiffusionModel(
    rin=rin,
    train_schedule="cosine",
    inference_schedule="cosine",
    pred_type="eps",
    num_classes=10,
)

batch = [
    (torch.rand(3, 16, 16), 0),
    (torch.rand(3, 24, 12), 1),
]

collated = pack_batch_to_ragged(batch, patch_size=4, tape_dim=rin.tape_dim)
labels = torch.nn.functional.one_hot(collated["labels"], num_classes=10).float()

loss = model(
    collated["patches"],
    collated["token_pos_embs"],
    labels,
    collated["offsets"],
    collated["pos_offsets"],
    collated["document_ids"],
)
loss.backward()
print("loss", loss.item())