"""Quick test for pack_sequences function."""
import torch
from rin_pytorch.utils.data_utils import pack_sequences

# Create dummy batch with 3 images of different sizes
batch = [
    (torch.randn(3, 32, 32), 0),   # 32x32 image, label 0
    (torch.randn(3, 64, 32), 1),   # 64x32 image, label 1  
    (torch.randn(3, 32, 64), 2),   # 32x64 image, label 2
]

patch_size = 16
tape_dim = 64

result = pack_sequences(batch, patch_size=patch_size, tape_dim=tape_dim)

print("=== Output shapes ===")
for k, v in result.items():
    print(f"  {k}: {v.shape}")

print()
print("=== Values ===")
print(f"  labels: {result['labels']}")
print(f"  offsets: {result['offsets']}")
print(f"  doc_ids: {result['doc_ids']}")

print()
print("=== Expected token counts ===")
print("  Image 1 (32x32): 2x2 = 4 tokens")
print("  Image 2 (64x32): 4x2 = 8 tokens")
print("  Image 3 (32x64): 2x4 = 8 tokens")
print("  Total expected: 20 tokens")
print(f"  Actual total: {result['patches'].shape[0]} tokens")

# Verify
assert result['patches'].shape[0] == 20, "Total token count mismatch!"
assert result['offsets'].tolist() == [0, 4, 12, 20], f"Offsets mismatch: {result['offsets'].tolist()}"
assert result['doc_ids'][:4].tolist() == [0, 0, 0, 0], "First doc_ids should be 0"
assert result['doc_ids'][4:12].tolist() == [1]*8, "Middle doc_ids should be 1"
assert result['doc_ids'][12:].tolist() == [2]*8, "Last doc_ids should be 2"

print()
print("✓ All assertions passed!")

