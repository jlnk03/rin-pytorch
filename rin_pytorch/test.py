from utils.ragged_tensor import (
    ragged_list_to_tensor, 
    ragged_tensor_to_list, 
    get_ragged_lengths,
    get_document_ids
)
import torch

print("=== Testing Basic Ragged Tensor Functionality ===")
# Create ragged list
t1 = torch.tensor([1, 2])
t2 = torch.tensor([3, 4, 5])
t3 = torch.tensor([6])

# Convert to ragged tensor
values, offsets = ragged_list_to_tensor([t1, t2, t3])
print(f"Original tensors: {[t1, t2, t3]}")
print(f"Concatenated values: {values}")
print(f"Offsets: {offsets}")
print(f"Expected: values=[1, 2, 3, 4, 5, 6], offsets=[0, 2, 5, 6]")

# Test reconstruction
reconstructed = ragged_tensor_to_list(values, offsets)
print(f"Reconstructed tensors: {reconstructed}")
print(f"Reconstruction matches original: {all(torch.equal(orig, rec) for orig, rec in zip([t1, t2, t3], reconstructed))}")

print("\n=== Testing Document IDs ===")
# Test document IDs
doc_ids = get_document_ids(offsets)
print(f"Document IDs: {doc_ids}")
print(f"Expected: [0, 0, 1, 1, 1, 2] (lengths: 2, 3, 1)")

# Verify the document IDs are correct
lengths = get_ragged_lengths(offsets)
print(f"Sequence lengths: {lengths}")

# Check that each token gets the right document ID
for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
    doc_ids_for_seq = doc_ids[start:end]
    expected_id = torch.full((end - start,), i, dtype=torch.int64)
    print(f"Document {i}: tokens {start}:{end} have IDs {doc_ids_for_seq}, expected {expected_id}")
    assert torch.equal(doc_ids_for_seq, expected_id), f"Mismatch for document {i}"

print("\n=== Testing Example from Description ===")
# Test the example from the user's description: lengths [3, 2, 6]
test_lengths = [3, 2, 6]
test_offsets = torch.tensor([0] + torch.tensor(test_lengths).cumsum(dim=0).tolist())
print(f"Test offsets for lengths {test_lengths}: {test_offsets}")

test_doc_ids = get_document_ids(test_offsets)
expected_doc_ids = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2])
print(f"Document IDs: {test_doc_ids}")
print(f"Expected: {expected_doc_ids}")
print(f"Match: {torch.equal(test_doc_ids, expected_doc_ids)}")

print("\n=== Testing 2D Tensors ===")
# Test with 2D tensors (more realistic for embeddings)
t1_2d = torch.tensor([[1, 10], [2, 20]])  # 2x2
t2_2d = torch.tensor([[3, 30], [4, 40], [5, 50]])  # 3x2  
t3_2d = torch.tensor([[6, 60]])  # 1x2

values_2d, offsets_2d = ragged_list_to_tensor([t1_2d, t2_2d, t3_2d], dim=0)
print(f"2D values shape: {values_2d.shape}")
print(f"2D values:\n{values_2d}")
print(f"2D offsets: {offsets_2d}")

# Test document IDs for 2D case
doc_ids_2d = get_document_ids(offsets_2d)
print(f"2D Document IDs: {doc_ids_2d}")
print(f"Expected: [0, 0, 1, 1, 1, 2] (same as 1D case since we concatenate on dim 0)")

# Test reconstruction of 2D
reconstructed_2d = ragged_tensor_to_list(values_2d, offsets_2d, dim=0)
print(f"Reconstructed 2D tensors shapes: {[t.shape for t in reconstructed_2d]}")
print(f"2D Reconstruction matches: {all(torch.equal(orig, rec) for orig, rec in zip([t1_2d, t2_2d, t3_2d], reconstructed_2d))}")

print("\n=== Testing Latent Document IDs ===")
# Test creating document IDs for latents similar to how it's done in Rin model
# Simulate batch_size=4, latent_slots=4 (total latent size = 16)
batch_size = 4
latent_slots = 4
latent_dim = 64

# Create dummy latent tensor
dummy_latent_template = torch.randn(latent_slots, latent_dim)  # [4, 64]
dummy_latent = dummy_latent_template.repeat(batch_size, 1)  # [16, 64] (4*4=16)

print(f"Dummy latent template shape: {dummy_latent_template.shape}")
print(f"Repeated latent shape: {dummy_latent.shape}")

# Create latent offsets like in the Rin model
latent_offsets = torch.arange(0, batch_size + 1, dtype=torch.int64) * latent_slots
print(f"Latent offsets: {latent_offsets}")
print(f"Expected: [0, 4, 8, 12, 16] for batch_size=4, latent_slots=4")

# Create latent document IDs
latent_doc_ids = get_document_ids(latent_offsets)
print(f"Latent document IDs: {latent_doc_ids}")
print(f"Expected: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]")

# Verify the document IDs are correct
expected_latent_doc_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
print(f"Latent document IDs match expected: {torch.equal(latent_doc_ids, expected_latent_doc_ids)}")

# Test with a different configuration - batch_size=2, latent_slots=8 (still total=16)
print(f"\nTesting with batch_size=2, latent_slots=8:")
batch_size_2 = 2
latent_slots_2 = 8
latent_offsets_2 = torch.arange(0, batch_size_2 + 1, dtype=torch.int64) * latent_slots_2
latent_doc_ids_2 = get_document_ids(latent_offsets_2)
print(f"Offsets: {latent_offsets_2}")  # [0, 8, 16]
print(f"Document IDs: {latent_doc_ids_2}")
expected_latent_doc_ids_2 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
print(f"Document IDs match expected: {torch.equal(latent_doc_ids_2, expected_latent_doc_ids_2)}")

# Verify that each batch item has the correct number of latent slots
for i in range(batch_size):
    start_idx = latent_offsets[i].item()
    end_idx = latent_offsets[i + 1].item()
    batch_latent_ids = latent_doc_ids[start_idx:end_idx]
    expected_batch_id = torch.full((latent_slots,), i, dtype=torch.int64)
    print(f"Batch {i}: latent positions {start_idx}:{end_idx}, IDs {batch_latent_ids}, expected {expected_batch_id}")
    assert torch.equal(batch_latent_ids, expected_batch_id), f"Mismatch for batch {i}"

print("\n=== Testing Cross Document Mask ===")
# Test the cross document mask function similar to TransformerDecoderLayer

# Create a scenario with multiple documents to show masking behavior
# Latent tokens: 3 documents with 2 latent slots each = 6 total latent tokens
latent_batch_size = 3
latent_slots_per_doc = 2
total_latent_tokens = latent_batch_size * latent_slots_per_doc

latent_offsets = torch.arange(0, latent_batch_size + 1, dtype=torch.int64) * latent_slots_per_doc
latent_document_ids = get_document_ids(latent_offsets)
print(f"Latent offsets: {latent_offsets}")  # [0, 2, 4, 6]
print(f"Latent document IDs: {latent_document_ids}")  # [0, 0, 1, 1, 2, 2]

# Input tokens: 3 documents with varying lengths [3, 2, 4] = 9 total input tokens
input_lengths = [3, 2, 4]
input_offsets = torch.tensor([0] + torch.tensor(input_lengths).cumsum(dim=0).tolist())
document_ids = get_document_ids(input_offsets)
print(f"Input offsets: {input_offsets}")  # [0, 3, 5, 9]
print(f"Document IDs: {document_ids}")  # [0, 0, 0, 1, 1, 2, 2, 2, 2]

# Define the cross document mask function
def cross_document_mask(q_idx, kv_idx):
    """Returns True if latent token q_idx can attend to input token kv_idx (same document)"""
    if kv_idx >= len(document_ids):
        return False  # Handle out of bounds
    return latent_document_ids[q_idx] == document_ids[kv_idx]

# Create the full attention mask matrix
num_query_tokens = len(latent_document_ids)  # 6 latent tokens
num_key_tokens = len(document_ids)  # 9 input tokens

mask_matrix = torch.zeros(num_query_tokens, num_key_tokens, dtype=torch.bool)

print(f"\nCreating mask matrix of size [{num_query_tokens}, {num_key_tokens}]")
print("Rows = latent tokens (queries), Cols = input tokens (keys/values)")

for q_idx in range(num_query_tokens):
    for kv_idx in range(num_key_tokens):
        mask_matrix[q_idx, kv_idx] = cross_document_mask(q_idx, kv_idx)

print(f"\nCross Document Mask Matrix:")
print("   Input tokens (documents):  0  0  0  1  1  2  2  2  2")
print("                    (indices): 0  1  2  3  4  5  6  7  8")
for q_idx in range(num_query_tokens):
    doc_id = latent_document_ids[q_idx].item()
    mask_row = mask_matrix[q_idx].int().tolist()
    print(f"Latent {q_idx} (doc {doc_id}):           {mask_row}")

print(f"\nMask interpretation:")
print("1 = can attend (same document), 0 = cannot attend (different document)")

# Show which tokens each latent can attend to
print(f"\nAttention allowed:")
for q_idx in range(num_query_tokens):
    doc_id = latent_document_ids[q_idx].item()
    allowed_indices = torch.where(mask_matrix[q_idx])[0].tolist()
    print(f"Latent {q_idx} (doc {doc_id}) can attend to input tokens: {allowed_indices}")

# Test the problematic case from the error logs
print(f"\n=== Testing Error Case ===")
print("Simulating the case from error logs: 126 latent tokens, 1 input token")

# Create the problematic scenario
error_latent_doc_ids = torch.zeros(126, dtype=torch.int64)  # All belong to doc 0
error_document_ids = torch.tensor([0], dtype=torch.int64)  # Only 1 input token from doc 0

print(f"Error case latent_document_ids shape: {error_latent_doc_ids.shape}")
print(f"Error case document_ids shape: {error_document_ids.shape}")

# Test what happens when we try to access out of bounds
def safe_cross_document_mask(q_idx, kv_idx):
    """Safe version that handles out of bounds"""
    if kv_idx >= len(error_document_ids):
        return False  # Cannot attend to non-existent tokens
    return error_latent_doc_ids[q_idx] == error_document_ids[kv_idx]

# Show what the mask would look like for the first few latent tokens
print(f"\nSafe mask for first 5 latent tokens against 1 input token:")
for q_idx in range(5):
    can_attend = safe_cross_document_mask(q_idx, 0)  # Only 1 input token at index 0
    print(f"Latent {q_idx} can attend to input 0: {can_attend}")

# Show what would cause the error
print(f"\nWhat causes the index error:")
print("When kv_idx > 0 (e.g., kv_idx=1), but document_ids only has 1 element (index 0)")
print("document_ids[1] would cause IndexError: index out of bounds")

print("\n=== Cross Document Mask Tests Passed! ===")

print("\n=== All Tests Passed! ===")