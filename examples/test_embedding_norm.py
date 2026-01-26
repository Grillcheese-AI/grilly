"""Test embedding norm application"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from grilly.utils.vulkan_sentence_transformer import VulkanSentenceTransformer

model_name = "ibm-granite/granite-embedding-english-r2"
text = "test"

# Load models
st_model = SentenceTransformer(model_name)
vulkan_model = VulkanSentenceTransformer(model_name)

# Get underlying models
auto_model = st_model._modules['0'].auto_model
emb_module = auto_model.embeddings

# Tokenize
encoded_st = st_model.tokenizer(text, return_tensors='pt')
encoded_vulkan = vulkan_model._tokenize([text])

print("Input IDs match:", np.array_equal(encoded_st['input_ids'].numpy(), encoded_vulkan['input_ids']))

# Get original embeddings
with torch.no_grad():
    orig_emb = emb_module.tok_embeddings(encoded_st['input_ids'])
    print(f"\nOriginal tok_embeddings:")
    print(f"  Mean: {orig_emb.mean().item():.6f}, Std: {orig_emb.std().item():.6f}")
    
    orig_emb_norm = emb_module.norm(orig_emb)
    print(f"\nOriginal after norm:")
    print(f"  Mean: {orig_emb_norm.mean().item():.6f}, Std: {orig_emb_norm.std().item():.6f}")
    print(f"  Min: {orig_emb_norm.min().item():.6f}, Max: {orig_emb_norm.max().item():.6f}")

# Get Vulkan embeddings
vulkan_emb = vulkan_model._embedding_lookup(encoded_vulkan['input_ids'])
print(f"\nVulkan tok_embeddings:")
print(f"  Mean: {vulkan_emb.mean():.6f}, Std: {vulkan_emb.std():.6f}")

vulkan_emb_norm = vulkan_model._apply_modernbert_embedding_norm(vulkan_emb)
print(f"\nVulkan after norm:")
print(f"  Mean: {vulkan_emb_norm.mean():.6f}, Std: {vulkan_emb_norm.std():.6f}")
print(f"  Min: {vulkan_emb_norm.min():.6f}, Max: {vulkan_emb_norm.max():.6f}")

# Compare
diff = np.abs(orig_emb_norm.detach().cpu().numpy() - vulkan_emb_norm)
print(f"\nDifference after norm:")
print(f"  Mean: {diff.mean():.6f}, Max: {diff.max():.6f}")
print(f"  First 10 elements diff: {diff[0, 0, :10]}")

# Check norm weights
norm = emb_module.norm
norm_weight = norm.weight.detach().cpu().numpy().astype(np.float32)
norm_bias = norm.bias.detach().cpu().numpy().astype(np.float32) if norm.bias is not None else np.zeros(norm.weight.shape[0], dtype=np.float32)
print(f"\nNorm weights:")
print(f"  Weight mean: {norm_weight.mean():.6f}, std: {norm_weight.std():.6f}")
print(f"  Bias mean: {norm_bias.mean():.6f}, std: {norm_bias.std():.6f}")
print(f"  Eps: {norm.eps}")
