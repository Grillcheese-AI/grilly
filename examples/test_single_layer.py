"""Test a single layer to find where divergence occurs"""

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

# Tokenize
encoded_st = st_model.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
encoded_vulkan = vulkan_model._tokenize([text])

# Get embeddings with norm
with torch.no_grad():
    orig_x = auto_model.embeddings.tok_embeddings(encoded_st['input_ids'])
    orig_x = auto_model.embeddings.norm(orig_x)

vulkan_x = vulkan_model._embedding_lookup(encoded_vulkan['input_ids'])
vulkan_x = vulkan_model._apply_modernbert_embedding_norm(vulkan_x)

print("After embedding norm:")
print(f"  Original: mean={orig_x.mean().item():.6f}, std={orig_x.std().item():.6f}")
print(f"  Vulkan: mean={vulkan_x.mean():.6f}, std={vulkan_x.std():.6f}")
diff = np.abs(orig_x.detach().cpu().numpy() - vulkan_x)
print(f"  Difference: mean={diff.mean():.6f}, max={diff.max():.6f}")

# Test first layer only
print("\n=== Testing Layer 0 ===")
layer0 = auto_model.layers[0]
layer_weights = vulkan_model.layers[0]

# Get Q, K, V weights
with torch.no_grad():
    attn = layer0.attn
    wqkv = attn.Wqkv.weight.detach().cpu().numpy().astype(np.float32)  # (2304, 768)
    wqkv_bias = attn.Wqkv.bias.detach().cpu().numpy().astype(np.float32) if attn.Wqkv.bias is not None else np.zeros(2304, dtype=np.float32)
    
    # Split
    hidden_size = 768
    orig_q_weight = wqkv[:hidden_size, :]
    orig_k_weight = wqkv[hidden_size:2*hidden_size, :]
    orig_v_weight = wqkv[2*hidden_size:, :]
    
    orig_q_bias = wqkv_bias[:hidden_size]
    orig_k_bias = wqkv_bias[hidden_size:2*hidden_size]
    orig_v_bias = wqkv_bias[2*hidden_size:]

print("Q weights:")
print(f"  Original: mean={orig_q_weight.mean():.6f}, std={orig_q_weight.std():.6f}")
print(f"  Vulkan: mean={layer_weights['attn_q_weight'].mean():.6f}, std={layer_weights['attn_q_weight'].std():.6f}")
q_diff = np.abs(orig_q_weight - layer_weights['attn_q_weight'])
print(f"  Difference: mean={q_diff.mean():.6f}, max={q_diff.max():.6f}")

# Compute Q, K, V
with torch.no_grad():
    # Original: attn_norm is Identity, so x_norm = x
    x_norm_orig = orig_x
    # Wqkv projects to 3*hidden_size, then we split
    qkv_orig = torch.nn.functional.linear(x_norm_orig, attn.Wqkv.weight, attn.Wqkv.bias)
    qkv_orig = qkv_orig.view(1, -1, 3, 12, 64)  # (batch, seq, 3, heads, head_dim)
    q_orig = qkv_orig[:, :, 0, :, :].contiguous().view(1, -1, 768)  # (batch, seq, hidden)
    k_orig = qkv_orig[:, :, 1, :, :].contiguous().view(1, -1, 768)
    v_orig = qkv_orig[:, :, 2, :, :].contiguous().view(1, -1, 768)

# Vulkan
from grilly import functional
x_norm_vulkan = vulkan_x
q_vulkan = functional.linear(x_norm_vulkan, layer_weights['attn_q_weight'], layer_weights['attn_q_bias'])
k_vulkan = functional.linear(x_norm_vulkan, layer_weights['attn_k_weight'], layer_weights['attn_k_bias'])
v_vulkan = functional.linear(x_norm_vulkan, layer_weights['attn_v_weight'], layer_weights['attn_v_bias'])

print("\nQ, K, V after projection:")
print(f"  Original Q: mean={q_orig.mean().item():.6f}, std={q_orig.std().item():.6f}")
print(f"  Vulkan Q: mean={q_vulkan.mean():.6f}, std={q_vulkan.std():.6f}")
q_proj_diff = np.abs(q_orig.detach().cpu().numpy() - q_vulkan)
print(f"  Q difference: mean={q_proj_diff.mean():.6f}, max={q_proj_diff.max():.6f}")
