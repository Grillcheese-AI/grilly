"""Test QKV computation to verify it matches ModernBERT"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

model_name = "ibm-granite/granite-embedding-english-r2"
text = "test"

# Load model
st_model = SentenceTransformer(model_name)
auto_model = st_model._modules['0'].auto_model

# Tokenize
encoded = st_model.tokenizer(text, return_tensors='pt', padding=True, truncation=True)

# Get embeddings
with torch.no_grad():
    x = auto_model.embeddings.tok_embeddings(encoded['input_ids'])
    x = auto_model.embeddings.norm(x)
    
    # Get layer 0
    layer0 = auto_model.layers[0]
    attn = layer0.attn
    
    # ModernBERT way: Wqkv projects to 3*hidden_size, then split
    x_norm = x  # attn_norm is Identity for layer 0
    qkv = torch.nn.functional.linear(x_norm, attn.Wqkv.weight, attn.Wqkv.bias)
    # qkv shape: (batch, seq, 3*hidden_size) = (1, seq, 2304)
    
    # Reshape: (batch, seq, 3, num_heads, head_dim)
    batch_size, seq_len = qkv.shape[0], qkv.shape[1]
    num_heads = 12
    head_dim = 64
    qkv_reshaped = qkv.view(batch_size, seq_len, 3, num_heads, head_dim)
    
    # Split: unbind along dim=2
    q_combined, k_combined, v_combined = qkv_reshaped.unbind(dim=2)
    # Each is (batch, seq, num_heads, head_dim)
    
    # Transpose to (batch, num_heads, seq, head_dim) for attention
    q_torch = q_combined.transpose(1, 2)  # (batch, num_heads, seq, head_dim)
    k_torch = k_combined.transpose(1, 2)
    v_torch = v_combined.transpose(1, 2)
    
    print("ModernBERT QKV computation:")
    print(f"  qkv shape: {qkv.shape}")
    print(f"  qkv_reshaped shape: {qkv_reshaped.shape}")
    print(f"  q_combined shape: {q_combined.shape}")
    print(f"  q_torch shape (after transpose): {q_torch.shape}")
    print(f"  Q mean: {q_torch.mean().item():.6f}, std: {q_torch.std().item():.6f}")

# Our way: split Wqkv into separate Q, K, V weights
wqkv = attn.Wqkv.weight.detach().cpu().numpy().astype(np.float32)  # (2304, 768)
wqkv_bias = attn.Wqkv.bias.detach().cpu().numpy().astype(np.float32)
hidden_size = 768

q_weight = wqkv[:hidden_size, :]
q_bias = wqkv_bias[:hidden_size]
k_weight = wqkv[hidden_size:2*hidden_size, :]
k_bias = wqkv_bias[hidden_size:2*hidden_size]
v_weight = wqkv[2*hidden_size:, :]
v_bias = wqkv_bias[2*hidden_size:]

x_np = x.detach().cpu().numpy()
q_our = (x_np @ q_weight.T) + q_bias
k_our = (x_np @ k_weight.T) + k_bias
v_our = (x_np @ v_weight.T) + v_bias

# Reshape
q_our = q_our.reshape(batch_size, seq_len, num_heads, head_dim)
k_our = k_our.reshape(batch_size, seq_len, num_heads, head_dim)
v_our = v_our.reshape(batch_size, seq_len, num_heads, head_dim)

# Transpose to (batch, num_heads, seq, head_dim)
q_our_t = q_our.transpose(0, 2, 1, 3)

print("\nOur QKV computation:")
print(f"  q_our shape: {q_our.shape}")
print(f"  q_our_t shape (after transpose): {q_our_t.shape}")
print(f"  Q mean: {q_our_t.mean():.6f}, std: {q_our_t.std():.6f}")

# Compare
diff = np.abs(q_torch.numpy() - q_our_t)
print(f"\nQ difference: mean={diff.mean():.6f}, max={diff.max():.6f}")

if diff.mean() < 1e-5:
    print("  OK: QKV computation matches!")
else:
    print("  ERROR: QKV computation differs!")
