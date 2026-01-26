"""Test attention implementation against PyTorch reference"""

import numpy as np
import torch

def attention_pytorch(q, k, v, mask, head_dim):
    """PyTorch reference attention implementation"""
    batch_size, num_heads, seq_len, _ = q.shape
    
    # Scale
    scale = head_dim ** -0.5
    
    # Compute attention scores: Q @ K^T * scale
    attn_weights = torch.matmul(q, k.transpose(2, 3)) * scale
    
    # Apply mask (mask has -inf for padding, 0 for valid)
    if mask is not None:
        attn_weights = attn_weights + mask
    
    # Softmax
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    
    # Apply to values
    attn_output = torch.matmul(attn_weights, v)
    
    return attn_output, attn_weights

def attention_our_way(q, k, v, mask, head_dim):
    """Our attention implementation"""
    batch_size, seq_len, num_heads, _ = q.shape
    
    # Scale
    scale = 1.0 / np.sqrt(head_dim)
    
    # Transpose to (batch, num_heads, seq, head_dim)
    q_t = q.transpose(0, 2, 1, 3)
    k_t = k.transpose(0, 2, 1, 3)
    v_t = v.transpose(0, 2, 1, 3)
    
    # Compute scores: (batch, num_heads, seq_q, head_dim) @ (batch, num_heads, head_dim, seq_k)
    scores = np.einsum('bhqd,bhkd->bhqk', q_t, k_t) * scale
    
    # Apply mask
    if mask is not None:
        mask_expanded = mask.astype(np.float32)
        mask_expanded = (1.0 - mask_expanded) * -1e9
        mask_expanded = mask_expanded[:, None, None, :]
        scores = scores + mask_expanded
    
    # Softmax
    scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-8)
    
    # Apply to values
    out = np.einsum('bhqk,bhkd->bhqd', scores, v_t)
    
    # Transpose back
    out = out.transpose(0, 2, 1, 3)
    
    return out, scores

# Test
batch_size, seq_len, num_heads, head_dim = 1, 8, 2, 4

q_torch = torch.randn(batch_size, num_heads, seq_len, head_dim)
k_torch = torch.randn(batch_size, num_heads, seq_len, head_dim)
v_torch = torch.randn(batch_size, num_heads, seq_len, head_dim)
mask_torch = torch.ones(batch_size, seq_len, dtype=torch.float32)
mask_torch[0, 6:] = 0  # Mask last 2 tokens
mask_torch = (1.0 - mask_torch) * -1e9
mask_torch = mask_torch[:, None, None, :]

# PyTorch
attn_out_torch, attn_weights_torch = attention_pytorch(q_torch, k_torch, v_torch, mask_torch, head_dim)

# Ours (convert to our format: batch, seq, heads, dim)
q_np = q_torch.numpy().transpose(0, 2, 1, 3)
k_np = k_torch.numpy().transpose(0, 2, 1, 3)
v_np = v_torch.numpy().transpose(0, 2, 1, 3)
mask_np = (mask_torch[0, 0, 0, :].numpy() == 0).astype(np.float32)  # Convert back to 1/0 mask

attn_out_our, attn_weights_our = attention_our_way(q_np, k_np, v_np, mask_np, head_dim)

# Convert back
attn_out_our_torch = torch.from_numpy(attn_out_our).permute(0, 2, 1, 3)

# Compare
diff = torch.abs(attn_out_torch - attn_out_our_torch)
print("Attention Implementation Comparison:")
print(f"  Output difference: mean={diff.mean().item():.6f}, max={diff.max().item():.6f}")

if diff.mean() < 1e-4:
    print("  OK: Attention implementations match!")
else:
    print("  ERROR: Attention implementations differ!")
