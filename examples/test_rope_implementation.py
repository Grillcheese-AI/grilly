"""Test RoPE implementation against PyTorch reference"""

import numpy as np
import torch

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_pytorch(q, k, cos, sin):
    """PyTorch reference implementation"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rope_our_way(q_or_k, position_ids, rope_base=10000.0, rope_scaling=1.0):
    """Our current implementation"""
    batch_size, seq_len, num_heads, head_dim = q_or_k.shape
    result = q_or_k.copy()
    
    for b in range(batch_size):
        for s in range(seq_len):
            pos = int(position_ids[b, s])
            for h in range(num_heads):
                for dim_pair in range(head_dim // 2):
                    dim_even = dim_pair * 2
                    dim_odd = dim_pair * 2 + 1
                    
                    freq_exp = -2.0 * dim_pair / head_dim
                    freq = rope_base ** freq_exp
                    theta = (pos / rope_scaling) * freq
                    
                    x_even = q_or_k[b, s, h, dim_even]
                    x_odd = q_or_k[b, s, h, dim_odd]
                    
                    cos_val = np.cos(theta)
                    sin_val = np.sin(theta)
                    result[b, s, h, dim_even] = x_even * cos_val - x_odd * sin_val
                    result[b, s, h, dim_odd] = x_even * sin_val + x_odd * cos_val
    
    return result

# Test with small example
batch_size, seq_len, num_heads, head_dim = 1, 4, 2, 8
rope_base = 10000.0

# Create test data
q_torch = torch.randn(batch_size, num_heads, seq_len, head_dim)
k_torch = torch.randn(batch_size, num_heads, seq_len, head_dim)

# Compute cos/sin using ModernBERT method
position_ids = torch.arange(seq_len).unsqueeze(0)
inv_freq = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
emb = torch.cat((freqs, freqs), dim=-1)
cos = emb.cos()
sin = emb.sin()

# Apply PyTorch RoPE
q_embed_torch, k_embed_torch = apply_rotary_pos_emb_pytorch(q_torch, k_torch, cos.unsqueeze(1), sin.unsqueeze(1))

# Convert to numpy and apply our RoPE
q_np = q_torch.numpy().transpose(0, 2, 1, 3)  # (batch, seq, heads, dim)
k_np = k_torch.numpy().transpose(0, 2, 1, 3)
position_ids_np = position_ids.numpy()

q_embed_our = apply_rope_our_way(q_np, position_ids_np, rope_base)
k_embed_our = apply_rope_our_way(k_np, position_ids_np, rope_base)

# Convert back to torch format
q_embed_our_torch = torch.from_numpy(q_embed_our).permute(0, 2, 1, 3)  # (batch, heads, seq, dim)
k_embed_our_torch = torch.from_numpy(k_embed_our).permute(0, 2, 1, 3)

# Compare
diff_q = torch.abs(q_embed_torch - q_embed_our_torch)
diff_k = torch.abs(k_embed_torch - k_embed_our_torch)

print("RoPE Implementation Comparison:")
print(f"  Q difference: mean={diff_q.mean().item():.6f}, max={diff_q.max().item():.6f}")
print(f"  K difference: mean={diff_k.mean().item():.6f}, max={diff_k.max().item():.6f}")

# Check if they're equivalent
if diff_q.mean() < 1e-5 and diff_k.mean() < 1e-5:
    print("  ✓ RoPE implementations match!")
else:
    print("  ✗ RoPE implementations differ - need to fix!")
