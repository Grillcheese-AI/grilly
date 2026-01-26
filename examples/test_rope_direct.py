"""Test RoPE directly to verify it matches PyTorch"""

import numpy as np
import torch
from grilly.backend.attention import VulkanAttention
from grilly.backend.core import VulkanCore

# Create attention backend
core = VulkanCore()
attention = VulkanAttention(core)

# Test data
batch_size, seq_len, num_heads, head_dim = 1, 4, 2, 8
rope_base = 10000.0

q_torch = torch.randn(batch_size, num_heads, seq_len, head_dim)
position_ids = torch.arange(seq_len).unsqueeze(0)

# PyTorch RoPE
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

inv_freq = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
emb = torch.cat((freqs, freqs), dim=-1)
cos = emb.cos()
sin = emb.sin()

q_embed_torch = (q_torch * cos.unsqueeze(1)) + (rotate_half(q_torch) * sin.unsqueeze(1))

# Our RoPE (CPU)
q_np = q_torch.numpy().transpose(0, 2, 1, 3)  # (batch, seq, heads, dim)
position_ids_np = position_ids.numpy()
q_embed_our = attention._rope_cpu(q_np, position_ids_np, rope_base)

# Convert back
q_embed_our_torch = torch.from_numpy(q_embed_our).permute(0, 2, 1, 0)  # Wait, need to fix this
q_embed_our_torch = torch.from_numpy(q_embed_our).transpose(0, 2).transpose(1, 2)  # (batch, heads, seq, dim)

# Actually, let's just compare the numpy arrays directly
q_embed_torch_np = q_embed_torch.numpy().transpose(0, 2, 1, 3)  # (batch, seq, heads, dim)

diff = np.abs(q_embed_torch_np - q_embed_our)
print("RoPE Direct Comparison:")
print(f"  Difference: mean={diff.mean():.6f}, max={diff.max():.6f}")

if diff.mean() < 1e-5:
    print("  OK: RoPE matches PyTorch!")
else:
    print("  ERROR: RoPE differs from PyTorch!")
