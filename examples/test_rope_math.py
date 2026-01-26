"""Test if our RoPE math matches PyTorch rotate_half approach"""

import numpy as np
import torch

# Test the mathematical equivalence
head_dim = 8
dim_pair = 0  # First pair

# Our way: q'[2i] = q[2i] * cos(θ) - q[2i+1] * sin(θ)
#          q'[2i+1] = q[2i] * sin(θ) + q[2i+1] * cos(θ)

# PyTorch way: rotate_half(q) = [-q[head_dim//2:], q[:head_dim//2]]
#              q_embed = q * cos + rotate_half(q) * sin

# For head_dim=8, dim_pair=0:
# Our way: q'[0] = q[0] * cos - q[1] * sin
#          q'[1] = q[0] * sin + q[1] * cos

# PyTorch way: rotate_half(q) = [-q[4], -q[5], -q[6], -q[7], q[0], q[1], q[2], q[3]]
#              For first pair: q_embed[0] = q[0] * cos + (-q[4]) * sin
#                              q_embed[1] = q[1] * cos + (-q[5]) * sin

# Wait, these are different! Let me check the actual implementation...

# Create test vector
q = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
cos = torch.tensor([0.5, 0.6, 0.7, 0.8])
sin = torch.tensor([0.866, 0.8, 0.714, 0.6])

# Expand cos/sin to full head_dim (they're computed for head_dim//2 pairs)
cos_full = torch.cat([cos, cos])
sin_full = torch.cat([sin, sin])

# PyTorch way
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

q_embed_pytorch = (q * cos_full) + (rotate_half(q) * sin_full)
print("PyTorch rotate_half approach:")
print(f"  q: {q}")
print(f"  rotate_half(q): {rotate_half(q)}")
print(f"  q_embed: {q_embed_pytorch}")

# Our way (pair-wise rotation)
q_our = q.clone()
for dim_pair in range(head_dim // 2):
    dim_even = dim_pair * 2
    dim_odd = dim_pair * 2 + 1
    x_even = q_our[dim_even]
    x_odd = q_our[dim_odd]
    cos_val = cos_full[dim_even]  # Should use same cos for pair
    sin_val = sin_full[dim_even]
    q_our[dim_even] = x_even * cos_val - x_odd * sin_val
    q_our[dim_odd] = x_even * sin_val + x_odd * cos_val

print("\nOur pair-wise rotation approach:")
print(f"  q_embed: {q_our}")

print(f"\nDifference: {torch.abs(q_embed_pytorch - q_our)}")
print(f"Are they the same? {torch.allclose(q_embed_pytorch, q_our, atol=1e-5)}")
