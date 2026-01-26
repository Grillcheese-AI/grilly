"""Test fixed RoPE implementation"""

import numpy as np
import torch

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_pytorch(q, cos, sin):
    """PyTorch reference implementation"""
    return (q * cos) + (rotate_half(q) * sin)

def apply_rope_our_way_fixed(q_or_k, position_ids, rope_base=10000.0, rope_scaling=1.0):
    """Our fixed implementation using rotate_half approach"""
    batch_size, seq_len, num_heads, head_dim = q_or_k.shape
    result = q_or_k.copy()
    
    # Compute inv_freq (same as ModernBERT)
    inv_freq = 1.0 / (rope_base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    
    for b in range(batch_size):
        for s in range(seq_len):
            pos = float(position_ids[b, s]) / rope_scaling
            for h in range(num_heads):
                # Compute cos/sin for this position
                freqs = pos * inv_freq
                freqs_full = np.concatenate([freqs, freqs])
                cos_vals = np.cos(freqs_full)
                sin_vals = np.sin(freqs_full)
                
                # Get current q/k vector
                qk_vec = q_or_k[b, s, h, :]
                
                # rotate_half: split into two halves and rotate
                qk_first_half = qk_vec[:head_dim//2]
                qk_second_half = qk_vec[head_dim//2:]
                rotated = np.concatenate([-qk_second_half, qk_first_half])
                
                # Apply: q_embed = (q * cos) + (rotate_half(q) * sin)
                result[b, s, h, :] = (qk_vec * cos_vals) + (rotated * sin_vals)
    
    return result

# Test
batch_size, seq_len, num_heads, head_dim = 1, 4, 2, 8
rope_base = 10000.0

q_torch = torch.randn(batch_size, num_heads, seq_len, head_dim)
position_ids = torch.arange(seq_len).unsqueeze(0)

# Compute cos/sin using ModernBERT method
inv_freq = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
emb = torch.cat((freqs, freqs), dim=-1)
cos = emb.cos()
sin = emb.sin()

# Apply PyTorch RoPE
q_embed_torch = apply_rotary_pos_emb_pytorch(q_torch, cos.unsqueeze(1), sin.unsqueeze(1))

# Convert to numpy and apply our RoPE
q_np = q_torch.numpy().transpose(0, 2, 1, 3)  # (batch, seq, heads, dim)
position_ids_np = position_ids.numpy()
q_embed_our = apply_rope_our_way_fixed(q_np, position_ids_np, rope_base)

# Convert back to torch format
q_embed_our_torch = torch.from_numpy(q_embed_our).permute(0, 2, 1, 3)  # (batch, heads, seq, dim)

# Compare
diff = torch.abs(q_embed_torch - q_embed_our_torch)
print("Fixed RoPE Implementation Comparison:")
print(f"  Difference: mean={diff.mean().item():.6f}, max={diff.max().item():.6f}")

if diff.mean() < 1e-5:
    print("  OK: RoPE implementations match!")
else:
    print("  ERROR: RoPE implementations still differ!")
