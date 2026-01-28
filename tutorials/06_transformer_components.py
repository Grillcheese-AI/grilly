"""
Transformer Components Tutorial
===============================

Demonstrates GPU-accelerated transformer building blocks:
- LayerNorm
- Linear projections
- GELU activation
- Softmax attention

These are the core operations used in BERT, GPT, LLaMA, etc.
"""

import _path_setup  # noqa: F401 - must be first to ensure grilly is importable

import numpy as np
import time

from grilly import Compute

compute = Compute()
print(f"GPU: {compute.core.device_properties.deviceName}")

# Transformer dimensions
batch_size = 8
seq_len = 128
d_model = 256
n_heads = 4
d_k = d_model // n_heads  # 64

print(f"\nTransformer config: batch={batch_size}, seq={seq_len}, d_model={d_model}, heads={n_heads}")
print("=" * 70)

# Create sample input (like token embeddings)
np.random.seed(42)
x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)

# LayerNorm parameters
gamma = np.ones(d_model, dtype=np.float32)
beta = np.zeros(d_model, dtype=np.float32)

# ============ LayerNorm ============
print("\n1. LayerNorm")
x_flat = x.reshape(-1, d_model)  # (batch*seq, d_model)

start = time.perf_counter()
normed = compute.fnn.layernorm(x_flat, gamma, beta)
layernorm_time = (time.perf_counter() - start) * 1000

print(f"   Input shape: {x_flat.shape}")
print(f"   Output mean: {normed.mean():.6f} (should be ~0)")
print(f"   Output std:  {normed.std():.6f} (should be ~1)")
print(f"   Time: {layernorm_time:.2f}ms")

# ============ Linear Projection (Q, K, V) ============
print("\n2. Linear Projections (Q, K, V)")

# Initialize projection weights
w_q = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
w_k = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
w_v = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
b_q = np.zeros(d_model, dtype=np.float32)
b_k = np.zeros(d_model, dtype=np.float32)
b_v = np.zeros(d_model, dtype=np.float32)

start = time.perf_counter()
Q = compute.fnn.linear(normed, w_q, b_q)
K = compute.fnn.linear(normed, w_k, b_k)
V = compute.fnn.linear(normed, w_v, b_v)
proj_time = (time.perf_counter() - start) * 1000

print(f"   Q shape: {Q.shape}")
print(f"   K shape: {K.shape}")
print(f"   V shape: {V.shape}")
print(f"   Time (3 projections): {proj_time:.2f}ms")

# ============ Attention Scores ============
print("\n3. Attention Scores")

# Reshape for multi-head attention
Q_heads = Q.reshape(batch_size, seq_len, n_heads, d_k).transpose(0, 2, 1, 3)  # (B, H, S, D)
K_heads = K.reshape(batch_size, seq_len, n_heads, d_k).transpose(0, 2, 1, 3)
V_heads = V.reshape(batch_size, seq_len, n_heads, d_k).transpose(0, 2, 1, 3)

# Compute attention scores: Q @ K^T / sqrt(d_k)
scale = 1.0 / np.sqrt(d_k)
start = time.perf_counter()
scores = np.matmul(Q_heads, K_heads.transpose(0, 1, 3, 2)) * scale  # (B, H, S, S)
score_time = (time.perf_counter() - start) * 1000

print(f"   Scores shape: {scores.shape} (batch, heads, seq, seq)")
print(f"   Time: {score_time:.2f}ms")

# ============ Softmax ============
print("\n4. Softmax Attention Weights")

# Flatten for softmax: (B*H*S, S)
scores_flat = scores.reshape(-1, seq_len)

start = time.perf_counter()
attn_weights = compute.fnn.activation_softmax(scores_flat)
softmax_time = (time.perf_counter() - start) * 1000

print(f"   Weights sum per row: {attn_weights.sum(axis=-1)[:3]}... (should be 1.0)")
print(f"   Time: {softmax_time:.2f}ms")

# ============ FFN with GELU ============
print("\n5. Feed-Forward Network (Linear + GELU)")

d_ff = d_model * 4  # Standard FFN expansion
w_ff1 = np.random.randn(d_ff, d_model).astype(np.float32) * 0.02
b_ff1 = np.zeros(d_ff, dtype=np.float32)

start = time.perf_counter()
# Using fused Linear+GELU operation
ffn_out = compute.fnn.fused_linear_gelu(normed, w_ff1, b_ff1)
fused_time = (time.perf_counter() - start) * 1000

print(f"   Input:  {normed.shape}")
print(f"   Output: {ffn_out.shape}")
print(f"   Time (fused): {fused_time:.2f}ms")

# Compare with separate operations
start = time.perf_counter()
ffn_sep = compute.fnn.linear(normed, w_ff1, b_ff1)
ffn_sep = compute.fnn.activation_gelu(ffn_sep)
separate_time = (time.perf_counter() - start) * 1000

print(f"   Time (separate): {separate_time:.2f}ms")
print(f"   Fused speedup: {separate_time/fused_time:.2f}x")

# ============ Summary ============
print("\n" + "=" * 70)
print("Summary: Transformer Component Times")
print(f"  LayerNorm:    {layernorm_time:.2f}ms")
print(f"  Q/K/V proj:   {proj_time:.2f}ms")
print(f"  Attn scores:  {score_time:.2f}ms")
print(f"  Softmax:      {softmax_time:.2f}ms")
print(f"  FFN (fused):  {fused_time:.2f}ms")

# Buffer pool stats
pool = compute.fnn.buffer_pool
if pool:
    stats = pool.get_stats()
    print(f"\nBuffer pool: {stats['total_acquired']} ops, {stats['hit_rate']:.1%} hit rate")
