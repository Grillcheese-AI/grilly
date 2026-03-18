# HYLA Attention

HYLAAttention is a softmax-free attention mechanism. Instead of the standard QK^T softmax attention, HYLA uses a linear attention formulation that avoids the O(N^2) softmax bottleneck.

---

## Usage

```python
from grilly import nn

hyla = nn.HYLAAttention(embed_dim=512, num_heads=8)
output = hyla(query, key, value)
```

---

## How It Works

Standard attention:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

HYLA replaces the softmax with a kernel function that allows computing the attention output in O(N) time:

```
HYLA(Q, K, V) = phi(Q) (phi(K)^T V) / (phi(Q) phi(K)^T 1)
```

Where `phi` is a feature map. The key insight: by computing `phi(K)^T V` first (a `d x d` matrix), we avoid the `N x N` attention matrix entirely.

**Benefits:**

- O(N) time and memory complexity (linear in sequence length)
- No attention matrix materialized
- Well-suited for long sequences where standard attention is prohibitive

---

## FNetMixing

FNetMixing replaces attention entirely with Fourier transforms:

```python
fnet = nn.FNetMixing(embed_dim=512)
output = fnet(x)  # x: (batch, seq_len, embed_dim)
```

Uses FFT along the sequence dimension for token mixing, and FFT along the feature dimension for channel mixing. No learnable parameters in the mixing step.

**Backed by**: `fft.glsl` and `ifft.glsl` compute shaders.

---

## When to Use What

| Mechanism | Complexity | Best For |
|-----------|-----------|----------|
| Flash Attention 2/3 | O(N) memory, O(N^2) compute | Standard transformers, moderate seq lengths |
| HYLA | O(N) time and memory | Very long sequences, linear-time models |
| FNetMixing | O(N log N) | Speed-critical, no learnable attention needed |
| Standard MHA | O(N^2) | When you need attention weights |
