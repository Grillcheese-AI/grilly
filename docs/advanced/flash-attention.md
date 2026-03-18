# Flash Attention

Grilly implements Flash Attention 2 and Flash Attention 3 as Vulkan compute shaders, providing memory-efficient attention without materializing the full N x N attention matrix.

---

## Flash Attention 2

```python
from grilly import nn
import grilly.functional as F

# Module API
fa = nn.FlashAttention2(embed_dim=512, num_heads=8)
output = fa(query, key, value)

# Functional API
output = F.flash_attention2(q, k, v)
```

Flash Attention 2 tiles the attention computation into blocks that fit in GPU shared memory. Instead of computing the full `(seq_len, seq_len)` attention matrix, it processes blocks of Q and KV, accumulating the output incrementally.

**Key properties:**

- O(N) memory instead of O(N^2) for the attention matrix
- Numerically equivalent to standard attention (online softmax trick)
- Dispatched via `flash-attention2.glsl` compute shader

---

## Flash Attention 3

Flash Attention 3 adds subgroup (wave/warp) acceleration:

- Uses Vulkan subgroup operations for intra-wave reductions
- Cooperative matrix operations where supported (NVIDIA, Intel)
- Further tiling optimizations for long sequences

Available through the backend:

```python
backend = grilly.Compute()
output = backend.attention.flash_attention3(q, k, v, num_heads=8, head_dim=64)
```

---

## Standard Multi-Head Attention

For cases where you need the attention weights (visualization, analysis):

```python
mha = nn.MultiheadAttention(embed_dim=512, num_heads=8)
output, attn_weights = mha(query, key, value, mask=causal_mask)
```

Uses three shaders: `attention-scores.glsl`, `activation-softmax.glsl`, `attention-output.glsl`.

---

## RoPE (Rotary Position Embeddings)

```python
rope = nn.RoPE(dim=64, max_seq_len=2048)
q_rot, k_rot = rope(q, k, offset=0)
```

Applies rotary embeddings to query and key tensors. Used by modern transformer architectures (LLaMA, Mistral).

---

## Causal Masking

```python
# Create causal mask
mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)

output, weights = mha(query, key, value, mask=mask)
```

The attention shaders support arbitrary masks via `attention-mask.glsl`.

---

## Architecture-Specific Variants

The shader registry provides architecture-specific attention output shaders:

| Architecture | Shader | Notes |
|---|---|---|
| Generic | `attention-output.glsl` | Default |
| GPT | `attention-output-gpt.glsl` | Causal, autoregressive |
| T5 | `attention-output-t5.glsl` | Encoder-decoder relative position |
