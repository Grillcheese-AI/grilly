# SympFormer

`SympFormerBlock` applies symplectic (momentum-based) integration to transformer attention, treating the attention update as a Hamiltonian system.

---

## Usage

```python
from grilly import nn

block = nn.SympFormerBlock(
    embed_dim=512,
    num_heads=8,
    h_x_init=0.1,   # Position step size
    h_y_init=0.1,   # Momentum step size
    c_log=3.0,       # Logarithmic damping
    c_lin=0.1,       # Linear damping
)

output = block(x)  # x: (batch, seq_len, embed_dim)
```

---

## How It Works

Standard transformer layers apply a residual update:

```
x = x + Attention(x)
```

SympFormer treats this as a symplectic integrator. It maintains a "momentum" state `y` alongside the position `x`:

```
y_{n+1} = y_n - h_y * (c_log * y_n / (1 + |y_n|) + c_lin * y_n) + h_y * grad_x H
x_{n+1} = x_n + h_x * y_{n+1}
```

The damping terms (`c_log`, `c_lin`) provide stable convergence. The Hamiltonian `H` is defined by the attention function.

**Benefits:**

- Better convergence on long training runs
- Built-in momentum prevents oscillation
- Symplectic structure preserves energy (stable dynamics)

---

## Selective Momentum

`SelectiveMomentumAttention` combines SympFormer with TAPPA q-similarity to apply momentum only where it helps:

```python
from grilly.nn.selective_momentum import SelectiveMomentumAttention

sel_attn = SelectiveMomentumAttention(
    attention=nn.MultiheadAttention(512, 8),
    embed_dim=512,
    num_heads=8,
    threshold=0.5,  # q-similarity threshold
)

output = sel_attn(query, key, value)
```

Heads with q-similarity below the threshold (unpredictable, retrieval heads) get momentum acceleration. Predictable heads skip the overhead. This gives SympFormer's convergence benefit at roughly 1x wall-clock cost instead of 2.4x.

---

## Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `embed_dim` | required | Model dimension |
| `num_heads` | required | Number of attention heads |
| `h_x_init` | `0.1` | Position step size (learnable) |
| `h_y_init` | `0.1` | Momentum step size (learnable) |
| `c_log` | `3.0` | Logarithmic damping coefficient |
| `c_lin` | `0.1` | Linear damping coefficient |
