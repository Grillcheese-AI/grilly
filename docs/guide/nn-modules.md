# Neural Network Modules

All layers inherit from `nn.Module`, providing `parameters()`, `train()`/`eval()`, `state_dict()`, and `load_state_dict()`.

---

## Linear Layers

```python
from grilly import nn

linear = nn.Linear(in_features=784, out_features=256)
output = linear(x)  # x: (batch, 784) -> output: (batch, 256)
```

`nn.Embedding` maps integer indices to dense vectors:

```python
embed = nn.Embedding(num_embeddings=10000, embedding_dim=512)
output = embed(token_ids)  # token_ids: (batch, seq_len) -> (batch, seq_len, 512)
```

---

## Activations

| Module | Description |
|--------|-------------|
| `nn.ReLU()` | Rectified linear unit |
| `nn.GELU()` | Gaussian error linear unit |
| `nn.SiLU()` | Sigmoid linear unit (Swish) |
| `nn.SwiGLU()` | SwiGLU gated activation |
| `nn.GCU()` | Growing cosine unit |
| `nn.RoSwish()` | Rotary Swish |
| `nn.Softmax()` | Softmax normalization |
| `nn.Softplus()` | Smooth approximation to ReLU |

```python
model = nn.Sequential(
    nn.Linear(512, 512),
    nn.GELU(),
    nn.Linear(512, 256),
)
```

---

## Convolution

```python
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
output = conv(x)  # x: (batch, 3, H, W) -> output: (batch, 64, H, W)
```

Available: `nn.Conv1d`, `nn.Conv2d`.

---

## Normalization

```python
ln = nn.LayerNorm(normalized_shape=512)
rms = nn.RMSNorm(normalized_shape=512)
bn1 = nn.BatchNorm1d(num_features=256)
bn2 = nn.BatchNorm2d(num_features=64)
```

---

## Attention

```python
# Standard multi-head attention
mha = nn.MultiheadAttention(embed_dim=512, num_heads=8)
output = mha(query, key, value)

# Flash Attention 2 (memory-efficient)
fa = nn.FlashAttention2(embed_dim=512, num_heads=8)
output = fa(query, key, value)
```

See [Flash Attention](../advanced/flash-attention.md) and [HYLA Attention](../advanced/hyla.md) for advanced attention mechanisms.

---

## Recurrent Layers

```python
lstm = nn.LSTM(input_size=256, hidden_size=512, num_layers=2)
gru = nn.GRU(input_size=256, hidden_size=512)

# Cell-level access
lstm_cell = nn.LSTMCell(input_size=256, hidden_size=512)
gru_cell = nn.GRUCell(input_size=256, hidden_size=512)
```

---

## Pooling

```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)
avg = nn.AvgPool2d(kernel_size=2, stride=2)
adaptive = nn.AdaptiveMaxPool2d(output_size=(1, 1))
```

---

## Loss Functions

```python
mse = nn.MSELoss()
ce = nn.CrossEntropyLoss()
bce = nn.BCELoss()

loss = ce(logits, targets)
grad = ce.backward(np.ones_like(loss), logits, targets)
```

---

## Containers

```python
# Sequential: chain modules
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

# Residual: skip connection
block = nn.Residual(
    nn.Sequential(
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
    )
)
```

---

## Dropout

```python
drop = nn.Dropout(p=0.1)
output = drop(x)  # Zeros elements with probability 0.1 during training
```

---

## Transformer Layers

```python
encoder = nn.TransformerEncoderLayer(d_model=512, nhead=8)
decoder = nn.TransformerDecoderLayer(d_model=512, nhead=8)

# With rotary position embeddings
rope = nn.RoPE(dim=64, max_seq_len=2048)
```

---

## LoRA (Low-Rank Adaptation)

```python
from grilly.nn import LoRALinear, LoRAModel, LoRAConfig

config = LoRAConfig(r=8, alpha=16, dropout=0.1)
lora_linear = LoRALinear(base_linear, config)

# Or apply LoRA to an entire model
lora_model = LoRAModel(model, config, target_modules=["linear"])
```

---

## Multimodal Fusion

```python
from grilly.nn import PerceiverIO, CrossModalAttentionFusion, FlamingoFusion

# Perceiver IO for arbitrary input modalities
perceiver = PerceiverIO(input_dim=768, latent_dim=512, num_latents=64)

# Cross-modal attention
fusion = CrossModalAttentionFusion(dim=512, num_heads=8)
```

Available: `BottleneckFusion`, `CrossModalAttentionFusion`, `FlamingoFusion`, `ImageBindFusion`, `PerceiverIO`, `PerceiverResampler`, `VisionLanguageModel`.

---

## Full Module List

See [API Reference: nn](../api/nn.md) for the complete list of all exported classes and functions.
