# API Reference: nn

`grilly.nn` provides PyTorch-like neural network modules.

Source: [`nn/`](https://github.com/grillcheese-ai/grilly/tree/main/nn)

---

## Base

| Class | Description | Source |
|-------|-------------|--------|
| `Module` | Base class for all modules. Provides `parameters()`, `train()`, `eval()`, `state_dict()`, `load_state_dict()`, `zero_grad()`, `backward()`. | `nn/module.py` |
| `Parameter` | Tensor wrapper that registers as a module parameter. | `nn/parameter.py` |

---

## Linear Layers

| Class | Signature | Description |
|-------|-----------|-------------|
| `Linear` | `Linear(in_features, out_features, bias=True)` | Fully connected layer. |
| `Embedding` | `Embedding(num_embeddings, embedding_dim)` | Lookup table for integer indices. |
| `Dropout` | `Dropout(p=0.5)` | Randomly zeros elements during training. |

---

## Activations

| Class | Description |
|-------|-------------|
| `ReLU` | Rectified linear unit |
| `GELU` | Gaussian error linear unit |
| `SiLU` | Sigmoid linear unit (Swish) |
| `SwiGLU` | Gated linear unit with SiLU |
| `GCU` | Growing cosine unit |
| `RoSwish` | Rotary Swish |
| `Softmax` | Softmax normalization |
| `Softplus` | Smooth approximation to ReLU |

---

## Convolution

| Class | Signature | Description |
|-------|-----------|-------------|
| `Conv1d` | `Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0)` | 1D convolution. |
| `Conv2d` | `Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)` | 2D convolution. |

---

## Normalization

| Class | Signature | Description |
|-------|-----------|-------------|
| `LayerNorm` | `LayerNorm(normalized_shape, eps=1e-5)` | Layer normalization. |
| `RMSNorm` | `RMSNorm(normalized_shape, eps=1e-5)` | Root mean square normalization. |
| `BatchNorm1d` | `BatchNorm1d(num_features)` | Batch norm for 1D inputs. |
| `BatchNorm2d` | `BatchNorm2d(num_features)` | Batch norm for 2D inputs. |

---

## Attention

| Class | Signature | Description |
|-------|-----------|-------------|
| `MultiheadAttention` | `MultiheadAttention(embed_dim, num_heads, dropout=0.0)` | Standard multi-head attention. Returns `(output, weights)`. |
| `FlashAttention2` | `FlashAttention2(embed_dim, num_heads)` | Memory-efficient attention (no full N x N matrix). |
| `HYLAAttention` | `HYLAAttention(embed_dim, num_heads)` | Softmax-free linear attention. |
| `FNetMixing` | `FNetMixing(embed_dim)` | Fourier transform token mixing. |
| `SympFormerBlock` | `SympFormerBlock(embed_dim, num_heads, ...)` | Symplectic attention with momentum. |

---

## Recurrent

| Class | Signature | Description |
|-------|-----------|-------------|
| `LSTM` | `LSTM(input_size, hidden_size, num_layers=1)` | Long short-term memory. |
| `LSTMCell` | `LSTMCell(input_size, hidden_size)` | Single LSTM step. |
| `GRU` | `GRU(input_size, hidden_size)` | Gated recurrent unit. |
| `GRUCell` | `GRUCell(input_size, hidden_size)` | Single GRU step. |

---

## Pooling

| Class | Signature | Description |
|-------|-----------|-------------|
| `MaxPool2d` | `MaxPool2d(kernel_size, stride=None)` | Max pooling. |
| `AvgPool2d` | `AvgPool2d(kernel_size, stride=None)` | Average pooling. |
| `AdaptiveMaxPool2d` | `AdaptiveMaxPool2d(output_size)` | Adaptive max pooling. |
| `AdaptiveAvgPool2d` | `AdaptiveAvgPool2d(output_size)` | Adaptive average pooling. |

---

## Loss Functions

| Class | Signature | Description |
|-------|-----------|-------------|
| `MSELoss` | `MSELoss()` | Mean squared error. |
| `CrossEntropyLoss` | `CrossEntropyLoss()` | Cross-entropy (log-softmax + NLL). |
| `BCELoss` | `BCELoss()` | Binary cross-entropy. |

---

## Containers

| Class | Description |
|-------|-------------|
| `Sequential` | Chain modules in order. |
| `Residual` | Wrap a module with a skip connection: `output = x + module(x)`. |

---

## Transformer

| Class | Signature | Description |
|-------|-----------|-------------|
| `TransformerEncoderLayer` | `TransformerEncoderLayer(d_model, nhead)` | Standard transformer encoder layer. |
| `TransformerDecoderLayer` | `TransformerDecoderLayer(d_model, nhead)` | Transformer decoder with cross-attention. |
| `RoPE` | `RoPE(dim, max_seq_len)` | Rotary position embeddings. |

---

## LoRA

| Class | Signature | Description |
|-------|-----------|-------------|
| `LoRAConfig` | `LoRAConfig(r=8, alpha=16, dropout=0.1)` | LoRA configuration. |
| `LoRALinear` | `LoRALinear(base_linear, config)` | LoRA-adapted linear layer. |
| `LoRAAttention` | `LoRAAttention(base_attention, config)` | LoRA-adapted attention. |
| `LoRAModel` | `LoRAModel(model, config, target_modules)` | Apply LoRA to a full model. |
| `LoRAEmbedding` | `LoRAEmbedding(base_embedding, config)` | LoRA-adapted embedding. |

---

## SNN Neurons

| Class | Description |
|-------|-------------|
| `IFNode` | Integrate-and-Fire neuron. |
| `LIFNode` | Leaky Integrate-and-Fire with configurable `tau`. |
| `ParametricLIFNode` | LIF with learnable time constant. |
| `BaseNode` | Base class for spiking neurons. |
| `MemoryModule` | Base class for stateful SNN modules. |

---

## SNN Utilities

| Class | Description |
|-------|-------------|
| `ATan`, `Sigmoid`, `FastSigmoid` | Surrogate gradient functions. |
| `MultiStepContainer` | Run module over T time steps. |
| `SeqToANNContainer` | Wrap ANN layer for temporal data. |
| `Converter`, `VoltageScaler` | ANN-to-SNN conversion. |
| `Monitor` | Record neuron states during simulation. |
| `SynapseFilter`, `STPSynapse`, `DualTimescaleSynapse` | Synapse models. |

---

## Multimodal

| Class | Description |
|-------|-------------|
| `BottleneckFusion` | Bottleneck fusion of modalities. |
| `CrossModalAttentionFusion` | Cross-attention between modalities. |
| `FlamingoFusion` | Flamingo-style gated cross-attention. |
| `ImageBindFusion` | ImageBind-style joint embedding. |
| `PerceiverIO` | Perceiver IO for arbitrary modalities. |
| `PerceiverResampler` | Perceiver resampler (fixed-size output). |
| `VisionLanguageModel`, `VLMLayer` | Vision-language model components. |

---

## Autograd

| Symbol | Description |
|--------|-------------|
| `Variable` | Tracked tensor with automatic differentiation. |
| `tensor(data)` | Create a Variable from a list or numpy array. |
| `no_grad()` | Context manager to disable gradient tracking. |
| `enable_grad()` | Context manager to re-enable gradient tracking. |
| `Function` | Base class for custom autograd functions. |
| `zeros`, `ones`, `randn`, `rand`, `eye`, `full`, `arange`, `linspace` | Factory functions. |

See [Autograd Guide](../guide/autograd.md) for usage details.

---

## Projection Heads

| Class | Description |
|-------|-------------|
| `ProjectionHeads` | Multi-head subspace projection embeddings. |
