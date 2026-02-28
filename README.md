# Grilly

<p align="center">
  <img src="https://raw.githubusercontent.com/grillcheese-ai/grilly/main/assets/grilly_mascott_github.png" alt="Grilly" width="400">
</p>

*Deep learning, well done.*

[![CI](https://github.com/grillcheese-ai/grilly/actions/workflows/ci.yml/badge.svg)](https://github.com/grillcheese-ai/grilly/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/grilly)](https://pypi.org/project/grilly/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

GPU-accelerated neural network framework built on Vulkan compute shaders. Runs on **any GPU** — AMD, NVIDIA, Intel — no CUDA required. Provides a PyTorch-like `nn.Module` API backed by 161 SPIR-V shaders and a native C++ dispatch layer.

> **Alpha software.** APIs may change between minor versions. We welcome early adopters and feedback.

**Howto Guides:** [`howtos/`](howtos/) (self-contained HTML tutorials)

---

## Quick Start

```python
import numpy as np
from grilly import nn

# Define a model — same patterns as PyTorch
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

# Forward pass
x = np.random.randn(32, 784).astype(np.float32)
logits = model(x)
print(logits.shape)  # (32, 10)

# Loss + backward + optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = nn.optim.AdamW(model.parameters(), lr=1e-3)

targets = np.random.randint(0, 10, (32,))
loss = loss_fn(logits, targets)
grad = loss_fn.backward(np.ones_like(loss), logits, targets)

model.zero_grad()
model.backward(grad)
optimizer.step()
```

### Autograd

```python
from grilly import nn

x = nn.Variable(nn.randn(32, 128), requires_grad=True)
layer = nn.Linear(128, 10)

logits = x @ nn.Variable(layer.weight.T) + nn.Variable(layer.bias)
loss = logits.sum()
loss.backward()

print(x.grad.shape)  # (32, 128)
```

---

## Installation

### From PyPI

```bash
pip install grilly
```

### From Source (with C++ backend)

The C++ backend (`grilly_core`) is **required** — it provides the native Vulkan dispatch layer for all GPU operations.

```bash
git clone https://github.com/grillcheese-ai/grilly.git
cd grilly
pip install -e ".[dev]"

# Build the C++ backend
cmake -B build -DPYBIND11_FINDPYTHON=ON
cmake --build build --config Release
cp build/Release/grilly_core.*.pyd .   # Windows
# cp build/grilly_core.*.so .          # Linux
```

Verify:

```bash
python -c "import grilly_core; print('C++ backend OK')"
python -c "import grilly; b = grilly.Compute(); print('GPU:', b.device_name)"
```

See [INSTALL.md](INSTALL.md) for full setup (Vulkan SDK, Ubuntu, CI environments, troubleshooting).

### Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.12+ | 3.12 |
| GPU VRAM | 8 GB | 12 GB+ |
| System RAM | 32 GB | 64 GB |
| Vulkan | 1.2+ drivers | Latest drivers |

Supported GPUs: **AMD** (RX 5000+), **NVIDIA** (GTX 1060+), **Intel** (Arc A-series).

---

## Features

### PyTorch-like nn.Module API

Standard layers with GPU-accelerated forward and backward passes:

| Category | Modules |
|----------|---------|
| **Linear** | `Linear`, `Embedding`, `Dropout` |
| **Convolution** | `Conv1d`, `Conv2d` |
| **Recurrent** | `LSTM`, `LSTMCell`, `GRU`, `GRUCell` |
| **Pooling** | `MaxPool2d`, `AvgPool2d`, `AdaptiveMaxPool2d`, `AdaptiveAvgPool2d` |
| **Normalization** | `LayerNorm`, `RMSNorm`, `BatchNorm1d`, `BatchNorm2d` |
| **Activations** | `ReLU`, `GELU`, `SiLU`, `SwiGLU`, `GCU`, `RoSwish`, `Softmax`, `Softplus` |
| **Attention** | `MultiheadAttention`, `FlashAttention2`, `RoPE` |
| **Loss** | `MSELoss`, `CrossEntropyLoss`, `BCELoss` |
| **Containers** | `Sequential`, `Residual` |

### Spiking Neural Networks

Full SNN framework with surrogate gradient training:

- **Neuron models**: `IFNode`, `LIFNode`, `ParametricLIFNode`
- **Surrogate gradients**: `ATan`, `Sigmoid`, `FastSigmoid`
- **Temporal containers**: `SeqToANNContainer`, `MultiStepContainer`
- **Normalization**: `BatchNormThroughTime`, `TemporalEffectiveBatchNorm`, `NeuNorm`
- **Synapses**: `STPSynapse`, `DualTimescaleSynapse`, `SynapseFilter`
- **Attention**: `SpikingSelfAttention`, `TemporalWiseAttention`, `QKAttention`
- **ANN-to-SNN conversion**: `Converter`, `VoltageScaler`

### Multimodal Fusion

- `PerceiverIO` — Modality-agnostic input compression
- `PerceiverResampler` — Flamingo-style visual token resampling
- `FlamingoFusion` — Cross-attention VLM fusion
- `CrossModalAttentionFusion` — Bidirectional cross-modal attention
- `ImageBindFusion` — Joint embedding with contrastive loss
- `BottleneckFusion` — Multimodal Bottleneck Transformer
- `VisionLanguageModel` — Complete VLM with visual conditioning

### Transformer Components

- Flash Attention 2 (tiled, O(seq) memory)
- Rotary Position Embeddings (RoPE)
- LoRA fine-tuning (`LoRALinear`, `LoRAAttention`, `LoRAModel`)
- Transformer encoder/decoder layers
- Fused operations: SwiGLU FFN, RMSNorm+Linear, QKV projection

### Inference Optimizations

- Fused RMSNorm shader (Llama, Gemma)
- Grouped Query Attention (GQA) decode against KV-cache
- INT8 GEMM (weight-only, FP32 accumulation)
- 4-bit block quantization (per-block scale + zero-point)

### Optimizers

`AdamW`, `Adam`, `SGD`, `NLMS`, `NaturalGradient`, `AutoHypergradientAdamW` (OSGM-style auto LR tuning), plus LR schedulers (`StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`).

### Functional API

Stateless functions mirroring `torch.nn.functional`:

```python
import grilly.functional as F

F.linear(x, weight, bias)
F.relu(x)
F.softmax(x, dim=-1)
F.cross_entropy(logits, targets)
F.flash_attention2(q, k, v)
```

### Autograd

Full computation graph with automatic differentiation:

```python
from grilly.nn import Variable, no_grad, tensor

x = Variable(tensor([1.0, 2.0, 3.0]), requires_grad=True)
y = (x * x).sum()
y.backward()
print(x.grad)  # [2.0, 4.0, 6.0]
```

---

## C++ Backend (grilly_core)

The native C++ extension (`grilly_core`) wraps all Vulkan compute dispatch via pybind11. It provides 16 operation modules:

| Op | Description |
|----|-------------|
| `linear` | Dense matrix multiply (GEMM) |
| `conv` | 2D convolution (im2col + GEMM) |
| `activations` | ReLU, GELU, SiLU, Tanh |
| `layernorm` | Layer normalization |
| `rmsnorm` | Root mean square normalization |
| `batchnorm` | Batch normalization (2D) |
| `attention` | Flash Attention 2 |
| `attention_ops` | RoPE, KV-cache ops |
| `embedding` | Token + position embeddings |
| `pooling` | MaxPool2d, AvgPool2d |
| `loss` | Cross-entropy, MSE, BCE |
| `snn` | LIF/IF neuron step kernels |
| `optimizer` | Adam, AdamW, SGD step kernels |
| `learning` | STDP, Hebbian, EWC |
| `kv_cache` | Paged KV-cache management |
| `swizzle` | Memory layout transforms |

Build instructions: see [INSTALL.md](INSTALL.md#c-backend-grilly_core).

---

## Ecosystem

| Package | Description |
|---------|-------------|
| [optimum-grilly](https://github.com/grillcheese-ai/optimum-grilly) | HuggingFace Optimum backend — `from_pretrained` → Vulkan inference (Llama, Mistral, BERT, GPT-2) |
| [GrillyInference](https://github.com/grillcheese-ai/GrillyInference) | Native fp16 inference engine (paged KV-cache, INT8/4-bit quantization) |
| [GrillyCompression](https://github.com/grillcheese-ai/GrillyCompression) | DCT activation compression, KV-cache page compression |
| [GrillyDistil](https://github.com/grillcheese-ai/GrillyDistil) | SA-KD adaptive temperature distillation trainer |

```bash
pip install grilly optimum-grilly    # HuggingFace integration
pip install grilly grillyinference   # Native inference engine
```

---

## Examples

See [`examples/`](examples/) for runnable scripts:

- `hello_grilly.py` — Autograd forward + backward
- `train_mlp.py` — Full training loop with AdamW and cross-entropy
- `benchmark_gemm.py` — GPU vs CPU GEMM throughput
- `classifier.py` — Simple classifier example
- 13 experimental examples (VSA, MoE, capsules, cognitive control, and more)

---

## Architecture

```
grilly/
├── backend/        # Vulkan GPU dispatch (core.py, compute.py, pipelines.py, autograd_core.py)
├── cpp/            # C++ pybind11 extension (grilly_core) — 16 native ops
├── nn/             # PyTorch-like nn.Module layers, SNN framework, multimodal fusion
├── functional/     # Stateless F.* API (mirrors torch.nn.functional)
├── optim/          # Optimizers (AdamW, Adam, SGD, NLMS, NaturalGradient, Hypergradient)
├── utils/          # DataLoader, Dataset, HuggingFaceBridge, VulkanTensor, checkpointing
├── shaders/        # 161 GLSL compute shaders
│   └── spv/        # Compiled SPIR-V bytecode
├── experimental/   # Unstable: VSA, MoE routing, temporal reasoning, cognitive controller
├── howtos/         # 8 self-contained HTML tutorials
├── examples/       # Runnable example scripts
└── tests/          # Test suite (1000+ tests)
```

### Design Principles

- **Pure Vulkan** — no CUDA, no vendor lock-in
- **Hardware-agnostic** — AMD, NVIDIA, Intel on the same codebase
- **C++ dispatch layer** — pybind11 extension for low-overhead GPU calls
- **Zero-copy GPU memory** — `VulkanTensor` keeps data GPU-resident between ops
- **All data is `np.float32`** — numpy arrays in, numpy arrays out

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VK_GPU_INDEX` | Select GPU by index (multi-GPU systems) | `0` |
| `GRILLY_DEBUG` | Enable debug logging (`1` = on) | off |
| `ALLOW_CPU_VULKAN` | Allow Mesa llvmpipe software Vulkan (CI) | off |

---

## Testing

```bash
# All tests (requires Vulkan)
uv run pytest tests/ -v

# CPU-only (no GPU required)
uv run pytest tests/ -m "not gpu" -v

# With coverage
uv run pytest tests/ --cov=. --cov-report=term

# Single test
pytest tests/test_snn.py -k "test_lif"
```

---

## CI/CD

- **CI** (on push/PR): Lint (ruff, black), test (CPU-only on Mesa llvmpipe), build
- **CD** (on GitHub Release): Build and publish to PyPI via [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC, no API tokens)

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Run `ruff check .` and `pytest tests/ -v`
5. Submit a pull request

---

## License

MIT License — see [LICENSE](LICENSE) for details.
