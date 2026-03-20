# Grilly

<p align="center">
  <img src="https://raw.githubusercontent.com/grillcheese-ai/grilly/main/assets/grilly_mascott_github.png" alt="Grilly" width="400">
</p>

*Deep learning, well done.*

[![CI](https://github.com/grillcheese-ai/grilly/actions/workflows/ci.yml/badge.svg)](https://github.com/grillcheese-ai/grilly/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/grilly)](https://pypi.org/project/grilly/)
[![Tests](https://img.shields.io/badge/tests-1820%20passing-brightgreen)](https://github.com/grillcheese-ai/grilly/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-grilly.org-blue)](https://grillcheese-ai.github.io/grilly/getting-started/installation/)

GPU-accelerated neural network framework using Vulkan compute shaders. PyTorch-like API that runs on **any GPU** — AMD, NVIDIA, Intel — no CUDA dependency. 190 GLSL compute shaders compiled to SPIR-V, dispatched through a native C++ layer.

> **Alpha software.** APIs may change between minor versions.

---

## Installation

```bash
pip install grilly
```

For GPU acceleration (requires [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) and C++ toolchain):

```bash
git clone https://github.com/grillcheese-ai/grilly.git
cd grilly
pip install -e ".[dev]"
cmake -B build -DPYBIND11_FINDPYTHON=ON
cmake --build build --config Release
cp build/Release/grilly_core.*.pyd .   # Windows
# cp build/grilly_core.*.so .          # Linux
```

**Pre-built C++ extension (Windows x64 only):**

Download `grilly_core.cp312-win_amd64.pyd` from the [latest release](https://github.com/grillcheese-ai/grilly/releases) and place it in your grilly install directory:

```bash
# Find where grilly is installed
python -c "import grilly; print(grilly.__file__)"
# Copy the .pyd to that directory
cp grilly_core.cp312-win_amd64.pyd /path/to/grilly/
```

Without the C++ extension, grilly works fully via pure Python + numpy fallbacks — just without GPU acceleration.

See [INSTALL.md](INSTALL.md) for full setup, Ubuntu instructions, and troubleshooting.

### Requirements

| | Minimum | Recommended |
|---|---|---|
| Python | 3.12+ | 3.12 |
| GPU VRAM | 8 GB | 12 GB+ |
| System RAM | 32 GB | 64 GB |
| Vulkan | 1.1+ | Latest drivers |

Supported GPUs: AMD (RX 5000+), NVIDIA (GTX 1060+), Intel (Arc A-series).

---

## Quick Start

```python
import numpy as np
from grilly import nn
from grilly.optim import AdamW

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

optimizer = AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

x = np.random.randn(32, 784).astype(np.float32)
targets = np.random.randint(0, 10, (32,))

logits = model(x)
loss = loss_fn(logits, targets)
grad = loss_fn.backward(np.ones_like(loss), logits, targets)

model.zero_grad()
model.backward(grad)
optimizer.step()
```

### Autograd

```python
from grilly.nn import Variable, tensor

x = Variable(tensor([1.0, 2.0, 3.0]), requires_grad=True)
y = (x * x).sum()
y.backward()
print(x.grad)  # [2.0, 4.0, 6.0]
```

### Functional API

```python
import grilly.functional as F

F.linear(x, weight, bias)
F.relu(x)
F.softmax(x, dim=-1)
F.flash_attention2(q, k, v)
```

---

## Architecture

```
Python (VulkanTensor) → C++ Bridge (grilly_core) → Vulkan Compute Shaders
  nn/ modules            pybind11 bindings           190 SPIR-V shaders
  functional/ ops        dual-validity GPU/CPU        AMD / NVIDIA / Intel
  optim/                 zero CPU↔GPU ping-pong       No CUDA needed
```

**Package layout:**

```
grilly/
├── backend/        # Vulkan GPU dispatch (core, compute, pipelines, autograd)
├── cpp/            # C++ pybind11 extension — grilly_core native ops
├── nn/             # nn.Module layers, SNN framework, multimodal fusion, autograd
├── functional/     # Stateless F.* API (mirrors torch.nn.functional)
├── optim/          # Optimizers and LR schedulers
├── utils/          # DataLoader, VulkanTensor, HuggingFaceBridge, checkpointing
├── shaders/        # 190 GLSL compute shaders + compiled SPIR-V
├── experimental/   # VSA, MoE routing, temporal reasoning, cognitive controller
└── tests/          # 1,820 tests
```

---

## What's New in 0.5.0 "GPU-First"

- **C++ Tensor with dual-validity tracking** — data stays GPU-resident between ops; no CPU ping-pong
- **Flash Attention 3** with subgroup acceleration
- **HYLAAttention** (softmax-free), **FNetMixing**, **SympFormerBlock**
- **TAPPA q-similarity** for adaptive KV cache eviction
- **HDC packed ops** — 32x memory compression + block-code circular convolution
- **Sanger GHA** for neurogenesis
- **DisARM gradient estimator**
- **JIT compilation framework** (`@grilly.jit`)
- **Automatic Mixed Precision** (`autocast` + `GradScaler`)
- **ProjectionHeads** for structured embeddings
- **StreamingPipeline** for batched embed + upload
- `bindings.cpp` refactored into 11 focused files

---

## Features

### Layers

| Category | Modules |
|----------|---------|
| Linear | `Linear`, `Embedding`, `Dropout` |
| Convolution | `Conv1d`, `Conv2d` |
| Recurrent | `LSTM`, `LSTMCell`, `GRU`, `GRUCell` |
| Normalization | `LayerNorm`, `RMSNorm`, `BatchNorm1d`, `BatchNorm2d` |
| Activations | `ReLU`, `GELU`, `SiLU`, `SwiGLU`, `GCU`, `RoSwish` |
| Attention | `FlashAttention2/3`, `HYLAAttention`, `MultiheadAttention`, `RoPE` |
| LoRA | `LoRALinear`, `LoRAAttention`, `LoRAModel` |
| Pooling | `MaxPool2d`, `AvgPool2d`, `AdaptiveMaxPool2d` |
| Loss | `MSELoss`, `CrossEntropyLoss`, `BCELoss` |
| Containers | `Sequential`, `Residual` |

### Spiking Neural Networks

- Neuron models: `IFNode`, `LIFNode`, `ParametricLIFNode`
- Surrogate gradients: `ATan`, `Sigmoid`, `FastSigmoid`
- Temporal containers: `SeqToANNContainer`, `MultiStepContainer`
- ANN-to-SNN conversion: `Converter`, `VoltageScaler`

### Optimizers

`AdamW`, `Adam`, `SGD`, `NLMS`, `NaturalGradient`, `AutoHypergradientAdamW` (OSGM-style auto LR), plus schedulers: `StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`.

---

## Ecosystem

| Package | Description |
|---------|-------------|
| [optimum-grilly](https://github.com/grillcheese-ai/optimum-grilly) | HuggingFace Optimum backend — `from_pretrained` → Vulkan inference |
| [CubeMind](https://github.com/grillcheese-ai/cubemind) | Neuro-vector-symbolic reasoning powered by grilly 0.5.0 |

---

## Testing

```bash
uv run pytest tests/ -v                          # all tests (requires Vulkan)
uv run pytest tests/ -m "not gpu" -v             # CPU-only
uv run pytest tests/ --cov=. --cov-report=term   # with coverage
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VK_GPU_INDEX` | Select GPU by index | `0` |
| `GRILLY_DEBUG` | Enable debug logging (`1` = on) | off |
| `ALLOW_CPU_VULKAN` | Allow Mesa llvmpipe software Vulkan | off |

---

## Contributing

1. Fork the repo and create a feature branch
2. Add tests for new features
3. Run `ruff check .` and `uv run pytest tests/ -v`
4. Submit a pull request

---

## License

MIT License — see [LICENSE](LICENSE) for details.
