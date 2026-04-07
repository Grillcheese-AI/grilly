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

GPU-accelerated neural network framework using **Vulkan compute shaders**. PyTorch-like API that runs on **any GPU** -- AMD, NVIDIA, Intel -- no CUDA dependency. 231 GLSL compute shaders compiled to SPIR-V, dispatched through a native C++ layer with automatic CPU fallback.

> ## ⚠️ Pre-v1.0 release — massive changes ahead
>
> The current `main` branch is **substantially rewritten** since the last
> tagged release (`v0.6.1`). The changes touch the C++ Vulkan dispatch
> layer, the VMA buffer allocation strategy, the autograd graph, the
> `nn` module forward signatures, the `torch_api` facade, and several
> shaders. Existing user code that depends on `v0.6.1` semantics may
> need updates.
>
> **Highlights of what changed:**
>
> - **40x faster `nn.Linear`** on AMD/Windows via a new staging-buffer
>   pattern (DEVICE_LOCAL VRAM compute + WC stage-in + HOST_CACHED
>   readback). The same pattern was applied across `linear` /
>   `linear_backward` / `layernorm` / `embedding` / `activations` /
>   `optimizer` / `loss` / `dropout`.
> - **Cooperative-matrix GEMM** (`gemm-coopmat-shared.glsl`) — fp16
>   GEMM via `VK_KHR_cooperative_matrix`. Hits hardware tensor cores on
>   RDNA3 / NVIDIA RTX, runs through the driver emulation path on
>   RDNA2. New `LinearParams.elemSize` field + generic `py::array`
>   bindings let `nn.Linear` accept fp32 OR fp16 input transparently.
> - **Causal Linear-RNN prefix scan** — new `prefix-scan-causal` /
>   `prefix-scan-causal-backward` shaders + C++ dispatcher + Python
>   autograd wrapper. `grilly.nn.prefix_scan.CausalSequenceMixer` is a
>   drop-in subgroup-parallel replacement for sequence pooling that
>   strictly respects autoregressive causality.
> - **Real autograd through `nn.Linear`, `nn.LayerNorm`, `nn.Embedding`** —
>   their `forward` methods now wrap numpy outputs in `Variable` with a
>   `GradFn` that calls the existing `backward()` so `loss.backward()`
>   actually updates layer parameters. Before this fix, optimizer steps
>   silently no-op'd on every Module subclass that used the standard
>   `self.weight = nn.Parameter(...)` idiom.
> - **`Module.__setattr__` auto-registration** — `Parameter` and child
>   `Module` attribute assignments now populate `_parameters` /
>   `_modules` automatically. Standard PyTorch idiom; was previously
>   silently broken (every subclass returned 0 parameters).
> - **`.grl` checkpoint roundtrip** — `torch.save({'model': sd, 'step':
>   N}, path); ck = torch.load(path)` now returns exactly what was
>   saved (matches PyTorch semantics). Previously the loader
>   force-wrapped content under a fixed `'model'` key.
> - **VMA fix** — `BufferPool::allocateBuffer` now uses
>   `requiredFlags = DEVICE_LOCAL_BIT` instead of `preferredFlags`, so
>   the allocator actually selects VRAM instead of silently falling
>   back to slow host memory.
> - **`Variable.__array__`** — numpy interop. `np.matmul(tensor, w)` /
>   `np.dot(tensor, w)` / `np.asarray(tensor)` now work transparently.
> - **PEP 660 editable install fix** — `import grilly_core` now works
>   under modern editable installs (path hook didn't add the package
>   dir to `sys.path`, so the Vulkan probe silently reported
>   `VULKAN_AVAILABLE = False` even on machines with working Vulkan).
>
> See [the "What's new since 0.6.1" section below](#whats-new-since-061)
> for the full list. Tag will land as a `v1.0.0-rc.1` once the
> remaining causal-RNN training validation lands.

---

## Why Grilly?

- **Any GPU**: Vulkan runs on AMD, NVIDIA, Intel, and Apple (via MoltenVK). No CUDA lock-in.
- **PyTorch-like API**: `nn.Module`, `F.relu`, `AdamW` -- familiar patterns, new backend.
- **Always works**: Pure-Python numpy fallback if no GPU is available. Same code, same results.
- **Research-ready**: Spiking neural networks, Vector Symbolic Architectures, Mixture of Experts, cognitive controllers, temporal reasoning -- all GPU-accelerated.
- **Lightweight**: Core dependency is numpy only. Optional extras for torch, HuggingFace, ONNX.

---

## Installation

### Option 1: Python-only (no GPU acceleration)

```bash
pip install grilly
```

Works immediately with numpy. No GPU, no Vulkan SDK, no C++ compiler needed.

### Option 2: With Vulkan GPU acceleration

#### Linux / Google Colab (one-liner)

```bash
# Full build (~30 min — includes validation layers, all SDK tools)
curl -sSL https://raw.githubusercontent.com/Grillcheese-AI/grilly/main/scripts/install.sh | bash

# Fast build (~5 min — shaderc + loader only, recommended for Colab/CI)
curl -sSL https://raw.githubusercontent.com/Grillcheese-AI/grilly/main/scripts/install.sh | bash -s -- --fast
```

On Colab:

```python
# Recommended: Colab mode (Vulkan 1.3 + fast build + NVIDIA ICD, ~5 min)
!wget -qO- https://raw.githubusercontent.com/Grillcheese-AI/grilly/main/scripts/install.sh | bash -s -- --colab
```

This installs system deps, downloads and builds Vulkan SDK 1.4, compiles the grilly C++ extension, and installs the Python package. The `--fast` flag builds only the components grilly needs (shaderc, loader, headers) and skips validation layers.

#### Linux (manual step-by-step)

```bash
# 1. System dependencies (Ubuntu/Debian)
sudo apt-get install -y cmake g++ ninja-build pkg-config \
    libxcb-dri3-0 libxcb-present0 libpciaccess0 libpng-dev \
    libxcb-keysyms1-dev libxcb-dri3-dev libx11-dev libwayland-dev \
    libxrandr-dev libxcb-randr0-dev libx11-xcb-dev wayland-protocols

# 2. Vulkan SDK (download from https://vulkan.lunarg.com/sdk/home)
wget https://sdk.lunarg.com/sdk/download/1.4.341.1/linux/vulkansdk-linux-x86_64-1.4.341.1.tar.xz
tar xf vulkansdk-linux-x86_64-1.4.341.1.tar.xz
cd 1.4.341.1 && ./vulkansdk all -j $(nproc)
export VULKAN_SDK=$(pwd)/x86_64
export PATH=$VULKAN_SDK/bin:$PATH
export LD_LIBRARY_PATH=$VULKAN_SDK/lib:$LD_LIBRARY_PATH

# 3. Build grilly
git clone --recurse-submodules https://github.com/grillcheese-ai/grilly.git
cd grilly
pip install -e ".[dev]"
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)

# 4. Install the compiled extension
cp build/grilly_core.*.so $(python -c "import grilly; print(grilly.__path__[0])")/
```

#### Windows

```powershell
# 1. Install Vulkan SDK from https://vulkan.lunarg.com/sdk/home (Windows installer)
# 2. Install Visual Studio 2022 with C++ workload

git clone --recurse-submodules https://github.com/grillcheese-ai/grilly.git
cd grilly
pip install -e ".[dev]"
cmake -B build -DPYBIND11_FINDPYTHON=ON
cmake --build build --config Release
cp build\Release\grilly_core.*.pyd .
```

**Pre-built binary (Windows x64, Python 3.12):** Download `grilly_core.cp312-win_amd64.pyd` from the [latest release](https://github.com/grillcheese-ai/grilly/releases) and copy it into your grilly install directory.

#### macOS

```bash
# 1. Install Vulkan SDK from https://vulkan.lunarg.com/sdk/home#mac
brew install cmake ninja
# 2. Follow the Linux build steps above (uses MoltenVK)
```

### Verify installation

```python
import grilly
print(f"grilly {grilly.__version__}")

# Check GPU backend
try:
    from grilly.backend import _bridge
    print(f"Vulkan: {'enabled' if _bridge.is_available() else 'not available'}")
except ImportError:
    print("Vulkan: not installed (numpy fallback active)")
```

### Requirements

| | Minimum | Recommended |
|---|---|---|
| Python | 3.12+ | 3.12 |
| GPU VRAM | 8 GB | 12 GB+ |
| System RAM | 32 GB | 64 GB |
| Vulkan | 1.1+ | 1.4 (latest SDK) |

Supported GPUs: AMD (RX 5000+), NVIDIA (GTX 1060+), Intel (Arc A-series), Apple (M1+ via MoltenVK).

---

## Quick Start

```python
import numpy as np
from grilly import nn
from grilly.optim import AdamW

# Build a model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

# Train
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

out = F.linear(x, weight, bias)
out = F.relu(out)
out = F.softmax(out, dim=-1)
attn = F.flash_attention2(q, k, v)
```

See `notebooks/01_getting_started.ipynb` for a complete walkthrough.

---

## Features

### Layers (100+)

| Category | Modules |
|----------|---------|
| **Linear** | `Linear`, `Embedding`, `CapsuleEmbedding`, `Dropout` |
| **Convolution** | `Conv1d`, `Conv2d` |
| **Recurrent** | `LSTM`, `LSTMCell`, `GRU`, `GRUCell` |
| **Normalization** | `LayerNorm`, `RMSNorm`, `BatchNorm1d/2d` |
| **Activations** | `ReLU`, `GELU`, `SiLU`, `SwiGLU`, `GCU`, `RoSwish` |
| **Attention** | `FlashAttention2/3`, `HYLAAttention`, `MultiheadAttention`, `RoPE` |
| **LoRA** | `LoRALinear`, `LoRAAttention`, `LoRAModel` |
| **Pooling** | `MaxPool2d`, `AvgPool2d`, `AdaptiveMaxPool2d/AvgPool2d` |
| **Loss** | `MSELoss`, `CrossEntropyLoss`, `BCELoss` |
| **Containers** | `Sequential`, `Residual` |
| **Multimodal** | `PerceiverIO`, `ImageBindFusion`, `FlamingoFusion`, `VisionLanguageModel` |
| **Memory** | `MemoryRead`, `MemoryWrite`, `MemoryContextAggregate` |
| **Routing** | `DomainRouter`, `DomainPredictor`, `ExpertCombiner` |

### Spiking Neural Networks

Full SNN framework with GPU-accelerated spike dynamics:

- **Neurons**: `IFNode`, `LIFNode`, `ParametricLIFNode`
- **Surrogate gradients**: `ATan`, `Sigmoid`, `FastSigmoid`
- **Synapses**: `STPSynapse`, `DualTimescaleSynapse`, `SynapseFilter`
- **Temporal containers**: `SeqToANNContainer`, `MultiStepContainer`
- **Spiking attention**: `SpikingSelfAttention`, `QKAttention`, `TemporalWiseAttention`
- **ANN-to-SNN conversion**: `Converter`, `VoltageScaler`

### Optimizers

| Optimizer | Description |
|-----------|-------------|
| `Adam` | Classic Adam |
| `AdamW` | Adam with decoupled weight decay |
| `SGD` | Stochastic gradient descent |
| `NLMS` | Normalized Least Mean Squares |
| `NaturalGradient` | Fisher-preconditioned |
| `HypergradientAdamW` | OSGM-style auto learning rate |
| `AutoHypergradientAdamW` | Fully automatic hypergradient |
| `AffectAdam` | Emotion-weighted updates |

**Schedulers**: `StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`, `OneCycleLR`.

### Experimental Modules

| Module | Description |
|--------|-------------|
| `experimental.vsa` | Vector Symbolic Architectures (binary, holographic, block-codes, resonator networks) |
| `experimental.moe` | Mixture of Experts (relational encoder, resonator routing) |
| `experimental.temporal` | Temporal reasoning (causal chains, counterfactuals, world models) |
| `experimental.cognitive` | Cognitive controller (working memory, simulation, understand-think-speak) |
| `experimental.language` | Language processing (encoding, generation, parsing) |

---

## Architecture

```
Python API                    C++ Bridge                  GPU Shaders
─────────────                 ──────────                  ───────────
nn.Module layers              pybind11 bindings           231 SPIR-V kernels
F.* stateless ops      →      dual-validity tensors  →    AMD / NVIDIA / Intel
optim.* optimizers            zero CPU↔GPU ping-pong      No CUDA dependency
autograd engine               buffer pool management      Vulkan 1.1+ compute
```

### 3-Level GPU Fallback

Every operation has automatic fallback:

1. **grilly C++ / Vulkan** -- native compute shaders (fastest)
2. **PyTorch CUDA** -- if torch is available (fast)
3. **NumPy CPU** -- always available (correct)

Same API, same results, different speed. Your code never changes.

### GPU Kernels (231 operations)

| Category | Count | Examples |
|----------|-------|---------|
| Linear algebra | 20+ | GEMM, FFT, SVD, matmul |
| Attention | 15+ | flash attention, multi-head, spiking |
| Convolution | 10+ | conv2d forward/backward, im2col |
| Learning | 20+ | Adam, STDP, Hebbian, EWC, NLMS |
| VSA | 10+ | bind, bundle, similarity, resonator |
| SNN | 15+ | LIF/IF neuron, synapse, spike generation |
| Normalization | 10+ | layer norm, batch norm, RMS norm |
| Activation | 15+ | ReLU, GELU, SiLU, softmax (all with backward) |
| Memory/FAISS | 10+ | similarity search, place/time cells |

---

## Ecosystem

| Package | Description |
|---------|-------------|
| [optimum-grilly](https://github.com/grillcheese-ai/optimum-grilly) | HuggingFace Optimum backend -- `from_pretrained` on Vulkan |
| [CubeMind](https://github.com/grillcheese-ai/cubemind) | Neuro-vector-symbolic cognitive architecture powered by grilly |

---

## Notebooks & Tutorials

| Notebook | Description |
|----------|-------------|
| `notebooks/01_getting_started.ipynb` | Installation verification, first model, GPU check |
| `notebooks/02_training_loop.ipynb` | Full training loop: data loading, loss, optimization, checkpointing |
| `notebooks/03_spiking_neural_networks.ipynb` | SNN neurons, STDP learning, ANN-to-SNN conversion |
| `notebooks/04_vector_symbolic_architectures.ipynb` | VSA ops: bind, bundle, similarity, resonator networks |
| `notebooks/05_attention_and_transformers.ipynb` | Flash attention, RoPE, PerceiverIO, multi-head attention |

See also `tutorials/` for standalone Python scripts covering every feature.

---

## Testing

```bash
# All tests
uv run pytest tests/ -v

# CPU-only (no GPU required)
uv run pytest tests/ -m "not gpu" -v

# With coverage
uv run pytest tests/ --cov=. --cov-report=term

# Single module
uv run pytest tests/test_linear.py -v
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VK_GPU_INDEX` | Select GPU by index | `0` |
| `GRILLY_DEBUG` | Enable debug logging (`1` = on) | off |
| `ALLOW_CPU_VULKAN` | Allow Mesa llvmpipe software Vulkan | off |

---

## What's New

### 0.6.x (current)

- **MindForge** adapter hypernetwork integration (via CubeMind)
- **Synaptic shaders**: `synapsis-stdp-update.glsl`, `bridge-spike-to-continuous.glsl`
- **JIT shader fusion** with shaderc runtime compilation
- **Perceiver IO** with IndexCache K/V pre-projection
- **MoQE** Gumbel-Softmax router shader
- 215 compute shaders (up from 190)

### 0.5.x

- C++ Tensor with dual-validity tracking -- GPU-resident data, no CPU ping-pong
- Flash Attention 3 with subgroup acceleration
- HYLAAttention (softmax-free), FNetMixing, SympFormerBlock
- HDC packed ops -- 32x memory compression
- Sanger GHA for neurogenesis
- JIT compilation framework (`@grilly.jit`)
- Automatic Mixed Precision (`autocast` + `GradScaler`)

---

## Contributing

1. Fork the repo and create a feature branch
2. Add tests for new features
3. Run `ruff check .` and `uv run pytest tests/ -v`
4. Submit a pull request

---

## License

MIT License -- see [LICENSE](LICENSE) for details.
