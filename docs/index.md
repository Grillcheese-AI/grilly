# Grilly

**Deep learning, well done.**

GPU-accelerated neural network framework using Vulkan compute shaders. PyTorch-like API that runs on **any GPU** -- AMD, NVIDIA, Intel -- no CUDA dependency.

---

## Why Grilly?

CUDA locks you into NVIDIA hardware. Grilly uses Vulkan compute shaders, which run on every major GPU vendor. Same training code, any GPU.

| | PyTorch (CUDA) | Grilly (Vulkan) |
|---|---|---|
| AMD GPUs | ROCm (Linux data center only) | Full support |
| NVIDIA GPUs | Full support | Full support |
| Intel Arc | No | Full support |
| Windows consumer GPUs | NVIDIA only | All vendors |

## At a Glance

- **194 GLSL compute shaders** compiled to SPIR-V
- **1,820 tests** passing
- **C++ backend** (`grilly_core`) with pybind11 bindings -- zero CPU-GPU ping-pong
- **PyTorch-like API**: `nn.Module`, `nn.Linear`, `F.relu`, `AdamW`
- **Autograd engine** with `Variable`, reverse-mode autodiff, full operator overloading
- **JIT compilation** via `@grilly.jit` for fused GPU dispatch
- **AMP** with `autocast()` and `GradScaler`

## Quick Install

```bash
pip install grilly
```

For GPU acceleration, build the C++ backend:

```bash
git clone https://github.com/grillcheese-ai/grilly.git
cd grilly
pip install -e ".[dev]"
cmake -B build -DPYBIND11_FINDPYTHON=ON
cmake --build build --config Release
```

See [Installation](getting-started/installation.md) for full details.

## Quick Example

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

## Architecture

```
Python (nn.Module)  -->  C++ Bridge (grilly_core)  -->  Vulkan Compute Shaders
  nn/ modules              pybind11 bindings              194 SPIR-V shaders
  functional/ ops          VMA persistent mapping          AMD / NVIDIA / Intel
  optim/                   BufferPool allocation            No CUDA needed
```

```
grilly/
├── backend/        # Vulkan GPU dispatch, autograd core, JIT, AMP
├── cpp/            # C++ pybind11 extension (grilly_core)
├── nn/             # nn.Module layers, SNN, multimodal, LoRA, autograd
├── functional/     # Stateless F.* API (mirrors torch.nn.functional)
├── optim/          # Optimizers and LR schedulers
├── utils/          # DataLoader, VulkanTensor, HuggingFaceBridge
├── shaders/        # 194 GLSL compute shaders + compiled SPIR-V
├── experimental/   # VSA, MoE routing, temporal reasoning
└── tests/          # 1,820 tests
```

## Ecosystem

| Package | Description |
|---------|-------------|
| [optimum-grilly](https://github.com/grillcheese-ai/optimum-grilly) | HuggingFace Optimum backend -- `from_pretrained` to Vulkan inference |
| [CubeMind](https://github.com/grillcheese-ai/cubemind) | Neuro-vector-symbolic reasoning powered by grilly |

## License

MIT -- see [LICENSE](https://github.com/grillcheese-ai/grilly/blob/main/LICENSE).
