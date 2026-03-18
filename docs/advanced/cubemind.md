# CubeMind Integration

[CubeMind](https://github.com/grillcheese-ai/cubemind) is a neuro-vector-symbolic reasoning system built on grilly. It combines hyperdimensional computing (HDC) with neural networks for structured symbolic reasoning on GPU.

---

## Overview

CubeMind uses grilly's HDC operations (block codes, circular convolution, packed binary vectors) and neural network layers to build a reasoning system that:

- Represents knowledge as high-dimensional binary vectors
- Performs symbolic operations (binding, bundling, sequence encoding) on GPU
- Trains neural components with grilly's autograd and optimizers

---

## Grilly Features Used by CubeMind

| Feature | CubeMind Usage |
|---------|---------------|
| HDC packed ops | State representation as 10,240-bit hypervectors |
| Block-code circular conv | Sequence encoding for move histories |
| DisARM gradient estimator | Training through discrete block-code operations |
| Sanger GHA | Neurogenesis -- growing network capacity |
| `nn.Linear`, `nn.ReLU` | Neural prediction heads |
| `optim.AdamW` | Training neural components |
| VulkanTensor | GPU-resident state vectors |

---

## C++ Backend Support

The C++ backend includes CubeMind-specific bindings in `cpp/src/cubemind/` for high-performance dispatch of HDC operations that would be slow in pure Python.

---

## Getting Started

```bash
pip install cubemind
```

CubeMind depends on grilly >= 0.5.0. See the [CubeMind repository](https://github.com/grillcheese-ai/cubemind) for documentation and examples.
