# Grilly (The missing AI tool for AI developers on AMD, NVIDIA, and Intel GPUs)
## A Vulkan based SDK with a familiar Pytorch-like API and HuggingFace Vulkan wrapper
### Still early stage not ready for production at all !

GPU-accelerated neural network operations using Vulkan compute shaders.

Grilly provides high-performance GPU acceleration for neural network operations, including spiking neural networks (SNN), feedforward networks (FNN), attention mechanisms, memory operations, and more. Built on Vulkan for cross-platform GPU support (AMD, NVIDIA, Intel).

## Features

- **Spiking Neural Networks (SNN)**: LIF neurons, Hebbian learning, STDP
- **Feedforward Networks (FNN)**: Linear layers, activations, layer normalization, dropout
- **Attention Mechanisms**: Flash Attention 2, multi-head attention
- **Memory Operations**: Key-value memory, context aggregation
- **FAISS Integration**: GPU-accelerated vector similarity search
- **Learning Algorithms**: EWC, NLMS, whitening, natural gradients
- **Place & Time Cells**: Spatial and temporal encoding
- **Cross-Platform**: Works on AMD, NVIDIA, and Intel GPUs

## Installation

```bash
pip install grilly
```

### Requirements

- Python >= 3.10
- Vulkan drivers installed on your system
- NumPy >= 1.24.0

## Quick Start

```python
import grilly
import numpy as np

# Initialize compute backend
backend = grilly.Compute()

# Run LIF neuron step
input_current = np.random.randn(1000).astype(np.float32)
membrane = np.zeros(1000, dtype=np.float32)
refractory = np.zeros(1000, dtype=np.float32)

membrane, refractory, spikes = backend.lif_step(
    input_current, membrane, refractory,
    dt=0.001, tau_mem=20.0, v_thresh=1.0
)

# Use high-level SNN interface
snn = grilly.SNNCompute(n_neurons=1000)
result = snn.process(embedding)
```

## API Overview

### Core Classes

- `grilly.Compute` (alias for `VulkanCompute`): Main compute backend
- `grilly.SNNCompute`: High-level SNN interface
- `grilly.Learning`: Learning operations (EWC, NLMS, etc.)

### Operations

#### SNN Operations
- `lif_step()`: Leaky integrate-and-fire neuron dynamics
- `hebbian_learning()`: Hebbian weight updates
- `stdp_learning()`: Spike-timing-dependent plasticity

#### FNN Operations
- `linear()`: Linear transformation
- `activation_relu()`, `activation_gelu()`, `activation_silu()`: Activations
- `layernorm()`: Layer normalization
- `dropout()`: Dropout regularization

#### Attention
- `flash_attention2()`: Flash Attention 2 implementation
- `attention_scores()`, `attention_output()`: Attention components

#### Memory
- `memory_read()`, `memory_write()`: Key-value memory operations

#### FAISS
- `faiss_compute_distances()`: Compute pairwise distances
- `faiss_topk()`: Top-k selection

## Shader Compilation

Grilly uses pre-compiled SPIR-V shaders. If you need to recompile shaders:

```bash
# Compile a shader
glslc shader.glsl -o spv/shader.spv

# Or use the provided compilation script
python compile_shaders.py
```

## GPU Selection

By default, Grilly selects the best available GPU (preferring discrete NVIDIA/AMD GPUs). You can override this:

```bash
# Select specific GPU by index
export VK_GPU_INDEX=0

# Allow CPU/llvmpipe fallback (not recommended)
export ALLOW_CPU_VULKAN=1
```

## Examples

See the `examples/` directory for more usage examples.

## Documentation

Full API documentation: https://docs.grillcheeseai.com/grilly

## License

MIT License

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## Links

- **Homepage**: https://grillcheeseai.com
- **GitHub**: https://github.com/grillcheese-ai/grilly
- **Documentation**: https://docs.grillcheeseai.com/grilly
