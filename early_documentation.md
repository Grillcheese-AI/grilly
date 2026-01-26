⚠️ **Development Status**: Grilly is under active development and not production-ready. This is an early-stage alpha project (v0.1.0) with unstable APIs, incomplete features, and ongoing architectural changes. Use for research and experimentation only.

Grilly is a GPU-accelerated neural network library built on Vulkan compute shaders, providing a PyTorch-like API for cross-platform GPU acceleration. In active development, Grilly is designed to support spiking neural networks (SNNs), feedforward neural networks (FNNs), attention mechanisms, and biological learning algorithms on AMD, NVIDIA, and Intel GPUs.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [API Reference](#api-reference)
7. [Core Components](#core-components)
8. [Advanced Usage](#advanced-usage)
9. [GPU Selection and Optimization](#gpu-selection-and-optimization)
10. [Contributing](#contributing)

---

## Overview

### Purpose

Grilly is the missing AI development tool for researchers and engineers building neuromorphic and biologically-inspired AI systems. It provides GPU acceleration through Vulkan compute shaders while maintaining an intuitive, PyTorch-compatible interface that developers already know.

### Target Users

- AI/ML researchers exploring spiking neural networks and biologically-inspired learning
- Engineers deploying models on AMD, NVIDIA, and Intel GPUs
- Developers needing cross-platform GPU acceleration without vendor lock-in
- Teams building neuromorphic computing systems

### Key Design Philosophy

- **PyTorch-like API**: Familiar interface with minimal learning curve
- **Cross-platform**: Single codebase for AMD, NVIDIA, Intel GPUs via Vulkan
- **Biologically-inspired**: Native support for SNNs, STDP, Hebbian learning, and place/time cells
- **Research-focused**: Experimental pre-compiled SPIR-V shaders, ongoing memory optimization
- **HuggingFace integration**: Bridge between Transformers and Vulkan computation (in development)

⚠️ **Breaking Changes**: APIs are not stable and may change significantly between versions.

---

## Architecture

### System Overview

User Code (Python)
        ↓
   Grilly API (nn, functional, optim, utils)
        ↓
   Backend Layer (VulkanCompute, SNNCompute, Learning)
        ↓
   Shader Layer (Pre-compiled SPIR-V)
        ↓
   Vulkan Runtime
        ↓
   GPU Hardware (AMD/NVIDIA/Intel)

### Core Components

**Backend Modules:**
- `VulkanCompute`: Low-level GPU compute operations
- `SNNCompute`: High-level spiking neural network interface
- `VulkanLearning`: Biological learning algorithms
- `CapsuleTransformer`: Hippocampal-inspired memory architecture

**API Modules:**
- `nn`: Neural network layers and modules
- `functional`: Functional operations (activation, linear, attention, memory)
- `optim`: Optimization algorithms (Adam, SGD, NLMS, NaturalGradient)
- `utils`: Device management, HuggingFace bridge, PyTorch compatibility

### Data Flow

1. **Input preparation**: NumPy arrays or PyTorch tensors
2. **Device transfer**: Conversion to Vulkan-compatible format
3. **Compute execution**: GPU kernel dispatch via Vulkan
4. **Output retrieval**: Results returned as NumPy or PyTorch tensors

---

## Features

### Spiking Neural Networks (SNN)

Grilly provides native support for neuromorphic computation:

- **LIF Neurons**: Leaky integrate-and-fire neuron dynamics with customizable time constants
- **Hebbian Learning**: Weight updates based on pre- and post-synaptic activity correlation
- **STDP**: Spike-timing-dependent plasticity for temporal learning rules
- **Batched Operations**: Process multiple timesteps and neuron populations efficiently

**Use Case**: Temporal sequence learning, event-driven processing, low-power inference

### Feedforward Networks (FNN)

Classical neural network components optimized for GPU:

- **Linear Layers**: Matrix multiplication with bias
- **Activations**: ReLU, GELU, SiLU, and more
- **Layer Normalization**: Stable training with normalized activations
- **Dropout**: Regularization with configurable drop rates

**Use Case**: Feature extraction, dimensionality reduction, standard deep learning

### Attention Mechanisms

Modern transformer-based attention:

- **Flash Attention 2**: Memory-efficient attention with IO-aware computation
- **Multi-Head Attention**: Parallel attention subspaces
- **Attention Scores**: Computing similarity matrices
- **Attention Output**: Weighted aggregation of values

**Use Case**: Sequence modeling, transformer architectures, cross-attention fusion

### Memory Operations

Biologically-inspired memory systems:

- **Key-Value Memory**: Retrieve information based on query similarity
- **Context Aggregation**: Combine multiple memory contexts
- **Associative Recall**: Query-based information retrieval
- **Episodic Encoding**: Store and retrieve episodic memories

**Use Case**: Few-shot learning, meta-learning, episodic memory integration

### FAISS Integration

GPU-accelerated vector similarity search:

- **Pairwise Distances**: Compute L2 distances between vectors
- **Top-K Selection**: Efficient k-nearest neighbor search
- **Batch Operations**: Process multiple queries simultaneously

**Use Case**: Semantic search, nearest neighbor classification, similarity-based retrieval

### Place & Time Cells

Spatial and temporal encoding:

- **Grid Cell Encoding**: Represent spatial coordinates
- **Time Cell Population**: Encode temporal information
- **Hippocampal-inspired**: Biologically-plausible spatial navigation

**Use Case**: Spatial reasoning, temporal sequence understanding, navigation tasks

### Learning Algorithms

Advanced optimization and learning:

- **EWC** (Elastic Weight Consolidation): Continual learning without catastrophic forgetting
- **NLMS** (Normalized Least Mean Square): Online adaptive filtering
- **Whitening**: Decorrelation of features
- **Natural Gradients**: Second-order optimization

**Use Case**: Continual learning, online adaptation, multi-task learning

---

## Installation

### System Requirements

- **Python**: 3.10, 3.11, or 3.12
- **OS**: Linux, Windows, or macOS
- **GPU Driver**: Latest Vulkan drivers installed
- **Dependencies**: NumPy >= 1.24.0

### From PyPI

pip install grilly

### From Source

git clone https://github.com/grillcheese-ai/grilly.git
cd grilly
pip install -e ".[dev]"

### Verifying Installation

import grilly

# Check Vulkan availability
print(f"Vulkan available: {grilly.VULKAN_AVAILABLE}")

# Initialize compute backend
backend = grilly.Compute()
print("Backend initialized successfully")

---

## Quick Start

### Basic Compute Operations

import grilly
import numpy as np

# Initialize compute backend
backend = grilly.Compute()

# Prepare input data
input_current = np.random.randn(1000).astype(np.float32)
membrane = np.zeros(1000, dtype=np.float32)
refractory = np.zeros(1000, dtype=np.float32)

# Run LIF neuron step (spike generation)
membrane, refractory, spikes = backend.lif_step(
    input_current, 
    membrane, 
    refractory,
    dt=0.001,           # Timestep in seconds
    tau_mem=20.0,       # Membrane time constant in ms
    v_thresh=1.0        # Spike threshold
)

print(f"Spike count: {np.sum(spikes)}")
print(f"Membrane potential range: [{membrane.min():.3f}, {membrane.max():.3f}]")

### SNN Processing Pipeline

import grilly
import numpy as np

# Initialize SNN with 1000 neurons
snn = grilly.SNNCompute(n_neurons=1000)

# Generate temporal input (timesteps × features)
input_sequence = np.random.randn(10, 1000).astype(np.float32)

# Process sequence through SNN
for timestep in range(input_sequence.shape[0]):
    output_spikes = snn.process(input_sequence[timestep])
    print(f"Timestep {timestep}: {np.sum(output_spikes)} spikes")

### Using High-Level Modules

import grilly
import numpy as np

# Neural network layer
linear_layer = grilly.nn.Linear(in_features=768, out_features=512)

# Functional operations
x = np.random.randn(32, 768).astype(np.float32)
output = grilly.functional.linear(x, linear_layer.weight, linear_layer.bias)
activated = grilly.functional.relu(output)

# Attention mechanism
attention = grilly.nn.Attention(embed_dim=512, num_heads=8)
attended_output = attention(activated)

print(f"Output shape: {attended_output.shape}")

---

## API Reference

### Main Classes

#### VulkanCompute (alias: Compute)

Primary GPU compute backend for low-level operations.

**Constructor:**
backend = grilly.Compute()

**Key Methods:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `lif_step()` | input, membrane, refractory, dt, tau_mem, v_thresh | (membrane, refractory, spikes) | Single LIF neuron timestep |
| `hebbian_learning()` | weights, pre_activity, post_activity, learning_rate | weights | Hebbian weight updates |
| `stdp_learning()` | weights, pre_spikes, post_spikes, dt, A_plus, A_minus | weights | STDP weight updates |
| `linear()` | input, weight, bias | output | Linear transformation |
| `activation_relu()` | input | output | ReLU activation |
| `layernorm()` | input, weight, bias, eps | output | Layer normalization |
| `flash_attention2()` | query, key, value | output | Memory-efficient attention |

#### SNNCompute

High-level interface for spiking neural networks.

**Constructor:**
snn = grilly.SNNCompute(n_neurons=1000, dt=0.001, tau_mem=20.0)

**Key Methods:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `process()` | input | spikes | Process single timestep |
| `reset()` | - | - | Reset neuron states |
| `get_state()` | - | dict | Get membrane/refractory state |
| `set_state()` | state | - | Set neuron states |

#### VulkanLearning

Learning algorithm operations.

**Constructor:**
learning = grilly.Learning()

**Key Methods:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `ewc_penalty()` | params, fisher_info, weight | loss | Elastic weight consolidation penalty |
| `nlms_update()` | weights, input, error, step_size | weights | NLMS adaptive filtering |
| `whitening()` | data, eps | whitened | Feature whitening |
| `natural_gradient()` | gradient, fisher_info | ng | Natural gradient computation |

#### CapsuleTransformer

Hippocampal-inspired memory architecture.

**Constructor:**
capsule = grilly.CapsuleTransformer(config)

**Features:**
- Episodic memory encoding
- Spatial representation learning
- Temporal context integration
- Associative recall

### Submodule APIs

#### grilly.nn

Neural network layers and modules:

import grilly.nn as nn

# Available layers
linear = nn.Linear(768, 512)
attention = nn.Attention(embed_dim=512, num_heads=8)
norm = nn.LayerNorm(512)
dropout = nn.Dropout(p=0.1)
embedding = nn.Embedding(vocab_size=10000, embedding_dim=768)

#### grilly.functional

Functional operations (non-module):

import grilly.functional as F

# Activations
y = F.relu(x)
y = F.gelu(x)
y = F.silu(x)

# Linear operations
y = F.linear(x, weight, bias)
y = F.matmul(x, y)

# Normalization
y = F.layer_norm(x, normalized_shape, weight, bias)

# Attention
y = F.flash_attention2(query, key, value)
y = F.attention_scores(query, key)

#### grilly.optim

Optimizers:

import grilly.optim as optim

# Initialize optimizer
optimizer = optim.Adam(params, lr=1e-3, betas=(0.9, 0.999))
optimizer = optim.SGD(params, lr=0.01, momentum=0.9)
optimizer = optim.NLMS(params, step_size=0.01)
optimizer = optim.NaturalGradient(params, lr=1e-3)

# Training loop
for epoch in range(num_epochs):
    loss = forward_pass(model, data)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

#### grilly.utils

Utility functions:

import grilly.utils as utils

# Device management
device = utils.get_device_manager()
backend = utils.get_vulkan_backend()

# HuggingFace integration
bridge = utils.get_huggingface_bridge()
vulkan_model = bridge.convert_transformer(hf_model)

# Tensor conversion
vulkan_tensor = utils.to_vulkan(numpy_array)
numpy_array = utils.from_vulkan(vulkan_tensor)

# PyTorch compatibility
tensor = utils.tensor([1, 2, 3])
zeros = utils.zeros((batch_size, features))
randn = utils.randn((hidden_dim,))

---

## Core Components

### Compute Backend

The compute backend is the foundation of Grilly, managing:

1. **GPU Selection**: Automatically selects best available GPU
2. **Memory Management**: Efficient allocation and deallocation
3. **Kernel Dispatch**: Vulkan command buffer execution
4. **Synchronization**: Proper GPU-CPU synchronization

### Shader Layer

Grilly uses pre-compiled SPIR-V shaders for:

- **Performance**: Zero-overhead abstraction to GPU kernels
- **Portability**: SPIR-V runs on all Vulkan-capable devices
- **Reliability**: Tested and validated compute kernels

**Shader Directory Structure:**
shaders/
├── nn/                 # Neural network operations
│   ├── linear.comp
│   ├── activation.comp
│   └── layernorm.comp
├── snn/                # Spiking neural network
│   ├── lif.comp
│   ├── stdp.comp
│   └── hebbian.comp
├── attention/          # Attention mechanisms
│   ├── flash_attn2.comp
│   └── multihead_attn.comp
├── memory/             # Memory operations
│   └── kv_memory.comp
└── spv/                # Compiled SPIR-V binaries

### Memory Architecture

Grilly implements multi-tier memory strategy:

1. **GPU Global Memory**: Large tensors, activations
2. **GPU Local Memory**: Temporary computations, shared buffers
3. **Staging Buffers**: CPU ↔ GPU transfers
4. **Persistent Mappings**: Zero-copy for compatible hardware

---

## Advanced Usage

### Custom Shader Compilation

For development or custom kernels:

# Compile single shader
glslc shaders/nn/custom.comp -o shaders/spv/custom.spv

# Compile all shaders
python scripts/compile_shaders.py

# Verify compilation
vulkaninfo | grep "Compute"

### GPU Selection

Control which GPU Grilly uses:

import os
import grilly

# Select specific GPU by index
os.environ['VK_GPU_INDEX'] = '1'

# Allow CPU fallback (not recommended)
os.environ['ALLOW_CPU_VULKAN'] = '1'

# Initialize after setting environment
backend = grilly.Compute()

### HuggingFace Integration

Seamlessly use transformers with Vulkan backend:

import grilly
from transformers import AutoModel
from grilly.utils import get_huggingface_bridge

# Load HuggingFace model
hf_model = AutoModel.from_pretrained("bert-base-uncased")

# Convert to Vulkan
bridge = get_huggingface_bridge()
vulkan_model = bridge.convert_transformer(hf_model)

# Use with Grilly
embedding = vulkan_model(input_ids)

### PyTorch Interoperability

Use Grilly alongside PyTorch:

import torch
import grilly
from grilly.utils import tensor_conversion

# PyTorch model
pt_model = torch.nn.Linear(768, 512)

# Convert weights to Grilly
grilly_weights = tensor_conversion.to_vulkan(pt_model.weight)

# Compute on GPU via Vulkan
output = grilly.functional.linear(input_data, grilly_weights)

# Convert back to PyTorch
pt_output = tensor_conversion.from_vulkan(output)

### Batched Operations

Process multiple samples efficiently:

import grilly
import numpy as np

backend = grilly.Compute()

# Batched inputs (batch_size × features)
batch_input = np.random.randn(32, 1000).astype(np.float32)

# Single kernel call processes entire batch
output = backend.linear(batch_input, weights, bias)
print(output.shape)  # (32, 512)

### Profiling and Optimization

Measure performance:

import grilly
import time

backend = grilly.Compute()

# Warmup
for _ in range(10):
    _ = backend.linear(input_data, weights, bias)

# Benchmark
start = time.perf_counter()
for _ in range(100):
    _ = backend.linear(input_data, weights, bias)
elapsed = time.perf_counter() - start

print(f"Average time: {elapsed / 100 * 1000:.2f} ms")

---

## GPU Selection and Optimization

### Automatic Selection Strategy

Grilly selects GPUs in this priority order:

1. **NVIDIA CUDA-capable**: Highest priority
2. **AMD RDNA/RDNA2**: High-performance alternatives
3. **Intel Arc**: Intel discrete GPUs
4. **iGPU**: Integrated graphics (fallback)
5. **CPU/llvmpipe**: Software rendering (not recommended)

### Manual Selection

import os

# By index
os.environ['VK_GPU_INDEX'] = '0'

# By device name
os.environ['VK_GPU_NAME'] = 'NVIDIA GeForce RTX 4090'

### Performance Considerations

- **Memory bandwidth**: Use larger batch sizes on high-bandwidth GPUs
- **Compute cores**: SNNs benefit from many-core GPUs (AMD, NVIDIA)
- **Tensor cores**: Attention mechanisms leverage specialized hardware
- **Power efficiency**: Intel Arc offers excellent power/performance ratio

---

## Contributing

### Development Setup

git clone https://github.com/grillcheese-ai/grilly.git
cd grilly
pip install -e ".[dev]"

### Running Tests

# All tests
pytest

# Specific test file
pytest tests/test_lif.py -v

# With coverage
pytest --cov=grilly tests/

### Code Style

Grilly follows strict code quality standards:

# Format code
black grilly/

# Sort imports
isort grilly/

# Lint
ruff check grilly/

# Type checking
mypy grilly/

### Contributing Guidelines

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Follow code style (Black, Ruff, MyPy)
4. Add tests for new functionality
5. Submit pull request with clear description

---

## Project Information

**Current Version**: 0.1.0 (Alpha)

**Status**: 🚧 Under Active Development - Alpha (v0.1.0)
- Not production-ready
- APIs unstable and subject to change
- Features incomplete or experimental
- For research and experimentation only

**License**: MIT

**Author**: Nick (nick@grillcheeseai.com)

**Repository**: [https://github.com/grillcheese-ai/grilly](https://github.com/grillcheese-ai/grilly)

**Homepage**: [https://grillcheeseai.com](https://grillcheeseai.com)

**Documentation**: [https://docs.grillcheeseai.com/grilly](https://docs.grillcheeseai.com/grilly)

---

## Roadmap

### Near-term (Q1-Q2 2026)

- Core SNN architecture stabilization
- Basic LIF neuron implementation
- STDP learning rule validation
- Flash Attention 2 implementation
- HuggingFace integration foundation
- Comprehensive test suite
- API documentation

### Medium-term (Q3-Q4 2026)

- Advanced neuromorphic learning rules
- Continual learning mechanisms
- Capsule memory architecture
- Multi-GPU scaling
- Performance optimization
- Stability improvements

### Long-term (2027+)

- **ONNX Export**: Convert Grilly models to portable format
- **WebGPU Support**: Run in web browsers
- **Mobile Support**: Android/iOS GPU acceleration
- **Distributed Training**: Multi-GPU scaling
- **Quantization**: INT8/INT4 inference optimization
- Production-ready release (v1.0.0)

### Research Areas

- Advanced neuromorphic learning rules
- Continual learning mechanisms
- Biological plausibility metrics
- Energy efficiency optimization
- Neuromorphic-transformer hybrids

---

## Known Limitations & Issues

### Current Limitations (Alpha Status)

- **API Stability**: APIs subject to breaking changes in minor versions
- **Feature Completeness**: Many features are incomplete or experimental
- **Performance**: Not yet optimized; focus is on correctness
- **Documentation**: Work in progress; examples may be incomplete
- **Testing**: Test coverage is improving but not comprehensive
- **Debugging**: Limited error messages and debugging tools

### Common Issues

**Issue**: "Vulkan not available" error
- **Solution**: Ensure Vulkan drivers are installed and up-to-date
- **Status**: Known issue; needs better error handling

**Issue**: "No suitable GPU found"
- **Solution**: Check GPU support and drivers with `vulkaninfo`
- **Status**: GPU detection in development

**Issue**: Out of memory errors
- **Solution**: Reduce batch size; memory management is being optimized
- **Status**: Memory allocation strategy being refined

**Issue**: Slow performance
- **Solution**: This is expected in alpha; performance optimization is planned
- **Status**: Not a priority until API stabilization

**Issue**: API changes between versions
- **Solution**: Review CHANGELOG.md for breaking changes
- **Status**: Expected during alpha development

### Getting Help

- **GitHub Issues**: Report bugs at [https://github.com/grillcheese-ai/grilly/issues](https://github.com/grillcheese-ai/grilly/issues)
- **Discussions**: Ask questions at [https://github.com/grillcheese-ai/grilly/discussions](https://github.com/grillcheese-ai/grilly/discussions)
- **Email**: Contact nick@grillcheeseai.com for development inquiries

---

## References and Related Work

Grilly builds on decades of neuromorphic computing research:

- Spiking Neural Networks: O'Connor & Welling (2016), Neftci et al. (2019)
- Spike-Timing-Dependent Plasticity (STDP): Gerstner & Kistler (2002)
- Hebbian Learning: Hebb (1949), Modern implementations in SNNs
- Attention Mechanisms: Vaswani et al. (2017) - Transformer architecture
- Biological Learning Rules: Bi & Poo (2001), Markram et al. (2012)
- GPU Acceleration: Vulkan specification, Khronos Group
- Vector Search: Johnson et al. (2019) - FAISS: Billion-Scale Similarity Search

---

## License

Grilly is released under the MIT License. See LICENSE file for details.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

---

**Last Updated**: January 26, 2026

**Maintainer**: Grillcheese AI Team