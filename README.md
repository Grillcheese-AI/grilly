# Grilly

GPU-accelerated neural network framework using Vulkan compute shaders. Supports AMD, NVIDIA, and Intel GPUs.

## Features

### Neural Network Operations
- **Feedforward Networks**: Linear layers, activations (ReLU, GELU, SiLU, SoftMax, SwiGLU, RoSwish, GCU)
- **Convolutional Networks**: Conv2D, MaxPool2D, AvgPool2D, BatchNorm2D (forward and backward)
- **Recurrent Networks**: LSTM cells
- **Attention Mechanisms**: Flash Attention 2, multi-head attention, RoPE, prosody modulation
- **Normalization**: LayerNorm, RMSNorm, BatchNorm
- **Activations**: GELU, SiLU, ReLU, SoftMax, SoftPlus, SwiGLU, GEGLU, ReGLU, RoSwish, GCU
- **Fused Operations**: Linear+activation fusion, QKV projection, layer normalization+linear

### Spiking Neural Networks
- **Neuron Models**: LIF (Leaky Integrate-and-Fire), GIF (Generalized Integrate-and-Fire)
- **Learning**: STDP (Spike-Timing-Dependent Plasticity), Hebbian learning
- **Synaptic Dynamics**: Forward propagation, STDP traces, weight updates
- **Bridges**: Continuous-to-spike, spike-to-continuous conversion
- **Operations**: SNN matmul, softmax, readout, expert readout

### Memory & Retrieval
- **Memory Operations**: Read, write, context aggregation
- **Memory Injection**: Concatenation, gating, residual connections
- **Capsule Networks**: Capsule projection, dentate gyrus sparse expansion
- **FAISS Integration**: Distance computation, top-k selection, IVF filtering, quantization, k-means

### Learning Algorithms
- **Optimization**: Adam, natural gradients, Fisher information matrix
- **Continual Learning**: EWC (Elastic Weight Consolidation), Fisher penalties
- **Adaptive Filtering**: NLMS (Normalized Least Mean Squares), ensemble, prediction
- **Regularization**: Dropout, whitening transforms

### Specialized Operations
- **Place & Time Cells**: Spatial encoding, temporal encoding, theta-gamma oscillations
- **FFT**: Bit-reversal, butterfly operations, magnitude, power spectrum
- **Domain Adaptation**: Domain classification, routing, expert combination
- **Embeddings**: Lookup, position encoding, attention, FFN, pooling, normalization
- **Loss Functions**: Cross-entropy, BCE, contrastive loss
- **Semantic Encoding**: Affect MLP, affective processing

### Transformer Support
- **Architecture-Specific Optimizations**: BERT, GPT, T5, RoBERTa, DistilBERT, MPNet, XLM-RoBERTa, ALBERT
- **HuggingFace Bridge**: Load pre-trained models without PyTorch runtime
- **Model Components**: Multi-head attention, positional encoding, layer normalization
- **Fine-Tuning**: LoRA (Low-Rank Adaptation), gradient checkpointing

### LoRA Fine-Tuning
- Parameter-efficient fine-tuning for transformers
- Backward pass support for LoRA layers
- Memory-efficient training on 12GB VRAM

## Installation

### From PyPI (when published)

```bash
pip install grilly
```

### From Source

```bash
git clone https://github.com/grillcheese-ai/grilly.git
cd grilly
make install

# Or with development dependencies
make install-dev

# Or manually
pip install -e .
```

## Requirements

- Python >= 3.10
- Vulkan drivers
- NumPy >= 1.24.0
- Supported GPUs: AMD (tested on RX 6750 XT), NVIDIA, Intel Arc

## Quick Start

```python
import grilly
import numpy as np

# Initialize compute backend
backend = grilly.Compute()

# Spiking neural network example
input_current = np.random.randn(1000).astype(np.float32)
membrane = np.zeros(1000, dtype=np.float32)
refractory = np.zeros(1000, dtype=np.float32)

membrane, refractory, spikes = backend.snn.lif_step(
    input_current, membrane, refractory,
    dt=0.001, tau_mem=20.0, v_thresh=1.0
)

# Feedforward network example
x = np.random.randn(32, 384).astype(np.float32)
weight = np.random.randn(384, 128).astype(np.float32)
bias = np.zeros(128, dtype=np.float32)

output = backend.fnn.linear(x, weight, bias)
activated = backend.fnn.swiglu(output)

# Flash Attention 2
q = np.random.randn(32, 8, 64, 64).astype(np.float32)  # (batch, heads, seq, dim)
k = np.random.randn(32, 8, 64, 64).astype(np.float32)
v = np.random.randn(32, 8, 64, 64).astype(np.float32)

attention_out = backend.attention.flash_attention2(q, k, v)

# FAISS similarity search
query = np.random.randn(1, 384).astype(np.float32)
database = np.random.randn(10000, 384).astype(np.float32)

distances = backend.faiss.compute_distances(query, database)
top_k_distances, top_k_indices = backend.faiss.topk(distances, k=10)
```

## API Reference

### Core Interfaces

- `grilly.Compute()` - Main compute backend (alias for VulkanCompute)
- `grilly.SNNCompute()` - High-level spiking neural network interface
- `grilly.Learning()` - Learning algorithms (EWC, NLMS, etc.)

### Backend Namespaces

- `backend.snn.*` - Spiking neural network operations
- `backend.fnn.*` - Feedforward network operations
- `backend.attention.*` - Attention mechanisms
- `backend.memory.*` - Memory operations
- `backend.faiss.*` - Vector similarity search
- `backend.learning.*` - Learning algorithms
- `backend.cells.*` - Place and time cells

## Shader Statistics

- Total GLSL shaders: 137
- Compiled SPIR-V shaders: 138
- Categories: 12+ operation types

## Compiling Shaders

Shaders are pre-compiled and included. To recompile:

```bash
# Compile all shaders (cross-platform)
make compile-shaders

# Verify compilation
make verify-shaders

# Or manually:
# Windows: .\scripts\compile_all_shaders.ps1
# Linux/Mac: ./compile_shaders.sh

# Single shader
glslc shader.glsl -o spv/shader.spv
```

## GPU Selection

```bash
# Set GPU index (if multiple GPUs)
export VK_GPU_INDEX=0

# Enable debug logging
export GRILLY_DEBUG=1

# Allow CPU fallback
export ALLOW_CPU_VULKAN=1
```

## Testing

```bash
# All tests
make test

# CPU-only tests (skip GPU)
make test-cpu

# GPU tests only
make test-gpu

# With coverage report
make test-coverage

# Or use pytest directly
pytest grilly/tests/ -v
```

## Architecture

Grilly uses Vulkan compute shaders for cross-platform GPU acceleration. Each operation is implemented as a GLSL compute shader compiled to SPIR-V bytecode.

### Design Principles

- Pure Vulkan backend (no CUDA dependency)
- Hardware-agnostic (AMD, NVIDIA, Intel)
- Zero-copy GPU memory operations
- Minimal CPU-GPU transfers
- CPU fallback for unsupported operations

## Performance

Tested on AMD RX 6750 XT (12GB VRAM):
- LIF neuron simulation: 1M neurons at >1000 FPS
- Flash Attention 2: 32 batch, 8 heads, 512 seq length at ~50ms
- FAISS top-k: 10K vectors, 384D, k=10 at ~5ms

## Examples

See `examples/` directory for detailed usage:
- Transformer fine-tuning with LoRA
- Spiking neural network training
- FAISS similarity search
- Continual learning with EWC

## Development

### Quick Start

```bash
# Clone and setup
git clone https://github.com/grillcheese-ai/grilly.git
cd grilly

# Install with dev dependencies
make install-dev

# Run tests
make test

# Format code
make format

# Run linters
make lint

# Build package
make build
```

### Project Structure

```
grilly/
├── backend/            # Vulkan backend implementation
├── nn/                 # High-level neural network modules
├── shaders/            # GLSL compute shaders
│   └── spv/           # Compiled SPIR-V bytecode
├── tests/             # Test suite
├── utils/             # HuggingFace bridge, utilities
└── Makefile           # Build automation
```

### Makefile Commands

Run `make help` to see all available commands:
- `make install` - Install package
- `make test` - Run tests
- `make compile-shaders` - Compile shaders
- `make build` - Build distribution
- `make publish-test` - Publish to Test PyPI
- `make publish` - Publish to PyPI
- `make format` - Format code
- `make lint` - Run linters
- `make clean` - Clean build artifacts

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Run `make check` to verify
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## References

- Vulkan Compute Shaders: https://www.khronos.org/vulkan/
- Flash Attention 2: https://arxiv.org/abs/2307.08691
- STDP Learning: Bi & Poo (1998)
- EWC: Kirkpatrick et al. (2017)
- LoRA: Hu et al. (2021)
