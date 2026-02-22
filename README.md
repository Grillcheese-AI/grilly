# Grilly

<p align="center">
  <img src="https://raw.githubusercontent.com/grillcheese-ai/grilly/main/assets/grilly_mascott_github.png" alt="Grilly" width="400">
</p>

*Deep learning, well done.*

[![CI](https://github.com/grillcheese-ai/grilly/actions/workflows/ci.yml/badge.svg)](https://github.com/grillcheese-ai/grilly/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/grilly)](https://pypi.org/project/grilly/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Alpha software.** Not production-ready. APIs may change. We welcome early adopters and feedback.

GPU-accelerated neural network framework using Vulkan compute shaders. No CUDA required. Supports AMD, NVIDIA, and Intel GPUs.

**Documentation:** <https://grilly.readthedocs.io/>

## Release Status

- Current release line: **v0.4.0**
- Package name: `grilly`
- Python support: `>=3.12`
- Release channel: PyPI

Versioning is automated via [setuptools-scm](https://github.com/pypa/setuptools_scm) from git tags (e.g. `v0.4.0` → `0.4.0`).

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

### Inference & RDNA2 Optimizations (v0.4.0)
- **RMSNorm**: GPU-accelerated 2-pass RMSNorm shader (used by Llama, Gemma)
- **GQA Decode Attention**: Fused Grouped Query Attention for single-token generation against KV-cache
- **Fused SwiGLU FFN**: gate_proj + up_proj + SiLU in one GPU dispatch
- **Fused RMSNorm+Linear**: Eliminates intermediate buffer in pre-norm transformer layers
- **INT8 GEMM**: Weight-only INT8 with FP32 accumulation and per-group scales
- **4-bit Block Quantization**: Packed 4-bit weights with per-block scale + zero-point

### LoRA Fine-Tuning
- Parameter-efficient fine-tuning for transformers
- Backward pass support for LoRA layers
- Memory-efficient training on 12GB VRAM

## Installation

### From PyPI

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

- Python >= 3.12
- Vulkan drivers
- NumPy
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

## Grilly Ecosystem

Optional extension modules for inference, compression, HuggingFace integration, and distillation:

| Module | Package | Description |
|--------|---------|-------------|
| [GrillyInference](https://github.com/grillcheese-ai/GrillyInference) | `grillyinference` | Native fp16 inference engine (Llama, paged KV-cache, INT8/4-bit quantization, 100B on 12GB) |
| [GrillyCompression](https://github.com/grillcheese-ai/GrillyCompression) | `grillycompression` | DCT activation compression (30-60% VRAM), KV-cache page compression (3-5x) |
| [GrillyOptimum](https://github.com/grillcheese-ai/GrillyOptimum) | `grillyoptimum` | HuggingFace Optimum Vulkan backend (`from_pretrained` + `generate`) |
| [GrillyDistil](https://github.com/grillcheese-ai/GrillyDistil) | `grillydistil` | SA-KD adaptive temperature distillation trainer |

```bash
# Install with inference support
pip install grilly grillyinference

# Full ecosystem
pip install grilly grillyinference grillycompression grillyoptimum grillydistil
```

## Shader Statistics

- Total GLSL shaders: 160
- Compiled SPIR-V shaders: 160
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
# All tests (requires Vulkan)
make test

# CPU-only tests (no GPU required - for CI)
make test-cpu

# GPU tests only
make test-gpu

# With coverage report
make test-coverage

# Or use pytest directly
pytest tests/ -v                    # all tests
pytest tests/ -m "not gpu" -v       # CPU-only
pytest tests/ -m "gpu" -v          # GPU-only
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

## Built for GrillCheese AI

Grilly powers [GrillCheese AI](https://github.com/grillcheese-ai/grillcheese), a neuromorphic language system that replaces pure transformer stacks with brain-inspired modules — hippocampal memory, thalamic routing, amygdala affect, and Oja-rule online plasticity — all running on Vulkan compute. The research explores four hypotheses:

- **H1 (Architecture)**: Modular neuromorphic design can match transformers while enabling episodic memory, continual learning, and affect-driven routing.
- **H2 (Efficiency)**: Vulkan-accelerated SSM training can reach >10,000 tok/s on a single consumer GPU — no CUDA or cloud required.
- **H3 (Memory)**: Capsule encoding (768D to 32D) with dentate gyrus sparse expansion preserves information for hippocampal retrieval via Matryoshka representation learning.
- **H4 (Plasticity)**: Online Oja-rule weight updates enable continual adaptation without catastrophic forgetting.

Grilly v1.0 will ship alongside the GrillCheese AI public release.

## Examples

A minimal forward + backward pass:

```python
import grilly.nn as nn

layer = nn.Linear(128, 10)
x = nn.randn(32, 128, requires_grad=True)

logits = x @ nn.Variable(layer.weight.T) + nn.Variable(layer.bias)
loss = logits.sum()
loss.backward()

print(x.grad.shape)  # (32, 128)
```

See [`examples/`](examples/) for more:
- `hello_grilly.py` — Autograd forward + backward
- `train_mlp.py` — Full training loop with AdamW and cross-entropy
- `benchmark_gemm.py` — GPU vs CPU GEMM throughput table
- 14 experimental examples (VSA, MoE, capsules, cognitive control, and more)

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
├── .github/workflows/  # CI (lint, test, build) and CD (PyPI publish)
├── backend/            # Vulkan backend implementation
├── mcp-servers/        # MCP servers for AI coders
│   ├── grilly/         # TypeScript MCP server (grilly_docs, grilly_example, etc.)
│   └── elephant-coder/ # Codebase memory (Python)
├── nn/                 # High-level neural network modules
├── shaders/            # GLSL compute shaders
│   └── spv/            # Compiled SPIR-V bytecode
├── tests/              # Test suite
├── utils/              # HuggingFace bridge, utilities
└── Makefile            # Build automation
```

### MCP Server for AI Coders

The **grilly** MCP server (`mcp-servers/grilly/`) helps AI assistants use Grilly:

- `grilly_docs` — API docs (overview, quickstart, snn, fnn, attention, faiss)
- `grilly_example` — Example code snippets
- `grilly_list_ops` — List backend operations
- `grilly_run_python` — Execute Python snippets

```bash
cd mcp-servers/grilly && npm install && npm run build
```

### Makefile Commands

Run `make help` to see all available commands:
- `make install` - Install package
- `make test` - Run tests
- `make compile-shaders` - Compile shaders
- `make build` - Build distribution
- `make format` - Format code
- `make lint` - Run linters
- `make clean` - Clean build artifacts

## CI/CD

- **CI** (on push/PR): Lint (ruff), test (CPU-only), build
- **CD** (on release): Build, publish to PyPI via [Trusted Publishing](https://docs.pypi.org/trusted-publishers/)

Releases are published automatically when you create a GitHub Release with a tag (e.g. `v0.3.1`). **No API token needed** — uses PyPI Trusted Publishing (OIDC).

### One-time setup: Trusted Publisher on PyPI

1. Go to [pypi.org/manage/projects](https://pypi.org/manage/projects/) → **Manage** → **Publishing**
2. Add a **GitHub** publisher:
   - **Owner:** `grillcheese-ai`
   - **Repository:** `grilly`
   - **Workflow name:** `publish.yml`

### Manual publish (local)

```bash
make build
twine upload dist/*
# Requires PyPI API token (create at pypi.org/manage/account/token/)
```

For Test PyPI: `twine upload --repository testpypi dist/*`

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Run `make check` to verify
5. Submit a pull request

## Roadmap and Community

Open an issue. Tell us what to implement or optimize.

Current priorities:
- Native fp16 inference engine (GrillyInference) — Llama 3.2 3B at 30+ tok/s on RX 6750 XT
- SmoothQuant INT8 and 4-bit block quantization for inference
- Paged KV-cache with H2O eviction for 128k context
- Training throughput (GEMM tiling, fused backward shaders)
- Backward pass coverage for all operations
- HuggingFace Optimum integration (GrillyOptimum)

## License

MIT License - see LICENSE file for details.

