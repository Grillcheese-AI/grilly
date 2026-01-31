# Changelog

All notable changes to Grilly will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-31

### Added
- Initial release of Grilly framework
- Vulkan compute backend for GPU acceleration
- Support for AMD, NVIDIA, and Intel GPUs via Vulkan
- Spiking Neural Network (SNN) operations
  - LIF (Leaky Integrate-and-Fire) neurons
  - GIF (Generalized Integrate-and-Fire) neurons
  - STDP (Spike-Timing-Dependent Plasticity)
  - Hebbian learning
  - Synaptic connections
  - Continuous-to-spike and spike-to-continuous bridges
- Feedforward Neural Network (FNN) operations
  - Linear layers with multiple activation functions
  - Activations: ReLU, GELU, SiLU, SoftMax, SoftPlus, SwiGLU, GEGLU, ReGLU, RoSwish, GCU
  - Layer normalization, RMS normalization, batch normalization
  - Flash Attention 2 with RoPE support
  - Convolutional networks (Conv2D, MaxPool2D, AvgPool2D)
  - LSTM cells
- Learning algorithms
  - EWC (Elastic Weight Consolidation)
  - NLMS (Normalized Least Mean Squares) with ensemble support
  - Fisher Information Matrix computation
  - Natural gradients
  - Adam optimizer
  - Whitening transforms
- Memory operations
  - FAISS-based similarity search (distance, top-k, IVF, k-means)
  - GPU-accelerated memory read/write
  - Context aggregation
  - Memory injection (concatenation, gating, residual)
  - Capsule networks with dentate gyrus expansion
- Transformer support
  - VulkanSentenceTransformer for embedding models
  - Architecture-specific optimizations (BERT, GPT, T5, RoBERTa, DistilBERT, MPNet, XLM-RoBERTa, ALBERT)
  - HuggingFace model bridge (load weights without PyTorch runtime)
  - Fused operations (QKV projection, linear+activation)
  - Prosody-modulated attention
- Specialized operations
  - Place and time cells
  - Theta-gamma encoding
  - FFT operations (bit-reversal, butterfly, magnitude, power spectrum)
  - Domain adaptation (classification, routing, expert combination)
  - Semantic and affective encoding
- 137 GLSL compute shaders (138 compiled SPIR-V)
- LoRA (Low-Rank Adaptation) for efficient fine-tuning with backward pass
- Gradient checkpointing for memory optimization
- CPU fallback for unsupported operations
- Comprehensive test suite with GPU/CPU markers

### Features
- **Hardware Agnostic**: Works on AMD RX 6750 XT, NVIDIA RTX, Intel Arc
- **No PyTorch Runtime**: Load HuggingFace models as pure Vulkan tensors
- **Memory Efficient**: 12GB VRAM fine-tuning via LoRA
- **Bio-Inspired**: SNN operations for neuromorphic computing
- **Production Ready**: FastAPI integration examples

### Documentation
- Installation guide (uv and pip)
- Architecture-specific shader guide
- CLAUDE.md for AI assistant integration
- Example notebooks for common use cases

### Known Issues
- Some advanced SNN features still in development (PLIF, LSNN, Izhikevich)
- ANN→SNN conversion tool pending
- Multi-GPU support not yet implemented

### Upcoming (0.2.0)
- PLIF (Parametric LIF) neurons
- LSNN (Long Short-Term Memory neurons)
- Surrogate gradients for SNN training
- ANN→SNN conversion tool
- Multi-GPU distributed training
- More architecture-specific optimizations
