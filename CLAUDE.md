# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Grilly is a GPU-accelerated neural network framework using Vulkan compute shaders. It provides a PyTorch-like API but runs on any GPU (AMD, NVIDIA, Intel) via Vulkan — no CUDA dependency. Each neural network operation is implemented as a GLSL compute shader compiled to SPIR-V bytecode.

## Common Commands

```bash
# Install
pip install -e .                    # editable install
pip install -e ".[dev]"             # with dev dependencies (ruff, black, mypy, pytest-cov)

# Testing
pytest tests/ -v                    # all tests
pytest tests/ -m "not gpu"          # CPU-only tests (no Vulkan required)
pytest tests/ -k "gpu"              # GPU-only tests
pytest tests/test_snn.py            # single test file
pytest tests/test_snn.py -k "test_lif"  # single test

# Linting & Formatting
ruff check .                        # lint (line-length=100, rules: E,F,W,I,N,UP; E501 ignored)
black . --check                     # format check (line-length=100, target py312)
isort . --check-only                # import sort check (profile=black)
mypy .                              # type check (py3.12, ignore_missing_imports=true)

# Shaders
glslc shader.glsl -o spv/shader.spv         # compile single shader
.\scripts\compile_all_shaders.ps1            # compile all (Windows)

# Build & publish
python -m build                              # build distribution
powershell -ExecutionPolicy Bypass -File .\scripts\publish_pypi.ps1  # publish to PyPI
```

## Architecture

### Package layout (pyproject.toml maps `grilly.*` to top-level dirs)

The repo root **is** the `grilly` package. `pyproject.toml` uses `tool.setuptools.package-dir` to map `grilly` -> `.`, `grilly.backend` -> `backend/`, etc.

### Layer stack

1. **`backend/`** — Low-level Vulkan GPU dispatch. Each file wraps a category of SPIR-V shaders:
   - `core.py` — Vulkan instance/device init, buffer alloc, shader loading, compute dispatch
   - `pipelines.py` — Pipeline/descriptor-set creation and LRU caching
   - `compute.py` — `VulkanCompute` composes all operation modules (`snn`, `fnn`, `attention`, `memory`, `faiss`, `cells`, `learning`, `fft`, `conv`, `normalization`, `lora`, etc.) into a single entry point
   - `shader_registry.py` — Selects architecture-specific shaders (BERT, GPT, T5, etc.) with generic fallback
   - `autograd_core.py` — `GradientTape`, `ComputationNode`, backward ops

2. **`nn/`** — PyTorch-like `Module` subclasses. `module.py` defines the base `Module` class (with `parameters()`, `train()`/`eval()`, `state_dict`, etc.). Submodules: standard layers, SNN neurons, memory, capsules, transformers, LoRA, multimodal fusion, autograd (`Variable` with full backward graph).

3. **`functional/`** — Stateless functional API (`grilly.functional.*`), mirrors `torch.nn.functional`. Thin wrappers that instantiate `VulkanCompute` and call backend methods.

4. **`optim/`** — Optimizers: `Adam`, `AdamW`, `SGD`, `NLMS`, `NaturalGradient`, plus LR schedulers.

5. **`utils/`** — `DataLoader`/`Dataset` classes, `HuggingFaceBridge` (load pretrained weights without PyTorch runtime), `VulkanTensor`/tensor conversion, `pytorch_compat` (drop-in Tensor API), checkpointing, device management.

6. **`shaders/`** — GLSL compute shaders (137+). Compiled SPIR-V in `shaders/spv/`. Experimental VSA shaders in `shaders/experimental/`.

7. **`experimental/`** — Unstable features: VSA (Vector Symbolic Architecture), MoE routing, temporal reasoning, cognitive controller, language system.

### Key patterns

- **Entry point**: `grilly.Compute()` (alias for `VulkanCompute`) → namespaced ops like `backend.snn.lif_step()`, `backend.fnn.linear()`, `backend.attention.flash_attention2()`.
- **All data is `np.float32` numpy arrays**. The backend handles GPU upload/download transparently.
- **GPU tests auto-skip** when Vulkan is unavailable — the `gpu_backend` pytest fixture in `tests/conftest.py` handles this.
- **Environment variables**: `VK_GPU_INDEX` (GPU selection), `GRILLY_DEBUG=1` (debug logging), `ALLOW_CPU_VULKAN=1` (allow llvmpipe fallback).

## Requirements

- Python >= 3.10 (3.12 recommended)
- Vulkan drivers installed
- Tested on Windows 11 and Ubuntu 24.04
- Minimum: 8-10GB VRAM GPU, 32GB RAM
