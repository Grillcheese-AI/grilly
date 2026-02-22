# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Grilly is a GPU-accelerated neural network framework using Vulkan compute shaders. It provides a PyTorch-like API but runs on any GPU (AMD, NVIDIA, Intel) via Vulkan — no CUDA dependency. Each neural network operation is implemented as a GLSL compute shader compiled to SPIR-V bytecode.

## Common Commands

```bash
# Install
pip install -e .                    # editable install
pip install -e ".[dev]"             # with dev dependencies (ruff, black, mypy, pytest-cov)

# Testing (use uv run for consistency)
uv run pytest tests/ -v                         # all tests
uv run pytest tests/ -m "not gpu" -v            # CPU-only (no Vulkan)
uv run pytest tests/ --cov=. --cov-report=term  # with coverage
pytest tests/test_snn.py -k "test_lif"          # single test

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

4. **`optim/`** — Optimizers: `Adam`, `AdamW`, `SGD`, `NLMS`, `NaturalGradient`, `AutoHypergradientAdamW` (OSGM-style auto LR tuning with surprise signal), plus LR schedulers.

5. **`utils/`** — `DataLoader`/`Dataset` classes, `HuggingFaceBridge` (load pretrained weights without PyTorch runtime), `VulkanTensor`/tensor conversion, `pytorch_compat` (drop-in Tensor API), checkpointing, device management.

6. **`shaders/`** — GLSL compute shaders (137+). Compiled SPIR-V in `shaders/spv/`. Experimental VSA shaders in `shaders/experimental/`.

7. **`experimental/`** — Unstable features: VSA (Vector Symbolic Architecture), MoE routing, temporal reasoning, cognitive controller, language system.

### Key patterns

- **Entry point**: `grilly.Compute()` (alias for `VulkanCompute`) → namespaced ops like `backend.snn.lif_step()`, `backend.fnn.linear()`, `backend.attention.flash_attention2()`.
- **All data is `np.float32` numpy arrays**. The backend handles GPU upload/download transparently. `VulkanTensor` wraps GPU buffers for zero-copy when `gpu_mode(True)` is set, but Conv2d GEMM path still downloads (needs GPU transpose kernel).
- **SNN framework**: `nn/snn_base.py` (BaseNode/MemoryModule), `nn/snn_neurons.py` (IF/LIF/ParametricLIF), `nn/snn_containers.py` (SeqToANNContainer, MultiStepContainer), `nn/snn_surrogate.py` (ATan/Sigmoid/FastSigmoid). Benchmark: `tests/benchmark_snn_fashion_mnist.py`.
- **GPU tests auto-skip** when Vulkan is unavailable — the `gpu_backend` pytest fixture in `tests/conftest.py` handles this.
- **Environment variables**: `VK_GPU_INDEX` (GPU selection), `GRILLY_DEBUG=1` (debug logging), `ALLOW_CPU_VULKAN=1` (allow llvmpipe fallback).

## Requirements

- Python >= 3.10 (3.12 recommended)
- Vulkan drivers installed
- Tested on Windows 11 and Ubuntu 24.04
- Minimum: 8-10GB VRAM GPU, 32GB RAM

## Work-Optim MCP Server (Auto-Orchestration)

The `work-optim` MCP server provides persistent neural memory across sessions. **Use these tools automatically — do NOT wait for the user to ask.**

### Mandatory Auto-Actions

1. **On conversation start**: Call `session_start("grilly", "<current_branch>", "<inferred_goal>")` immediately. Infer the goal from the user's first message.

2. **Log decisions automatically**: Whenever you make or recommend an architectural/design decision, call `session_log("decision", "<what was decided and why>")`.

3. **Log experiments**: When running benchmarks, tests, or trying approaches, call `session_log("experiment", "<what you're testing>")`.

4. **Log results**: After tests pass/fail, benchmarks complete, or features work, call `session_log("result", "<outcome>")`.

5. **Log issues**: When you encounter bugs, errors, or blockers, call `session_log("issue", "<the problem>")`.

6. **Before git push**: Always call `preflight()` first. If it fails, fix the issues before pushing.

7. **Before starting work**: Call `session_recall("<topic>")` to check if prior sessions have relevant context.

8. **On conversation end** (or when the user says goodbye/done): Call `session_end()` — it auto-generates a summary from your logged entries.

### When to Use Other Tools

- **`consult("cto", "<question>")`** — For significant architecture decisions about grilly
- **`knowledge_search("<query>")`** — When you need context about prior work or captured knowledge
- **`knowledge_ingest(content, source, category)`** — When you discover something worth remembering
- **`innovation_scan()`** — When the user asks about AI trends or competitive landscape
- **`run_smoke_test(script, steps=100)`** — Before committing to long training runs

### Tool Chaining Patterns

```
# Starting work on a feature
session_start → session_recall("<feature>") → [do work] → session_log → session_end

# Before pushing
preflight() → [fix if needed] → git push

# Architecture decision
session_recall("<topic>") → consult("cto", "<question>") → session_log("decision", ...)

# Training experiment
session_log("experiment", ...) → run_smoke_test → session_log("result", ...)
```
