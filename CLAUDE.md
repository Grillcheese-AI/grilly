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


---

# Resident Autograd (branch: `autograd-resident-backward`)

> Added June 2026. The sections above describe the Python/Vulkan framework. This
> section covers the **C++ resident autograd engine** ? a reverse-mode autograd
> that runs the whole training step on-GPU (forward outputs and gradient buffers
> never leave VRAM). **`AUTOGRAD_STATE.md` is the detailed source of truth**;
> this is the orientation map.

## Where the work lives

- **C++ engine**: `cpp/src/autograd.cpp` + `cpp/include/grilly/autograd/`
  (`BackwardEngine`, `TapeContext`, `BufferRegistry`).
- **Compiled Python binding surface**: `cpp/python/bindings_autograd.cpp`.
  **`cpp/python/bindings.cpp` is DEAD CODE** ? it is NOT in the build. Editing it
  does nothing. Confirm against `CMakeLists.txt` before touching any binding.
- **Tests / milestones**: `experimental/resident_train/`
  - `train_full.py` ? composed Cubby block, **numpy forward + resident backward**.
    Stable baseline. Online training (fresh random batch/step), `gradcheck` mode
    vs finite-diff. ~0.945 train / ~0.891 test acc (chance 0.25).
  - `train_full_resident.py` ? same block, **fully resident** (forward also on-GPU
    via `forward_rmsnorm`/`forward_linear`/`forward_swiglu`). Same accuracy.
  - `test_backward_*.py` (8 files: linear, chain, ce, silu, fanout, rmsnorm,
    mingru, swiglu) ? per-op gradient unit tests. Keep all green.
  - `ttr3.py` ? minimal resident-forward repro (was the crash repro; passes now).
  - `ttr8.py` ? proof the multi-tape t/t2/t3 backward runs clean.

## Build & run (Windows, AMD RX 6750 XT / RADV)

There is **no CUDA**. The engine is C++ compiled to `grilly_core.cp312-win_amd64.pyd`.

```powershell
# Rebuild after C++/header changes (skip shader recompile):
Get-Process python | Stop-Process -Force      # a live python LOCKS the .pyd and blocks the copy
powershell -NoProfile -ExecutionPolicy Bypass -File ".\rebuild.ps1" -SkipShaders
# Omit -SkipShaders only when a .glsl changed. Build prints "Build OK" then copies the .pyd.

# Run against the engine with the cubby-lm venv python (grilly editable-installed there):
& "C:\Users\grill\Documents\GitHub\cubby-lm\.venv\Scripts\python.exe" experimental\resident_train\train_full_resident.py
```

Crash exit codes: `0` ok, `1` Python exception, `-1073741819` (0xC0000005) =
access violation / heap corruption (C++ level). For crashes, redirect to a temp
file (`> $env:TEMP\out.txt 2>&1`) and check `$LASTEXITCODE` ? piping through
Select-String can drop the tail.

## The Vulkan validation layer is the primary debugger

Installed, no rebuild needed. Enable per-run:

```powershell
$env:VK_INSTANCE_LAYERS="VK_LAYER_KHRONOS_validation"
```

It reports invalid/destroyed `VkBuffer` handles, descriptor mismatches, and
push-constant range violations with the offending handle. **Use it before
guessing.** A `VkBuffer` handle that decodes to ASCII (e.g. a value ending
`...2eadc26e` = `"part."`) means a C++ struct field was overwritten by string
data ? a **dangling reference / use-after-free**, not a GPU bug.

## Invariants ? do not regress these

- **`BufferRegistry` stores entries in a `std::deque`, NOT `std::vector`**
  (`cpp/include/grilly/autograd/buffer_registry.h`). `resolve()` hands out
  `GrillyBuffer&` references held across later `alloc()` calls. A vector would
  reallocate on `push_back` and dangle those references ? 0xC0000005 during
  backward once enough buffers are registered. This was the root-cause crash;
  deque keeps element references stable. (Commit f74e58e.)
- **Barrier discipline in `BackwardEngine::backward`**: a node reads
  `grad_output_buffer` that may have been built by accumulation writes from
  downstream nodes. Handlers that `fillZero` (a TRANSFER write) need
  `transferComputeBarrier` (not plain `barrier`) so the zero is ordered against a
  downstream shader read. Two real bugs lived here (MinGRU?Linear race;
  three-branch RMSNorm accumulation race). Don't weaken these barriers.
- **`getOrCreate` caches pipelines by NAME only** (`cpp/src/pipeline_cache.cpp`):
  the first creation's `numBuffers`/`pushConstSize` wins; later calls with
  different sizes are ignored. Keep each shader's push-constant struct size
  consistent across every call site.

## Known open issue

- A **non-fatal push-constant validation warning** (`VUID-vkCmdPushConstants-offset-01795`):
  a 20-byte push (rms-norm `RMSNormParams`) is flagged against a layout reported
  as having no COMPUTE range, even though `rms-norm` is created with
  `pushConstSize=20`. Forward is numerically correct (gradchecks pass), so it's
  latent, not breaking ? RADV tolerates it. Under investigation in
  `CommandBatch::dispatch` (`cpp/src/command_batch.cpp` ~L94): the order of
  `vkCmdBindPipeline` / `vkCmdBindDescriptorSets` / `vkCmdPushConstants` and
  which layout the push uses.
- **NOTE for whoever picks this up**: there is currently temporary
  `[PIPE-CREATE]` stderr instrumentation (an `fprintf` + `#include <cstdio>`) in
  `getOrCreate` in `cpp/src/pipeline_cache.cpp`, built into the active `.pyd`.
  **Remove it before the final commit** for this issue.

## Relationship to cubby-lm

This branch builds the **resident GPU training path** that the sibling `cubby-lm`
repo needs for its `0.0.1 perf #2` (resident activations ? the ~25 ms/dispatch
floor). `cubby-lm` imports grilly as an editable path source and loads this
`.pyd` automatically. See `../cubby-lm/CLAUDE.md`.
