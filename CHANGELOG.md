# Changelog

All notable changes to this project will be documented in this file.

This changelog follows the spirit of **Keep a Changelog** and uses the terms **Added**, **Changed**, **Fixed**, and **Performance**.

---

## [0.3.1] - 2026-02-10

### Added
- **CI/CD**: GitHub Actions workflows for CI (lint, test, build) and CD (PyPI publish on release)
- **Build provenance**: Sigstore attestation for release artifacts via `actions/attest-build-provenance`
- **Versioning**: setuptools-scm from git tags (e.g. `v0.3.1` → `0.3.1`)

### Changed
- **Python**: Requires Python >= 3.12 (was >= 3.10)
- **Ruff**: Config moved to `[tool.ruff.lint]`, rule ignores for N806, F403, F405, etc., per-file ignores for `__init__.py`, `tests/**`, `utils/numba_ops.py`, `scripts/*`, `tutorials/*`
- **Tests**: Registered `gpu` marker in `conftest.py`; Vulkan-dependent tests use `@pytest.mark.gpu` so CPU-only runs (`pytest -m "not gpu"`) skip them in CI
- **README**: Updated release status, badges (CI, PyPI), requirements, testing commands, CI/CD docs, project structure

### Fixed
- Ruff deprecation warnings and import/sorting issues across backend, utils, nn, experimental
- Missing imports (e.g. `numpy`, `struct` in `backend/_glu_append.py`)
- `up_out` → `input_out` in `utils/vulkan_sentence_transformer.py`
- `TYPE_CHECKING` imports for `SimulationResult`, `LoRAModel`, `VulkanCore`, etc.

---

## [0.3.0] - 2026-02-07

### Added
- `scripts/ingest_svc.py`: streaming SVC ingestion to avoid double-encoding (ingest once through `CognitiveController`) and support chunked processing for large JSONL files.
- `tests/experimental/test_svc_ingest_speed.py`: regression test to ensure `InstantLanguage.ingest_svc(..., verbose=False)` stays quiet (no per-entry print spam).

### Changed
- Ingestion defaults are now friendlier for large corpora: optional `--no-ngrams` switch to bypass expensive n-gram HRR word construction during bulk ingestion.

### Fixed
- Test helper construction of `SVCEntry` now matches the current dataclass signature (`svc_s/svc_v/svc_c` instead of a nested `svc={...}` dict).

## 2026-02 RDNA2 performance series

### Added
- RDNA2-friendly block-wise GEMM kernels and shader variants for matrix-heavy paths.
- Top-k selection upgrade on GPU paths.

### Fixed
- Shader + test packaging to ship only modified kernels and their validation tests.

### Performance

### Added
- **RDNA2-friendly GEMM similarity path** for VSA dot-products (codebook projection / similarity) via the `gemm_mnk` shader (block-tiled matmul).
- **GPU top-k routing primitives**:
  - Row-wise argmax shader (`vsa-argmax-rows.glsl`)
  - Winner masking shader for iterative top-k (`vsa-mask-selected.glsl`)
- **Tiled GEMM VSA shader**: `vsa-similarity-gemm.glsl` (scores = Q · Cᵀ), used for large codebooks and batch queries.
- **GPU backend tests**:
  - `tests/experimental/test_backend_vsa_similarity_gemm.py`
  - `tests/experimental/test_backend_vsa_topk.py`
- **Fast streaming SVC ingestion script**: `scripts/ingest_svc.py` (chunked ingestion, progress reporting, optional n-gram disabling).

### Changed
- `backend/experimental/vsa.py`
  - Prefers **GEMM-based similarity** when `gemm_mnk` is available.
  - Adds/extends API for GEMM similarity and (iterative) top-k selection without full matrix readback.
- `experimental/language/svc_loader.py`
  - Adds a **batch sentence encoding fast path** (pad → bind_batch → bind_batch → bundle_batch).
  - Bipolarizes word/role/pos vectors for consistent Hadamard binding in the fast path.
- `experimental/cognitive/controller.py` and `experimental/language/system.py`
  - Ingestion paths updated to support faster bulk ingestion and optional feature disabling (templates/realm vectors/ngrams).
- **Ingestion workflow**
  - Avoids doing two full passes (language ingestion + controller ingestion) in the default script; controller ingestion is now the single entry point for large datasets.

### Fixed
- `backend/fnn.py::VulkanFNN.gemm()`
  - Corrected buffer upload/download usage and consistent shader key usage for `gemm_mnk`.

### Performance
- Bulk SVC ingestion no longer “hangs” due to duplicated work and excessive console output:
  - Chunked ingestion + progress throttling in `scripts/ingest_svc.py`.
  - Removes per-entry spam printing when `verbose=False` (guarded by a regression test).
- Large-codebook similarity and resonator projections move from many small reductions to **block-wise GEMM**, improving RDNA2 occupancy and memory reuse.

### Notes / Migration
- If you rely on experimental shaders, ensure the new/updated GLSL files are compiled to SPIR-V and present under your `shaders/**/spv` folder with matching names.
- GPU top-k selection is implemented as **k rounds** of (argmax → mask). This is ideal for small k (e.g., 1–8).

---

## 2026-02-06

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

### New Features (0.2.0)
- PLIF (Parametric LIF) neurons
- LSNN (Long Short-Term Memory neurons)
- Surrogate gradients for SNN training
- ANN→SNN conversion tool
- Multi-GPU distributed training
- More architecture-specific optimizations


### Upcoming (0.3.x)