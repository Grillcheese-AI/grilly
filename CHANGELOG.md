# Changelog

All notable changes to this project will be documented in this file.

This changelog follows the spirit of **Keep a Changelog** and uses the terms **Added**, **Changed**, **Fixed**, and **Performance**.

---

## [Unreleased]

### Added

- **`tests/sentencepiece/test_sentencepiece_parity.py`** — Adaptive SentencePiece parity tests vs Hugging Face references (T5/MT5-style tokenizers; auto-skip until `grilly.tokenizers` SentencePiece support is available).
- **`tests/tokenizers/test_gpu_tokenizer_parity.py`** — Adaptive real tokenizer parity tests vs Hugging Face references (auto-skip until `grilly.tokenizers` lands and assets are available).
- **`tests/sentence_transformers/`**, **`tests/transformers_compat/`**, **`tests/converter/`**, **`tests/autograd_chain/`**, **`tests/moe_quant/`** — CI-safe skip-marked scaffold suites to make pre-v1.0 roadmap targets executable and incrementally fillable.
- **`docs/pre-v1.0/OPTIMIZATION_PARITY_TASKLIST.md`** — Post-upgrade pre-v1.0 optimization + feature parity execution plan (P0/P1/P2 workstreams, acceptance criteria, and verification commands).
- **`backend/fnn_chain.py`** — `FnnChainRecorder` / `ChainBufferHandle`: batched `linear` / `relu` / `softmax` recording with `read()` / `read_multiple()` for one submit+wait (then one or many downloads); `VulkanCompute.record_commands(fnn_chain=True)` or `VulkanFNN.chain_record()`. See `docs/PERF_DISPATCH.md`.
- **`tests/parity/`** — Numerical parity tests for `grilly.functional` (numpy reference; optional PyTorch `F.linear` / `F.relu` when `torch` is installed). See `tests/parity/README.md`.
- **`tests/parity/test_optimizers_parity.py`** — SGD and Adam (CPU) stepping vs `torch.optim`.
- **`docs/MIGRATION_PYTORCH.md`** — PyTorch → Grilly migration cookbook (device model, functional layout, module backend lifecycle, debugging).
- **`docs/PERF_DISPATCH.md`** — Vulkan dispatch/batching (`record_commands`, async dispatch), Sequential fusion notes, pybind GIL checklist.

### Changed

- **`docs/pre-v1.0/OPTIMIZATION_PARITY_TASKLIST.md`** — Re-prioritized roadmap to feature-delivery order: GPU tokenizer -> sentence-transformers -> transformers compatibility -> PyTorch→Grilly converter, with supporting throughput track; expanded tokenizer interoperability to include SentencePiece compatibility targets.
- **`utils/tensor_conversion.py`**, **`backend/base.py`** — `VulkanTensor.prepare_for_dispatch()` and C++ `Tensor.gpu_handle_if_valid()` binding so GPU-resident tensors reuse buffers in `BufferMixin._prepare_input` without redundant uploads when possible.
- **`docs/PYTORCH_PARITY_TASKLIST.md`**, **`docs/PYTORCH_PARITY_STATUS.md`**, **`docs/GPU_OPTIMIZATION_REVIEW.md`** — Workstream C (kernel/throughput milestone) marked complete with scoped deliverables; follow-ups consolidated under **Workstream C — future**.
- **`docs/api/functional.md`** — PyTorch parity notes and links to migration docs and parity tests.
- **`backend/compute.py`** — `VulkanCompute` exposes `record_commands`, `dispatch_compute`, `dispatch_compute_async`, `wait_async`, `wait_fence`.
- **`backend/fnn.py`** — `_linear_relu_recorded_chain`: Linear→ReLU in one submit when `fused-linear-relu` shader is absent.
- **`cpp/python/bindings_{activations,attention,linear}.cpp`** — GIL release around heavy ops; `require_c_contiguous_float` on `linear` inputs.

---

## [0.5.0] — 2026-03-18 — "GPU-First"

### Added

#### GPU-First Architecture
- **C++ Tensor exposed to Python** — `grilly_core.Tensor` with dual-validity GPU/CPU tracking, lazy sync, autograd support (`requires_grad`, `grad`, `grad_fn`, `backward()`)
- **VulkanTensor wraps C++ Tensor** — data stays on GPU between operations, `.numpy()` for explicit download
- **bindings.cpp split** — 4,221-line monolith → 11 focused files (core, linear, activations, conv, attention, normalization, optim, loss, snn, pooling, misc)
- **JIT compilation framework** — `@grilly.jit` decorator, trace-based graph capture with `TracedGraph`, auto-retrace on shape change
- **Automatic Mixed Precision** — `autocast` context manager + `GradScaler` for float16 compute paths

#### New Attention Architectures
- **Flash Attention 3** — subgroup-accelerated online softmax (wave64), double-buffered shared memory, register-resident queries
- **HYLAAttention** — Hypernetwork Linear Attention with local nonlinearity (RMSNorm + ReLU), no global softmax sync
- **FNetMixing** — parameter-free FFT token mixing, O(d log d)
- **SympFormerBlock** — Nesterov-accelerated attention with momentum stream and learned step sizes
- **SelectiveMomentumAttention** — TAPPA + SympFormer: momentum only on unpredictable heads

#### TAPPA Integration (ICLR 2026)
- **q-similarity shader** — cosine similarity between consecutive queries per attention head
- **Adaptive KV cache** — `compute_head_budgets()`, `classify_heads()`, `eviction_priority()` blending H2O with q-similarity

#### CubeMind / NVSA Support
- **Bit-packed HDC ops** — uint32 XOR bind, popcount bundle/similarity, cyclic permute (32x memory compression)
- **Block-code circular convolution** — per-block one-hot binding with zero magnitude drift after 1000+ chained ops
- **Sanger GHA shader** — streaming O(d) PCA for neurogenesis whitening at 75% capacity
- **DisARM gradient estimator** — antithetic sampling through discrete block-code ops (7.5x lower variance vs REINFORCE)
- **BatchVSAEncoder** — GPU batch text→hypervector encoding via VulkanVSA
- **BatchEmbedder** — batch ONNX inference for 1000+ sentences/call

#### New GLSL Compute Shaders (16 new, 190 total)
- `flash-attention3.glsl`, `attention-q-similarity.glsl`
- `tensor-transpose.glsl`, `tensor-cat.glsl`, `tensor-where.glsl`, `tensor-index-select.glsl`, `tensor-bmm.glsl`
- `hdc-bind-packed.glsl`, `hdc-bundle-packed.glsl`, `hdc-similarity-packed.glsl`, `hdc-permute-packed.glsl`
- `blockcode-bind.glsl`, `blockcode-unbind.glsl`, `blockcode-similarity.glsl`
- `sanger-gha.glsl`

#### Framework Features
- **ProjectionHeads** — multi-head structured embeddings (semantic 256D + emotion 128D + intent 128D + context 128D = 640D)
- **StreamingPipeline** — producer-consumer GPU embed + parallel upload to vector stores

### Changed
- **nn/modules.py split** — 1,939 lines → 8 focused files + re-export shim
- **nn/ modules GPU-first** — `_convert_input()` always passes VulkanTensor through
- **functional/ uses C++ bridge** — `_bridge.linear()` etc. with `_to_numpy()` safety

### Deprecated
- **`Compute()`** — emits `DeprecationWarning`. Use `nn.*` modules or `backend._bridge` functions. Removal planned for 0.6.0.

### Performance
- 1,820 tests passing (up from 1,528), 0 regressions
- Eliminated CPU↔GPU ping-pong at C++ binding boundary
- JIT trace captures op graph for future OpGraph fusion

---

## [0.4.0] - 2026-02-22

### Added

#### New Vulkan Shaders (Inference & RDNA2 Optimization)
- **`rms-norm.glsl`**: RMSNorm shader (2-pass: compute mean(x^2), then normalize+scale). Used by Llama, Gemma, and modern architectures that skip mean subtraction.
- **`gqa-attention.glsl`**: Grouped Query Attention decode shader. Fuses repeat_kv expansion into the attention kernel — maps query heads to KV heads via integer division. Designed for single-token decode against KV-cache.
- **`swiglu-fused.glsl`**: Fused SwiGLU FFN shader. Combines gate_proj + up_proj + SiLU + elementwise multiply into one dispatch (eliminates 2 intermediate buffers).
- **`rms-norm-linear-fused.glsl`**: Fused RMSNorm → Linear projection. Eliminates intermediate buffer between normalization and linear transform (common in pre-norm Llama layers).
- **`int8-gemm.glsl`**: INT8 weight-only GEMM with FP32 accumulation. Activations are fp32, weights are int8 packed 4 per uint32, with per-group fp32 scales. RDNA2 optimized.
- **`dequant-4bit.glsl`**: 4-bit block-quantized GEMM. Weights packed 8 per uint32, with per-block scale and zero-point dequantization fused into the matmul kernel.

#### New Backend Methods
- **`VulkanFNN.rms_norm()`**: GPU-accelerated RMSNorm with VulkanTensor and device-local buffer support. 2-pass algorithm with CPU/Numba fallback.
- **`VulkanFNN.swiglu_fused()`**: Fused SwiGLU FFN that takes gate and up projection weights, produces `SiLU(x @ gate.T) * (x @ up.T)` in one GPU dispatch.
- **`VulkanFNN.gemm_int8()`**: INT8 weight-only GEMM. Accepts int8 weights with per-group scales, packs to uint32 for GPU, accumulates in FP32.
- **`VulkanAttention.gqa_decode_attention()`**: GQA decode attention for single-token generation. Maps query heads to KV heads for Grouped Query Attention without materializing expanded KV tensors.

#### Grilly Ecosystem Modules (Separate Repos)
- **[GrillyInference](https://github.com/grillcheese-ai/GrillyInference)**: Native fp16 inference engine for Llama-family models. Includes paged KV-cache with H2O eviction and VSA multi-scale summaries, SmoothQuant INT8 and 4-bit block quantization, layer offloading for 100B models on 12GB VRAM, and text generation with top-k/top-p sampling.
- **[GrillyCompression](https://github.com/grillcheese-ai/GrillyCompression)**: Block-wise DCT codec for activation compression (30-60% VRAM savings), KV-cache page compression (3-5x), and communication compression for future multi-GPU support.
- **[GrillyOptimum](https://github.com/grillcheese-ai/GrillyOptimum)**: HuggingFace Optimum-compatible Vulkan backend. `VulkanModelForCausalLM.from_pretrained()` and `generate()` compatible with HF pipelines.
- **[GrillyDistil](https://github.com/grillcheese-ai/GrillyDistil)**: SA-KD (Simulated Annealing-based Adaptive Temperature) distillation trainer with Metropolis acceptance, 50+ seed prompts per domain, and KL-divergence + CE combined loss.

### Changed
- **Version**: Bumped to 0.4.0 (from 0.3.x series)
- **Shader count**: 154 → 160 GLSL compute shaders
- **README**: Updated with ecosystem modules section and inference capabilities

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