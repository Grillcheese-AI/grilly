# Pre-v1.0 Optimization + Parity Tasklist

This tasklist is a post-upgrade pass focused on shipping a stable, fast pre-v1.0 runtime with clearer PyTorch parity guarantees.

Scope:
- Python Vulkan runtime orchestration (`backend/*`)
- C++ bridge integration (`cpp/python/*`)
- Functional + module parity tests (`tests/parity`, targeted GPU tests)
- Performance/throughput work that directly affects user-visible training/inference latency

---

## Baseline (already landed)

- `FnnChainRecorder` with `linear` / `relu` / `softmax`, `read`, and `read_multiple` (`backend/fnn_chain.py`)
- `VulkanTensor.prepare_for_dispatch()` residency bind path (`utils/tensor_conversion.py`)
- `_prepare_input()` GPU-resident fast path (`backend/base.py`)
- Conv backward-weight GPU GEMM path and parity test (`backend/conv.py`, `tests/test_conv_backward_weight_gemm.py`)
- MoE backward stability fix at production shape (`cpp/src/ops/moe_forward.cpp`): corrected output-projection grad matmul and added bounds checks; Python binding wired via `moe_backward_gpu(...)` entrypoint with safe CPU fallback (`cpp/python/bindings_moe.cpp`)
- VSA-LM fused C++ forward/backward (`grilly_core.vsa_lm_*`): AdditionLinear FFN (L1 distance shader) + sign activation + LayerNorm + output projection in one C++ call. CPU Eigen backward with full AdditionLinear gradient (STE for sign). New files: `cpp/src/ops/vsa_lm_forward.cpp`, `cpp/include/grilly/ops/vsa_lm_forward.h`, `cpp/python/bindings_vsa_lm.cpp`, `shaders/sign-activation.glsl`. Tests: `tests/test_vsa_lm_forward.py` (shape + parity).

---

## Priority Feature Roadmap (user-directed)

Order locked by product priority:
1. GPU tokenizer
2. Sentence-transformers
3. Transformers compatibility (target: near 1:1 behavior/signature coverage)
4. PyTorch -> Grilly converter

---

## P0: GPU Tokenizer (highest priority)

### P0.1 Core GPU tokenizer runtime
- Status: `[~]` (CPU parity path landed; native GPU tokenization still open)
- Goal:
  - Tokenization and detokenization run on GPU-backed buffers for high-throughput inference/training pipelines.
- Tasks:
  - [x] Add `grilly.tokenizers` module with `Tokenizer` interface (`encode`, `decode`, `batch_encode`). Implementation: `tokenizer_impl/` → `grilly.tokenizers`; default `Tokenizer` is `FastTokenizer` (Rust `tokenizers` library, not `transformers`).
  - [ ] Implement BPE/WordPiece fast path with GPU kernels for pretokenized merge/scoring stages.
  - [x] Keep exact CPU fallback for unsupported edge cases (identical outputs). Current fallback is the Rust `tokenizers` CPU path; optional `numpy_to_input_ids_buffers` in `tokenizer_impl/gpu.py` for staging.
  - [ ] Add `VulkanTensor`-friendly API: accept list[str] and return ids/attention masks in GPU-friendly layout (`wrap_ids_as_vulkan_tensors` exists behind `GRILLY_GPU_TOKENIZER=1` until wired).
- Acceptance:
  - Deterministic token IDs vs reference tokenizer on supported models.
  - >=2x throughput improvement vs CPU tokenizer on large batch benchmarks.

### P0.2 HF tokenizer interoperability
- Status: `[~]` (Rust `tokenizer.json` + Hub fallbacks + BERT/DistilBERT/T5/mT5 parity tests; raw `spiece.model`-only repos still open)
- Tasks:
  - [x] Load Hugging Face–compatible tokenizer assets: `tokenizer.json` from Hub (`huggingface_hub`), local dir, or file path (`tokenizer_impl/loader.py`). No `transformers` dependency in the Grilly tokenizer package.
  - [x] Hub repos without root `tokenizer.json` (e.g. `google/mt5-small`): fall back to `onnx/tokenizer.json` when present — same Rust `tokenizers` pipeline, parity vs HF reference.
  - [ ] Add loader for **only** SentencePiece assets (`spiece.model` / no JSON export) with encode/decode parity vs HF.
  - [~] Validation suite: `tests/tokenizers/test_gpu_tokenizer_parity.py` (BERT, DistilBERT); `tests/sentencepiece/test_sentencepiece_parity.py` (`t5-small`, `google/mt5-small`, special-token cases). Current run: all 6 tests passing in local env.
- Acceptance:
  - Asset compatibility documented and tested for top target checkpoints (including SentencePiece-backed models like T5/LLaMA-family tokenizers).

---

## P1: Sentence-Transformers support

### P1.1 Inference parity and API surface
- Status: `[ ]`
- Goal:
  - `SentenceTransformer`-style embedding API with drop-in ergonomics for common usage.
- Tasks:
  - [ ] Add `grilly.sentence_transformers` wrapper with `encode()` behavior-compatible options (`batch_size`, `normalize_embeddings`, device semantics).
  - [ ] Implement pooling strategies (`mean`, `cls`, `max`) and normalization parity.
  - [ ] Validate cosine similarity/semantic search outputs against reference pipelines.
- Acceptance:
  - Embedding outputs within tolerance across target ST models; API docs include known deltas.

### P1.2 GPU-first embedding pipeline
- Status: `[ ]`
- Tasks:
  - [ ] Route tokenizer -> encoder -> pooling through chain recorder where possible.
  - [ ] Add `rec.embedding_lookup(ids, table) -> handle` so embed -> first layer stays GPU-resident (no CPU round-trip).
  - [ ] Add `read_multiple` fan-out examples for MoE-style encoder blocks.
- Acceptance:
  - Single-submit batching demonstrated in benchmarked ST inference path.

---

## P2: Transformers compatibility (1:1 target)

### P2.1 Signature and config compatibility
- Status: `[ ]`
- Goal:
  - Match `transformers` module signatures and config behavior for core models.
- Tasks:
  - [ ] Add compatibility matrix by model family (BERT, RoBERTa, MiniLM, GPT2-class decoder).
  - [ ] Match key forward signatures (`input_ids`, `attention_mask`, `token_type_ids`, `position_ids`, `past_key_values` where applicable).
  - [ ] Align output objects (`last_hidden_state`, `pooler_output`, logits) and shape conventions.
- Acceptance:
  - Core families pass reference compatibility tests with documented exceptions.

### P2.2 Numerical and behavioral parity
- Status: `[ ]`
- Tasks:
  - [ ] Golden parity tests versus HF forward outputs (fp32 tolerances per op/family).
  - [ ] Attention behavior policy and masks parity (`causal`, `padding`, mixed masks).
  - [ ] Tokenizer-model handshake tests (special tokens, truncation/padding behavior).
- Acceptance:
  - "1:1" means no user-visible API breakage for covered families in documented scenarios.

---

## P3: PyTorch -> Grilly converter (after compatibility)

### P3.1 Converter core
- Status: `[ ]`
- Goal:
  - Convert PyTorch/HF checkpoints and model graphs into runnable Grilly modules.
- Tasks:
  - [ ] Add `grilly.convert.from_pytorch(...)` entrypoint.
  - [ ] Implement state_dict key mapping + tensor layout transforms.
  - [ ] Generate conversion report (mapped/unmapped params, warnings, unsupported ops).
- Acceptance:
  - Supported architectures convert and run inference with parity checks.

### P3.2 CLI + migration UX
- Status: `[ ]`
- Tasks:
  - [ ] Add CLI (`python -m grilly.convert ...`) with dry-run and validation modes.
  - [ ] Add migration cookbook examples from PyTorch/HF to Grilly runtime.
- Acceptance:
  - Users can convert, validate, and run model with a single documented workflow.

---

## Supporting Throughput Track (parallel, non-blocking)

### T1 Chain recorder dependency-aware barriers
- Status: `[ ]`
- Tasks:
  - [ ] Remove unconditional post-dispatch barriers where no RAW hazard exists.
  - [ ] Add `force_barrier=True` debug mode.
  - [ ] Add `rec.linear_backward(grad_out, input, weights) -> (grad_input_handle, grad_weight_handle)` for GPU backward matmul chaining.
  - [ ] Add MoE fan-out microbench (4/8 experts).

### T2 VulkanTensor residency hardening
- Status: `[ ]`
- Tasks:
  - [ ] Guard for older `grilly_core` binaries lacking `gpu_handle_if_valid`.
  - [ ] Add residency counters (fallback uploads/downloads) for profiling.

### T3 C2+/C4+ infrastructure
- Status: `[ ]`
- Tasks:
  - [ ] INT8 GEMM tiling and tuning (C2+) with MoQE-focused benchmarks (4-bit/8-bit expert paths).
  - [ ] Add dequant-in-shader path so quantized weights do not require FP32 shadow copies during inference.
  - [ ] Transfer queue + staging ring + overlap experiments (C4+).

### T4 Autograd GPU-resident graph execution
- Status: `[ ]`
- Tasks:
  - [ ] Make `Variable.backward()` detect chainable matmul -> relu -> matmul regions.
  - [ ] Route detected backward regions through chain recorder (batched GPU dispatches) instead of one bridge call per topo node.
  - [ ] Add fallback path to current traversal for unsupported ops while preserving gradient correctness.
- Acceptance:
  - Backward pass correctness parity vs current autograd; fewer fence waits in profiled training steps.

### T5 Fused CE + softmax chain op
- Status: `[ ]`
- Tasks:
  - [ ] Add `rec.cross_entropy(logits_handle, targets) -> (loss_handle, grad_logits_handle)` fused forward+backward on GPU.
  - [ ] Integrate with chain recorder API so LM training step can stay in a single submit region through loss/grad.
  - [ ] Add correctness tests vs reference CE+softmax backward and benchmark on `(seq, vocab)`-sized logits.
- Acceptance:
  - Fused CE op matches reference gradients within tolerance and reduces per-step synchronization overhead.

---

## Verification Commands

```bash
# Existing runtime/perf guardrails
uv run pytest tests/test_fnn_chain.py -v
uv run pytest tests/test_vulkan_tensor_residency.py -v
uv run pytest tests/test_conv_backward_weight_gemm.py -v
uv run pytest tests/parity/ -m parity -v
uv run python benchmarks/benchmark_conv_backward_weight.py
uv run python benchmarks/benchmark_int8_gemm.py

# Tokenizer parity (Rust CPU path vs HF reference in tests)
uv run pytest tests/tokenizers/ -v
uv run pytest tests/sentencepiece/ -v
# VSA-LM fused forward/backward
uv run pytest tests/test_vsa_lm_forward.py -v
uv run pytest tests/test_moe_forward.py -v

# Planned (new feature suites to add as implemented)
# uv run pytest tests/sentence_transformers/ -v
# uv run pytest tests/transformers_compat/ -v
# uv run pytest tests/converter/ -v
# uv run pytest tests/autograd_chain/ -v
# uv run pytest tests/moe_quant/ -v
```

---

## Exit Criteria for pre-v1.0 cut

- GPU tokenizer shipped and validated against reference tokenizer outputs. *(Current: Rust `tokenizers` CPU path + parity tests for BERT/DistilBERT/T5/mT5; GPU merge/BPE kernels still to ship.)*
- Sentence-transformers pipeline shipped with documented parity bounds.
- Transformers compatibility target achieved for declared core families (documented matrix).
- PyTorch -> Grilly converter ships with dry-run + validation report.
- No known correctness regressions in chain recorder/residency paths.
- Performance regressions detectable with benchmark baselines committed in docs.
