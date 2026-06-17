# Grilly Parity+ Tasklist (PyTorch Parity and Beyond)

This tasklist is an execution plan to bring Grilly to practical parity with PyTorch for common production workflows, then exceed PyTorch in cross-vendor GPU portability and selected performance/research areas.

Use with:
- `docs/PYTORCH_PARITY_STATUS.md` (feature status)
- `docs/GPU_OPTIMIZATION_REVIEW.md` (C++/shader/backend optimization findings)

---

## Objectives

1. Reach high-confidence parity for core deep learning workflows.
2. Eliminate major performance regressions at Python↔GPU boundaries.
3. Strengthen reliability, observability, and migration ergonomics.
4. Build “better than PyTorch” differentiators in Vulkan portability + specialized kernels.

---

## Status Legend

- `[ ]` Not started
- `[-]` In progress
- `[x]` Done
- `(!)` Blocked

Priority:
- `P0` Critical
- `P1` High
- `P2` Medium

---

## Workstream A: Core Training/Inference Parity (P0)

### A1. Standard model parity matrix validation
- Priority: `P0`
- Status: `[-]`
- Tasks:
  - [x] Scaffold `tests/parity/` with README, pytest marker `parity`, numpy + optional PyTorch references (`linear`, `relu`, small chain).
  - [x] Optimizer stepping vs `torch.optim` (SGD, Adam CPU): `tests/parity/test_optimizers_parity.py`.
  - [ ] Define full canonical parity suites: MLP, CNN, Transformer encoder, seq2seq-lite (extend beyond functional smoke tests).
  - [ ] Add numerical parity tests (forward + backward) against PyTorch references for modules.
  - [x] Document float32 tolerance defaults in `tests/parity/README.md` (iterate per-op policy later).
- Deliverable:
  - `tests/parity/` test suite with pass/fail dashboard.
- Acceptance criteria:
  - >=95% parity pass on core models (CPU and GPU paths).
- Progress note (2026-04-03):
  - Initial functional parity tests landed; CI can run them without `torch`. Install `torch` locally or in optional CI job for cross-checks.

### A2. Functional API parity hardening
- Priority: `P0`
- Status: `[-]`
- Tasks:
  - [ ] Audit `grilly.functional` signatures vs `torch.nn.functional` equivalents (remaining symbols).
  - [ ] Align argument names (`dim`, `axis`, defaults) where feasible.
  - [x] Document intentional differences and fallback behavior (functional parity section + migration doc).
- Deliverable:
  - Updated `docs/api/functional.md` parity annotations.
- Acceptance criteria:
  - All core functions (`linear`, activations, softmax, dropout, common losses) have clear parity notes and tested behavior.
- Progress note (2026-04-03):
  - Added PyTorch parity notes and links on `docs/api/functional.md`; full per-function audit still open.

### A3. Optimizer and scheduler parity verification
- Priority: `P0`
- Status: `[-]`
- Tasks:
  - [x] Snapshot tests: SGD (no momentum) and Adam (CPU) vs `torch.optim` — `tests/parity/test_optimizers_parity.py`.
  - [ ] Verify AdamW/SGD+momentum and full scheduler curves vs PyTorch.
  - [ ] Add `tests/parity/test_schedulers.py` when scheduler parity is scoped.
- Deliverable:
  - `tests/parity/test_optimizers_parity.py` (initial), `tests/parity/test_schedulers.py` (planned)
- Acceptance criteria:
  - Drift is within agreed tolerances across canonical optimizer scenarios.

---

## Workstream B: Python/Bindings Performance (P0)

### B1. Remove per-dispatch forced waits in Python core
- Priority: `P0`
- Status: `[-]`
- Tasks:
  - [x] Async dispatch mode in `backend/core.py` (`_dispatch_compute_async`, `_wait_async`; existing).
  - [x] Public aliases on `VulkanCompute`: `dispatch_compute`, `dispatch_compute_async`, `wait_async`, `wait_fence`, `record_commands`.
  - [x] Migration notes: `docs/PERF_DISPATCH.md`.
  - [ ] Broader adoption in Python hot paths + measured throughput gains (25%+ goal).
- Deliverable:
  - Non-blocking dispatch API + migration notes.
- Acceptance criteria:
  - >=25% end-to-end throughput gain on multi-op inference pipelines.
- Progress note (2026-04-03):
  - Documented APIs; default `_dispatch_compute` remains synchronous for correctness.

### B2. Expand command recording usage for multi-op chains
- Priority: `P0`
- Status: `[-]`
- Tasks:
  - [x] Linear→ReLU fallback: `VulkanFNN._linear_relu_recorded_chain` (one submit via `record_commands` when fused shader missing).
  - [ ] Migrate additional hot paths (`conv`, `linear backward` chains) to command chaining.
  - [ ] Reduce begin/submit/wait frequency to one submit per chain where possible (ongoing).
- Deliverable:
  - Refactored backend modules with grouped dispatches.
- Acceptance criteria:
  - Queue idle ratio reduced substantially in profiler traces.
- Progress note (2026-04-03):
  - FlashAttention2 and RMSNorm already use `record_commands`; Sequential+fused shaders unchanged.

### B3. Standardize pybind GIL release on heavy kernels
- Priority: `P0`
- Status: `[x]`
- Tasks:
  - [x] Audit remaining `cpp/python/bindings_*.cpp` entrypoints: conv, normalization, loss, SNN, pooling, optim, **misc** (dropout, embedding, KV swizzle), **fusion** (`ShaderFusionEngine::fuse`), **OpGraph** (`optimize` / `execute`).
  - [x] `py::gil_scoped_release` on activations (`register_activations_ops`), attention (`register_attention_ops`), fused MLP/LayerNorm+Linear.
  - [x] Checklist in `docs/PERF_DISPATCH.md` (bindings / GIL section).
- Deliverable:
  - GIL policy compliance across binding modules.
- Acceptance criteria:
  - No long-running compute wrapper holds GIL unnecessarily.

### B4. Enforce contiguous-array contracts at binding boundaries
- Priority: `P1`
- Status: `[x]`
- Tasks:
  - [x] `require_c_contiguous_float` in `bindings_core.h`; applied to `linear` (`x`, `weights`, `bias`).
  - [x] Extended: activations, conv, norm, loss, SNN, pooling, optim, **misc** (dropout, embedding ids+table, KV append/decode, eviction/train helpers, swizzle), **Hamming** (`require_c_contiguous_int8`), **SigLIP** / **Perceiver** / **MoQE** uploads and forwards.
- Deliverable:
  - Safer, deterministic binding inputs.
- Acceptance criteria:
  - No stride-related correctness bugs in parity suite.

---

## Workstream C: Kernel and Backend Throughput (P0/P1)

**Workstream status: `[x]` closed (2026-04-03)** for the scoped deliverables below. Follow-ups that require new Vulkan infrastructure (atomic 1x1 wiring, INT8 shared-memory tiling, dedicated transfer queue) are tracked under **Workstream C — future**.

### C1. Conv backward weight atomic bottleneck rewrite
- Priority: `P0`
- Status: `[x]`
- Tasks:
  - [x] GPU-side GEMM for `grad_weight` without host im2col round-trip when `gemm_mnk` + `tensor-transpose` are present (`backend/conv.py`; test `tests/test_conv_backward_weight_gemm.py`).
  - [x] Baseline benchmark: `benchmarks/benchmark_conv_backward_weight.py`.
  - [x] Non-GEMM path: `conv2d-backward-weight.glsl` (per-weight-slot accumulation, no global atomics).
- Deferred (see **C — future**): two-phase reduction or C++ dispatch wiring for experimental `conv1x1-backward-weight.glsl`.
- Deliverable:
  - Training backward conv path without host im2col matmul when GEMM shaders are available; documented fallback.
- Acceptance criteria:
  - GPU GEMM path validated vs PyTorch reference; benchmark entrypoint for regression tracking.
- Progress note (2026-04-03):
  - GEMM path: GPU im2col → transpose → `gemm_mnk`; only `grad_weight` read back. Without those shaders, NumPy matmul after download remains.

### C2. INT8 GEMM tiling/vectorization
- Priority: `P0`
- Status: `[x]` (P0 vectorization + benchmark; tiled SM tuning deferred)
- Tasks:
  - [x] Inner K-loop vectorization: 4-wide packed weight loads in `int8-gemm.glsl` (recompile `shaders/spv/int8-gemm.spv`).
  - [x] Baseline benchmark: `benchmarks/benchmark_int8_gemm.py`.
- Deferred (see **C — future**): tiled shared-memory strategy + per-device tuned workgroups.
- Deliverable:
  - Packed-load INT8 GEMM path and repeatable benchmark.
- Acceptance criteria:
  - Shader + SPIR-V in tree; benchmark runs when `int8-gemm` loads; further >1.5x speedups expected from tiling work, not gated on this milestone.

### C3. Attention phase restructuring (correctness + performance)
- Priority: `P0`
- Status: `[x]`
- Tasks:
  - [x] Flash-style online softmax path: `flash_attention2` + batched dispatches (`record_commands`).
  - [x] Bridge path: scores → softmax → output and fused `attention_scores_softmax_output` where available (`docs/GPU_OPTIMIZATION_REVIEW.md` optimization steps 2–4).
  - [x] Long-sequence GPU smoke tests: `tests/test_attention_long_sequence.py` (S = 128/256/512; `@slow` S = 1024).
- Progress note (2026-02-10 / 2026-04-03):
  - Shader micro-opts on `flash-attention2.glsl` / `attention-output.glsl`; `tests/test_attention.py` validation.
  - `benchmarks/profile_gpu_bottlenecks.py` for CPU-side dispatch/transfer hotspots.
  - FA2 vs decomposed-path numeric parity is **not** claimed; long-seq tests assert shape + finiteness only.
- Deliverable:
  - Production attention stack with documented batching and fused paths.
- Acceptance criteria:
  - No ad-hoc multi-phase barrier hacks in the primary FA2 path; smoke coverage at medium/long S.

### C4. Persistent transfer context for staged copies
- Priority: `P1`
- Status: `[x]` (documented baseline; advanced transfer path deferred)
- Tasks:
  - [x] Document current reuse: single `_cmd_buffer` / `_fence` per `VulkanCore`; `record_commands` batch path (`docs/GPU_OPTIMIZATION_REVIEW.md` Workstream C).
- Deferred (see **C — future**): dedicated transfer queue + VMA staging ring; async overlap with compute.
- Deliverable:
  - Clear documentation of current submit/wait and batching behavior for perf work.
- Acceptance criteria:
  - Engineers can reason about transfer vs compute from `GPU_OPTIMIZATION_REVIEW.md` + core code paths.

### Workstream C — future (not blocking C closure)
- **C1+**: Wire `shaders/conv1x1-backward-weight.glsl` in `cpp` dispatch for 1×1 kernels (experimental atomic path).
- **C2+**: INT8 shared-memory tiling + tuned workgroups; optional perf CI threshold vs baseline.
- **C4+**: Dedicated transfer queue + VMA staging ring + optional async overlap (see Workstream D overlap with `VulkanTensor` residency).

---

## Workstream D: Reliability and Developer Experience (P1)

### D1. Bridge strict mode and fallback telemetry
- Priority: `P1`
- Status: `[x]`
- Tasks:
  - [x] Add `GRILLY_BRIDGE_STRICT=1` option to raise on GPU path failures.
  - [x] Add fallback counters for silent CPU fallback events.
- Progress note (2026-02-10):
  - Added strict mode + fallback telemetry API in `backend/_bridge.py` (`get_fallback_stats()`, `reset_fallback_stats()`), wired across bridge GPU op wrappers.
  - Added regression tests in `tests/test_bridge_strict_mode.py` to verify strict-mode raising and fallback counter increment behavior.
- Deliverable:
  - Better observability for production/perf regressions.
- Acceptance criteria:
  - Clear logs/metrics reveal when and why fallback occurs.

### D2. Functional backend singleton for fallback mode
- Priority: `P1`
- Status: `[-]`
- Tasks:
  - [x] Avoid repeated `Compute()` initialization in functional fallback paths.
  - [x] Add lifecycle + cleanup guidance (`docs/MIGRATION_PYTORCH.md` — module backend via `DeviceManager`, reused Vulkan backend).
- Progress note (2026-02-10):
  - Legacy `Compute()` fallback path was removed from `functional/*.py`; functional runtime now prefers `_bridge` with explicit CPU fallback logic.
- Progress note (2026-04-03):
  - Documented lazy `Module._get_backend()` / `get_device_manager().vulkan` singleton pattern for long-running processes.
- Deliverable:
  - Reduced init overhead and more stable fallback behavior.
- Acceptance criteria:
  - Measurable latency reduction in fallback-only environments.

### D4. GPU dispatch batching and async execution
- Priority: `P1`
- Status: `[-]`
- Tasks:
  - [x] Add `_dispatch_compute_async` and `_wait_async` primitives to `backend/core.py`.
  - [x] Add `wait_previous` flag to skip redundant fence waits.
  - [x] `CommandRecorder` integration for Linear→ReLU when fused shader missing (`_linear_relu_recorded_chain`); `VulkanCompute` exposes `record_commands`.
  - [x] `VulkanTensor` residency / dispatch prep: `gpu_handle_if_valid` on C++ `Tensor`, `_try_bind_cpp_gpu_buffer`, `prepare_for_dispatch()`; `BufferMixin._prepare_input` uses them to reuse pooled buffers or C++ VkBuffer without redundant upload when data is already GPU-resident.
  - [x] Fusion patterns for Linear+activation in `nn.Sequential` (existing fused shaders + recorded fallback).
- Progress note (2026-02-10):
  - Profiler identified fence waits as dominant bottleneck (40-100 waits per 10-20 iters).
  - Async primitives added; measurable gains require batching multiple kernels per submit.
- Progress note (2026-04-03):
  - GPU-resident `VulkanTensor` inputs to pooled-buffer ops now call `prepare_for_dispatch()` before dispatch so C++-backed GPU tensors bind without an extra host upload when the handle is valid.
- Deliverable:
  - Reduced per-op CPU overhead for chained GPU operations.
- Acceptance criteria:
  - 2x+ speedup on chained small ops (e.g., MLP block) vs current serial dispatch.

### D3. Documentation parity map and migration cookbook
- Priority: `P1`
- Status: `[-]`
- Tasks:
  - [x] Add “PyTorch -> Grilly migration cookbook” with common patterns (`docs/MIGRATION_PYTORCH.md`).
  - [ ] Expand known differences and workarounds by feature (ongoing; link from parity status).
- Deliverable:
  - User-facing migration docs for teams porting codebases.
- Acceptance criteria:
  - New users can port baseline projects without deep source-diving.

---

## Workstream E: Deprecated API Removal and `_bridge` Migration (P0/P1)

### E1. Remove legacy `Compute()` fallback usage from functional API
- Priority: `P0`
- Status: `[x]`
- Tasks:
  - [x] Replace `Compute()` fallback paths in:
    - `functional/activations.py`
    - `functional/linear.py`
    - `functional/normalization.py`
    - `functional/attention.py`
  - [x] Route all GPU-first paths through `backend/_bridge.py`.
  - [x] Keep explicit CPU fallback behavior (numpy/numba) where needed, but remove legacy Vulkan Python path dependency.
- Progress note (2026-02-10):
  - Completed full `functional/*.py` migration off `Compute()` including `dropout`, `loss`, `embedding`, `faiss`, `memory`, `learning`, `cells`, `bridge`, and `fft`.
  - Added `_bridge` support helpers for previously unbridged functional module families and kept explicit CPU fallback paths.
  - Added targeted JIT/snippet optimizations: true LRU graph cache in `backend/jit.py` and vectorized CPU fallback hotspots.
- Deliverable:
  - Functional modules no longer import `Compute()` for normal execution paths.
- Acceptance criteria:
  - No `from grilly import Compute` in `functional/*.py` except explicitly documented temporary shims.

### E2. Migrate module-level backend initialization off `Compute()`
- Priority: `P0`
- Status: `[-]`
- Tasks:
  - [x] Refactor `nn/module.py` backend initialization to use bridge-native device/context instead of `Compute()`.
  - [x] Refactor `utils/tensor_conversion.py` fallback callsites that instantiate `Compute()`.
  - [ ] Add compatibility wrappers where needed to avoid breaking existing user code.
- Progress note (2026-02-10):
  - Removed direct `Compute()` usage from both `nn/module.py` and `utils/tensor_conversion.py`.
  - Preserved container compatibility by keeping `_get_backend()` return shape aligned with existing `backend.fnn.*` callers.
- Deliverable:
  - Core runtime no longer depends on deprecated `Compute()` for primary path.
- Acceptance criteria:
  - Runtime path for `nn` + `functional` uses `_bridge`/`grilly_core` first and does not instantiate deprecated APIs in steady-state.

### E3. Deprecation enforcement and removal policy
- Priority: `P1`
- Status: `[-]`
- Tasks:
  - [ ] Define deprecation timeline (warn -> hard warn -> removal) for legacy APIs.
  - [x] Add CI check that fails on new references to deprecated APIs (`Compute()`, legacy bindings source).
  - [x] Update docs and changelog with migration examples (`docs/MIGRATION_PYTORCH.md`, `CHANGELOG.md` [Unreleased]).
- Progress note (2026-02-10):
  - Added CI guard in `.github/workflows/ci.yml` to fail on deprecated `Compute()` usage in runtime migration targets (`functional`, `nn/module.py`, `utils/tensor_conversion.py`).
- Deliverable:
  - Enforced migration policy with automated guardrails.
- Acceptance criteria:
  - No new deprecated API usage introduced after policy activation.

### E4. Legacy binding and backend path quarantine
- Priority: `P1`
- Status: `[ ]`
- Tasks:
  - [ ] Quarantine or remove `cpp/python/bindings.cpp` (legacy monolith, non-compiled).
  - [ ] Remove stale docs that recommend deprecated entrypoints where replacement exists.
  - [ ] Tag remaining legacy-only codepaths with explicit removal issue IDs.
- Deliverable:
  - Lower maintenance burden and reduced confusion in code/docs.
- Acceptance criteria:
  - Single clear binding architecture and single recommended runtime path.

---

## Workstream F: Ecosystem and Integration Parity (P1/P2)

### E1. ONNX operator coverage expansion
- Priority: `P1`
- Status: `[ ]`
- Tasks:
  - [ ] Identify top missing operators from common transformer/CNN exports.
  - [ ] Prioritize by model prevalence and implementation complexity.
- Deliverable:
  - ONNX compatibility matrix + incremental op support.
- Acceptance criteria:
  - Increased pass rate on selected ONNX model import tests.

### E2. HuggingFace model compatibility tiers
- Priority: `P1`
- Status: `[ ]`
- Tasks:
  - [ ] Define compatibility tiers (full/partial/experimental).
  - [ ] Publish tested model families and known limitations.
- Deliverable:
  - Compatibility table with reproducible test scripts.
- Acceptance criteria:
  - Stable documented support for at least target model families.

### E3. Distributed training roadmap
- Priority: `P2`
- Status: `[ ]`
- Tasks:
  - [ ] Decide architecture (single-node first, gradient sync strategy).
  - [ ] Define minimum viable distributed API surface.
- Deliverable:
  - Technical design doc + milestone breakdown.
- Acceptance criteria:
  - Approved design and prototype milestone committed.

---

## Workstream G: “Better than PyTorch” Targets (P1/P2)

### F1. Cross-vendor performance profiles
- Priority: `P1`
- Status: `[ ]`
- Tasks:
  - [ ] Add vendor-specific tuning profiles (AMD/NVIDIA/Intel).
  - [ ] Auto-select workgroup/tile variants at runtime from device caps.
- Deliverable:
  - Profile-driven kernel selection.
- Acceptance criteria:
  - Consistent gains across at least 2 vendor families.

### F2. GPU-first SNN and VSA benchmark leadership
- Priority: `P2`
- Status: `[ ]`
- Tasks:
  - [ ] Define benchmark suite where Grilly outperforms common PyTorch baselines.
  - [ ] Publish reproducible scripts and numbers.
- Deliverable:
  - Public benchmark report for SNN/VSA workloads.
- Acceptance criteria:
  - Demonstrated advantage in selected workloads.

### F3. Model graph fusion cost model
- Priority: `P2`
- Status: `[ ]`
- Tasks:
  - [ ] Expand pairwise fusion to pattern-based multi-op fusion.
  - [ ] Add cost model to prevent regressions due to over-fusion.
- Deliverable:
  - Fusion planner with performance-aware decisions.
- Acceptance criteria:
  - Net positive benchmark impact across representative models.

---

## Cross-Cutting Testing and Metrics

### Required Benchmark Set
- [ ] MLP inference/training
- [ ] CNN inference/training
- [ ] Transformer encoder inference/training
- [ ] Quantized INT8 inference
- [-] Attention long-sequence stress

### Required Metrics
- [-] End-to-end throughput (samples/s or tokens/s)
- [ ] P50/P95 latency
- [-] GPU kernel time by op
- [ ] Queue idle ratio
- [ ] Host↔device transfer bandwidth
- [ ] CPU overhead per dispatch
- [ ] Fallback rate (GPU->CPU)

Progress note (2026-02-10):
- `GPU kernel time by op` now has a repeatable profiler entrypoint (`python benchmarks/profile_gpu_bottlenecks.py`) that surfaces dispatch/fence/transfer hotspots for `Linear` and `MultiheadAttention`; Vulkan timestamp-query granularity is still pending.
- Added async dispatch primitives (`_dispatch_compute_async`, `_wait_async`, `wait_previous` flag) to `backend/core.py` to enable future batching optimizations.

### Parity Quality Gates
- [-] Numerical parity thresholds documented for functional smoke tests (`tests/parity/README.md`); per-op class policy TBD
- [ ] Regression thresholds defined for perf CI
- [ ] CI matrix includes CPU fallback + Vulkan-enabled runs

---

## Suggested Milestones

### Milestone 1 (Parity Foundation) — 4 to 6 weeks
- A1, A2, B1, B3, D1, E1

### Milestone 2 (Performance Stabilization) — 4 to 8 weeks
- B2, **C (closed)**, D2, E2 — Workstream C kernel/throughput items for this milestone are complete per scoped checklist; see **Workstream C — future** for follow-ups.

### Milestone 3 (Parity Expansion + Beyond) — 6 to 10 weeks
- **C — future** (INT8 tiling, transfer queue/VMA, optional conv1x1 atomic), E3, E4, F1, F3

---

## Ownership Template (fill in)

For each item, assign:
- **Owner**
- **ETA**
- **Dependencies**
- **Risk level**
- **Validation benchmark**

Example:
- `B1` Owner: ___ / ETA: ___ / Depends on: ___ / Risk: ___ / Benchmark: ___

---

## Notes

- Keep this tasklist synchronized with `docs/PYTORCH_PARITY_STATUS.md`.
- When an item completes, update both:
  1. task status here
  2. parity status in the parity matrix document.

