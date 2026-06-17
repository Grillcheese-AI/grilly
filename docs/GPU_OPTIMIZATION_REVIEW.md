# GPU Optimization Code Review (C++ + GLSL)

This review focuses on the native Vulkan backend (`cpp/`) and compute shaders (`shaders/`) with emphasis on end-to-end throughput, kernel occupancy, synchronization overhead, and memory traffic.

## Executive Priority List

1. Remove per-op CPU/GPU synchronization and host round-trips in C++ op wrappers.
2. Replace global atomic accumulation patterns in gradient shaders with hierarchical reductions.
3. Introduce specialization/pipeline variants for workgroup sizes and common dimensions.
4. Improve attention kernels to avoid global barriers and reduce O(seq^2) scratch traffic.
5. Rework INT8 GEMM and fused MLP kernels to be tiled/subgroup-based instead of scalar loops.

---

## Critical Findings (Highest Impact)

### 1) Frequent submit+wait+download in op wrappers prevents overlap

**Where**
- `cpp/src/ops/linear.cpp`
- `cpp/src/ops/conv.cpp`
- `cpp/src/ops/attention.cpp`
- `cpp/src/ops/attention_ops.cpp`
- `cpp/src/ops/fused.cpp`

**What**
- Most wrappers do:
  - `batch.begin() -> dispatch -> submitDeferred() -> waitForCompletion()`
  - followed by immediate `pool.download(...)`.
- This serializes CPU and GPU, destroys queue depth, and blocks inter-op overlap.

**Why it hurts**
- Every op becomes a hard synchronization point.
- No opportunity for batching across layers or fusing chains at runtime.
- Even if kernel time is small, fixed fence/wait latency dominates.

**Recommendation**
- Introduce a graph-first execution path for native ops similar to `VulkanBackend::graphMode`.
- Keep intermediate tensors device-resident across op chains; download only at explicit boundaries.
- Add async API variant (`enqueue` + `finalize`) so callers can schedule multiple kernels before one wait.

**Expected gain**
- High (often 1.5x-3x in inference/training step throughput for small/medium kernels).

---

### 2) Staged upload/download recreates command pool + fence per transfer

**Where**
- `cpp/src/buffer_pool.cpp` (`uploadStaged`, `downloadStaged`)

**What**
- Each transfer allocates/destroys:
  - command pool
  - command buffer
  - fence
  - staging buffer
- Then waits immediately on fence.

**Why it hurts**
- Heavy driver overhead and allocator churn.
- Converts DMA-capable transfers into synchronous micro-transactions.

**Recommendation**
- Create persistent transfer context:
  - one transfer command pool
  - small ring of command buffers + timeline semaphore/fences
  - reusable staging buffer arena (or chunked suballocations).
- Batch copies and signal once per batch.
- Prefer async copy + compute overlap when queue family supports transfer/async compute.

**Expected gain**
- High for data-heavy workloads and frequent host interaction.

---

### 3) Atomic float accumulation bottleneck in conv backward weight

**Where**
- `shaders/conv1x1-backward-weight.glsl`

**What**
- Inner loops perform `atomicAdd(grad_weights[w_idx], dy * x)` from many invocations.

**Why it hurts**
- Heavy contention on shared output addresses.
- Poor scaling with channels/spatial size; atomics dominate runtime.

**Recommendation**
- Two-phase reduction:
  1. Per-workgroup partial gradients in shared memory (or partitioned global workspace).
  2. Second kernel reduce partials into final gradient.
- Optionally tile by `(c_out, c_in)` blocks to maximize locality and reduce collision domain.

**Expected gain**
- High in training backward pass (can be multiple-x depending on contention).

---

### 4) Attention kernel structure has global-memory synchronization anti-patterns

**Where**
- `shaders/gqa-attention.glsl`

**What**
- Single shader has three phases with `barrier()` + `memoryBarrierBuffer()` between global-score writes and reads.
- Softmax and weighted sum operate through global `scores` buffer.

**Why it hurts**
- Workgroup barriers do not synchronize across all workgroups, yet algorithm behavior implies global phase ordering.
- Large global scratch traffic (`scores`) is expensive and cache-unfriendly.

**Recommendation**
- Split into explicit kernels:
  1. score compute
  2. row-wise softmax
  3. weighted value accumulation
  with command-buffer barriers between dispatches.
- Or adopt online softmax tiled formulation (flash-style) to avoid materialized full scores.
- Use subgroup reductions for row max/sum in-kernel.

**Expected gain**
- High correctness/performance improvement and better scalability to long cache lengths.

---

### 5) Fused transformer shaders are memory-efficient but compute-scalar in inner loops

**Where**
- `shaders/fused-mlp-gelu.glsl`
- `shaders/fused-layernorm-linear.glsl`

**What**
- Large inner loops are scalar (`for` over full hidden/input dimensions).
- Workgroup-size fixed at 256, static shared arrays sized for specific dims.

**Why it hurts**
- Underutilizes subgroup/tensor-like instructions.
- Register pressure + long scalar loops limit occupancy and throughput.

**Recommendation**
- Move to tiled MMA-like pattern:
  - cooperative loading of weight/input tiles into shared memory
  - per-thread micro-tiles
  - subgroup-level partial reductions.
- Generate specialized variants per common dims (e.g., 768/3072/1024/4096).
- Consider cooperative matrix extension path where available.

**Expected gain**
- Medium to high on transformer-heavy inference.

---

## High-Value C++ Backend Opportunities

### A) Descriptor and pipeline caching can be improved for dynamic buffer reuse

**Where**
- `cpp/src/pipeline_cache.cpp`

**Observation**
- Descriptor cache key includes concrete buffer handles and ranges.
- This misses reuse opportunities when same layout is used with new transient buffers.

**Recommendation**
- Add bindless-like strategy where feasible, or cache by descriptor layout + slot pattern and update dynamic descriptors per dispatch.
- Explore descriptor indexing/update templates (if target Vulkan profile permits).

**Impact**
- Medium; reduces CPU overhead in dispatch-heavy runs.

---

### B) Queue family selection currently picks first compute queue

**Where**
- `cpp/src/device.cpp`

**Observation**
- Selection does not prioritize dedicated compute queues or async transfer-capable pairings.

**Recommendation**
- Score queue families:
  - dedicated compute preferred
  - transfer queue pairing preferred
  - queue count > 1 preferred.
- Keep compatibility fallback to current behavior.

**Impact**
- Medium; helps overlap transfer/compute on capable GPUs.

---

### C) OpGraph fusion currently handles pairwise only

**Where**
- `cpp/src/op_graph.cpp`

**Observation**
- Current fusion is pairwise and rule-based (`A+B -> Fused`).

**Recommendation**
- Extend to multi-op patterns:
  - `linear -> bias -> activation`
  - `norm -> linear -> activation`
  - `qk -> mask -> softmax -> av`
- Include cost model (bytes saved, extra registers, expected occupancy).

**Impact**
- Medium to high depending on model structure.

---

## Shader-Specific Opportunities

### 1) `int8-gemm.glsl` is scalar and likely memory-bound

**Where**
- `shaders/int8-gemm.glsl`

**Recommendation**
- Tile over M/N/K.
- Use packed vector loads and shared-memory staging.
- Unpack int8 in vectorized chunks; accumulate with subgroup reduction/micro-tiles.
- Consider dedicated dp4a-like path where supported.

**Impact**
- High for quantized inference paths.

---

### 2) `hmm-forward.glsl` has unused shared helper and scalar reductions

**Where**
- `shaders/hmm-forward.glsl`

**Recommendation**
- Replace per-thread full loops with subgroup-assisted logsumexp reduction.
- If state count is large, block/tile state-space and reduce in stages.

**Impact**
- Medium.

---

### 3) `adaptive-avgpool-3x3.glsl` uses tiny fixed local size (3x3x16)

**Where**
- `shaders/adaptive-avgpool-3x3.glsl`

**Recommendation**
- Consider 1D/2D larger workgroups for better warp/wave utilization.
- Benchmark variant set (e.g., 64/128/256-thread forms) and choose by device profile.

**Impact**
- Low to medium (kernel likely not dominant but easy win).

---

### 4) Flash-attention-style kernel can reduce barriers and specialize tile config

**Where**
- `shaders/flash-attention3.glsl`

**Recommendation**
- Device-tuned tile selection (`tile_k`, subgroup width, head-dim specialization).
- Consider separating paths by head_dim bucket (64/80/96/128).
- Add occupancy guardrails for shared-memory usage and register pressure.

**Impact**
- Medium to high for long-sequence attention.

---

## Correctness + Performance Risk Items

1. **Cross-workgroup phase assumptions in single-dispatch multi-phase kernels** (notably `gqa-attention`) should be validated; global ordering requires dispatch boundaries, not only local barriers.
2. **Atomic-heavy gradient accumulation** may be numerically non-deterministic and slower on some vendors.
3. **Fixed-dimension shared arrays** in fused kernels can silently underperform or limit portability for non-default dimensions.

---

## Recommended Implementation Order

### Phase 1 (Immediate, high ROI)
1. Remove forced waits/downloads from hot op wrappers; expose async/batched path.
2. Rework staging copy pipeline (persistent transfer context).
3. Replace `vulkan-utils` CI package already done; keep Vulkan tooling healthy for profiling.

### Phase 2 (Kernel hotspots)
1. Conv backward weight: two-phase reduction (remove global atomics in inner loop).
2. INT8 GEMM: tiled/vectorized kernel.
3. GQA attention: split phases or switch to online softmax tiled implementation.

### Phase 3 (Advanced tuning)
1. Multi-op fusion and runtime cost model in `OpGraph`.
2. Device-specific kernel variant selection (workgroup/tile specialization).
3. Optional cooperative matrix path rollout with robust fallback.

---

## Measurement Plan (Required to Validate Gains)

Track these before/after metrics:

- **Kernel time** (per op) via timestamp queries.
- **CPU submit overhead** (dispatch-to-submit latency).
- **Queue idle ratio** (time GPU idle between kernels).
- **Transfer bandwidth** (host↔device effective GB/s).
- **End-to-end tokens/s or samples/s** on representative models.

Recommended benchmark matrix:
- Short/medium/long sequence lengths.
- Small and large batch sizes.
- At least one AMD and one NVIDIA GPU profile.

## Benchmark Snapshot (2026-02-10, AMD RX 6750 XT)

These are current-branch measurements after the shader/JIT edits in this cycle (`fnn-linear`, `attention-output`, `flash-attention2`, JIT LRU/kwargs fixes). They are useful as a reproducible snapshot, but they are not a strict before/after A-B because pre-change runs were not captured in this same report.

### Linear (`benchmarks/bench_linear.py`)

- Standard mode (numpy input path):
  - `Small (32,128)->64`: `0.524 ms` (GPU), `0.016 ms` (CPU)
  - `Medium (64,512)->512`: `5.64 ms` (GPU), `0.254 ms` (CPU)
  - `Large (256,1024)->2048`: `88.43 ms` (GPU), `6.37 ms` (CPU)
  - `XL (512,2048)->4096`: `365.29 ms` (GPU), `25.33 ms` (CPU)
- GPU-resident mode:
  - Similar timings (`0.497 ms`, `5.88 ms`, `88.61 ms`, `365.55 ms`)
- Peak observed GPU throughput in this run: `23.5 GFLOP/s` (XL case).

### Attention scores (`benchmarks/bench_attention.py`)

- `B=1 S=32 H=4 D=32`: `0.396 ms` (GPU)
- `B=2 S=64 H=8 D=64`: `1.07 ms` (GPU)
- `B=4 S=128 H=8 D=64`: `6.65 ms` (GPU)
- `B=2 S=256 H=12 D=64`: `17.41 ms` (GPU)
- `B=1 S=512 H=12 D=64`: `35.72 ms` (GPU)

### Attention output + FlashAttention2 (bridge microbench)

- `attention_output`
  - `B=1 H=8 S=128 D=64`: `10.561 ms`
  - `B=2 H=8 S=128 D=64`: `20.673 ms`
  - `B=1 H=12 S=256 D=64`: `31.491 ms`
- `flash_attention2`
  - `B=1 H=8 S=128 D=64`: `11.338 ms`
  - `B=2 H=8 S=128 D=64`: `21.419 ms`
  - `B=1 H=12 S=256 D=64`: `34.059 ms`

### Immediate interpretation

- The optimized shaders compile cleanly and pass attention/functional tests.
- Current end-to-end GPU timings are still slower than CPU baselines for many tested shapes in this environment.
- This reinforces that the next largest wins are still orchestration-level:
  - reduce synchronization frequency,
  - keep tensors GPU-resident across op chains,
  - reduce upload/download boundaries.

### Python/Vulkan profile snapshot (2026-02-10)

Ran:

- `python benchmarks/profile_gpu_bottlenecks.py`

Observed hotspots:

- `backend/core.py:_dispatch_compute` + `vkWaitForFences` dominate cumulative time in parity-critical paths.
  - Linear profile (20 iters): `vkWaitForFences` appears 40 times and consumes a large fraction of runtime.
  - Attention profile (10 iters): `vkWaitForFences` appears 100 times across chained kernels.
- Buffer lifecycle and transfer overhead are still significant:
  - `backend/base.py:_acquire_buffer` / `backend/buffer_pool.py:acquire`
  - `backend/core.py:_create_buffer`, `vkAllocateMemory`
  - `backend/core.py:_upload_buffer`, `backend/core.py:_download_buffer`
- Python-side Vulkan wrapper overhead is visible (`vulkan._vulkan._callApi`), indicating many small API calls per op.
- Attention forward still spends substantial time in CPU NumPy (`numpy.einsum`) for parts of the path, confirming partial CPU residency in module-level execution.

Current timing comparison (RX 6750 XT):

- **Linear**: 20 iterations = ~41ms total, CPU baseline = ~5.4ms. The `choose_fastest` policy routes small shapes to CPU (correct decision).
- **Attention**: 10 iterations = ~155ms, with significant time in `numpy.einsum` (49ms) showing CPU fallback paths.

### Optimization step 1: Add async dispatch primitives (2026-02-10)

Implemented:

1. Added `_dispatch_compute_async()` to `backend/core.py` — submits work without waiting, returns fence handle.
2. Added `_wait_async()` for explicit synchronization.
3. Added `wait_previous: bool` parameter to `_dispatch_compute()` to optionally skip the "before" fence wait (useful when queue is known-idle).

Next steps to realize gains:

- Use `CommandRecorder` context manager to batch multiple kernels into single submit (e.g., Linear → ReLU → Linear chains).
- Keep intermediate tensors GPU-resident via `VulkanTensor` to eliminate upload/download between ops.
- Add fusion patterns for common sequences (Linear+Activation, Attention QKV projection).

Implication:

- For currently tested shapes, bottlenecks are primarily **dispatch synchronization and data movement overhead**, not just shader ALU efficiency.
- The fence-wait reduction changes are in place but need to be paired with **batching** to show measurable gains — single-op latency is dominated by fixed submission overhead.

### Optimization step 2: FlashAttention2 single submit (2026-02-10)

Implemented in `backend/attention.py` (`VulkanAttention.flash_attention2`):

- Replaced **one `_dispatch_compute` per pass** (init + each Q/K tile + finalize) with a single `with core.record_commands() as rec:` block.
- Inserts `rec.barrier()` between dependent dispatches (same semantics as sequential submits, fewer host round-trips).
- Reduces Python/Vulkan overhead from **O(tiles) fence waits** to **one** submit + wait per forward call.

`tests/test_attention.py` passes after this change.

### Optimization step 3: MultiheadAttention uses C++ bridge pipeline (2026-02-10)

Implemented in `nn/attention.py` (`MultiheadAttention.forward`):

- When `seq_len_q == seq_len_k` and the C++ bridge is available, the path is:
  `_bridge.attention_scores` → optional **CPU** padding mask (same rules as `FlashAttention2` CPU fallback) → `_bridge.softmax` → `_bridge.attention_output`.
- Avoids the previous mix of Python Vulkan `attention_scores` plus **CPU** softmax plus **`np.einsum`** for the weighted value projection (major hotspot in profiling).
- Cross-attention with `seq_len_q != seq_len_k` still uses the legacy Vulkan + NumPy fallback (kernel contract is square `S×S` scores).

The old `backend.attention.attention_mask(scores, mask)` call was incorrect (second positional is `use_causal`, not the mask tensor). Padding masks are now applied with explicit `np.where` / additive mask, matching `FlashAttention2` forward.

### Optimization step 4: Fused attention scores + softmax + output (2026-04-03)

- **C++**: `grilly::ops::attentionScoresSoftmaxOutput` records attention-scores → softmax (multi-pass) → attention-output in **one** `CommandBatch` / submit (see `cpp/src/ops/attention_ops.cpp`).
- **Python**: `_bridge.attention_scores_softmax_output` wraps the binding; `MultiheadAttention.forward` uses it when **`mask is None`** and **`seq_len_q == seq_len_k`** (unmasked self-attention with equal Q/K/V sequence length). Any mask or unequal lengths falls back to the step-3 bridge or legacy path.

**Cross-attention (`Sq ≠ Sk`) and masks**

- The fused and standard C++ score kernels allocate an **`S×S`** attention matrix with a single `S` from Q/K layout; they assume **self-attention** (same sequence index for queries and keys).
- For **encoder–decoder** or other cross-attention, use the **legacy** `MultiheadAttention` path (Python/Vulkan + NumPy) or implement a dedicated kernel that supports rectangular `(Sq, Sk)` score tensors.
- **Padding masks**: when a mask is provided, the module uses the non-fused bridge (CPU application of mask to scores, then GPU softmax + output). Masks must broadcast to **`(batch, num_heads, seq_len_q, seq_len_k)`** — e.g. `(batch, seq_q, seq_k)` expanded with `[:, None, :, :]` or an additive mask of that rank.

---

## Summary

The codebase already has strong foundations (buffer pooling, descriptor LRU, op-graph fusion hooks, several fused shaders). The biggest remaining bottlenecks are **synchronization granularity** and **scalar/atomic-heavy kernel sections**. Prioritizing async batching + atomic reduction redesign + attention/INT8 kernel rework should produce the largest practical speedups.

---

## Bindings + Python Layer Review

This section analyzes the pybind bindings (`cpp/python/`) and Python backend paths (`backend/`, `functional/`, `utils/`) for avoidable overhead between model code and GPU kernels.

## Critical Findings (Bindings + Python)

### 1) Python core dispatch path is still fence-synchronous per kernel

**Where**
- `backend/core.py` (`_dispatch_compute`)

**What**
- `_dispatch_compute` performs `vkWaitForFences` before submit and again immediately after submit for every dispatch.

**Why it hurts**
- Completely serializes command execution from Python path.
- Removes queue depth and overlap opportunities even when kernels are independent.

**Recommendation**
- Add a non-blocking dispatch mode (`submit_only`) and explicit `wait()` boundaries.
- Route multi-op Python code to `CommandRecorder` chains by default.
- Keep existing synchronous behavior only for compatibility paths.

**Impact**
- High for legacy Python backend and any operation sequences.

---

### 2) Many Python modules force upload->dispatch->download per op

**Where**
- `backend/fnn.py`
- `backend/attention.py`
- `backend/snn.py`
- `backend/pooling.py`
- `backend/tensor_ops.py`

**What**
- Typical pattern is flatten numpy -> upload all inputs -> dispatch one kernel -> immediate download -> release.

**Why it hurts**
- Repeated host-device copies dominate for medium/small kernels.
- Prevents GPU-resident tensor flow across layers.

**Recommendation**
- Promote GPU-resident `VulkanTensor` flow for intermediate values.
- Add fused Python-level execution helpers that keep outputs on GPU until terminal read.
- For multi-pass algorithms (attention, backward chains), use one recorded command buffer submission.

**Impact**
- High in end-to-end model throughput.

---

### 3) GIL release coverage is inconsistent across heavy pybind entrypoints

**Where**
- `cpp/python/bindings_conv.cpp`
- `cpp/python/bindings_attention.cpp`
- `cpp/python/bindings_misc.cpp` (many paths)
- (Positive examples already exist in `bindings_linear.cpp`, `bindings_moqe_train.cpp`, `bindings_perceiver.cpp`)

**What**
- Some heavy GPU wrappers run with GIL held; only selected functions use `py::gil_scoped_release`.

**Why it hurts**
- Blocks Python-side concurrency (input pipeline, orchestration threads, async workers).

**Recommendation**
- Standardize call guards:
  - `py::call_guard<py::gil_scoped_release>()` for long-running compute wrappers
  - or explicit `py::gil_scoped_release` around GPU dispatch sections.
- Keep GIL held only around Python object manipulation.

**Impact**
- Medium to high in multi-threaded training/inference hosts.

---

### 4) pybind array signatures allow strided input but code assumes contiguous

**Where**
- Multiple bindings use `py::array_t<float>` + `request().ptr` and treat memory as dense contiguous.

**What**
- Many wrappers do not enforce `c_style|forcecast`, and ignore `strides`.

**Why it hurts**
- Potential correctness hazards for strided/non-contiguous arrays.
- Hidden conversions/copies may occur unpredictably and hurt performance.

**Recommendation**
- Use explicit signatures:
  - `py::array_t<float, py::array::c_style | py::array::forcecast>`
- Or validate contiguity and materialize contiguous buffers once at boundary.

**Impact**
- Medium (performance consistency + correctness hardening).

---

### 5) Functional API fallback can create fresh backend objects per call

**Where**
- `functional/attention.py`
- `functional/linear.py` (fallback path)

**What**
- Functional fallback calls `Compute()` directly; in patterns where bridge is unavailable this can repeatedly initialize backend state.

**Why it hurts**
- High startup overhead and repeated shader/device setup costs.

**Recommendation**
- Add a module-level singleton backend cache for functional fallback path.
- Reuse backend/device across functional calls.

**Impact**
- Medium to high in environments not using native bridge.

---

## Additional High-Value Opportunities

### A) Descriptor cache key in Python pipelines uses object identity, not handle value

**Where**
- `backend/pipelines.py` (`get_cached_descriptor_set`)

**What**
- Cache key uses `id(buf)` for buffer handles.

**Why it hurts**
- Two Python objects wrapping the same Vulkan handle can miss cache.
- Causes avoidable descriptor churn and lower hit rates.

**Recommendation**
- Use normalized handle values (`int(buf)`) in cache key.
- Preserve `(shader, sizes, handle_values)` as cache key.

**Impact**
- Medium CPU overhead reduction under dispatch-heavy workloads.

---

### B) Broad exception swallowing in bridge can mask GPU performance regressions

**Where**
- `backend/_bridge.py` (many wrappers return `None` on any exception)

**What**
- Fallback-to-CPU behavior can silently trigger if GPU path errors.

**Why it hurts**
- Throughput collapses without obvious signal.
- Hard to profile because errors are debug-log only.

**Recommendation**
- Keep fallback, but add optional strict mode env:
  - `GRILLY_BRIDGE_STRICT=1` to raise on GPU path failure.
- Add lightweight counters for fallback frequency.

**Impact**
- Medium for performance observability and reliability.

---

### C) Python-side causal mask construction uses nested loops

**Where**
- `backend/attention.py` (`attention_mask`)

**What**
- Causal mask is created with Python nested loops.

**Why it hurts**
- O(S^2) Python loop overhead for larger sequence lengths.

**Recommendation**
- Replace with vectorized construction (`np.tril`) and contiguous cast once.
- Prefer generating mask in shader when possible.

**Impact**
- Low to medium (but easy win).

---

### D) Legacy monolithic binding file remains in tree and risks divergence

**Where**
- `cpp/python/bindings.cpp` (legacy, not compiled per `CMakeLists.txt`)

**What**
- New split binding architecture is active, but old monolith remains.

**Why it matters**
- Easy source of stale behavior assumptions and duplicate maintenance effort.

**Recommendation**
- Keep as archive only if needed; otherwise remove or clearly gate as reference-only with CI checks.

**Impact**
- Low runtime impact, medium maintainability impact.

---

## Workstream C (kernel / throughput) — status

**Milestone status:** Workstream C is **closed** for the scoped parity milestone (see `docs/PYTORCH_PARITY_TASKLIST.md`). Remaining throughput ideas (atomic 1×1 wiring, INT8 tiling, dedicated transfer queue) are listed there under **Workstream C — future**.

**C1 — Conv backward weight**  
The shipped Vulkan path uses `conv2d-backward-weight.glsl` (per-weight-slot accumulation, **no** global atomics). When `gemm_mnk` and `tensor-transpose` shaders are available, `_conv2d_backward_weight_gemm` keeps im2col on device: `convd_im2col` → transpose `cols` → `gemm_mnk` for `grad_weight` (only `grad_weight` is read back). If those shaders are missing, the code falls back to downloading `cols` and NumPy `grad_out @ cols.T`. The standalone `conv1x1-backward-weight.glsl` (atomic float) is not wired in the C++ dispatch table; treat as experimental follow-up. Baseline: `benchmarks/benchmark_conv_backward_weight.py`. Parity: `tests/test_conv_backward_weight_gemm.py`.

**C2 — INT8 GEMM**  
`shaders/int8-gemm.glsl` inner K-loop processes **4 K steps per iteration** (one packed `uint32` weight load per four activations). Rebuild `shaders/spv/int8-gemm.spv` after GLSL edits. Baseline: `benchmarks/benchmark_int8_gemm.py` (`VulkanFNN.gemm_int8`). Shared-memory tiling is future work.

**C3 — Attention**  
Long-sequence smoke tests: `tests/test_attention_long_sequence.py` (S = 128, 256, 512; optional slow S = 1024). FlashAttention2 uses online softmax; do not assume bit-identical agreement with a decomposed scores→softmax→output path without a dedicated parity investigation.

**C4 — Transfers**  
`VulkanCore` already reuses a single command buffer and fence per dispatch (`_cmd_buffer`, `_fence`); `record_commands` uses `_batch_fence`. Further gains: dedicated transfer queue + VMA staging buffers for host↔device copies (tracked as C4+ follow-up, not required for C closure).

---

## Bindings + Python Implementation Order

### Phase A (Immediate)
1. Add async/non-blocking dispatch path in Python core + explicit wait points.
2. Apply GIL-release guards consistently across heavy bindings.
3. Enforce contiguous pybind array contracts at entrypoints.

### Phase B (Dataflow)
1. Make VulkanTensor GPU-resident path default for intermediate tensors.
2. Reduce mandatory download points in Python modules.
3. Improve descriptor-key normalization in `backend/pipelines.py`.

### Phase C (Reliability + maintainability)
1. Add strict-mode fallback diagnostics in bridge.
2. Vectorize residual Python loops (e.g., causal mask creation).
3. Retire or quarantine legacy monolithic bindings source.

---

## Bindings + Python Summary

The C++ kernel side is advancing quickly, but Python/binding orchestration still introduces major serialization and copy overhead in common paths. The fastest wins are to **remove per-dispatch waits**, **keep tensors GPU-resident**, and **standardize pybind/GIL handling**. These changes will let existing optimized kernels deliver their intended end-to-end gains.

