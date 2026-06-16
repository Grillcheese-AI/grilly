# GEMM autotuner — findings + C++ integration spec

> ## ⚠️ CORRECTION (2026-06-16) — coopmat kernel was BROKEN when first benched
>
> The `gemm-coopmat-shared` kernel these timings were measured against was
> **numerically wrong** at the time the table below was produced. A parity test
> (identity-matrix probe) showed the fp16 path returned the **per-tile transpose**
> of the correct result: it staged the weight matrix B with the wrong stride,
> reading `W` (stored N×K) as if it were a K×N row-major matrix. fp32 matched the
> CPU reference to 5.6e-7; fp16 had relerr ≈ 1.41 (uncorrelated).
>
> **Root cause:** `bench_gemm` (autograd.cpp) and `ops::linear` (linear.cpp) fed
> the *same* kernel under **contradictory** layout assumptions — bench passes B as
> a plain K×N GEMM operand, while `ops::linear` passes the weight matrix N×K and
> needs `y = x·Wᵀ`. The kernel was written to bench's assumption, so it timed fine
> but computed garbage in the real linear op. Nothing caught it because the
> resident path runs fp32 and never invoked coopmat for real work.
>
> **Fix:** B-staging now reads `W` transposed (`B[(tile_col+b_c)*K + (k+b_r)]`),
> matching the `y = x·Wᵀ` convention of the linear op. Re-validated: identity probe
> exact; random-data parity **2.9e-04** vs fp32 across 512×1024×1024, ×3072, ×8192,
> 512×4096×1024, 256×2048×2048, 512×1024×64 — all PASS at <1e-2.
>
> **Consequences for THIS doc:**
> 1. The fp16-ms column below timed a broken kernel. The *timings* are probably
>    still roughly representative (same dispatch shape, same work), but the
>    **winner column is not trustworthy** until re-benched against the fixed
>    kernel. Re-run the autotuner sweep before relying on any per-shape decision.
> 2. The fixed kernel now implements `x·Wᵀ`, NOT plain `A·B`. This matters for the
>    backward reuse — see FP16_BACKWARD_SPEC.md: `grad_input = g·W` wants the same
>    `x·Wᵀ` convention (good, reuses directly), but `grad_weight = gᵀ·x` wants a
>    plain `A·B` (x is a data matrix, not weights) — so the two backward GEMMs need
>    DIFFERENT conventions and a single un-parameterized kernel can't serve both.
> 3. `gemm_autotune.json` cached entries predate the fix → stale. Delete and
>    re-tune.

---

# (original notes below — winner column pending re-bench)


Python-level autotuner that measures, per GEMM shape, whether the fp32 tiled
kernel (`gemm_tiled`) or the fp16 cooperative-matrix kernel
(`gemm-coopmat-shared`) is faster on the actual device, and caches the winner.
Module: `grilly/backend/gemm_autotune.py`. Cache: `~/.grilly/gemm_autotune.json`
(device-scoped). Status: **built, guarded, validated on RX 6750 XT.** It is a
measurement table only — it changes nothing about training until a dispatch hook
consults it (see "C++ integration", below).

## Why

The resident trunk GEMMs run fp32 today:
- `TapeContext::forward_linear` -> `ops::batchedLinear` -> fp32 path.
- `BackwardEngine::backward_linear` is **hardcoded** to `gemm_tiled` for both
  grad_input and (transpose +) grad_weight (`autograd.cpp`).

`bench_gemm(M,K,N,iters)` (already bound to Python) times `gemm_tiled` vs
`gemm-coopmat-shared` for one shape. The autotuner drives it across the shapes
the model actually issues and records the faster kernel. The crossover is real
and shape-dependent, so a per-shape table beats any single blanket choice.

## Findings (AMD Radeon RX 6750 XT, RADV)

Coopmat **is** supported on this GPU (`VK_KHR_cooperative_matrix` present). The
measured winners, batch=1:

| shape M×K×N        | role (0.0.1 / tiny)        | fp32 ms | fp16 ms | winner  |
|--------------------|----------------------------|---------|---------|---------|
| 512×1024×3072      | MinGRU G/V/D proj          | ~0.91   | ~0.84   | coopmat |
| 512×1024×8192      | SwiGLU gate+up             | 2.32    | 2.29    | **tiled** |
| 512×4096×1024      | SwiGLU down                | 1.58    | 1.21    | coopmat (1.31×) |
| 512×8192×1024      | SwiGLU up grad_input       | 3.17    | 2.43    | coopmat (1.31×) |
| 512×1024×65536     | tied output head           | 21.4    | 20.1    | tiled (by policy — capped) |

Takeaways:
- Coopmat wins ~1.3× on most trunk GEMMs, **but loses on 1024×8192** (gate+up).
  That single inversion is the whole justification for per-shape selection: a
  blanket "always coopmat" would regress that shape.
- The head (N=65536) measured a marginal coopmat win (1.07×) pre-cap, but it is
  recorded as **tiled by policy** — see safety cap.

## Safety cap (why the GPU crashed, and what prevents a repeat)

During the first full sweep the GPU dropped its display signal (RADV TDR / queue
wedge). The crasher was a vocab-sized **M** dimension: `bench_gemm`'s coopmat
path dispatches `gy = M/16` workgroups deep, so the `65536×512×1024` grad_weight
transpose = 4096 workgroups deep over fp16 buffers, which hung the queue.

Guard (`_DEFAULT_COOPMAT_MAX_DIM = 32768`): if `M > cap` **or** `N > cap`, the
autotuner does **not** issue the coopmat bench dispatch — it records `tiled`
(the resident default) with `capped: true` and never touches the GPU for that
shape. The cap sits below any vocab-sized (65536) dimension, so neither the
M=65536 transpose (the crasher) nor the N=65536 head re-benches. `_load` also
re-asserts the cap over already-cached entries, so a coopmat result cached
before the cap existed is demoted to tiled on next construction. Verified:
unit test confirms M/N=65536 shapes never call `bench_gemm`.

Operational rule going forward: **bench one shape per process, low iters
(≤40).** The MCP bridge also hangs (separately from the GPU) on long
synchronous GPU work; one-shot keeps each call short.

## API

```python
from grilly.backend.gemm_autotune import GemmAutotuner, cubby_gemm_shapes
t = GemmAutotuner(iters=30)          # loads device-scoped cache
t.decide(M, K, N)                    # -> "tiled"|"coopmat" (benches+caches once)
t.lookup(M, K, N)                    # -> cached value or None (no GPU)
t.tune_shapes(cubby_gemm_shapes(cfg))# batch; respects cap
```

`cubby_gemm_shapes(cfg, batch=1)` derives the distinct trunk GEMMs (MinGRU proj,
SwiGLU gate+up/down, head, plus the backward companions) from a `SparseCubbyConfig`.

## C++ integration — the real picture (NOT a one-line shader swap)

The original framing ("flip backward_linear from gemm_tiled to coopmat per the
table") is **too optimistic**. What's actually required, by path:

**`ops::linear` (forward standalone op)** already has full coopmat selection
(`linear.cpp`), but gated on `inElem == 2` (fp16 input) + device support +
shape alignment (M%16, K%16, N%64). It only takes coopmat when the *caller*
provides fp16 data. The resident path hands it fp32, so it never triggers.

**To make the resident trunk use coopmat** you cannot just rename the pipeline.
You must, per GEMM the table marks `coopmat`:
1. Provide **fp16** A/B buffers (convert the resident fp32 weight + activation
   buffers to fp16 — a real data-path change, with conversion cost and a
   numerical-accuracy check vs the fp32 baseline).
2. Satisfy alignment **M%16, K%16, N%64**, else coopmat under-covers the output
   (the dispatch uses integer division `N/64, M/16` with no ceil — a
   correctness trap, not just a perf one). Most trunk shapes are aligned
   (512, 1024, 4096, 8192) but the head N=65536 and any odd batch are not.
3. For backward: there is **no fp16 backward shader yet** — `linearBackward`
   hard-rejects `elemSize != 4` ("TODO: coopmat backward shader"). So backward
   coopmat is blocked until that shader lands. Forward-only coopmat is the
   reachable first increment.

**Recommended sequencing (smallest safe increments):**
1. *(done)* Python table of per-shape winners.
2. Forward-only, fp16 coopmat for the aligned trunk GEMMs the table favors
   (MinGRU proj, SwiGLU down), behind a flag, with an fp32-parity test gating
   each shape. Skip anything unaligned or over the cap.
3. Measure end-to-end step time (not just per-GEMM µbench) — the fp32->fp16
   conversion and bias post-pass eat into the 1.3× kernel win; confirm net gain
   before widening.
4. Backward coopmat only after the fp16 backward shader exists and passes
   gradient-parity. Separate effort.

The table is the input to step 2's "which shapes". Wiring lives in Claude Code's
lane (C++ + rebuild), deliberately separate from this measurement work.

## Honest caveats

- **µbench ≠ training speedup.** `bench_gemm` times the kernel only, on garbage
  buffers, no fp32<->fp16 conversion, no staging. The 1.3× is an upper bound on
  the per-GEMM win; net step speedup will be lower.
- **Cap is conservative on the head.** The N=65536 head is forced to tiled even
  though it micro-benched ~1.07× faster in fp16. That GEMM is also the single
  biggest one (~20 ms); revisiting it safely (capped iters, watchdog) could be
  worth it later, but not at the cost of another TDR.
- **Coverage gap in coopmat dispatch.** `bench_gemm` / `ops::linear` coopmat use
  `N/64, M/16` (floor) grids. Unaligned shapes are silently under-covered. The
  alignment gate in `ops::linear` handles this for the forward op, but any new
  hook MUST replicate that gate.
- **Device-scoped cache.** Table tuned on the RX 6750 XT must never be applied
  to another GPU; the JSON is keyed by device name and the loader only reads its
  own device's section.
