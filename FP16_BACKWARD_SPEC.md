# Spec — fp16 cooperative-matrix backward for resident Linear

**Goal:** unblock coopmat in the resident *training* path. Forward coopmat
already exists (`ops::linear` + `gemm-coopmat-shared`); the blocker is backward.
`ops::linearBackward` hard-rejects `elemSize != 4` ("fp16 backward needs a
cooperative matrix backward shader — TODO"), and `BackwardEngine::backward_linear`
(autograd.cpp) is hardcoded to fp32 `gemm_tiled`. This spec defines exactly what
to build.

## Key finding (REVISED 2026-06-16 after fixing + validating the kernel)

The forward coopmat kernel `gemm-coopmat-shared` is now **fixed and numerically
validated** (was returning the per-tile transpose; see GEMM_AUTOTUNE.md
correction). Crucially, the fix means the kernel now computes **`C = A · Bᵀ`**
where B is the weight matrix stored `(N, K)` — i.e. it matches the linear op's
`y = x·Wᵀ` convention, NOT a plain `A·B`. Validated parity vs fp32: ~2.9e-04
across all trunk shapes.

This changes the backward reuse story. The two backward GEMMs do NOT share a
convention:

```
grad_input  = g @ W      # g (BS,out), W stored (out,in).
                         #   = g · W with W as (K=out, N=in) but stored (out,in)
                         #   = EXACTLY the kernel's A·Bᵀ form (B=W is the (N,K)=(in?,..))
                         #   -> reuses the FIXED kernel directly. ✓
grad_weight = gᵀ @ x     # gᵀ (out,BS), x (BS,in), plain row-major A·B.
                         #   x is a DATA matrix, not weights -> needs plain A·B,
                         #   which the fixed kernel NO LONGER does. ✗
```

So:
- **grad_input** can reuse the fixed `gemm-coopmat-shared` as-is (its `x·Wᵀ`
  convention is exactly `g·W` with W stored row-major (out,in)). Confirm with a
  parity probe before trusting this — the index algebra is plausible but unproven
  for the backward operand shapes.
- **grad_weight** needs a *plain* `A·B` coopmat (no Bᵀ). Options:
  (a) a second kernel variant `gemm-coopmat-shared-nt` that stages B without the
      transpose (the ORIGINAL staging, before the fix), or
  (b) materialize `xᵀ` and feed `gᵀ · (xᵀ)ᵀ` through the Bᵀ kernel — wasteful,
      two transposes, not worth it, or
  (c) parameterize the kernel with a push-constant `transpose_b` flag selecting
      which staging index to use. **(c) is cleanest** — one kernel, one bit.

Recommended: **(c)** — add a `uint transpose_b` push constant to
`gemm-coopmat-shared`; `transpose_b=1` (weights, `x·Wᵀ`) for forward + grad_input,
`transpose_b=0` (plain `A·B`) for grad_weight. The two staging index expressions
already exist (pre-fix = plain, post-fix = transposed); the flag just selects.

Helper shaders still needed (written + compiled):
1. `cast-f32-f16.glsl` — elementwise fp32 → fp16 copy. ✓ compiled.
2. `transpose-f32-f16.glsl` — fused transpose+cast for building `gᵀ` (fp16). ✓
   compiled. (Still needed for grad_weight's gᵀ regardless of the flag approach.)

## The three backward quantities (unchanged math)

Linear: `y = x @ W^T (+b)`, W is `(outputDim, inputDim)`, x is
`(batchSeq, inputDim)`, grad_output `g` is `(batchSeq, outputDim)`.

```
grad_input  = g   @ W            # (BS,out)·(out,in)        -> (BS,in)
grad_weight = g^T @ x            # (out,BS)·(BS,in)         -> (out,in)
grad_bias   = sum(g, axis=0)     # (out,)
```

Note W is stored `(out,in)` = exactly the `(K,N)` operand `grad_input` needs
(K=out, N=in) with NO transpose. `grad_weight` needs `g^T` (transpose of g).

## Dispatch plan (fp16 path), per pass

All coopmat GEMMs require M%16, K%16, N%64 on the **operand as the kernel sees
it** (M=rows of A/C, K=contraction, N=cols of B/C). Cast inputs to fp16 first.

**Pass 0 — grad_input = g @ W**
- A = g (BS×out) fp16, B = W (out×in) fp16, C = grad_input (BS×in) fp32.
- M=BS, K=out, N=in. Coopmat-legal iff BS%16, out%16, in%64.
- Prep: `cast-f32-f16` on g and W (W cast once per step, cacheable).
- Dispatch `gemm-coopmat-shared` gx=in/64, gy=BS/16.

**Pass 1 — grad_weight = g^T @ x**
- Need g^T (out×BS). Use `transpose-f32-f16` on g → gT_f16 (out×BS), fp16.
- Cast x → x_f16 (BS×in).
- A = gT (out×BS), B = x (BS×in), C = grad_weight (out×in) fp32.
- M=out, K=BS, N=in. Coopmat-legal iff out%16, BS%16, in%64.
- Dispatch gx=in/64, gy=out/16.

**Pass 2 — grad_bias** = unchanged. Reuse the existing fp32 reduction
(fnn-linear-backward pass 2, or a standalone column-sum). Bias stays fp32
everywhere (matches forward). No coopmat.

## Alignment fallback (REQUIRED — correctness, not just perf)

`gemm-coopmat-shared` uses floor division grids (`N/64`, `M/16`) and the loop
`for k in 0..K step 16`. If a dimension isn't aligned it **silently drops the
remainder** — wrong results, not a crash. So the dispatch wrapper MUST check
M%16==0 && K%16==0 && N%64==0 for the pass's operand layout and fall back to the
fp32 `gemm_tiled` (or the fnn-linear-backward scalar pass) when unaligned.

Cubby trunk dims are friendly: BS=512, in/out ∈ {1024, 4096, 8192} are all
%64. The danger dims are the vocab head (out=65536 in grad_weight: 65536%16 ok,
but it's over the autotuner cap — keep it fp32 tiled) and any non-multiple batch
(B*S). Gate per-shape; do NOT assume.

## C++ wiring (where each piece lands)

`ops::linearBackward` (linear.cpp): add an fp16 branch parallel to the existing
fp32 one, selected when `p.elemSize == 2 && device.hasCooperativeMatrix() &&
aligned`. Allocate fp16 staging for g/x/gT + W; emit cast/transpose dispatches
into the same batched command buffer before the coopmat dispatch; keep the fp32
path as the `else`.

`BackwardEngine::backward_linear` (autograd.cpp): this is the resident hot path
and currently emits `gemm_tiled` directly (not via `ops::linearBackward`). Two
options:
- **(A, smaller)** leave it fp32 — only flip `ops::linearBackward` for non-
  resident callers. Lowest risk, but does NOT speed up resident training (the
  thing we care about). 
- **(B, the real win)** mirror the cast→coopmat chain inline in
  `backward_linear`'s pass 0 / pass 1 blocks, guarded by the same alignment +
  autotuner-table check. This is where the 1.3× actually lands. Higher risk
  (hot path, fp16 grads). Recommended only after (A) proves parity.

Autotuner table feeds the per-shape guard: only take the coopmat branch where
`gemm_autotune` says `coopmat` AND aligned AND under the dim cap.

## Numerical-parity gate (MUST pass before enabling)

fp16 inputs lose mantissa; grad_weight accumulates over K=BS=512 terms, the
riskiest for fp16 round-off (accumulation is fp32 in the coopmat accumulator,
which helps). Gate each shape:
- Compare fp16-coopmat grad_input / grad_weight against the fp32 `gemm_tiled`
  result on the same inputs.
- Threshold: relative L2 error < 1e-2 (fp16 GEMM with fp32 accumulate typically
  lands ~1e-3). If a shape exceeds it, keep that shape fp32.
- Add to test suite as a per-shape parametrized parity test, same spirit as
  test_checkpoint's logits-parity check.

## Build order (smallest safe increments)

1. ✓ DONE — `cast-f32-f16.glsl` + `transpose-f32-f16.glsl` written + compiled
   (SPVs in shaders/spv). ✓ DONE — fixed + validated `gemm-coopmat-shared`
   forward (parity ~2.9e-04).
2. ✓ DONE — added the `transpose_b` push-constant flag to `gemm-coopmat-shared`
   (transpose_b=1 -> A·Bᵀ weights/forward; transpose_b=0 -> plain A·B).
   Push grew 12→16 bytes; updated both C++ callers (`ops::linear` sets =1,
   `bench_gemm` sets =0 with a separate 12-byte pc3 for gemm_tiled) and rebuilt
   (Build OK). Forward RE-VALIDATED post-rebuild: ~2.9e-04 across all trunk
   shapes — the flag did not regress the forward path.
   NOTE: the transpose_b=0 *code path* itself is compiled but not yet exercised
   from Python (no binding sets the flag). Its plain-A·B math is validated
   indirectly (forward with pre-transposed W gives 2.9e-04), but its first true
   test is step 3, where grad_weight invokes it. Gate it there.
3. ✓ DONE — added `linearBackwardCoopmat` (fp32 in/out, fp16 coopmat GEMMs
   internally) + binding `linear_backward_coopmat`. Both backward GEMMs use
   transpose_b=0 (plain A·B): grad_input = g·W (B=W read plain as (out,in)),
   grad_weight = gᵀ·x (gᵀ built fp16 via transpose-f32-f16). grad_bias fp32 via
   fnn-linear-backward pass 2. Parity vs numpy ref, 6 shapes (incl. MinGRU-proj
   512x1024x3072, SwiGLU-down 512x4096x1024): grad_input/grad_weight ~7.7e-04,
   grad_bias EXACT — ALL PASS <1e-2. No autograd/hot-path change. NOTE: this
   means grad_input also uses transpose_b=0, NOT transpose_b=1 as the earlier
   key-finding guessed — because W is stored (out,in) which IS already the plain
   (K=out,N=in) operand, so no Báµ€ needed. The transpose_b=1 path remains the
   forward-only case.
   BONUS FIX: found + fixed a latent bug in the EXISTING fp32 `ops::linearBackward`
   — grad_bias pass 2 dispatched ceil(out/256) workgroups but the shader is a
   16-wide workgroup, so only the first 64 of 1024 bias entries were ever
   computed (15/16 left zero). Now ceil(out/16). This bug only affected the
   `ops::linearBackward` binding, not the resident autograd path (which has its
   own bias handling).
4. Wire `backward_linear` (autograd.cpp) option (B) behind the autotuner table,
   one shape at a time, parity-gated. Measure END-TO-END step time, not µbench —
   the cast+transpose dispatches eat into the kernel win.
5. Only widen to more shapes after net step-time improvement is confirmed.

## Honest risks

- **Hot path + fp16 grads.** Backward feeds AdamW; bad fp16 grads degrade
  training silently (loss drifts, not a crash). The parity gate is the guard;
  do not skip it.
- **Cast/transpose overhead.** Two extra dispatches + fp16 staging per linear.
  On small GEMMs this can erase the coopmat win — that's WHY it's autotuner-
  gated and end-to-end-measured, not applied blanket.
- **GPU stability.** Same coopmat kernel family that TDR'd the bench on a
  vocab-sized M. Keep the dim cap; never dispatch coopmat on the 65536 head.
- **Coverage floor.** The floor-division grid means alignment fallback is
  mandatory and must be tested with a deliberately unaligned shape.
- **Backward shader is reused, not new** — lower risk than a hand-written
  backward kernel, but it inherits `gemm-coopmat-shared`'s exact alignment
  contract, so the wrapper owns correctness.
