# Resident Autograd — Working State

Branch: `autograd-resident-backward` (off `2.0-dev`). Author: this work session.
Last verified: green on RX 6750 XT, grilly_core rebuilt clean.

## What this branch does
Turns grilly's STUBBED C++ `BackwardEngine` (cpp/src/autograd.cpp) into a real
resident reverse-mode autograd: gradients computed on-GPU, buffers never leave
VRAM, one command batch / one submit per backward pass.

## The load-bearing idea
The autograd graph (TensorRef/Node) addresses memory by opaque `uint32_t
buffer_id`, but BufferPool only deals in GrillyBuffer value types — there was no
id->buffer table, so every backward handler was a `// TODO` placeholder that
freed grad buffers without dispatching. The fix is a registry that bridges
id <-> GrillyBuffer; once that exists, each handler is "resolve inputs, alloc
grad, dispatch the existing backward shader, publish id".

## Done + VERIFIED (numpy-checked on hardware)
- `cpp/include/grilly/autograd/buffer_registry.h` — BufferRegistry. uint32 ids,
  external (forward-owned) vs owned (grad/temp) buffers, resolve(), alloc()
  (DEVICE_LOCAL), upload/download, step-scoped clear(). Shares the engine's
  BufferPool.
- `CommandBatch::fillZero` (vk_command_batch.h / command_batch.cpp) — GPU-side
  vkCmdFillBuffer zero-init. DEVICE_LOCAL buffers carry TRANSFER_DST so it's valid.
- `backward_linear` — real 3-pass fnn-linear-backward (grad_x, grad_w, grad_b)
  on resident buffers, fillZero before the atomic-accumulate passes. grad_x/grad_W
  correct to <1e-6 vs numpy. (test_backward_linear.py)
- `backward()` driver — single batch_.begin()/submit() around the tape walk;
  PULL-based node->node propagation (a node activates if a downstream consumer
  deposited a grad under one of its output buffer_ids); fan-out accumulation via
  grilly::ops::batchedAdd (elementwise-add, in-place). Loss node always activates
  even with grad_output_buffer==0.
  2-layer Linear chain: grad_W2/grad_W1/grad_x all correct. (test_backward_chain.py)
- `backward_cross_entropy` — real cross-entropy-backward, dL/dlogits =
  softmax-onehot. CE->Linear chain grad_W and grad_h correct to ~2e-7. This is the
  training entry point. (test_backward_ce.py)
- `backward_activation` (shared 3-buffer helper) -> backward_relu/gelu/silu wire
  the existing activation-*-backward shaders. SiLU grad correct to 1.2e-7.
  (test_backward_silu.py)
- `backward_add` (was already correct: routes grad_output to both inputs) +
  FAN-OUT accumulation VERIFIED: x -> two Linears -> Add backprops
  grad_x = grad_y@W1 + grad_y@W2 EXACTLY (0.0 err). This exercises the
  find_or_insert_grad batchedAdd path — the residual-connection pattern.
  (test_backward_fanout.py)
- `backward_rmsnorm` + NEW `rms-norm-backward.glsl` kernel (none existed). 2-pass,
  no atomics (one thread per output cell). grad_x 3.6e-7, grad_w 2.4e-7 vs numpy.
  Added OpType::RMSNorm to the dispatch switch (was hitting default no-op).
  (test_backward_rmsnorm.py)
- `backward_mingru` — wired the existing mingru-backward shader (8 buffers
  {gradH,G,V,D,H,gradG,gradV,gradD}). Node: 3 inputs (G,V,D), saves [G,V,D,H].
  grad_g/v/d all ~1e-7 vs numpy. Added OpType::MinGRU (routes BACKWARD only;
  not fused in forward OpGraph). (test_backward_mingru.py)
- `backward_swiglu` — wired activation-swiglu-backward (3 buffers, input is
  [x1|x2] 2*hidden wide -> hidden out). grad 6e-8 vs numpy. OpType::SwiGLU.
  (test_backward_swiglu.py)
- FULL CUBBY BLOCK composes, gradient-checks, and TRAINS:
  RMSNorm->3 Linears->MinGRU->residual->RMSNorm->Linear->SwiGLU->residual->
  head->CE. dL/dX matches finite-diff to rel 3.3e-4; loss 1.21->~0, acc->1.00
  by step 15 (AdamW). (experimental/resident_train/train_block.py;
  bisect_block.py is the A-E cross-op composition regression test.)
- Python surface: `cpp/python/bindings_autograd.cpp` (NEW, compiled file added to
  CMakeLists + bindings_core). Exposes OpType / TensorRef / TapeContext /
  AutogradNode + register_input (float32) / register_input_u32 / read_buffer.

## CRITICAL gotchas discovered
- `bindings.cpp` is DEAD CODE (CMake comment: "not compiled"). The autograd
  bindings only exist now because bindings_autograd.cpp was added to the build.
  Do NOT edit bindings.cpp expecting it to take effect.
- CE targets are read by the shader as `float[]` then cast to uint. Register CE
  targets via register_input (float32), NOT register_input_u32.
- cross-entropy-backward shader uses ONE WORKGROUP PER ROW (gl_WorkGroupID.x) →
  dispatch gx=batchSize. The existing loss.cpp::crossEntropyBackward has a latent
  bug here: gx=(batchSize+255)/256 under-computes for batchSize>1.
- TapeContext member init order: registry_ MUST be declared before engine_
  (engine holds a ref to it).
- MinGRU forward `a = 0.001 + 0.998*sigmoid(d)` per the SHADER (mingru-*.glsl).
  The mingru.h HEADER COMMENT says 0.05+0.9 — it is STALE/WRONG. Trust the shader.
- BACKWARD-PASS SYNC: the driver walk needs transferComputeBarrier in TWO places
  per node (both required; per-op unit tests don't catch these — only cross-op
  composition does, via bisect_block.py / full block):
    (1) BEFORE dispatch_node_backward — the node reads grad_output_buffer which
        may have been built by batchedAdd accumulation while processing
        downstream nodes (e.g. RMSNorm reading an n1 grad summed from 3 Linear
        branches). Without it, the read races the adds.
    (2) AFTER dispatch — the handler's grad writes (incl. fillZero TRANSFER_WRITE
        in backward_linear) must be visible before accumulation reads them and
        before the next batchedAdd in a same-buffer fan-out (e.g. MinGRU G==V==D).
  Use transferComputeBarrier (covers TRANSFER+COMPUTE), NOT plain barrier
  (compute-only) — fillZero is a transfer write.
- Gradient-check composed graphs against finite differences (or an independent
  analytic), not just "loss goes down". Both sync races above produced
  plausible-but-wrong grads that still partially trained; only the fd check at
  rel<1e-2 caught them. analytic-vs-fd agreeing to ~1e-5 confirms the numpy ref
  before blaming the engine.

## STILL STUBBED / MISSING (not on the trunk critical path)
- backward_mul (dL/da = dy*b, dL/db = dy*a) still `= 1` placeholders. Only needed
  if the trunk uses elementwise products outside the fused SwiGLU/MinGRU kernels.
- backward shape ops: sum/mean/transpose still `= 1` placeholders. Only needed if
  the trunk reduces/transposes on the autograd path.
- backward_layernorm, backward_softmax, backward_attention still stubbed (no
  matching backward shaders wired). Cubby uses RMSNorm + MinGRU, not LayerNorm/
  attention, so not on the trunk critical path.

## Trunk readiness — BACKWARD PATH CLOSED
ALL per-op kernels the Cubby trunk needs are implemented and verified vs numpy:
  embed/linear, cross-entropy, silu/relu/gelu, add + fan-out (residuals),
  rms-norm, mingru.
Plus driver: pull propagation, loss-node seeding, single-batch residency, fan-out
accumulation. The full
  embed -> [RMSNorm -> MinGRU -> RMSNorm -> SwiGLU]xL -> RMSNorm -> tied head -> CE
backward path can now be assembled and trained resident.

## Forward side (separate, not this branch)
Forward resident execution already works in C++: nn::Tensor + ComputeBackend
graphMode (beginBatch/dispatch/endBatch -> OpGraph.optimize()+execute(), single
fused submit). cubby-lm currently uses the NON-resident _bridge staging ops
(grilly::ops::linear, CPU-ptr in/out). Driving resident forward from Python would
need the ComputeBackend dispatch primitives bound (createBuffer/upload/dispatch/
beginBatch/endBatch/setGraphMode) — NOT done yet.

## Integration — RESIDENT GRADS ARE USABLE (training loops, not just pointwise)
experimental/resident_train/ — numpy forward + numpy AdamW, RESIDENT backward.
- train_linear_ce.py: Linear+CE on a 16x8->4 memorize task. loss 1.2565 -> 0.0204
  monotonic, acc -> 1.00 by step 40. Proves the read-back -> optimizer ->
  re-register loop and correct grad sign/scale. Confirms the /B mean-CE scaling.
- train_mingru_ce.py: train THROUGH the MinGRU scan. G,V,D learnable -> MinGRU ->
  mean-pool -> Linear head -> CE. loss 1.1049 -> ~0, acc -> 1.00 by step 15.
  Proves backward-in-time grad flows into all 3 projections with usable scale.
The two hardest trunk backward ops now both TRAIN, not just match numpy.
- train_block.py: the FULL block (RMSNorm->3Lin->MinGRU->res->RMSNorm->Lin->
  SwiGLU->res->head->CE) gradient-checks (dL/dX vs finite-diff rel 3.3e-4) AND
  trains (loss 1.21->~0, acc->1.00 by step 15, AdamW on all params). This is the
  composition proof — every op type + both residual fan-outs compose correctly.
  bisect_block.py = the A-E cross-op regression test that localized the two sync
  races fixed in this slice.

## RESIDENT FORWARD — opened (Linear), feeds the backward tape directly
TapeContext now stores batch_/cache_ refs and exposes a resident forward path
that reuses the SAME registry/batch as backward (no separate ComputeBackend
graphMode binding needed — the backward infra already had everything):
  forward_begin();  out_id = forward_linear(in_id, w_id, bias_id, M,K,N);  forward_submit();
forward_linear dispatches the existing fnn-linear shader (via batchedLinear) and
returns a RESIDENT output buffer id. That id flows straight into record_op +
backward — the forward output is never computed in numpy and never re-uploaded.
Verified: forward_linear == numpy exactly (0.0); resident-forward logits -> CE
backward gives grad_W 6e-8, grad_X 7e-9 (logits never touched by numpy).
Remaining forward ops are mechanical: rms-norm, mingru-forward, activation-swiglu
shaders all exist — add forward_rmsnorm/forward_mingru/forward_swiglu the same way
(dispatch existing shader on registry buffers, return resident id, barrier).

## FULL TRAINING MILESTONE ? resident backward generalizes (DONE)
experimental/resident_train/train_full.py. One Cubby block, ONLINE training
(fresh random batch every step, so memorization is impossible and falling loss
== generalization). Task: bucket the length-L token sum into C quartile classes
(monotonic aggregation -- learnable + generalizable, unlike the first attempt
sum-mod-C which is adversarial to smooth approximators and only memorized).
  RESULT: train_acc 0.945 / test_acc 0.891 (chance 0.25); train & test track
  each other (both are held-out under online training). 400 AdamW steps, exit 0,
  no crash. `gradcheck` mode PASSES all 8 param groups vs finite-diff (rel_err
  <8e-3) -- validates the hand-stitched 3-tape backward routing (mean-CE /B,
  mean-pool /L, the two residual additions, embedding scatter np.add.at).
This is the end-to-end proof: the resident reverse-mode engine trains a fully
composed block to GENERALIZE on a real aggregation task, gradients verified.
grads() uses the t/t2/t3 multi-tape structure with NUMPY forward seeds.

## RESIDENT-FORWARD CRASH (0xC0000005) ? root cause + fix path
Symptom: train_full.py's original grads() (resident forward feeding the backward
tape) crashes with exit -1073741819 (access violation / heap corruption) inside
the first grads() call. Non-deterministic across trivial code edits -- classic
undefined-behavior / memory-stomp signature.
Bisection (experimental/resident_train/ttr*.py, file-based, exit-code-gated):
  - ttr8: the EXACT t/t2/t3 backward structure with NUMPY-forward seeds runs 3
    full iterations CLEAN (exit 0). => the backward engine is NOT the culprit,
    even multi-tape.
  - ttr4/ttr5: a SINGLE resident forward op (or rmsnorm + up to 3 linears) then a
    backward runs clean in isolation.
  - ttr3: resident forward (rmsnorm+linear) with NO read-back of the forward
    outputs, then a backward -> CRASH. Adding a read_buffer of the forward output
    (ttr6/ttr7) makes that specific crash vanish (and exposes a correctness bug:
    grads come back 0.0 when the forward output is not read back).
  => The stomping code is one of the resident FORWARD shaders/dispatches, and
     whether it faults depends on what the shared BufferPool packs adjacent to the
     forward output (padding = clean, live backward struct = access violation).
Architecture context: every gc.TapeContext(dev) SHARES ctx.pool / ctx.batch /
ctx.cache (bindings_autograd.cpp ctor TapeContext(ctx.pool,ctx.batch,ctx.cache)).
BufferRegistry (buffer_registry.h) hands owned buffers back to the shared pool on
clear(); PipelineCache has an LRU descriptor-set cache keyed on raw buffer handle
(suspect for stale-handle aliasing across tapes). grilly uses VMA ("VMA allocator
initialized" at init), so VMA debug margins are available.
Leading hypothesis (tail-end vectorized overwrite): forward_linear with N=24
dispatches gx=(24+15)/16 = 2 workgroups = 32 columns, but only 24 are valid -- if
the linear shader does not clamp the column index, every row writes 8 garbage
floats past its end. Invisible in isolation (lands on padding), fatal when the
pool packs a live backward struct after it.
FIX PATH (deterministic, do this to make forward resident):
  1. Build with VMA_DEBUG_MARGIN 64 + VMA_DEBUG_DETECT_CORRUPTION 1. If the crash
     vanishes -> proof of tail-end overflow; the corruption-detector pinpoints the
     overflowing allocation.
  2. Run with VK_LAYER_KHRONOS_validation + VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT
     for the exact shader line + binding index (no validation-layer hook in the
     instance-creation code yet -- would need adding, or set via env/vkconfig).
  3. Audit the forward shaders' global-write guards (fnn-linear, rms-norm,
     activation-swiglu) for index < element_count clamps; fix the offender.
  4. Separately: invalidate / key the PipelineCache descriptor LRU by buffer
     generation (not raw handle) so a reused pool handle can't bind a stale set.
Once forward is safe: rebuild train_full.py grads() to run forward resident and
chain the resident intermediate ids into backward -> fully on-GPU training step.

## (history) backward path + block composition
Backward is COMPLETE and the full block both gradient-checks and trains; see the
"Done + VERIFIED" and "Integration" sections above. Resident forward was opened
and per-op verified; the crash above is the remaining blocker to a fully-resident
step.

## Tests (run from grilly root, cubby-lm venv) ? ALL GREEN
  test_backward_linear.py   linear (grad_x, grad_W)
  test_backward_chain.py    2-layer Linear chain (propagation)
  test_backward_ce.py       cross-entropy + CE->Linear
  test_backward_silu.py     SiLU (relu/gelu share the helper)
  test_backward_fanout.py   fan-out accumulation (residual pattern)
  test_backward_rmsnorm.py  RMSNorm (grad_x, grad_w)
  test_backward_mingru.py   MinGRU (grad_g, grad_v, grad_d)
  test_backward_swiglu.py   SwiGLU (2*hidden input split)
  experimental/resident_train/bisect_block.py   cross-op composition (A-E)
  experimental/resident_train/train_block.py    full block gradcheck + train
  experimental/resident_train/train_full.py     full training: gradcheck | (no-arg) online train
    - gradcheck: validates stitched 3-tape backward vs finite-diff (8 params)
    - no-arg: 400-step online train, expect test_acc>0.5 (got ~0.89)
Run e.g.:
  & "C:\Users\grill\Documents\GitHub\cubby-lm\.venv\Scripts\python.exe" test_backward_mingru.py
Rebuild after C++ edits (add full, no flag, when a .glsl changes):
  powershell -NoProfile -ExecutionPolicy Bypass -File .\rebuild.ps1 -SkipShaders
