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

## RESIDENT-FORWARD CRASH (0xC0000005) ? ROOT-CAUSED + FIXED
Symptom: resident forward feeding the backward tape crashed with exit -1073741819
(access violation) inside grads(). Non-deterministic across trivial edits ? a
heap-corruption signature.

ROOT CAUSE (confirmed with VK_LAYER_KHRONOS_validation ? installed; enable via
env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation, no code hook needed):
validation flagged vkCmdDispatch storage buffer descriptor ["GradOutput"] using
VkBuffer 0x0 "invalid or has been destroyed", and vkUpdateDescriptorSets Invalid
VkBuffer 0x706172742eadc26e. That hex decodes to ASCII "part." ? a VkBuffer handle
field overwritten with string data, i.e. a DANGLING C++ REFERENCE. BufferRegistry
stored entries in std::vector and resolve() returns GrillyBuffer&. The dispatch
code resolves several buffers into references, then alloc()s a grad buffer; that
push_back reallocated the vector and dangled the held references, so .handle read
freed memory. Resident forward adds enough owned buffers (forward outputs +
rms_vals + re-registered intermediates) to cross a realloc boundary DURING backward
that numpy-forward (ttr8) never hit ? exactly why ttr8 was clean and resident
forward crashed.

FIX (committed): std::vector<Entry> -> std::deque<Entry> in buffer_registry.h.
deque::push_back invalidates iterators but NOT references to existing elements.
One word, zero behavior change. After the fix: ttr3 shows ZERO invalid-VkBuffer
errors under validation, and the fully-resident step trains (below).

Note on the earlier VMA_DEBUG_MARGIN experiment: margin made ttr3 stop crashing,
which looked like proof of a tail overflow ? it was a RED HERRING. Margin only
shifted heap layout so the dangling read landed on non-fatal memory; it never
fixed the full BL=192 training (still crashed with margin). The shaders
(fnn-linear, rms-norm, activation-swiglu) were all correctly bounds-clamped all
along. Lesson: "margin makes it vanish" is necessary-not-sufficient for overflow;
confirm with validation before concluding. The margin was reverted.

REMAINING (separate, non-fatal): validation also shows
VUID-vkCmdPushConstants-offset-01795 ? the rms-norm dispatch pushes 20 bytes to a
pipeline layout whose push-constant range does not cover them. Forward is
numerically correct regardless (gradchecks pass), so it is latent, not breaking.
Likely a getOrCreate("rms-norm", ...) push-size vs RMSNormParams mismatch. Fix next.

## FULLY RESIDENT TRAINING ? works post-fix (DONE)
experimental/resident_train/train_full_resident.py ? same block + online task as
train_full.py, but grads() runs the forward resident (forward_rmsnorm /
forward_linear / forward_swiglu on-GPU) and feeds the backward tape, no numpy
forward. RESULT after the deque fix: gradcheck PASSES all 8 params (rel_err
<4e-3); 400-step online training exits 0 with train_acc 0.945 / test_acc 0.891
(identical to the numpy-forward milestone ? resident forward computes the same
values). This is the fully-on-GPU training step. (train_full.py keeps numpy
forward as the simple stable baseline; both are committed and pass.)

## (history) backward path + block composition
Backward is COMPLETE and the full block both gradient-checks and trains; see the
"Done + VERIFIED" and "Integration" sections above. Resident forward is opened,
per-op verified, AND the dangling-reference crash that blocked feeding it into the
backward tape is fixed (deque) ? the fully-resident step now trains (see "FULLY
RESIDENT TRAINING").

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
  experimental/resident_train/train_full.py     full training (NUMPY forward + resident backward): gradcheck | (no-arg) train
    - gradcheck: validates stitched 3-tape backward vs finite-diff (8 params)
    - no-arg: 400-step online train, expect test_acc>0.5 (got ~0.89)
  experimental/resident_train/train_full_resident.py  SAME but RESIDENT forward (on-GPU); trains post deque-fix (~0.89)
  Validation: env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation surfaces invalid-buffer / push-constant VUIDs (layer installed).
Run e.g.:
  & "C:\Users\grill\Documents\GitHub\cubby-lm\.venv\Scripts\python.exe" test_backward_mingru.py
Rebuild after C++ edits (add full, no flag, when a .glsl changes):
  powershell -NoProfile -ExecutionPolicy Bypass -File .\rebuild.ps1 -SkipShaders
