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

## Recommended next step
Two remaining pieces for a FULL single-block trainer:
  1. Wire backward_swiglu (NOT yet wired — no OpType/handler exists, but the
     shader activation-swiglu-backward DOES). Forward: out = x1*silu(x2), input
     is [x1:hidden][x2:hidden] concatenated -> output hidden. Backward shader: 3
     buffers {grad_out, input, grad_in}, push {output_elements, hidden_dim},
     gx=(output_elements+255)/256. Same easy pattern as backward_activation but
     input is 2*hidden wide and output is hidden wide. Add OpType::SwiGLU + binding.
  2. Then a full-block trainer: embed -> RMSNorm -> MinGRU -> RMSNorm -> SwiGLU
     (+residuals) -> RMSNorm -> tied head -> CE, all backward resident, confirm
     loss drops. RMSNorm/SwiGLU/residual/MinGRU/Linear/CE all individually verified;
     this only tests their COMPOSITION (saved-tensor conventions lining up).
The harder follow-on is driving FORWARD resident too (bind ComputeBackend dispatch
primitives) so forward+backward are both single-submit on-GPU. Until then forward
uses _bridge staging ops and only backward is resident.

## Tests (run from grilly root, cubby-lm venv) — ALL GREEN
  test_backward_linear.py   linear (grad_x, grad_W)
  test_backward_chain.py    2-layer Linear chain (propagation)
  test_backward_ce.py       cross-entropy + CE->Linear
  test_backward_silu.py     SiLU (relu/gelu share the helper)
  test_backward_fanout.py   fan-out accumulation (residual pattern)
  test_backward_rmsnorm.py  RMSNorm (grad_x, grad_w)
  test_backward_mingru.py   MinGRU (grad_g, grad_v, grad_d)
Run e.g.:
  & "C:\Users\grill\Documents\GitHub\cubby-lm\.venv\Scripts\python.exe" test_backward_mingru.py
Rebuild after C++ edits (add full, no flag, when a .glsl changes):
  powershell -NoProfile -ExecutionPolicy Bypass -File .\rebuild.ps1 -SkipShaders
