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

## STILL STUBBED / MISSING (the next slices)
- MinGRU: mingru-backward.spv ALREADY EXISTS. MinGRU is NOT an OpType; wire it as
  its own standalone op/handler (resolve, dispatch mingru-backward, publish), not
  through the OpType switch. This is a wiring slice, not a kernel-authoring slice.
  THIS IS THE LAST TRUNK PIECE.
- backward_mul (dL/da = dy*b, dL/db = dy*a) still `= 1` placeholders. Only needed
  if the trunk uses elementwise products outside SwiGLU.
- backward shape ops: sum/mean/transpose still `= 1` placeholders. Only needed if
  the trunk reduces/transposes on the autograd path.
- backward_layernorm still stubbed (no layernorm-backward shader). Cubby uses
  RMSNorm, not LayerNorm, so not on the critical path.

## Trunk readiness
ALL per-op KERNELS the Cubby trunk needs now exist and are verified: linear, CE,
silu/relu/gelu, add+fanout (residuals), rms-norm-backward. Plus driver
propagation + loss seeding + single-batch residency. The ONLY remaining trunk
piece is wiring MinGRU backward (its shader exists). After that the full
embed -> [RMSNorm -> MinGRU -> RMSNorm -> SwiGLU]xL -> RMSNorm -> tied head -> CE
backward path is closed and the trunk can train resident.

## Forward side (separate, not this branch)
Forward resident execution already works in C++: nn::Tensor + ComputeBackend
graphMode (beginBatch/dispatch/endBatch -> OpGraph.optimize()+execute(), single
fused submit). cubby-lm currently uses the NON-resident _bridge staging ops
(grilly::ops::linear, CPU-ptr in/out). Driving resident forward from Python would
need the ComputeBackend dispatch primitives bound (createBuffer/upload/dispatch/
beginBatch/endBatch/setGraphMode) — NOT done yet.

## Recommended next step
Wire MinGRU backward — the last trunk piece. mingru-backward.spv already exists;
read the forward mingru op (cpp/src/ops/mingru.cpp) + mingru-backward.glsl for the
buffer layout/params, then wire a handler that resolves the saved forward state,
dispatches mingru-backward on resident buffers, and publishes grad ids. MinGRU is
NOT an OpType (the scan isn't pointwise) — give it a dedicated path. After that,
build the first end-to-end resident trunk micro-step (one block + CE) and confirm
loss decreases over a few steps.

## Tests (run from grilly root, cubby-lm venv)
  & "C:\Users\grill\Documents\GitHub\cubby-lm\.venv\Scripts\python.exe" test_backward_linear.py
  ...test_backward_chain.py
  ...test_backward_ce.py
Rebuild after C++ edits:
  powershell -NoProfile -ExecutionPolicy Bypass -File .\rebuild.ps1 -SkipShaders
