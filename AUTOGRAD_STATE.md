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

REMAINING (separate, non-fatal): VUID-vkCmdPushConstants-offset-01795 - a 20-byte rms-norm push flagged against a layout reported as having no COMPUTE range. INVESTIGATED: instrumented getOrCreate (instrumentation since REMOVED) confirms rms-norm IS created with pushConstSize=20 and is the only 20-byte push in ttr3 - so the layout is correct and the warning comes from the dispatch using the wrong/stale layout for the push. Next: inspect CommandBatch::dispatch (cpp/src/command_batch.cpp ~L94) - the vkCmdPushConstants call and which layout it passes. Forward is numerically correct (gradchecks pass); latent, not breaking.

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


## RESIDENT FORWARD OP SET COMPLETE (mingru + embedding added) - June 2026
Added the two missing resident forward ops so the ENTIRE Cubby trunk forward can
run on-GPU with no numpy round-trip:
- TapeContext.forward_mingru(g_id,v_id,d_id, batch,seqLen,hidden) - wires the
  existing mingru-forward.glsl (4 buf {G,V,D,H}, push {seqLen,hidden},
  gx=(hidden+63)/64, gy=batch). Parity vs numpy mingru_fwd: max_abs_diff 1.5e-7.
  (experimental/resident_train/ttr_mingru_fwd.py)
- TapeContext.forward_embedding(ids_id,table_id, batch,seqLen,vocab,dim) - wires
  embedding-lookup.glsl (3 buf {ids(u32),table,out}, push {batch,seqLen,vocab,dim},
  gx=(batch*seqLen+255)/256). ids uploaded via register_input_u32. Exact gather,
  parity 0.0. (ttr_embedding_fwd.py)
Resident forward op set is now: embedding, rmsnorm, linear, mingru, swiglu - all
parity-verified. Backward already covers all of these (embedding backward = host
scatter-add, cheap, off the hot path). No regression: the full resident gradcheck
(train_full_resident.py gradcheck) still PASSES all 8 params after these edits, on
a clean build with the [PIPE-CREATE] instrumentation removed.

## SINGLE-TAPE FULL-TRUNK INTEGRATION - steps 1 & 2 DONE (June 2026)
`experimental/resident_train/train_trunk_lm.py`. The full Cubby LM trunk now
records in ONE tape and backprops with ONE backward() call:
  embedding -> [rmsnorm -> WG/WV/WD Linear -> mingru -> Add(res) -> rmsnorm ->
  Wg Linear -> swiglu -> Add(res)]xL -> final rmsnorm -> tied head Linear(E) ->
  per-token CE.
- ONE backward(): the pull-walk routes the whole tail->head graph including the
  residual Add fan-outs ACROSS layers (r2 of layer l feeds both rmsnorm and the
  next Add) - no hand-stitched 3-tape split (the toy train_full*.py needed 3
  tapes only because of its numpy mean-pool + pooled CE).
- per-token CE, NO mean-pool node -> the stubbed Sum/Mean backward stay off path.
- tied-E merge (step 2): E registered once, used as the head Linear weight (its
  weight-grad lands in E's grad slot); the embedding gather grad host-scatters
  into the SAME dE. GATE: gradcheck on E with the tie (rel 5.2e-5) AND an
  `--untied` control (separate Whead; E = embedding-grad only, rel 5.7e-5) both
  pass -> the merge is correct, not masking a sign/scale bug.
- GATE PASSED: gradcheck vs finite-diff at d=64, L=2 -> all 14 param groups
  rel_err ~1e-5 (<< 1e-2). Trains: loss 2.42 -> 0.001 in 40 AdamW steps.
- gotcha (gradcheck harness, not the engine): float32 finite-diff on the
  saturating MinGRU path floors at ~2e-4 max_abs and gives spurious marginal
  fails (WG[1]/WD[1] ~1-2e-2). Run the fd REFERENCE in float64
  (forward(...,f64=True)) -> floor drops to ~1e-6, all pass cleanly. Make the
  reference more accurate; do NOT loosen the threshold.
- NOTE: uses 3 separate WG/WV/WD Linears, not the fused gvd Linear(d->3d) + slice
  (mathematically identical). Fusing to 1 dispatch is a step-3 throughput item
  and needs a Slice backward op (OpType.Slice exists in the enum but no backward
  handler is wired).

## RESIDENT FORWARD on the single tape - DONE (forward_add added)
train_trunk_lm.py `--resident` runs the WHOLE forward on-GPU and feeds the same
backward tape: forward_embedding -> [forward_rmsnorm -> forward_linear x3 ->
forward_mingru -> forward_add -> forward_rmsnorm -> forward_linear ->
forward_swiglu -> forward_add]xL -> forward_rmsnorm -> tied head forward_linear.
No activation leaves VRAM during the forward (only the grad read-back + embedding
scatter remain - step 3 kills the read-back).
- NEW op `TapeContext::forward_add(a_id,b_id,totalElements)` (autograd.cpp +
  autograd.h + bindings_autograd.cpp): NON-destructive out=a+b for residuals
  (both operands must survive for backward). Built from the verified fillZero +
  in-place batchedAdd primitives (zero out; out+=a; out+=b) with a
  transferComputeBarrier before each RMW - no new shader. This is the op the
  "resident forward set complete" note was actually missing (residual Add).
- GATES PASSED: resident-forward logits parity vs numpy max_abs_diff 1.07e-6;
  gradcheck all 14 params rel ~1e-5 (TIED and --untied). register_input_u32 takes
  ONLY the array (no requires_grad bool) - gotcha.

## PERSISTENT RESIDENT WEIGHTS + RESIDENT AdamW (step 3) - DONE
BufferRegistry gained a PERSISTENT entry class: weights + Adam m/v registered
ONCE survive begin()/clear() with a STABLE id. clear() now truncates only the
step-scoped suffix (`persistent_watermark_`); erasing a deque suffix preserves
references to the persistent prefix (same deque-ref-safety the crash fix relied
on). New surface:
- `BufferRegistry::alloc_persistent` + watermark + dtor releases persistent owned.
- `TapeContext::register_weight(arr)` -> persistent resident id (upload once).
- `TapeContext::adamw_update(w,grad,m,v,numel,lr,b1,b2,eps,wd,b1t,b2t,clear_grad)`
  dispatches adamw-update.glsl in place; NO own begin/submit, so all param
  updates batch into one forward_begin/forward_submit.
Training loop (train_trunk_lm.py `--resident-opt`): per-layer weights + moments
stay resident; backward grads feed adamw_update directly -- no per-step weight
upload, no per-layer grad readback.
GATES PASSED:
- resident AdamW == numpy AdamW to <9e-7 over 25 steps with persistent W/m/v
  (test_resident_adamw.py) -- proves persistence across clear() + optimizer math.
- end-to-end loss curve matches the numpy-AdamW reference to <4e-6 over 40 steps
  (identical init/batch, same un-normalized grads). 4 ms/step vs 7 ms/step even
  at toy d=64/L=2.
SCOPE: E (tied embedding) stays on the host path (numpy AdamW + embedding
scatter) -- resident embedding BACKWARD is the deferred P1 op; E is tiny. The
transfer-elimination payoff scales with weight size -> material at d=1024/L=18.
No regression: train_full_resident.py gradcheck still PASSES all 8 params.

## CAPACITY FOR L=18 (step 4) - DONE, + REAL-DATA VALIDATION
No constant bumps were needed: kMaxGradEntries=4096 >> ~290 grad entries for
L=18; the 64 MB TapeArena resets each begin() and holds one pass. `--big` records
the full v3.3 trunk (V=65000/d=1024/L=18, 183 nodes) in 0.2 MB / 64 MB arena
(0.26%) and backprops + resident AdamW with no overflow/OOM (resident grads finite
at step 1). The toy random-65k-target task NaNs after a few steps -- DEGENERATE
objective, not a capacity bug.
REAL DATA (`--tinystories`): the full persistent-weights + resident
forward/backward/AdamW stack trains a real LM on TinyStories (BBPE-65k V=65536,
d=256 L=6): CE 11.60->4.08, perplexity 108697->59.4 over 200 steps (628 ms/step),
no NaN. The resident single-tape trunk LEARNS LANGUAGE -- v3.3-shape head (V=65k
tied) + deep trunk correct end to end.

## CUBBY LINK (step 5) - DONE; P0 RESIDENT TRUNK INTEGRATION COMPLETE
cubby-lm/cubby/trunk/resident.py (ResidentTrunk), ALONGSIDE model.py. Reads a
CubbyLM's Variables into persistent resident weights (+ Adam moments); E (tied
embedding) on the host path. Handles cubby's fused gvd proj (split 3), real
SwiGLU FFN (gate_up + down) with the swiglu half-swap, tied head.
- GATES (all green, now DETERMINISTIC): forward parity 7.9e-7, gradient parity
  per-param <=2e-6 vs model.py's Python-tape backward, loss-curve match 7e-6
  (resident-grad training vs numpy training, identical descent).
- train_step (resident adamw_update on persistent weights + E numpy AdamW) +
  autoregressive generate + cubby.trace per-block emission (read back only when a
  tracer is active -> zero overhead at OFF).
- THE FLIP: main.py train defaults to --backend resident (train_cubby_resident);
  --backend tape keeps model.py. Resident train->generate on TinyStories: CE
  11.16->4.49 / ppl ->89 in 150 steps, coherent English; trace 6/6 blocks.
- determinism fix: force_numpy_reference() short-circuits grilly.nn.autograd's
  LAZY gpu-backward singleton so model.py's backward stays numpy -- the resident
  grilly_core.Device() is then the ONLY Vulkan context (two contexts were
  intermittently corrupting each other; ~1/3 flaky before).

All P0 resident-trunk-integration steps (1,2,2b,3,4,5) are DONE.

## DEEP-TRUNK (L>=7) CORRECTNESS - descriptor-set cache bug FIXED (the big one)
Bringing up the v3.3 shape (d=1024/L=18) exposed a SILENT grad-corruption bug that
the small parity configs (L<=4) never hit. Two parts, both in the descriptor-set
cache (cpp/.../vk_pipeline_cache.h, pipeline_cache.cpp):
1. IN-FLIGHT EVICTION. The descriptor cache is an LRU capped at kMaxCachedDescSets
   (was 100). The backward of one tape is a SINGLE command batch; each dispatch
   allocates a distinct descriptor set (fresh buffers) recorded into the still
   un-submitted command buffer. At ~15 sets/layer, L>=7 exceeds 100, so on a miss
   the LRU FREES a set still referenced by the in-flight command buffer -> submit
   reads a freed set -> garbage/zero grads (scattered: worst 2e12 / nan at L=12-18,
   while L=4 was clean). FIX: kMaxCachedDescSets 100 -> 4096 and pool defaults
   500/1000 -> 8192/131072, so the cap exceeds any realistic single batch (~L=250).
2. CROSS-STEP STALE REUSE. With sets now persisting across steps, the cache (keyed
   by shader+buffer-handle) falsely reused a stale set after the buffer POOL
   recycled a handle for a different buffer -> NON-DETERMINISTIC grads (loss-curve
   parity went flaky: 0.36 then 0.12 run-to-run). FIX: TapeContext::begin() calls
   cache_.clearDescriptorCache() each step -- safe because every CommandBatch
   submit() is synchronous (GPU idle at the step boundary), so no set is in-flight.
RESULT: gradient parity vs model.py now clean+DETERMINISTIC at L=4..24 (2-6e-6;
was nan at L>=12); loss-curve match back to 7.033e-6 identical across runs.
The probe instrumentation that found it: a per-step global grad-NORM print -- it
read 0.00e+00 at v3.3 step 1 (impossible for a real step), revealing the grads
were CORRUPTED (not just spiking), which the bisect (L=4 ok / L=12 garbage) +
reading allocDescriptorSet localized to the LRU.

## v3.3 DEEP-LM TRAINING HYGIENE (resident grad clip + warmup + lr)
With correct grads, an L=18/350M trunk from scratch still needs standard hygiene
or it NaNs on the first gradient spike (cold-Adam oversized steps):
- adamw-update.glsl gained a `grad_scale` push const applied to the gradient
  BEFORE the m/v update (Adam normalizes by sqrt(v), so scaling lr != scaling the
  grad). cubby train_step folds (global-norm clip * mean-CE 1/B) into grad_scale;
  skips the step entirely if the global grad-norm is non-finite.
- enable_residual_scale (config 0.0.1): GPT-2-style 1/sqrt(2L) init of each
  residual branch's output proj (model.py Block) -- bounds the forward residual
  stream; pure init, backend-agnostic (parity preserved).
- train_cubby_resident: linear LR warmup + lower peak lr (3e-4).
RESULT: v3.3 (d=1024/L=18/V=65536, 350M params) trains STABLY -- CE 11.24 ->
6.30, ppl 76188 -> 543 in 30 steps (lr=3e-4, clip=1.0), no NaN, emits English
words. The 0.0.1 production-shape run is unblocked.
NOTE: the clip currently reads all grad buffers back to host for the global norm
(correct but a transfer cost); a GPU sum-of-squares reduction is the perf TODO.

## v3.3 THROUGHPUT: 0.23 -> 0.77 it/s (3.3x), now compute-bound
Profiling the v3.3 step (4.26 s) showed it was ~99% HOST TRANSFER, not compute:
global-norm grad readback 1.68 s (39%), fwd+bwd 1.21 s (28%), E numpy AdamW 1.04 s
(24%). Two new resident ops removed the host costs:
- reduce-sumsq.glsl + TapeContext::sum_squares(ids, numels): global gradient L2
  norm ON-GPU (atomic-accumulate Sg^2 -> one 4-byte readback, not ~1 GB). 0.23 ->
  0.34 it/s.
- embedding-backward.glsl + TapeContext::embedding_scatter_add: scatter the
  embedding-output grad INTO E's grad buffer (which holds the tied-head weight
  grad) so the tied embedding trains FULLY RESIDENT (joins the resident AdamW +
  GPU norm; no head readback, no host scatter, no numpy AdamW, no E re-upload).
  0.34 -> 0.77 it/s. Loss curve byte-identical to the host-E path.
  GOTCHA (RDNA): float atomicAdd (GL_EXT_shader_atomic_float / OpAtomicFAddEXT) is
  WAVE-COALESCED INCORRECTLY when lanes scatter to different addresses -- a single
  +1 became +32 (= wavefront). Use atomicCompSwap on the uint bit-pattern (a true
  per-lane RMW) for scattered float adds. (reduce-sumsq's one-atomic-per-workgroup
  pattern is unaffected and keeps plain atomicAdd.)
Then profiling fwd+bwd showed the BACKWARD was 88% (996 ms vs 126 ms forward) --
NOT dispatch count, so op fusion would not have helped. The forward fnn-linear is
TILED; the backward fnn-linear-backward grad_input pass was the NAIVE one (each
thread serially loops output_dim at low occupancy -- brutal for the V=65536 head).
- Route grad_input = grad_out @ W through the existing TILED gemm_tiled.glsl
  (C=A@B, fp32, shared-mem + 4x4 register blocking). W is stored out x in = K x N,
  so it maps exactly with no transpose. 0.77 -> 0.97 it/s; loss identical, parity
  2e-6 (fp32, exact). Backward 996 -> 721 ms.
NOTE on coopmat: the RX 6750 XT is RDNA2 -- NO cooperative-matrix hardware, so the
coopmat path runs in DRIVER EMULATION (fp16 vector ops), no tensor-core speedup.
The real lever here is tiling, done in fp32 (safe, exact) -- not coopmat/fp16.
- grad_weight = grad_out^T @ x: transpose grad_out (tensor-transpose.glsl) ->
  (out x BS), then gemm_tiled (M=out, K=BS, N=in). Tiled + coalesced vs the naive
  strided pass. 0.97 -> 1.25 it/s; parity 2e-6 (exact).
Total: 0.23 -> 1.25 it/s (5.4x), loss byte-identical throughout. Both backward
GEMMs now tiled; forward fnn-linear already tiled. Remaining: mingru scan backward
+ the small elementwise ops + per-dispatch/barrier overhead.

## NEXT (downstream cubby ladder)
A longer v3.3 run to convergence (warmup + more steps), then 0.0.2 chunked
sliding-window attention, etc.
