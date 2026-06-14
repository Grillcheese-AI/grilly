# TODO ? Road to a resident-trained Cubby

Cross-repo work board for the Cubby/grilly resident-training effort. This is the
ACTIONABLE checklist layer; the detailed source-of-truth docs are:
- grilly: `AUTOGRAD_STATE.md` (resident autograd internals + integration plan)
- grilly: `CLAUDE.md` (orientation map) / cubby-lm: `CLAUDE.md`
- cubby-lm: `cubby/ROADMAP.md` (the 0.0.x -> 0.1.0 version ladder)

Branch: `autograd-resident-backward` (grilly). Hardware: AMD RX 6750 XT / RADV.
Legend: [x] done+verified  [~] in progress / partial  [ ] not started
Priority: (P0) critical path to resident training  (P1) needed soon  (P2) later

---

## DONE ? verified on hardware

- [x] BufferRegistry (id <-> GrillyBuffer), deque-backed (stable resolve() refs)
- [x] Resident BACKWARD engine: linear, cross-entropy, silu/relu/gelu, add +
      fan-out (residuals), rmsnorm, mingru, swiglu ? all gradcheck vs numpy
- [x] backward() driver: single-batch residency, pull propagation, fan-out accum,
      the two transferComputeBarrier sync fixes
- [x] Resident FORWARD ops: linear, rmsnorm, swiglu, **mingru** (parity 1.5e-7),
      **embedding** (exact gather) ? full trunk-interior forward set complete
- [x] 0xC0000005 resident-forward crash root-caused + fixed (vector->deque)
- [x] Full resident training generalizes (train_full_resident.py: gradcheck PASS
      all 8 params; online train 0.945/0.891)
- [x] Push-constant warning investigated (rms-norm push=20 is correct; cause is in
      CommandBatch::dispatch layout) + [PIPE-CREATE] instrumentation removed
- [x] CLAUDE.md (both repos) + AUTOGRAD_STATE.md current; committed

---

## P0 ? RESIDENT TRUNK INTEGRATION  (the gating milestone)

The per-op resident forward+backward are all in place. Remaining: compose them
into ONE tape for Cubby's real architecture and switch cubby onto it. Build in
order; each step has a gate that must pass before the next.

- [x] **1. Single-tape full trunk forward+backward.** DONE+VERIFIED.
      `experimental/resident_train/train_trunk_lm.py`. Records, in ONE tape:
        embedding -> [ rmsnorm -> WG/WV/WD Linear -> mingru -> Add(residual) ->
                       rmsnorm -> Wg Linear -> swiglu -> Add(residual) ]xL ->
                       final rmsnorm -> tied head Linear(weight=E) -> per-token CE
      then calls backward() ONCE (no hand-stitched 3-tape split). Residuals are
      OpType.Add with fan-out across layers; per-token CE, NO mean-pool node (so
      the stubbed Sum/Mean backward stay off this path).
      GATE PASSED: gradcheck vs FINITE-DIFF at d=64, L=2 -> all 14 param groups
      rel_err ~1e-5 (<< 1e-2). Train sanity: loss 2.42 -> 0.001 in 40 AdamW steps.
      NOTE: uses 3 separate WG/WV/WD Linears, NOT the fused gvd Linear(d->3d) +
      slice -- mathematically identical, and the fuse (1 dispatch) is a step-3
      throughput optimization that needs a Slice backward op (not yet wired). The
      finite-diff reference runs in float64 (forward(...,f64=True)); float32 fd
      truncation on the saturating MinGRU path floored at ~2e-4 and produced
      spurious marginal fails -- float64 dropped the floor to ~1e-6.

- [x] **2. Tied embedding/head gradient merge.** DONE+VERIFIED (in the same
      train_trunk_lm.py). E is registered ONCE; the head Linear uses E as its
      weight so its weight-grad lands in E's grad slot; the embedding gather grad
      (host scatter-add from emb-leaf's grad) is ADDED to it.
      GATE PASSED: gradcheck on E with the tie (rel 5.2e-5) AND the `--untied`
      control (separate Whead; E gets embedding-grad only, rel 5.7e-5) both pass
      -- the merge is correct, not masking a sign/scale error.

- [x] **2b. Resident FORWARD on the single tape.** DONE+VERIFIED
      (train_trunk_lm.py `--resident`). The whole forward runs on-GPU
      (forward_embedding/rmsnorm/linear/mingru/add/swiglu) and feeds the same
      backward tape -- no activation leaves VRAM. Added NEW grilly op
      `TapeContext::forward_add` (non-destructive out=a+b for residuals; built
      from fillZero + batchedAdd, no new shader) -- the residual Add was the one
      op the "resident forward set complete" note actually lacked.
      GATE PASSED: resident logits parity vs numpy 1.07e-6; gradcheck all 14
      params rel ~1e-5 (TIED + --untied). Prereq for step 3 (numpy forward would
      force a per-step weight upload).

- [x] **3. Persistent resident weights + resident AdamW.** DONE+VERIFIED.
      BufferRegistry now has PERSISTENT entries (weights + Adam m/v) that survive
      begin()/clear() with a stable id (persistent_watermark_; clear() truncates
      the step-scoped suffix, keeping the persistent prefix -- deque suffix-erase
      preserves prefix refs). New ops: `register_weight` (persistent resident
      upload-once) + `adamw_update` (wraps adamw-update.glsl; no own submit, so
      all params update in ONE forward_begin/submit batch). Per-layer weights +
      moments stay resident across steps; resident AdamW updates them in place.
      GATES PASSED:
        - resident AdamW vs numpy AdamW: <9e-7 over 25 steps, persistent W/m/v
          (test_resident_adamw.py).
        - end-to-end (train_trunk_lm.py `--resident-opt`): the persistent-weights
          + resident-AdamW loss curve matches the numpy-AdamW reference to <4e-6
          over 40 steps (identical init/batch; same un-normalized grads, so the
          delta is purely optimizer impl). Even at toy d=64/L=2: 4 ms/step vs
          7 ms/step.
      SCOPE NOTE: E (tied embedding) stays on the host path (numpy AdamW + the
      embedding scatter) -- resident embedding BACKWARD is the deferred P1 op;
      E is tiny (V*d). The transfer-elimination win (no per-step weight upload /
      grad readback) is on the L per-layer weight matrices and SCALES with weight
      size, so the material wall-clock drop lands at d=1024/L=18 (step 4).

- [x] **4. Capacity for L=18.** DONE+VERIFIED. NO constant bumps needed:
      kMaxGradEntries=4096 vs ~290 grad entries for L=18; the 64 MB TapeArena is
      reset each begin() and holds one pass. train_trunk_lm.py `--big` records the
      full v3.3 trunk (V=65000, d=1024, L=18, 183 nodes) using 0.2 MB / 64 MB arena
      (0.26%) and backprops + runs resident AdamW with NO overflow/OOM; resident
      grads finite at step 1. (The toy random-65k-target task NaNs after a few
      steps -- training divergence on a DEGENERATE objective, not a capacity bug;
      see the real-data run below.)
      REAL-DATA VALIDATION (train_trunk_lm.py `--tinystories`): the full
      persistent-weights + resident forward/backward/AdamW stack trains a real LM
      on TinyStories (BBPE-65k, V=65536, d=256, L=6, B=8 S=64). CE 11.60 -> 4.08
      and perplexity 108697 -> 59.4 over 200 steps (~628 ms/step), monotone-ish
      descent, no NaN. Confirms the resident single-tape trunk LEARNS LANGUAGE --
      the v3.3-shape head (V=65k tied) + deep trunk are correct end to end.

- [ ] **5. Link into cubby-lm (parity-gated, non-destructive).**
      Add a resident execution path in `cubby-lm/cubby/trunk/` ALONGSIDE the
      existing Python-tape `model.py` (do not delete it). Wire `CubbyLM` forward
      to the resident TapeContext ops.
      GATE: forward parity vs the numpy trunk (max_abs_diff, per cubby port
      discipline) AND gradient parity; then a short TinyStories run matches the
      numpy-path loss curve. Only then flip the default.

---

## P1 ? open grilly items

- [ ] Push-constant VUID-vkCmdPushConstants-offset-01795: fix the wrong/stale
      layout in `CommandBatch::dispatch` (cpp/src/command_batch.cpp ~L94).
      Non-fatal today (forward numerically correct) ? clear it for a clean
      validation run.
- [ ] Resident embedding BACKWARD (scatter-add shader, OpType.Embedding) ? only if
      profiling shows the host scatter is a real cost. Cheap today; defer.
- [ ] Decide bf16 path for the trunk (config supports it) once fp32 resident
      trains; gate on parity + loss.

## P2 ? backward stubs (NOT on the 0.0.0 LM trunk path; implement when needed)

- [ ] backward_mul (dL/da=dy*b, dL/db=dy*a) ? needed only for elementwise products
      outside fused SwiGLU/MinGRU
- [ ] backward_sum / backward_mean / backward_transpose ? needed only if a
      reduction/transpose enters the autograd path (e.g. pooling)
- [ ] backward_layernorm / backward_softmax / backward_attention ? needed at 0.0.2
      (chunked sliding-window attention), not before

---

## CONV OPS (off the Cubby critical path; vision cortex is 0.1.0)

- [x] **conv2d FIXED** on the C++ core. Two dispatch bugs found + fixed in
      `cpp/src/ops/conv.cpp` (the GLSL shaders were correct):
        1. backward-weight grid mismatch (gx dropped the kH factor; gy had a
           spurious kH) -> wrong dW for any multi-channel / kH>1 kernel.
        2. forward GEMM output was (C_out, batch*HW) channel-major, not
           de-interleaved to (batch, C_out, HW) -> wrong forward for batch>1.
      Validated vs torch (fwd/dX/dW/dB ~1e-6, incl. batch>1, multi-channel,
      grouped). nn.Conv2d now routes through the C++ bridge; legacy ctypes path
      kept as fallback. xfail markers removed. Commit 0ae21e1.
- [x] **conv1d FIXED** for free (wraps Conv2d, height=1); validated vs torch.
- [ ] **conv3d: ADD (net-new).** Not present (`_core.conv3d` absent). Needs:
      conv3d-forward.glsl (+ backward-input / backward-weight) with a depth loop
      and depth push-constants; a `conv3d` C++ op in conv.cpp (direct path
      mirrors conv2d-forward, or extend im2col with a D axis for the GEMM path);
      `_core.conv3d` + backward bindings; an `nn.Conv3d` class (+ export in
      nn/__init__.py); tests. Full rebuild (shaders change -> no -SkipShaders).
      Validate each stage vs torch.nn.functional.conv3d (fwd, then dX, then dW).
- [ ] Test-infra: tests/test_conv2d.py uses a `Compute()` fixture (legacy ctypes
      backend) needing the `vulkan` pip package, absent in the C++-core venv ->
      the in-repo gpu conv tests don't execute there. Correctness was validated
      directly vs torch instead. Consider a bridge-based fixture.

## STRATEGIC ? VM-facing contracts to freeze EARLY (cubby-lm)

Not coupling the VM into trunk training (that repeats the v4 mistake), but pin the
shared interfaces now so the late CubeLang integration (0.0.8) is wiring, not
redesign. See the discussion captured around the binding head / VM.

- [ ] Freeze the VSA codebook spec: D=10240, MAP-bipolar, frozen, its seed.
- [ ] Freeze the (k,l) block-code projection seed (0xC0DEB00C) ? cross-process
      arena sharing depends on it never changing.
- [ ] Freeze the closed `OpcodeStmt` grammar the CubeLang head emits / the VM
      consumes.
- [ ] Keep `forward_features(tokens)->(B,S,D)` stable ? every later head attaches
      here.
- [ ] VM "walking skeleton": a standalone green test that hand-writes a 1-op
      CubeLang program over a toy world model, executes on the VM, asserts the
      result ? kept live, NOT in the training loop. De-risks the 0.0.8 big-bang.

---

## DOWNSTREAM ? cubby-lm version ladder

Tracked in `cubby-lm/cubby/ROADMAP.md`; the resident-trunk milestone above unblocks
the d=1024/L=18 run that 0.0.1 is gated on. High-level order (do NOT pull forward;
one validated component per rung, gate on generation quality):

- [~] 0.0.1 finish: resident throughput -> full v3.3-shape (d=1024,L=18,V=65k) run
- [ ] 0.0.2 chunked sliding-window attention (W=512, every 3rd layer)
- [ ] 0.0.3 sparse MoE-MinGRU
- [ ] 0.0.4 Hebbian growth
- [ ] 0.0.5 SegmentMemory
- [ ] 0.0.6 VSA binding head (uses the frozen codebook above)
- [ ] 0.0.7 MTP (decode-time only, frozen trunk)
- [ ] 0.0.8 CubeLang head + VM + WorldManager arena (wire to the walking skeleton)
- [ ] 0.0.9 MindForge adapter bank + cross-trunk fusion
- [ ] 0.1.0 afferent SNN gate (system integration)

---

## WORKING LOOP (Windows, this machine)

```powershell
$py = "C:\Users\grill\Documents\GitHub\cubby-lm\.venv\Scripts\python.exe"

# rebuild grilly after C++/header edits (-SkipShaders unless a .glsl changed):
Get-Process python | Stop-Process -Force        # unlock the .pyd
powershell -NoProfile -ExecutionPolicy Bypass -File ".\rebuild.ps1" -SkipShaders

# gradcheck / parity (run from grilly root or experimental/resident_train):
& $py experimental\resident_train\train_full_resident.py gradcheck
& $py experimental\resident_train\ttr_mingru_fwd.py
& $py experimental\resident_train\ttr_embedding_fwd.py

# validation layer (no rebuild needed) ? surfaces buffer / push-constant VUIDs:
$env:VK_INSTANCE_LAYERS="VK_LAYER_KHRONOS_validation"
```

Discipline: gradcheck composed graphs vs finite-diff (not "loss goes down"); the
sync races + tied-head merge are exactly the bugs only an fd check catches.
