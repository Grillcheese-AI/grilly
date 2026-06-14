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

- [ ] **1. Single-tape full trunk forward+backward.**
      Record, in ONE tape:
        embedding -> [ rmsnorm -> fused gvd Linear(d->3d) -> mingru ->
                       Add(residual) -> rmsnorm -> Linear -> swiglu ->
                       Add(residual) ]xL -> final rmsnorm -> tied head
                       Linear(weight=E) -> per-token CrossEntropy
      then call backward() ONCE (no hand-stitched 3-tape split).
      - Residuals are OpType.Add (real backward + fan-out accumulate ? present).
      - NOTE: per-token CE means NO mean-pool node (CE /N handles the mean), so
        the stubbed Sum/Mean backward are NOT on this path. Keep it that way.
      GATE: gradcheck vs finite-diff at small dims (d=64, L=2), rel < 1e-2, before
      trusting. Add `experimental/resident_train/train_trunk_lm.py` (gradcheck mode).

- [ ] **2. Tied embedding/head gradient merge.**
      Register E once; the head Linear's weight-grad accumulates into E's grad
      slot; the embedding grad (host scatter-add from the grad flowing into the
      emb-output buffer) is ADDED to it.
      GATE: gradcheck on E specifically passes with the tie (vs an untied control).

- [ ] **3. Persistent resident weights + resident AdamW.**
      Keep params + Adam moments resident across steps; run the update on resident
      grad+weight buffers via `adamw-update.glsl` (exists). No per-step weight
      upload / grad readback.
      GATE: step time drops materially vs the numpy-AdamW path; loss curve matches
      the numpy-AdamW reference for N steps. THIS is the 0.0.1 perf #2 payoff.

- [ ] **4. Capacity for L=18.**
      Raise TapeArena capacity; verify BackwardEngine `kMaxGradEntries` (4096) and
      the BufferRegistry deque hold an 18-layer trunk's node/buffer/grad count.
      GATE: full d=1024/L=18 trunk records + backprops without overflow/OOM.

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
