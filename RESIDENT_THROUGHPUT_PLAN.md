# Resident-trunk throughput + auto-LR plan

Target: the cubby-lm resident training step (`cubby/trunk/resident.py::ResidentTrunk.train_step`).
The path is **dispatch-bound** (per-step wall-clock ~ number of GPU dispatches, not
compute/transfer). Measured shape: mbpe_v33 L=8, d=1024, V=32768, B=4 S=128 →
~1.39 it/s, ~712 tok/s.

## Per-step dispatch budget (~295 dispatches)

| phase | dispatches | source |
|---|---|---|
| forward | ~113 | `_resident_forward` (11/layer ×5 + 18/attn-layer ×3 + embed/head/router) |
| backward | ~110 | `t.backward(nCE)` |
| **AdamW** | **67** | **one `adamw_update` per param**, Python loop (resident.py:280) |
| sum_squares / scatter | 2 | already fused ✓ |
| logits readback | — | **67 MB GPU→CPU per step** (resident.py:397) + host softmax |

## Work items (ranked)

### 1. CE loss-scalar via `loss-ce-fused.glsl` (kills the 67 MB readback)
`loss-ce-fused.glsl` already computes per-row `losses[BS]` AND `grad_logits` in ONE
dispatch (on-chip subgroup reduction, no full softmax to VRAM). It is NOT wired in:
`BackwardEngine::backward_cross_entropy` (autograd.cpp:876) uses the older
grad-only `cross-entropy-backward` shader.
- **grilly:** in `backward_cross_entropy`, swap to `loss-ce-fused`; allocate a
  persistent `losses[BS]` buffer on the CrossEntropy node and expose its id
  (e.g. `TapeContext::last_ce_losses()` or return via the node). Gradient is
  identical (softmax − onehot), so this is drop-in.
- **cubby:** in `_fb_run`/`train_step`, stop `read_buffer(logits, [BS,V])`; read
  back `losses[BS]` (~2 KB) and report `losses.mean()`. Drop the numpy
  `_ce`/`_sampled_ce`/`_dual_head_ce` from the hot path (reporting-only today —
  see finding below). Optionally reduce `losses` on GPU to one float.
- **win:** ~67 MB→2 KB readback/step + removes host softmax; reported loss
  becomes the TRUE full-softmax CE.

### 2. Fused AdamW (67 → 1 dispatch)
Today: `for p in self.opt: t.adamw_update(...)` = one dispatch/param.
- **Option A (least invasive):** multi-buffer `adamw_update_multi(weights[],
  grads[], ms[], vs[], ns[], hparams)` mirroring the existing multi-buffer
  `sum_squares` (resident.py:270) — one dispatch over a descriptor array.
- **Option B (optimal):** allocate ALL opt params into one contiguous persistent
  block (grad/m/v likewise) at registration (`buffer_registry` alloc_persistent),
  with per-param offsets; AdamW is then literally one dispatch over the whole
  block. Bigger refactor, best result; also makes the hypergradient reduction (#3)
  trivial (single buffer).
- **win:** ~66 fewer dispatches/step (~22% of the budget).

### 3. Hypergradient auto-LR (convergence; stacks on #2)
Use `optim/hypergradient.py`'s OSGM scheme (arXiv:2502.11229) as the design, but
keep weights resident — only the LR scalar adapts.
- Per step the global hypergradient needs two scalars: `⟨g_k, d_{k-1}⟩` and
  `‖d_{k-1}‖²`, where `d = m̂/(√v̂+eps)` is the Adam update direction.
- **grilly:** persist `d_prev` (one buffer, or reuse the contiguous block from #2);
  add a fused dot-product reduction `dot(g_k, d_prev)` (mirror `sum_squares`).
- **cubby:** `lr_k = lr_{k-1} − β_hyper · ⟨g_k,d_{k-1}⟩ / (‖d_{k-1}‖²+eps)`
  (β_hyper auto-stabilized per OSGM); feed `lr_k` into the existing resident
  `adamw_update`. ~1 extra reduction/step, no weight readback.
- **win:** removes manual LR tuning; faster convergence. `current_surprise`
  (grad-prediction error) is reusable for the SNN afferent gain later (0.1.0).

### 4. gvd projection fusion (3 → 1 forward + 3 → 1 backward / layer)
`_resident_forward` runs G/V/D as three separate `(d,d)` linears
(resident.py:344-346) because there is no slice of the fused `(3d,d)` output.
- **grilly:** `forward_mingru` variant consuming a packed `(BS,3d)` tensor
  (G=[:, :d], V=[:, d:2d], D=[:, 2d:3d]); + matching backward.
- **win:** ~16 fwd + ~16 bwd dispatches saved at L=8.

### 5. bf16 weights/activations (AFTER 1–4)
Won't speed ops (dispatch-bound; fp16 already null there) but halves activation
memory → N=1024 fits the fast regime instead of the VRAM-bound 0.7 it/s wall →
~2× tok/s. Payoff only after the dispatch count drops.

## Finding: dual-head / sampled-softmax / router are no-ops in the resident path
- The gradient is **entirely** the GPU full-softmax `CrossEntropy` over the
  combined 32768 vocab (resident.py:464). `use_sampled` and the dual-head/router
  weighting change only the **printed** loss, not the gradient. So sampled softmax
  buys ZERO compute here (forward still materializes full `(BS,V)` logits).
- The **router is frozen**: excluded from `self.opt` AND its `nRouter` branch is a
  dangling output `backward(nCE)` never traverses → grad 0. `w_lang` stays ≈0.5,
  which is why the printed `ppl` reads ~2× low.
- Consequence: #1 both speeds up the step AND makes the reported loss honest. A
  REAL sampled/dual-head win needs a GPU sampled-CE kernel (compute only K+1
  logits) — worth it only if V grows well past 32k.

## Suggested order
#1 (readback) → #2 (fused AdamW) → #3 (hypergradient) → #4 (gvd) → #5 (bf16).
All require a grilly rebuild; do while the trunk run is idle (shared 12 GB).
