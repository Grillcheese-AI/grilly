# HYPOTHESES — Event-driven SNN synaptic propagation (Tier 0)

## Claim under test
On the RX 6750 XT (RADV), replacing dense synaptic propagation
`I = spikes · W` (O(N²)/step) with an event-driven scatter over a compact
fired-neuron list reduces weight-read traffic in proportion to spike activity,
**without** requiring an indirect-dispatch path in `grilly_core`.

## Mechanism
1. `gif_neuron_emit.glsl` — unchanged GIF dynamics; on spike, `atomicAdd` the
   neuron index into `fired_idx` (counter in `fired_count`).
2. `synapse_scatter.glsl` — dispatched at worst-case grid `ceil(N*N/256)`; each
   invocation early-outs on `t >= fired_count[0]*N` before reading `W`. Only
   `fired*N` invocations touch weights / hit the float atomic.

## Predictions
- **P1 (correctness):** scatter result equals `spikes · W` to fp32 rounding
  (`max|Δ| < 1e-2`). — *reference test*
- **P2 (scaling):** weight-read traffic = `fired·N`, i.e. linear in activity;
  at biological activity (1–5%) that is a 20–100× read reduction. — *reference test*
- **P3 (on-GPU, pending):** measured Kernel-B time scales sub-linearly toward
  the activity floor; crossover where Tier-0 launch overhead dominates the dense
  matmul is at high activity / small N.

## Kill criterion
If on-GPU Kernel-B time does **not** fall below the dense matmul at ≤5% activity
for N ≥ 1024 (P3 fails), the energy story for SNN-on-this-GPU is launch/atomic-
bound, not compute-bound → event-driven is not worth the indirect-dispatch
surgery; keep SNN as a fidelity component and let VSA carry the efficiency claim.

## Results
- [x] Shaders written (`gif_neuron_emit`, `synapse_scatter`, `synapse_dense` baseline)
- [x] SPIR-V compile (glslc): both exit 0; GL_EXT_shader_atomic_float accepted.
- [x] Reference (NumPy): P1 PASS max|dense-sparse|=3.4e-5(N=1024)/1.45e-4(N=4096);
      P2 PASS reads=fired*N, linear in activity (1%->100x, 5%->20x).
- [x] Device+atomics probe: RX 6750 XT inits; VK_EXT_shader_atomic_float[2] ENABLED
      on-device. Float scatter supported.
- [x] On-GPU A/B timing (P3): DONE. Registered _core.spike_scatter op (gather
      form, no atomics — float atomicAdd gave wrong results despite the ext being
      enabled). Correct to fp32 (3.8e-6 / 3.05e-5 vs spikes@W).
      VERDICT: event-driven kernel BEATS dense at sparse activity (N=4096:
      1.26x @0.5%, 1.22x @5%; N=1024 crossover ~10%, dense wins above). But the
      end-to-end win is Amdahl-capped at ~1.1-1.3x because BOTH ops re-upload W
      (67MB @N=4096) per call -> transfer floor ~6.9ms dwarfs the kernel saving.
      The 20-100x traffic advantage is real at the kernel but gated on WEIGHT
      RESIDENCY (W resident across timesteps), not a better scatter kernel.
- [ ] CSR weights (Tier 2): fired*N -> fired*fan_out
- [x] RESIDENCY EXPERIMENT (resident_bench op, W uploaded once, 100 dispatches):
      CONFIRMED residency is the dominant lever. Per-step costs collapsed vs the
      upload-bound run (N=4096 @5%: scatter 7.18ms -> 0.378ms ~19x; dense 8.77 ->
      0.97ms ~9x). With W resident, event-driven scatter BEATS dense by:
        N=4096: 5.29x @0.5%, 4.74x @1%, 2.57x @5%, 2.13x @20%
        N=1024: 3.62x @0.5%, 2.73x @5%, 1.21x @20% (crossover near 20%)
      Real, on-RDNA2, at biological sparsity. NOT the 20-100x read-model ideal:
      (a) dense has a fixed floor (always loops N); (b) a per-timestep submit/sync
      floor (~0.15-0.2ms, one GPU submit + host wait per dispatch) now dominates
      the scatter at very low activity, capping speedup ~5x.
- [x] NEXT LEVER: batch multiple timesteps per GPU submit (CommandBatch/OpGraph,
      one begin / many dispatch / one wait) to remove the per-step submit floor
      and approach the kernel-limited ceiling.
      DONE (resident_bench batched=1, one submit + barriers). Batching cut the
      scatter per-step 5-10x (submit floor removed). FULL ADVANTAGE now visible:
      scatter vs dense, W-resident + batched, per-step:
        N=4096: 23.1x @0.5%, 24.4x @1%, 10.1x @5%, 6.2x @10%
        N=1024: 20.8x @0.5%, 18.5x @1%,  8.2x @5%, 6.1x @10%
      At 0.5-1% activity this matches the read-count model (20-100x) and realizes
      the brain module's "97% less than MACs" (~30x; measured 20-24x). Residual
      gap to ideal 1/activity = scatter's fixed per-dispatch cost (~0.03ms:
      N-element I_acc write, barrier, bind) — diminishing returns past here.

## VERDICT — isolated kernel (all measured on RX 6750 XT)
  read-count model ......... 20-100x   (idealized)
  per-call GPU (W re-upload)  1.1-1.3x  (W transfer floor)
  W resident, per-step submit 2.5-5.3x  (submit/host-sync floor)
  W resident + batched submit 6-24x     (fixed per-dispatch floor)
Holds ONLY in the isolated single-vector, W-resident, batched-dispatch regime.
See INTEGRATION FINDING below — this does NOT transfer to the real FFN workload.

## INTEGRATION FINDING (cubby-lm/brain/event_snn.py) — qualifies the verdict
Wired spike_propagate_batch (non-square, weighted, multi-bit) into a real
Synapsis drop-in. Correct to fp32. BUT at the realistic SNN-FFN operating point
(M=512 spike vectors, 2048x2048, ~2% density) the event-driven op is ~270ms vs
dense _bridge.linear ~42ms — i.e. 6x SLOWER, not faster.
Diagnosis (profile_propagate.py, reuse_test.py, scale_test.py):
  - compact 5.9ms, asarray 0.13ms — negligible. The op call itself is the cost.
  - NOT W-row reuse: shared vs independent fired lists equal (271 vs 280ms).
  - K-INDEPENDENT: M=512 costs 293ms@K=1, 287ms@K=40, 266ms@K=200. Sparser input
    does NOT help -> the gather loop is not the bottleneck.
  - M-SCALING: 7.4ms(M=16) / 19ms(M=64) / 287ms(M=512). Cost tracks M*N_out.
=> The batched multi-vector op is OVERHEAD-BOUND (per-call W re-upload + M*N_out
   readback + dispatch at the numpy<->GPU boundary), not compute-bound. Event-
   driven sparsity buys nothing when the kernel isn't compute-bound.
Why resident_bench showed 20x but this shows 0.16x: resident_bench measured a
tiny grid (gx=16), single reused vector, W resident across batched dispatches,
no per-call readback. The real propagation is a large batched matmul that crosses
the host boundary every call. The 20x regime does NOT transfer to the FFN.
REQUIREMENT to realize the advantage: GPU-RESIDENT FUSED execution — W persistent
on device, activations kept on-GPU across both synapses and across tokens (no
per-op numpy round-trip), i.e. an op-graph / resident-tensor path in grilly, not
a per-op drop-in. That is a substantial grilly infra piece (matches CLAUDE.md's
note that the resident path is still missing). Until then, dense GEMV wins for
the SNN-FFN and VSA should carry the efficiency claim.

## ROOT CAUSE + FIX (supersedes the "overhead-bound" diagnosis above)
The 270ms was NOT inherent overhead — it was a WRITE-COMBINED READBACK bug.
Phase timing inside the op (PPB_PROF=1) showed: upload 1.6ms, desc 0.0ms,
dispatch+wait 2.8ms (kernel is FAST), DOWNLOAD 292ms. The output buffer came
from pool.acquire() = DEVICE_LOCAL|HOST_VISIBLE write-combined memory; CPU reads
from WC run at ~14 MB/s. (The K-independence / M-scaling were the 4MB readback,
not the kernel.)
FIX: same staging pattern linear.cpp already uses — compute on acquireDeviceLocal
buffers, stage OUT through acquireReadback (HOST_CACHED ~7 GB/s), all copies +
compute in one command buffer (copy-in -> barrier -> dispatch -> barrier ->
copy-out). 
RESULT (M=512, 2048x2048, 2%): op 294ms -> 3.7ms (download 292 -> 0.22ms).
spike_propagate_batch is now ~14x FASTER than dense _bridge.linear (3.7 vs 52ms).
The event-driven kernel advantage is REAL at the production operating point.

## REMAINING TAX (module level, not the op)
EventDrivenSynapsis.forward end-to-end is ~break-even with dense (0.6-0.9x)
because host-side compaction (np.nonzero over the dense M x N_in spike matrix,
~6ms at M=512) now costs as much as the propagation. This is a dense-input
artifact: the spikes arrive materialized. Removing it = generate/keep spikes in
sparse (fired-list) form, or compact on GPU — i.e. avoid the dense host scan.
That is the real next lever for the FFN; the op itself is no longer the problem.
