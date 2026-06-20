# grilly scaling plan — toward ~1B data-parallel, hardware-agnostic

Goal (the Cubby thesis at scale): frontier-*class* small/mid models trained at a
fraction of datacenter energy, on **multi-GPU grilly/Vulkan** (no CUDA lock-in).
First concrete target chosen: **~1B params, ~100-300B tokens, data-parallel,
hardware-agnostic** (NVIDIA-via-Vulkan / AMD / commodity — don't assume any).

Compute sanity: 1B × 200B tok ≈ 6·1e9·2e11 = 1.2e21 FLOPs. At ~20-50 H100-equiv
(~3e16 effective FLOPS) ≈ half-a-day to a couple days *once the infra works*.
**The hard part is the infrastructure, not the compute.** The energy edge =
(more capability/FLOP via the sparse stack) × (more FLOP/watt-$ via Vulkan on
non-premium hardware).

## The sparse stack IS the energy lever (so correctness is on the critical path)
A dead component = a lost efficiency multiplier. Before any cloud spend these must
actually work (validated on the small rig):
- MoE top-2 (0.0.3) — activate a fraction of params/token
- Ternary/BitNet SwiGLU — multiply-free, the biggest energy/op win
- Chunked SWA + MinGRU — sub-quadratic context
- Fix the resident **dead router + no-op sampled-softmax** (see cubby
  `RESIDENT_THROUGHPUT_PLAN.md`) — currently no sparsity benefit at all.

## Phased path (dependency-ordered)

### Phase 0 — small-rig validation (single 12 GB GPU; in progress)
De-risk before scaling: sparse-stack correctness (above) + the throughput wins
(loss-ce-fused readback, fused AdamW, hypergradient) + depth stability
(clip/warmup/residual-scale, gnorm). Nothing here needs >1 GPU.

### Phase 1 — data parallelism in grilly (the make-or-break)
grilly today has NO distributed layer (enumerates devices, uses one). Build:
- **Gradient all-reduce** across replicas after backward. The resident grads are
  already readable in VRAM (the `sum_squares` pattern). v1: **host-staged ring
  all-reduce** — portable, works on any backend/cloud; no NCCL/RCCL. Optimize to
  intra-node Vulkan external-memory/semaphore P2P later.
- **Bucketed overlap:** all-reduce gradient buckets as backward produces them
  (overlap comms with compute). v1 may barrier after backward; bucket next.
- **Parity gate** (same discipline as every grilly op): 2-replica training must
  match 1-replica with averaged gradients (numerical parity), then loss-curve
  match. **Needs ≥2 GPUs to validate** — rent 2 cheap cloud GPUs for a few hours
  if not available locally.
- Optimizer-state note: pure replicate-everything DP holds Adam fp32 m/v on every
  GPU (~8 GB for 1B). Fine on ≥24 GB cloud GPUs; add **ZeRO-1 optimizer sharding**
  only if VRAM-bound. Not needed for the first 1B run on big cloud GPUs.

### Phase 2 — bf16 mixed precision (hardware-agnostic)
bf16 compute + **fp32 master weights**. GEMM dispatches **coopmat when present,
fp32-tiled otherwise** (grilly has both: `gemm-coopmat-shared.glsl` + `gemm_tiled`)
— so it degrades gracefully across NVIDIA/AMD/commodity. See `FP16_BACKWARD_SPEC.md`,
`GEMM_AUTOTUNE.md`. Parity-gate vs fp32.

### Phase 3 — 1B model shape
Bring the trunk to ~1B (e.g. d=2048, L~24, V=32k, MoE optional). The depth/
stability work from the small model transfers. Validate it trains stably at this
shape on one GPU before going multi-node.

### Phase 4 — cloud multi-node
Network transport for inter-node all-reduce (TCP/RDMA, gloo-style — NOT NCCL),
sharded + resumable checkpointing (extend the existing `.grl` crash-guardrail),
fault tolerance, multi-node launch/orchestration.

## Order of work
Phase 0 (small rig) → Phase 1 DP on 2 GPUs (parity-gated) → Phase 2 bf16 →
Phase 3 1B shape → Phase 4 cloud. Each gated like every grilly op: parity, then
loss-curve match, then scale.
