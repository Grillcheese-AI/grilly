# Grilly readback / compute-buffer optimization (2.0-dev)

## Root cause
`BufferPool::acquire()` returns DEVICE_LOCAL|HOST_VISIBLE memory that VMA maps
write-combined (WC) on AMD/Windows. WC memory is:
- ~0.05 GB/s for GPU **compute reads** (bypasses L2), and
- ~25 MB/s for CPU **readback** (uncached host reads).

Any op that binds `pool.acquire()` buffers as compute I/O and then
`pool.download()`s an output straight from one pays both taxes. Severity scales
with buffer size: catastrophic for large activation tensors, negligible for
scalar/tiny outputs.

## Proven fix (the staging pattern)
Reference impls: `linear.cpp`, `mingru.cpp`, `layernorm.cpp`, `loss.cpp`,
`optimizer.cpp`, `prefix_scan.cpp`, `embedding.cpp`, and `snn.cpp::spikePropagateBatch`.

- Compute buffers (shader reads/writes): `pool.acquireDeviceLocal()` — cached VRAM ~432 GB/s.
- GPU-only scratch: `acquireDeviceLocal()` too (no host staging).
- Stage-IN (CPU writes): `pool.acquire()` — WC is fine for sequential memcpy ~9 GB/s.
- Stage-OUT (CPU reads): `pool.acquireReadback()` — HOST_CACHED ~7 GB/s.
- One command buffer:
  `begin -> copyBuffer(stgIn->DL)... -> transferComputeBarrier -> dispatch(es)
   -> transferComputeBarrier -> copyBuffer(DL->stgOut) -> submitDeferred -> waitForCompletion`,
  then `pool.download(stgOut, ...)`. Compute-compute `barrier()` between multi-pass dispatches.

## Proof point
`rmsnorm` 8x512x2048 (32 MB output), RX 6750 XT:
- before: 2412 ms (median)
- after:  34.6 ms  -> ~70x, max_abs_diff vs fp64 unchanged at 8.583e-06.

## Op audit (WC-readback antipattern)

### Already correct (staging in place)
linear, mingru, layernorm, embedding, loss, optimizer, prefix_scan, bandit,
eggroll, snn::spikePropagateBatch,
rmsnorm [FIXED: 2412->34.6 ms, 70x],
activations [FIXED: softmax/softmaxBackward/mfSoftmax/mfSoftplus + the
activationForward helper funcs; softmax 32MB now 44 ms, max_abs_diff 6.1e-08].

### Still slow — ranked backlog
TIER 1 (Cubby / LM / SNN hot path):
- vsa_lm_forward.cpp  (VSA LM head; STATEFUL resident-weight handle + multi-layer
  forward; big logits download max_seq*vocab @387 — needs a careful dedicated pass)
- snn.cpp neuron ops (lifStep/snnNodeForward/Backward/hebbian/stdp/spikeScatter/
  residentBench) — LOW PRIORITY: outputs are nNeurons-sized (KB), readback tax sub-ms.

TIER 2 (bio learning + attention):
- learning.cpp  (STDP/Hebbian/EWC/NLMS/Whitening; ~10 downloads)
- attention.cpp (@156), attention_ops.cpp (@68,180,181,233,277,318,366)

TIER 3 (vision / MoE / misc — confirm still live):
- conv.cpp (~9), pooling.cpp (~6), batchnorm.cpp
- moe_forward.cpp (~10), moqe_train.cpp        [MoQE archived?]
- perceiver.cpp, perceiver_encoder.cpp, siglip_encoder.cpp
- swizzle.cpp, fused.cpp, kv_cache.cpp (persistent cache buffers — review separately)

## Per-op recipe
1. Classify each acquired buffer: INPUT (host->gpu), OUTPUT (gpu->host), SCRATCH (gpu-only).
2. INPUT/OUTPUT/SCRATCH compute buffer -> acquireDeviceLocal.
3. INPUT add stage = acquire(); OUTPUT add stage = acquireReadback().
4. upload() into stage-in; bind DL buffers in the descriptor set.
5. In one batch: copyBuffer(stgIn->DL) for each input; transferComputeBarrier;
   dispatch(es) (keep existing compute-compute barriers); transferComputeBarrier;
   copyBuffer(DL->stgOut) for each output; submitDeferred; waitForCompletion.
6. download() from stage-out; release all.
7. Rebuild, check max_abs_diff unchanged + time before/after.
