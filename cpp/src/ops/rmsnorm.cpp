#include "grilly/ops/rmsnorm.h"

#include <cstring>

namespace grilly {
namespace ops {

// ── RMSNorm forward (port of backend/normalization.py) ──────────────────
//
// RMSNorm is a 2-pass algorithm using the SAME shader with different
// pass_type values. This is a design pattern from the Python backend —
// one shader handles both phases via a uniform branch:
//
//   pass_type 0: Each thread accumulates x^2 for one position,
//                stores rms_vals[pos] = sum_sq / features
//   pass_type 1: Each thread normalizes one element:
//                out[i] = weight * x[i] * rsqrt(rms_vals + eps)
//
// The advantage: one pipeline, one descriptor set — just swap push constants.
// A pipeline barrier between passes ensures mean(x^2) is ready before normalize.

void rmsnorm(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
             const float* input, float* output,
             const float* weight,
             uint32_t batchSize, uint32_t seqLen, uint32_t features,
             float eps) {
    const uint32_t totalPositions = batchSize * seqLen;
    const uint32_t totalElements  = totalPositions * features;
    const size_t inputBytes  = size_t(totalElements) * sizeof(float);
    const size_t outputBytes = inputBytes;
    const size_t weightBytes = size_t(features) * sizeof(float);
    const size_t rmsBytes    = size_t(totalPositions) * sizeof(float);

    // DEVICE_LOCAL compute buffers (cached VRAM ~432 GB/s — the shader
    // reads/writes these). bufRms is GPU-only scratch (pass 0 writes it,
    // pass 1 reads it), so it never needs a host-side staging buffer.
    GrillyBuffer bufInput  = pool.acquireDeviceLocal(inputBytes);
    GrillyBuffer bufOutput = pool.acquireDeviceLocal(outputBytes);
    GrillyBuffer bufWeight = pool.acquireDeviceLocal(weightBytes);
    GrillyBuffer bufRms    = pool.acquireDeviceLocal(rmsBytes);

    // Stage-IN (CPU writes -> WC memory, fast sequential memcpy ~9 GB/s).
    GrillyBuffer stgInput  = pool.acquire(inputBytes);
    GrillyBuffer stgWeight = pool.acquire(weightBytes);
    // Stage-OUT (CPU reads -> MUST be HOST_CACHED via acquireReadback;
    // reading back from WC memory runs at ~25 MB/s — 2.4 s for 32 MB here).
    GrillyBuffer stgOutput = pool.acquireReadback(outputBytes);

    pool.upload(stgInput,  input,  inputBytes);
    pool.upload(stgWeight, weight, weightBytes);

    // Get pipeline: 4 buffers, 20 bytes push constants
    PipelineEntry pipe = cache.getOrCreate("rms-norm", 4,
                                           sizeof(RMSNormParams));

    // Descriptor set (same for both passes — same buffers)
    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufInput.handle,  0, inputBytes},
        {bufOutput.handle, 0, outputBytes},
        {bufWeight.handle, 0, weightBytes},
        {bufRms.handle,    0, rmsBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("rms-norm",
                                                        bufInfos);

    RMSNormParams push0{batchSize, seqLen, features, eps, 0};  // mean(x^2)
    RMSNormParams push1{batchSize, seqLen, features, eps, 1};  // normalize
    uint32_t gx0 = (totalPositions + 255) / 256;
    uint32_t gx1 = (totalElements  + 255) / 256;

    // Single command buffer: stage-in DMA -> 2-pass compute -> stage-out DMA.
    batch.begin();
    batch.copyBuffer(stgInput,  bufInput,  inputBytes);
    batch.copyBuffer(stgWeight, bufWeight, weightBytes);
    batch.transferComputeBarrier();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx0, 1, 1,
                   &push0, sizeof(push0));
    batch.barrier();  // mean(x^2) must be ready before normalize
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx1, 1, 1,
                   &push1, sizeof(push1));
    batch.transferComputeBarrier();
    batch.copyBuffer(bufOutput, stgOutput, outputBytes);
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(stgOutput, output, outputBytes);  // HOST_CACHED ~7 GB/s

    pool.release(bufInput);
    pool.release(bufOutput);
    pool.release(bufWeight);
    pool.release(bufRms);
    pool.release(stgInput);
    pool.release(stgWeight);
    pool.release(stgOutput);
}

}  // namespace ops
}  // namespace grilly
