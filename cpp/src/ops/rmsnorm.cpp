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

    // Acquire 4 buffers
    GrillyBuffer bufInput  = pool.acquire(inputBytes);
    GrillyBuffer bufOutput = pool.acquire(outputBytes);
    GrillyBuffer bufWeight = pool.acquire(weightBytes);
    GrillyBuffer bufRms    = pool.acquire(rmsBytes);

    // Upload input data
    pool.upload(bufInput, input, inputBytes);
    pool.upload(bufWeight, weight, weightBytes);

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

    // 2-pass dispatch
    batch.begin();

    // Pass 0: compute mean(x^2)
    RMSNormParams push0{batchSize, seqLen, features, eps, 0};
    uint32_t gx0 = (totalPositions + 255) / 256;
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx0, 1, 1,
                   &push0, sizeof(push0));
    batch.barrier();

    // Pass 1: normalize
    RMSNormParams push1{batchSize, seqLen, features, eps, 1};
    uint32_t gx1 = (totalElements + 255) / 256;
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx1, 1, 1,
                   &push1, sizeof(push1));

    batch.submitDeferred();
    batch.waitForCompletion();

    // Download result
    pool.download(bufOutput, output, outputBytes);

    // Release all buffers
    pool.release(bufInput);
    pool.release(bufOutput);
    pool.release(bufWeight);
    pool.release(bufRms);
}

}  // namespace ops
}  // namespace grilly
