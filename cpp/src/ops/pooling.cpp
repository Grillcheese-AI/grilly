#include "grilly/ops/pooling.h"

#include <cstring>

namespace grilly {
namespace ops {

// ═══════════════════════════════════════════════════════════════════════════
// Pooling ops
//
// Spatial pooling shaders use (8,8,1) workgroups — each thread computes
// one output spatial position. Dispatch is 3D: (outW+7)/8, (outH+7)/8,
// batch*channels.
//
// Mean pooling uses (256,1,1) — simple parallel reduction.
// ═══════════════════════════════════════════════════════════════════════════

// ── MaxPool2d forward ────────────────────────────────────────────────────

void maxpool2dForward(CommandBatch& batch, BufferPool& pool,
                      PipelineCache& cache,
                      const float* input, float* output, uint32_t* indices,
                      const MaxPool2dParams& p) {
    const size_t inBytes  = size_t(p.batchSize) * p.channels * p.inHeight *
                            p.inWidth * sizeof(float);
    const size_t outBytes = size_t(p.batchSize) * p.channels * p.outHeight *
                            p.outWidth * sizeof(float);
    const size_t idxBytes = size_t(p.batchSize) * p.channels * p.outHeight *
                            p.outWidth * sizeof(uint32_t);

    GrillyBuffer bufIn  = pool.acquire(inBytes);
    GrillyBuffer bufOut = pool.acquire(outBytes);
    GrillyBuffer bufIdx = pool.acquire(idxBytes);

    pool.upload(bufIn, input, inBytes);

    PipelineEntry pipe = cache.getOrCreate("maxpool2d-forward", 3,
                                           sizeof(MaxPool2dParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufIn.handle,  0, inBytes},
        {bufOut.handle, 0, outBytes},
        {bufIdx.handle, 0, idxBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("maxpool2d-forward",
                                                        bufInfos);

    uint32_t gx = (p.outWidth + 7) / 8;
    uint32_t gy = (p.outHeight + 7) / 8;
    uint32_t gz = p.batchSize * p.channels;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, gy, gz,
                   &p, sizeof(p));
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufOut, output, outBytes);
    pool.download(bufIdx, reinterpret_cast<float*>(indices), idxBytes);

    pool.release(bufIn);
    pool.release(bufOut);
    pool.release(bufIdx);
}

// ── MaxPool2d backward ──────────────────────────────────────────────────

void maxpool2dBackward(CommandBatch& batch, BufferPool& pool,
                       PipelineCache& cache,
                       const float* gradOutput, const uint32_t* indices,
                       float* gradInput, const MaxPool2dBackwardParams& p) {
    const size_t gradOutBytes = size_t(p.batchSize) * p.channels * p.outHeight *
                                p.outWidth * sizeof(float);
    const size_t idxBytes     = gradOutBytes;  // same spatial dims (uint32)
    const size_t gradInBytes  = size_t(p.batchSize) * p.channels * p.inHeight *
                                p.inWidth * sizeof(float);

    GrillyBuffer bufGradOut = pool.acquire(gradOutBytes);
    GrillyBuffer bufIdx     = pool.acquire(idxBytes);
    GrillyBuffer bufGradIn  = pool.acquire(gradInBytes);

    pool.upload(bufGradOut, gradOutput, gradOutBytes);
    pool.upload(bufIdx, reinterpret_cast<const float*>(indices), idxBytes);

    // Zero grad_input
    std::vector<float> zeros(p.batchSize * p.channels * p.inHeight * p.inWidth,
                             0.0f);
    pool.upload(bufGradIn, zeros.data(), gradInBytes);

    PipelineEntry pipe = cache.getOrCreate("maxpool2d-backward", 3,
                                           sizeof(MaxPool2dBackwardParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufGradOut.handle, 0, gradOutBytes},
        {bufIdx.handle,     0, idxBytes},
        {bufGradIn.handle,  0, gradInBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("maxpool2d-backward",
                                                        bufInfos);

    uint32_t gx = (p.outWidth + 7) / 8;
    uint32_t gy = (p.outHeight + 7) / 8;
    uint32_t gz = p.batchSize * p.channels;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, gy, gz,
                   &p, sizeof(p));
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufGradIn, gradInput, gradInBytes);

    pool.release(bufGradOut);
    pool.release(bufIdx);
    pool.release(bufGradIn);
}

// ── AvgPool2d forward ────────────────────────────────────────────────────

void avgpool2dForward(CommandBatch& batch, BufferPool& pool,
                      PipelineCache& cache,
                      const float* input, float* output,
                      const AvgPool2dParams& p) {
    const size_t inBytes  = size_t(p.batchSize) * p.channels * p.inHeight *
                            p.inWidth * sizeof(float);
    const size_t outBytes = size_t(p.batchSize) * p.channels * p.outHeight *
                            p.outWidth * sizeof(float);

    GrillyBuffer bufIn  = pool.acquire(inBytes);
    GrillyBuffer bufOut = pool.acquire(outBytes);

    pool.upload(bufIn, input, inBytes);

    PipelineEntry pipe = cache.getOrCreate("avgpool2d-forward", 2,
                                           sizeof(AvgPool2dParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufIn.handle,  0, inBytes},
        {bufOut.handle, 0, outBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("avgpool2d-forward",
                                                        bufInfos);

    uint32_t gx = (p.outWidth + 7) / 8;
    uint32_t gy = (p.outHeight + 7) / 8;
    uint32_t gz = p.batchSize * p.channels;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, gy, gz,
                   &p, sizeof(p));
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufOut, output, outBytes);

    pool.release(bufIn);
    pool.release(bufOut);
}

// ── AvgPool2d backward ──────────────────────────────────────────────────

void avgpool2dBackward(CommandBatch& batch, BufferPool& pool,
                       PipelineCache& cache,
                       const float* gradOutput, float* gradInput,
                       const AvgPool2dParams& p) {
    const size_t gradOutBytes = size_t(p.batchSize) * p.channels * p.outHeight *
                                p.outWidth * sizeof(float);
    const size_t gradInBytes  = size_t(p.batchSize) * p.channels * p.inHeight *
                                p.inWidth * sizeof(float);

    GrillyBuffer bufGradOut = pool.acquire(gradOutBytes);
    GrillyBuffer bufGradIn  = pool.acquire(gradInBytes);

    pool.upload(bufGradOut, gradOutput, gradOutBytes);

    PipelineEntry pipe = cache.getOrCreate("avgpool2d-backward", 2,
                                           sizeof(AvgPool2dParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufGradOut.handle, 0, gradOutBytes},
        {bufGradIn.handle,  0, gradInBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("avgpool2d-backward",
                                                        bufInfos);

    uint32_t gx = (p.inWidth + 7) / 8;
    uint32_t gy = (p.inHeight + 7) / 8;
    uint32_t gz = p.batchSize * p.channels;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, gy, gz,
                   &p, sizeof(p));
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufGradIn, gradInput, gradInBytes);

    pool.release(bufGradOut);
    pool.release(bufGradIn);
}

// ── Mean pooling ─────────────────────────────────────────────────────────

void meanPool(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
              const float* input, float* output, const MeanPoolParams& p) {
    const size_t inBytes  = size_t(p.batchSize) * p.seqLen * p.features *
                            sizeof(float);
    const size_t outBytes = size_t(p.batchSize) * p.features * sizeof(float);

    GrillyBuffer bufIn  = pool.acquire(inBytes);
    GrillyBuffer bufOut = pool.acquire(outBytes);

    pool.upload(bufIn, input, inBytes);

    PipelineEntry pipe = cache.getOrCreate("mean-pooling", 2,
                                           sizeof(MeanPoolParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufIn.handle,  0, inBytes},
        {bufOut.handle, 0, outBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("mean-pooling",
                                                        bufInfos);

    uint32_t gx = (p.batchSize * p.features + 255) / 256;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufOut, output, outBytes);

    pool.release(bufIn);
    pool.release(bufOut);
}

}  // namespace ops
}  // namespace grilly
