#include "grilly/ops/batchnorm.h"

#include <cstring>
#include <vector>

namespace grilly {
namespace ops {

// ═══════════════════════════════════════════════════════════════════════════
// BatchNorm2d
//
// Both forward and backward dispatch at (256,1,1) — one thread per channel.
// Each thread loops over the entire (batch, H, W) spatial domain for its
// channel, computing batch statistics in forward and accumulating gradients
// in backward.
//
// The forward pass stores batch_mean and batch_var for use in backward.
// Running stats are updated with EMA when training=1.
// ═══════════════════════════════════════════════════════════════════════════

// ── BatchNorm2d forward ──────────────────────────────────────────────────

void batchnorm2dForward(CommandBatch& batch, BufferPool& pool,
                        PipelineCache& cache,
                        const float* input, float* output,
                        const float* gamma, const float* beta,
                        float* runningMean, float* runningVar,
                        float* batchMean, float* batchVar,
                        const BatchNorm2dForwardParams& p) {
    const size_t dataBytes = size_t(p.batchSize) * p.numFeatures * p.height *
                             p.width * sizeof(float);
    const size_t featBytes = size_t(p.numFeatures) * sizeof(float);

    GrillyBuffer bufInput      = pool.acquire(dataBytes);
    GrillyBuffer bufOutput     = pool.acquire(dataBytes);
    GrillyBuffer bufGamma      = pool.acquire(featBytes);
    GrillyBuffer bufBeta       = pool.acquire(featBytes);
    GrillyBuffer bufRunMean    = pool.acquire(featBytes);
    GrillyBuffer bufRunVar     = pool.acquire(featBytes);
    GrillyBuffer bufBatchMean  = pool.acquire(featBytes);
    GrillyBuffer bufBatchVar   = pool.acquire(featBytes);

    pool.upload(bufInput, input, dataBytes);
    pool.upload(bufGamma, gamma, featBytes);
    pool.upload(bufBeta, beta, featBytes);
    pool.upload(bufRunMean, runningMean, featBytes);
    pool.upload(bufRunVar, runningVar, featBytes);

    PipelineEntry pipe = cache.getOrCreate("batchnorm2d-forward", 8,
                                           sizeof(BatchNorm2dForwardParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufInput.handle,     0, dataBytes},
        {bufOutput.handle,    0, dataBytes},
        {bufGamma.handle,     0, featBytes},
        {bufBeta.handle,      0, featBytes},
        {bufRunMean.handle,   0, featBytes},
        {bufRunVar.handle,    0, featBytes},
        {bufBatchMean.handle, 0, featBytes},
        {bufBatchVar.handle,  0, featBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("batchnorm2d-forward",
                                                        bufInfos);

    uint32_t gx = (p.numFeatures + 255) / 256;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.submit();

    pool.download(bufOutput, output, dataBytes);
    pool.download(bufRunMean, runningMean, featBytes);
    pool.download(bufRunVar, runningVar, featBytes);
    pool.download(bufBatchMean, batchMean, featBytes);
    pool.download(bufBatchVar, batchVar, featBytes);

    pool.release(bufInput);
    pool.release(bufOutput);
    pool.release(bufGamma);
    pool.release(bufBeta);
    pool.release(bufRunMean);
    pool.release(bufRunVar);
    pool.release(bufBatchMean);
    pool.release(bufBatchVar);
}

// ── BatchNorm2d backward ─────────────────────────────────────────────────

void batchnorm2dBackward(CommandBatch& batch, BufferPool& pool,
                         PipelineCache& cache,
                         const float* gradOutput, const float* input,
                         float* gradInput,
                         const float* batchMean, const float* batchVar,
                         const float* gamma,
                         float* gradGamma, float* gradBeta,
                         const BatchNorm2dBackwardParams& p) {
    const size_t dataBytes = size_t(p.batchSize) * p.numFeatures * p.height *
                             p.width * sizeof(float);
    const size_t featBytes = size_t(p.numFeatures) * sizeof(float);

    GrillyBuffer bufGradOut   = pool.acquire(dataBytes);
    GrillyBuffer bufInput     = pool.acquire(dataBytes);
    GrillyBuffer bufGradIn    = pool.acquire(dataBytes);
    GrillyBuffer bufBatchMean = pool.acquire(featBytes);
    GrillyBuffer bufBatchVar  = pool.acquire(featBytes);
    GrillyBuffer bufGamma     = pool.acquire(featBytes);
    GrillyBuffer bufGradGamma = pool.acquire(featBytes);
    GrillyBuffer bufGradBeta  = pool.acquire(featBytes);

    pool.upload(bufGradOut, gradOutput, dataBytes);
    pool.upload(bufInput, input, dataBytes);
    pool.upload(bufBatchMean, batchMean, featBytes);
    pool.upload(bufBatchVar, batchVar, featBytes);
    pool.upload(bufGamma, gamma, featBytes);

    // Zero grad outputs
    std::vector<float> zeros(p.numFeatures, 0.0f);
    pool.upload(bufGradGamma, zeros.data(), featBytes);
    pool.upload(bufGradBeta, zeros.data(), featBytes);

    PipelineEntry pipe = cache.getOrCreate("batchnorm2d-backward", 8,
                                           sizeof(BatchNorm2dBackwardParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufGradOut.handle,   0, dataBytes},
        {bufInput.handle,     0, dataBytes},
        {bufGradIn.handle,    0, dataBytes},
        {bufBatchMean.handle, 0, featBytes},
        {bufBatchVar.handle,  0, featBytes},
        {bufGamma.handle,     0, featBytes},
        {bufGradGamma.handle, 0, featBytes},
        {bufGradBeta.handle,  0, featBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("batchnorm2d-backward",
                                                        bufInfos);

    uint32_t gx = (p.numFeatures + 255) / 256;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.submit();

    pool.download(bufGradIn, gradInput, dataBytes);
    pool.download(bufGradGamma, gradGamma, featBytes);
    pool.download(bufGradBeta, gradBeta, featBytes);

    pool.release(bufGradOut);
    pool.release(bufInput);
    pool.release(bufGradIn);
    pool.release(bufBatchMean);
    pool.release(bufBatchVar);
    pool.release(bufGamma);
    pool.release(bufGradGamma);
    pool.release(bufGradBeta);
}

}  // namespace ops
}  // namespace grilly
