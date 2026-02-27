#include "grilly/ops/layernorm.h"

#include <cstring>

namespace grilly {
namespace ops {

// ── LayerNorm forward (port of backend/normalization.py) ──────────────────
//
// LayerNorm is a 3-pass algorithm using the SAME shader with different
// pass_type values. This is a design pattern from the Python backend —
// one shader handles all three phases via a uniform branch:
//
//   pass_type 0: Each thread accumulates elements for one position,
//                stores mean[pos] = sum / features
//   pass_type 1: Each thread accumulates (x - mean)^2,
//                stores var[pos] = sum / features
//   pass_type 2: Each thread normalizes one element:
//                out[i] = gamma * (x[i] - mean) / sqrt(var + eps) + beta
//
// The advantage: one pipeline, one descriptor set — just swap push constants.
// Pipeline barriers between passes ensure mean is ready before variance,
// and both are ready before normalize.

void layernorm(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
               const float* input, float* output,
               const float* gamma, const float* beta,
               uint32_t batchSize, uint32_t seqLen, uint32_t features,
               float eps) {
    const uint32_t totalPositions = batchSize * seqLen;
    const uint32_t totalElements  = totalPositions * features;
    const size_t inputBytes  = size_t(totalElements) * sizeof(float);
    const size_t outputBytes = inputBytes;
    const size_t gammaBytes  = size_t(features) * sizeof(float);
    const size_t betaBytes   = gammaBytes;
    const size_t meanBytes   = size_t(totalPositions) * sizeof(float);
    const size_t varBytes    = meanBytes;

    // Acquire 6 buffers
    GrillyBuffer bufInput  = pool.acquire(inputBytes);
    GrillyBuffer bufOutput = pool.acquire(outputBytes);
    GrillyBuffer bufGamma  = pool.acquire(gammaBytes);
    GrillyBuffer bufBeta   = pool.acquire(betaBytes);
    GrillyBuffer bufMean   = pool.acquire(meanBytes);
    GrillyBuffer bufVar    = pool.acquire(varBytes);

    // Upload input data
    pool.upload(bufInput, input, inputBytes);
    pool.upload(bufGamma, gamma, gammaBytes);
    pool.upload(bufBeta, beta, betaBytes);

    // Get pipeline: 6 buffers, 20 bytes push constants
    PipelineEntry pipe = cache.getOrCreate("fnn-layernorm", 6,
                                           sizeof(LayerNormParams));

    // Descriptor set (same for all 3 passes — same buffers)
    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufInput.handle,  0, inputBytes},
        {bufOutput.handle, 0, outputBytes},
        {bufGamma.handle,  0, gammaBytes},
        {bufBeta.handle,   0, betaBytes},
        {bufMean.handle,   0, meanBytes},
        {bufVar.handle,    0, varBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("fnn-layernorm",
                                                        bufInfos);

    // 3-pass dispatch
    batch.begin();

    // Pass 0: compute mean
    LayerNormParams push0{batchSize, seqLen, features, eps, 0};
    uint32_t gx0 = (totalPositions + 255) / 256;
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx0, 1, 1,
                   &push0, sizeof(push0));
    batch.barrier();

    // Pass 1: compute variance
    LayerNormParams push1{batchSize, seqLen, features, eps, 1};
    uint32_t gx1 = (totalPositions + 255) / 256;
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx1, 1, 1,
                   &push1, sizeof(push1));
    batch.barrier();

    // Pass 2: normalize + affine transform
    LayerNormParams push2{batchSize, seqLen, features, eps, 2};
    uint32_t gx2 = (totalElements + 255) / 256;
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx2, 1, 1,
                   &push2, sizeof(push2));

    batch.submit();

    // Download result
    pool.download(bufOutput, output, outputBytes);

    // Release all buffers
    pool.release(bufInput);
    pool.release(bufOutput);
    pool.release(bufGamma);
    pool.release(bufBeta);
    pool.release(bufMean);
    pool.release(bufVar);
}

// ── LayerNorm backward ───────────────────────────────────────────────────

void layernormBackward(CommandBatch& batch, BufferPool& pool,
                       PipelineCache& cache,
                       const float* gradOutput, const float* input,
                       const float* gamma, const float* mean,
                       const float* var,
                       float* gradInput, float* gradGamma, float* gradBeta,
                       uint32_t batchSize, uint32_t seqLen, uint32_t features,
                       float eps) {
    const uint32_t totalPositions = batchSize * seqLen;
    const uint32_t totalElements  = totalPositions * features;
    const size_t elemBytes    = size_t(totalElements) * sizeof(float);
    const size_t gammaBytes   = size_t(features) * sizeof(float);
    const size_t posBytes     = size_t(totalPositions) * sizeof(float);

    // 8 buffers for backward
    GrillyBuffer bufGradOut   = pool.acquire(elemBytes);
    GrillyBuffer bufInput     = pool.acquire(elemBytes);
    GrillyBuffer bufGamma     = pool.acquire(gammaBytes);
    GrillyBuffer bufMean      = pool.acquire(posBytes);
    GrillyBuffer bufVar       = pool.acquire(posBytes);
    GrillyBuffer bufGradIn    = pool.acquire(elemBytes);
    GrillyBuffer bufGradGamma = pool.acquire(gammaBytes);
    GrillyBuffer bufGradBeta  = pool.acquire(gammaBytes);

    pool.upload(bufGradOut, gradOutput, elemBytes);
    pool.upload(bufInput, input, elemBytes);
    pool.upload(bufGamma, gamma, gammaBytes);
    pool.upload(bufMean, mean, posBytes);
    pool.upload(bufVar, var, posBytes);

    // Zero grad outputs
    std::vector<float> zeros_elem(totalElements, 0.0f);
    std::vector<float> zeros_feat(features, 0.0f);
    pool.upload(bufGradIn, zeros_elem.data(), elemBytes);
    pool.upload(bufGradGamma, zeros_feat.data(), gammaBytes);
    pool.upload(bufGradBeta, zeros_feat.data(), gammaBytes);

    PipelineEntry pipe = cache.getOrCreate("fnn-layernorm-backward", 8,
                                           sizeof(LayerNormParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufGradOut.handle,   0, elemBytes},
        {bufInput.handle,     0, elemBytes},
        {bufGamma.handle,     0, gammaBytes},
        {bufMean.handle,      0, posBytes},
        {bufVar.handle,       0, posBytes},
        {bufGradIn.handle,    0, elemBytes},
        {bufGradGamma.handle, 0, gammaBytes},
        {bufGradBeta.handle,  0, gammaBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet(
        "fnn-layernorm-backward", bufInfos);

    batch.begin();

    // Pass 0: intermediate sums
    LayerNormParams push0{batchSize, seqLen, features, eps, 0};
    batch.dispatch(pipe.pipeline, pipe.layout, descSet,
                   (totalPositions + 255) / 256, 1, 1,
                   &push0, sizeof(push0));
    batch.barrier();

    // Pass 1: grad_input
    LayerNormParams push1{batchSize, seqLen, features, eps, 1};
    batch.dispatch(pipe.pipeline, pipe.layout, descSet,
                   (totalElements + 255) / 256, 1, 1,
                   &push1, sizeof(push1));
    batch.barrier();

    // Pass 2: grad_gamma, grad_beta
    LayerNormParams push2{batchSize, seqLen, features, eps, 2};
    batch.dispatch(pipe.pipeline, pipe.layout, descSet,
                   (features + 255) / 256, 1, 1,
                   &push2, sizeof(push2));

    batch.submit();

    pool.download(bufGradIn, gradInput, elemBytes);
    pool.download(bufGradGamma, gradGamma, gammaBytes);
    pool.download(bufGradBeta, gradBeta, gammaBytes);

    pool.release(bufGradOut);
    pool.release(bufInput);
    pool.release(bufGamma);
    pool.release(bufMean);
    pool.release(bufVar);
    pool.release(bufGradIn);
    pool.release(bufGradGamma);
    pool.release(bufGradBeta);
}

}  // namespace ops
}  // namespace grilly
