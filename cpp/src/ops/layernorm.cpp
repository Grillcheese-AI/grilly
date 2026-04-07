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

    // Staging pattern: 3 stage-in (input, gamma, beta), 1 stage-out (output).
    // mean and var are intermediate buffers — pure DEVICE_LOCAL, never touched
    // by the CPU, so no staging buffers needed for them.
    GrillyBuffer bufInputDL  = pool.acquireDeviceLocal(inputBytes);
    GrillyBuffer bufOutputDL = pool.acquireDeviceLocal(outputBytes);
    GrillyBuffer bufGammaDL  = pool.acquireDeviceLocal(gammaBytes);
    GrillyBuffer bufBetaDL   = pool.acquireDeviceLocal(betaBytes);
    GrillyBuffer bufMeanDL   = pool.acquireDeviceLocal(meanBytes);
    GrillyBuffer bufVarDL    = pool.acquireDeviceLocal(varBytes);

    GrillyBuffer bufInputStage  = pool.acquire(inputBytes);
    GrillyBuffer bufGammaStage  = pool.acquire(gammaBytes);
    GrillyBuffer bufBetaStage   = pool.acquire(betaBytes);
    GrillyBuffer bufOutputStage = pool.acquireReadback(outputBytes);

    pool.upload(bufInputStage, input, inputBytes);
    pool.upload(bufGammaStage, gamma, gammaBytes);
    pool.upload(bufBetaStage,  beta,  betaBytes);

    // Get pipeline: 6 buffers, 20 bytes push constants
    PipelineEntry pipe = cache.getOrCreate("fnn-layernorm", 6,
                                           sizeof(LayerNormParams));

    // Descriptor set bound to DEVICE_LOCAL buffers
    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufInputDL.handle,  0, inputBytes},
        {bufOutputDL.handle, 0, outputBytes},
        {bufGammaDL.handle,  0, gammaBytes},
        {bufBetaDL.handle,   0, betaBytes},
        {bufMeanDL.handle,   0, meanBytes},
        {bufVarDL.handle,    0, varBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("fnn-layernorm",
                                                        bufInfos);

    batch.begin();

    // Stage-in: copy 3 host-visible staging buffers to DL VRAM
    batch.copyBuffer(bufInputStage, bufInputDL, inputBytes);
    batch.copyBuffer(bufGammaStage, bufGammaDL, gammaBytes);
    batch.copyBuffer(bufBetaStage,  bufBetaDL,  betaBytes);
    batch.transferComputeBarrier();

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

    // Stage-out: DL output → HOST_CACHED readback staging
    batch.transferComputeBarrier();
    batch.copyBuffer(bufOutputDL, bufOutputStage, outputBytes);

    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufOutputStage, output, outputBytes);

    pool.release(bufInputDL);
    pool.release(bufOutputDL);
    pool.release(bufGammaDL);
    pool.release(bufBetaDL);
    pool.release(bufMeanDL);
    pool.release(bufVarDL);
    pool.release(bufInputStage);
    pool.release(bufGammaStage);
    pool.release(bufBetaStage);
    pool.release(bufOutputStage);
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

    // Staging pattern: 5 stage-in (gradOut, input, gamma, mean, var),
    // 3 stage-out (gradIn, gradGamma, gradBeta)
    GrillyBuffer bufGradOutDL   = pool.acquireDeviceLocal(elemBytes);
    GrillyBuffer bufInputDL     = pool.acquireDeviceLocal(elemBytes);
    GrillyBuffer bufGammaDL     = pool.acquireDeviceLocal(gammaBytes);
    GrillyBuffer bufMeanDL      = pool.acquireDeviceLocal(posBytes);
    GrillyBuffer bufVarDL       = pool.acquireDeviceLocal(posBytes);
    GrillyBuffer bufGradInDL    = pool.acquireDeviceLocal(elemBytes);
    GrillyBuffer bufGradGammaDL = pool.acquireDeviceLocal(gammaBytes);
    GrillyBuffer bufGradBetaDL  = pool.acquireDeviceLocal(gammaBytes);

    GrillyBuffer bufGradOutStage   = pool.acquire(elemBytes);
    GrillyBuffer bufInputStage     = pool.acquire(elemBytes);
    GrillyBuffer bufGammaStage     = pool.acquire(gammaBytes);
    GrillyBuffer bufMeanStage      = pool.acquire(posBytes);
    GrillyBuffer bufVarStage       = pool.acquire(posBytes);
    GrillyBuffer bufGradInStage    = pool.acquireReadback(elemBytes);
    GrillyBuffer bufGradGammaStage = pool.acquireReadback(gammaBytes);
    GrillyBuffer bufGradBetaStage  = pool.acquireReadback(gammaBytes);

    pool.upload(bufGradOutStage, gradOutput, elemBytes);
    pool.upload(bufInputStage,   input,      elemBytes);
    pool.upload(bufGammaStage,   gamma,      gammaBytes);
    pool.upload(bufMeanStage,    mean,       posBytes);
    pool.upload(bufVarStage,     var,        posBytes);

    // Zero the grad output staging buffers (atomic accumulation in shader).
    // Reuse the readback stage buffers as upload-zeros source.
    std::vector<float> zeros_elem(totalElements, 0.0f);
    std::vector<float> zeros_feat(features, 0.0f);
    pool.upload(bufGradInStage,    zeros_elem.data(), elemBytes);
    pool.upload(bufGradGammaStage, zeros_feat.data(), gammaBytes);
    pool.upload(bufGradBetaStage,  zeros_feat.data(), gammaBytes);

    PipelineEntry pipe = cache.getOrCreate("fnn-layernorm-backward", 8,
                                           sizeof(LayerNormParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufGradOutDL.handle,   0, elemBytes},
        {bufInputDL.handle,     0, elemBytes},
        {bufGammaDL.handle,     0, gammaBytes},
        {bufMeanDL.handle,      0, posBytes},
        {bufVarDL.handle,       0, posBytes},
        {bufGradInDL.handle,    0, elemBytes},
        {bufGradGammaDL.handle, 0, gammaBytes},
        {bufGradBetaDL.handle,  0, gammaBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet(
        "fnn-layernorm-backward", bufInfos);

    batch.begin();

    // Stage-in: copy all 8 stage buffers (5 inputs + 3 zeroed grads) to DL
    batch.copyBuffer(bufGradOutStage,   bufGradOutDL,   elemBytes);
    batch.copyBuffer(bufInputStage,     bufInputDL,     elemBytes);
    batch.copyBuffer(bufGammaStage,     bufGammaDL,     gammaBytes);
    batch.copyBuffer(bufMeanStage,      bufMeanDL,      posBytes);
    batch.copyBuffer(bufVarStage,       bufVarDL,       posBytes);
    batch.copyBuffer(bufGradInStage,    bufGradInDL,    elemBytes);
    batch.copyBuffer(bufGradGammaStage, bufGradGammaDL, gammaBytes);
    batch.copyBuffer(bufGradBetaStage,  bufGradBetaDL,  gammaBytes);
    batch.transferComputeBarrier();

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

    // Stage-out: copy 3 grad buffers from DL → HOST_CACHED readback staging
    batch.transferComputeBarrier();
    batch.copyBuffer(bufGradInDL,    bufGradInStage,    elemBytes);
    batch.copyBuffer(bufGradGammaDL, bufGradGammaStage, gammaBytes);
    batch.copyBuffer(bufGradBetaDL,  bufGradBetaStage,  gammaBytes);

    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufGradInStage,    gradInput,  elemBytes);
    pool.download(bufGradGammaStage, gradGamma,  gammaBytes);
    pool.download(bufGradBetaStage,  gradBeta,   gammaBytes);

    pool.release(bufGradOutDL);
    pool.release(bufInputDL);
    pool.release(bufGammaDL);
    pool.release(bufMeanDL);
    pool.release(bufVarDL);
    pool.release(bufGradInDL);
    pool.release(bufGradGammaDL);
    pool.release(bufGradBetaDL);
    pool.release(bufGradOutStage);
    pool.release(bufInputStage);
    pool.release(bufGammaStage);
    pool.release(bufMeanStage);
    pool.release(bufVarStage);
    pool.release(bufGradInStage);
    pool.release(bufGradGammaStage);
    pool.release(bufGradBetaStage);
}

}  // namespace ops
}  // namespace grilly
