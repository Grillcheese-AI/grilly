#include "grilly/ops/learning.h"

#include <cstring>
#include <vector>

namespace grilly {
namespace ops {

// ═══════════════════════════════════════════════════════════════════════════
// Learning / auxiliary ops
// ═══════════════════════════════════════════════════════════════════════════

// ── SSM fused math ───────────────────────────────────────────────────────

void ssmFusedMath(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                  const float* gate, const float* value, const float* decay,
                  float* scanOut, const SSMFusedParams& p) {
    const size_t dataBytes = size_t(p.batchSize) * p.seqLen * p.features *
                             sizeof(float);

    GrillyBuffer bufGate  = pool.acquire(dataBytes);
    GrillyBuffer bufValue = pool.acquire(dataBytes);
    GrillyBuffer bufDecay = pool.acquire(dataBytes);
    GrillyBuffer bufOut   = pool.acquire(dataBytes);

    pool.upload(bufGate, gate, dataBytes);
    pool.upload(bufValue, value, dataBytes);
    pool.upload(bufDecay, decay, dataBytes);

    PipelineEntry pipe = cache.getOrCreate("ssm-fused-math", 4,
                                           sizeof(SSMFusedParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufGate.handle,  0, dataBytes},
        {bufValue.handle, 0, dataBytes},
        {bufDecay.handle, 0, dataBytes},
        {bufOut.handle,   0, dataBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("ssm-fused-math",
                                                        bufInfos);

    uint32_t gx = (p.batchSize * p.features + 255) / 256;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.submit();

    pool.download(bufOut, scanOut, dataBytes);

    pool.release(bufGate);
    pool.release(bufValue);
    pool.release(bufDecay);
    pool.release(bufOut);
}

// ── Fisher information diagonal ──────────────────────────────────────────

void fisherInfoUpdate(CommandBatch& batch, BufferPool& pool,
                      PipelineCache& cache,
                      const float* grads, float* fisher,
                      const FisherInfoParams& p) {
    const size_t bytes = size_t(p.numParams) * sizeof(float);

    GrillyBuffer bufGrads  = pool.acquire(bytes);
    GrillyBuffer bufFisher = pool.acquire(bytes);

    pool.upload(bufGrads, grads, bytes);
    pool.upload(bufFisher, fisher, bytes);

    PipelineEntry pipe = cache.getOrCreate("fisher-info", 2,
                                           sizeof(FisherInfoParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufGrads.handle,  0, bytes},
        {bufFisher.handle, 0, bytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("fisher-info", bufInfos);

    uint32_t gx = (p.numParams + 255) / 256;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.submit();

    pool.download(bufFisher, fisher, bytes);

    pool.release(bufGrads);
    pool.release(bufFisher);
}

// ── EWC penalty ──────────────────────────────────────────────────────────

void ewcPenalty(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                const float* currentParams, const float* optimalParams,
                const float* fisher, float* penalty,
                const EWCPenaltyParams& p) {
    const size_t bytes = size_t(p.numParams) * sizeof(float);

    GrillyBuffer bufCurrent  = pool.acquire(bytes);
    GrillyBuffer bufOptimal  = pool.acquire(bytes);
    GrillyBuffer bufFisher   = pool.acquire(bytes);
    GrillyBuffer bufPenalty  = pool.acquire(bytes);

    pool.upload(bufCurrent, currentParams, bytes);
    pool.upload(bufOptimal, optimalParams, bytes);
    pool.upload(bufFisher, fisher, bytes);

    PipelineEntry pipe = cache.getOrCreate("fisher-ewc-penalty", 4,
                                           sizeof(EWCPenaltyParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufCurrent.handle, 0, bytes},
        {bufOptimal.handle, 0, bytes},
        {bufFisher.handle,  0, bytes},
        {bufPenalty.handle, 0, bytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("fisher-ewc-penalty",
                                                        bufInfos);

    uint32_t gx = (p.numParams + 255) / 256;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.submit();

    pool.download(bufPenalty, penalty, bytes);

    pool.release(bufCurrent);
    pool.release(bufOptimal);
    pool.release(bufFisher);
    pool.release(bufPenalty);
}

// ── NLMS predict ─────────────────────────────────────────────────────────

void nlmsPredict(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                 const float* x, const float* w, const float* bias,
                 float* yPred, const NLMSPredictParams& p) {
    const size_t xBytes    = size_t(p.batchSize) * p.nFeatures * sizeof(float);
    const size_t wBytes    = size_t(p.nFeatures) * sizeof(float);
    const size_t biasBytes = sizeof(float);
    const size_t yBytes    = size_t(p.batchSize) * sizeof(float);

    GrillyBuffer bufX    = pool.acquire(xBytes);
    GrillyBuffer bufW    = pool.acquire(wBytes);
    GrillyBuffer bufBias = pool.acquire(biasBytes);
    GrillyBuffer bufY    = pool.acquire(yBytes);

    pool.upload(bufX, x, xBytes);
    pool.upload(bufW, w, wBytes);
    pool.upload(bufBias, bias, biasBytes);

    PipelineEntry pipe = cache.getOrCreate("nlms-predict", 4,
                                           sizeof(NLMSPredictParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufX.handle,    0, xBytes},
        {bufW.handle,    0, wBytes},
        {bufBias.handle, 0, biasBytes},
        {bufY.handle,    0, yBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("nlms-predict",
                                                        bufInfos);

    uint32_t gx = (p.batchSize + 255) / 256;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.submit();

    pool.download(bufY, yPred, yBytes);

    pool.release(bufX);
    pool.release(bufW);
    pool.release(bufBias);
    pool.release(bufY);
}

// ── NLMS update ──────────────────────────────────────────────────────────

void nlmsUpdate(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                const float* x, const float* yPred, const float* yTrue,
                float* w, float* bias, float* mu, float* error,
                const NLMSUpdateParams& p) {
    const size_t featureBytes = size_t(p.nFeatures) * sizeof(float);
    const size_t scalarBytes  = sizeof(float);

    GrillyBuffer bufX     = pool.acquire(featureBytes);
    GrillyBuffer bufYPred = pool.acquire(scalarBytes);
    GrillyBuffer bufYTrue = pool.acquire(scalarBytes);
    GrillyBuffer bufW     = pool.acquire(featureBytes);
    GrillyBuffer bufBias  = pool.acquire(scalarBytes);
    GrillyBuffer bufMu    = pool.acquire(scalarBytes);
    GrillyBuffer bufError = pool.acquire(scalarBytes);

    pool.upload(bufX, x, featureBytes);
    pool.upload(bufYPred, yPred, scalarBytes);
    pool.upload(bufYTrue, yTrue, scalarBytes);
    pool.upload(bufW, w, featureBytes);
    pool.upload(bufBias, bias, scalarBytes);
    pool.upload(bufMu, mu, scalarBytes);

    PipelineEntry pipe = cache.getOrCreate("nlms-update", 7,
                                           sizeof(NLMSUpdateParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufX.handle,     0, featureBytes},
        {bufYPred.handle, 0, scalarBytes},
        {bufYTrue.handle, 0, scalarBytes},
        {bufW.handle,     0, featureBytes},
        {bufBias.handle,  0, scalarBytes},
        {bufMu.handle,    0, scalarBytes},
        {bufError.handle, 0, scalarBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("nlms-update",
                                                        bufInfos);

    uint32_t gx = (p.nFeatures + 255) / 256;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.submit();

    pool.download(bufW, w, featureBytes);
    pool.download(bufBias, bias, scalarBytes);
    pool.download(bufMu, mu, scalarBytes);
    pool.download(bufError, error, scalarBytes);

    pool.release(bufX);
    pool.release(bufYPred);
    pool.release(bufYTrue);
    pool.release(bufW);
    pool.release(bufBias);
    pool.release(bufMu);
    pool.release(bufError);
}

// ── Continuous-to-spike bridge ───────────────────────────────────────────

void continuousToSpikes(CommandBatch& batch, BufferPool& pool,
                        PipelineCache& cache,
                        const float* features, float* spikes,
                        const float* W, const float* b,
                        const float* random, float* temp,
                        const ContinuousToSpikeParams& p) {
    const size_t featBytes   = size_t(p.batchSize) * p.inputDim * sizeof(float);
    const size_t spikeBytes  = size_t(p.batchSize) * p.numTimesteps * p.spikeDim *
                               sizeof(float);
    const size_t wBytes      = size_t(p.inputDim) * p.spikeDim * sizeof(float);
    const size_t bBytes      = size_t(p.spikeDim) * sizeof(float);
    const size_t randomBytes = spikeBytes;
    const size_t tempBytes   = size_t(p.batchSize) * p.spikeDim * sizeof(float);

    GrillyBuffer bufFeat   = pool.acquire(featBytes);
    GrillyBuffer bufSpikes = pool.acquire(spikeBytes);
    GrillyBuffer bufW      = pool.acquire(wBytes);
    GrillyBuffer bufB      = pool.acquire(bBytes);
    GrillyBuffer bufRandom = pool.acquire(randomBytes);
    GrillyBuffer bufTemp   = pool.acquire(tempBytes);

    pool.upload(bufFeat, features, featBytes);
    if (p.useProjection) {
        pool.upload(bufW, W, wBytes);
        pool.upload(bufB, b, bBytes);
    }
    pool.upload(bufRandom, random, randomBytes);

    PipelineEntry pipe = cache.getOrCreate("bridge-continuous-to-spike", 6,
                                           sizeof(ContinuousToSpikeParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufFeat.handle,   0, featBytes},
        {bufSpikes.handle, 0, spikeBytes},
        {bufW.handle,      0, wBytes},
        {bufB.handle,      0, bBytes},
        {bufRandom.handle, 0, randomBytes},
        {bufTemp.handle,   0, tempBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet(
        "bridge-continuous-to-spike", bufInfos);

    uint32_t totalOut = p.batchSize * p.numTimesteps * p.spikeDim;
    uint32_t gx = (totalOut + 255) / 256;

    batch.begin();

    // Pass 0: project (if using projection)
    ContinuousToSpikeParams push0 = p;
    push0.passType = 0;
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &push0, sizeof(push0));
    batch.barrier();

    // Pass 1: encode
    ContinuousToSpikeParams push1 = p;
    push1.passType = 1;
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &push1, sizeof(push1));

    batch.submit();

    pool.download(bufSpikes, spikes, spikeBytes);

    pool.release(bufFeat);
    pool.release(bufSpikes);
    pool.release(bufW);
    pool.release(bufB);
    pool.release(bufRandom);
    pool.release(bufTemp);
}

// ── Spike-to-continuous bridge ───────────────────────────────────────────

void spikesToContinuous(CommandBatch& batch, BufferPool& pool,
                        PipelineCache& cache,
                        const float* spikes, float* features,
                        const float* weights, const float* W,
                        const float* b, float* temp,
                        const SpikeToContinuousParams& p) {
    const size_t spikeBytes = size_t(p.batchSize) * p.totalTime * p.spikeDim *
                              sizeof(float);
    const size_t featBytes  = size_t(p.batchSize) * p.outputDim * sizeof(float);
    const size_t wDecBytes  = size_t(p.timeWindow) * sizeof(float);
    const size_t wProjBytes = size_t(p.spikeDim) * p.outputDim * sizeof(float);
    const size_t bBytes     = size_t(p.outputDim) * sizeof(float);
    const size_t tempBytes  = size_t(p.batchSize) * p.spikeDim * sizeof(float);

    GrillyBuffer bufSpikes  = pool.acquire(spikeBytes);
    GrillyBuffer bufFeat    = pool.acquire(featBytes);
    GrillyBuffer bufWeights = pool.acquire(wDecBytes);
    GrillyBuffer bufW       = pool.acquire(wProjBytes);
    GrillyBuffer bufB       = pool.acquire(bBytes);
    GrillyBuffer bufTemp    = pool.acquire(tempBytes);

    pool.upload(bufSpikes, spikes, spikeBytes);
    pool.upload(bufWeights, weights, wDecBytes);
    if (p.useProjection) {
        pool.upload(bufW, W, wProjBytes);
        pool.upload(bufB, b, bBytes);
    }

    PipelineEntry pipe = cache.getOrCreate("bridge-spike-to-continuous", 6,
                                           sizeof(SpikeToContinuousParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufSpikes.handle,  0, spikeBytes},
        {bufFeat.handle,    0, featBytes},
        {bufWeights.handle, 0, wDecBytes},
        {bufW.handle,       0, wProjBytes},
        {bufB.handle,       0, bBytes},
        {bufTemp.handle,    0, tempBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet(
        "bridge-spike-to-continuous", bufInfos);

    uint32_t totalOut = p.batchSize * p.outputDim;
    uint32_t gx = (totalOut + 255) / 256;

    batch.begin();

    // Pass 0: encode spikes
    SpikeToContinuousParams push0 = p;
    push0.passType = 0;
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &push0, sizeof(push0));
    batch.barrier();

    // Pass 1: project
    SpikeToContinuousParams push1 = p;
    push1.passType = 1;
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &push1, sizeof(push1));

    batch.submit();

    pool.download(bufFeat, features, featBytes);

    pool.release(bufSpikes);
    pool.release(bufFeat);
    pool.release(bufWeights);
    pool.release(bufW);
    pool.release(bufB);
    pool.release(bufTemp);
}

}  // namespace ops
}  // namespace grilly
