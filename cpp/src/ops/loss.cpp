#include "grilly/ops/loss.h"

#include <cstring>
#include <vector>

namespace grilly {
namespace ops {

// ═══════════════════════════════════════════════════════════════════════════
// Loss functions
// ═══════════════════════════════════════════════════════════════════════════

// ── Cross-entropy loss forward ───────────────────────────────────────────
// 3-pass like softmax: max → sum_exp → loss.
// The shader uses log-sum-exp for numerical stability.

void crossEntropyLoss(CommandBatch& batch, BufferPool& pool,
                      PipelineCache& cache,
                      const float* logits, const uint32_t* targets,
                      float* losses, const CrossEntropyParams& p) {
    const uint32_t totalPositions = p.batchSize * p.seqLen;
    const size_t logitBytes  = size_t(totalPositions) * p.vocabSize * sizeof(float);
    const size_t targetBytes = size_t(totalPositions) * sizeof(uint32_t);
    const size_t lossBytes   = size_t(totalPositions) * sizeof(float);
    const size_t auxBytes    = size_t(totalPositions) * sizeof(float);

    GrillyBuffer bufLogits = pool.acquire(logitBytes);
    GrillyBuffer bufTarget = pool.acquire(targetBytes);
    GrillyBuffer bufLoss   = pool.acquire(lossBytes);
    GrillyBuffer bufMax    = pool.acquire(auxBytes);
    GrillyBuffer bufSumExp = pool.acquire(auxBytes);

    pool.upload(bufLogits, logits, logitBytes);
    pool.upload(bufTarget, reinterpret_cast<const float*>(targets), targetBytes);

    PipelineEntry pipe = cache.getOrCreate("loss-cross-entropy", 5,
                                           sizeof(CrossEntropyParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufLogits.handle, 0, logitBytes},
        {bufTarget.handle, 0, targetBytes},
        {bufLoss.handle,   0, lossBytes},
        {bufMax.handle,    0, auxBytes},
        {bufSumExp.handle, 0, auxBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("loss-cross-entropy",
                                                        bufInfos);

    uint32_t gx = (totalPositions + 255) / 256;

    batch.begin();

    // Pass 0: find max logit per position
    CrossEntropyParams push0 = p;
    push0.passType = 0;
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &push0, sizeof(push0));
    batch.barrier();

    // Pass 1: compute sum_exp
    CrossEntropyParams push1 = p;
    push1.passType = 1;
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &push1, sizeof(push1));
    batch.barrier();

    // Pass 2: compute loss
    CrossEntropyParams push2 = p;
    push2.passType = 2;
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &push2, sizeof(push2));

    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufLoss, losses, lossBytes);

    pool.release(bufLogits);
    pool.release(bufTarget);
    pool.release(bufLoss);
    pool.release(bufMax);
    pool.release(bufSumExp);
}

// ── Cross-entropy backward ───────────────────────────────────────────────

void crossEntropyBackward(CommandBatch& batch, BufferPool& pool,
                          PipelineCache& cache,
                          const float* logits, const uint32_t* targets,
                          float* gradLogits,
                          const CrossEntropyBackwardParams& p) {
    const size_t logitBytes = size_t(p.batchSize) * p.numClasses * sizeof(float);
    const size_t targetBytes = size_t(p.batchSize) * sizeof(uint32_t);

    GrillyBuffer bufLogits = pool.acquire(logitBytes);
    GrillyBuffer bufTarget = pool.acquire(targetBytes);
    GrillyBuffer bufGrad   = pool.acquire(logitBytes);

    pool.upload(bufLogits, logits, logitBytes);
    pool.upload(bufTarget, reinterpret_cast<const float*>(targets), targetBytes);

    PipelineEntry pipe = cache.getOrCreate("cross-entropy-backward", 3,
                                           sizeof(CrossEntropyBackwardParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufLogits.handle, 0, logitBytes},
        {bufTarget.handle, 0, targetBytes},
        {bufGrad.handle,   0, logitBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("cross-entropy-backward",
                                                        bufInfos);

    uint32_t gx = (p.batchSize + 255) / 256;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufGrad, gradLogits, logitBytes);

    pool.release(bufLogits);
    pool.release(bufTarget);
    pool.release(bufGrad);
}

}  // namespace ops
}  // namespace grilly
