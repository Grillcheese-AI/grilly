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

    // Staging pattern: 2 stage-in (logits, targets), 1 stage-out (losses).
    // max and sumExp are intermediate DL-only buffers (CPU never sees them).
    GrillyBuffer bufLogitsDL = pool.acquireDeviceLocal(logitBytes);
    GrillyBuffer bufTargetDL = pool.acquireDeviceLocal(targetBytes);
    GrillyBuffer bufLossDL   = pool.acquireDeviceLocal(lossBytes);
    GrillyBuffer bufMaxDL    = pool.acquireDeviceLocal(auxBytes);
    GrillyBuffer bufSumExpDL = pool.acquireDeviceLocal(auxBytes);

    GrillyBuffer bufLogitsStage = pool.acquire(logitBytes);
    GrillyBuffer bufTargetStage = pool.acquire(targetBytes);
    GrillyBuffer bufLossStage   = pool.acquireReadback(lossBytes);

    pool.upload(bufLogitsStage, logits, logitBytes);
    pool.upload(bufTargetStage, reinterpret_cast<const float*>(targets), targetBytes);

    PipelineEntry pipe = cache.getOrCreate("loss-cross-entropy", 5,
                                           sizeof(CrossEntropyParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufLogitsDL.handle, 0, logitBytes},
        {bufTargetDL.handle, 0, targetBytes},
        {bufLossDL.handle,   0, lossBytes},
        {bufMaxDL.handle,    0, auxBytes},
        {bufSumExpDL.handle, 0, auxBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("loss-cross-entropy",
                                                        bufInfos);

    uint32_t gx = (totalPositions + 255) / 256;

    batch.begin();
    batch.copyBuffer(bufLogitsStage, bufLogitsDL, logitBytes);
    batch.copyBuffer(bufTargetStage, bufTargetDL, targetBytes);
    batch.transferComputeBarrier();

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

    batch.transferComputeBarrier();
    batch.copyBuffer(bufLossDL, bufLossStage, lossBytes);

    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufLossStage, losses, lossBytes);

    pool.release(bufLogitsDL);
    pool.release(bufTargetDL);
    pool.release(bufLossDL);
    pool.release(bufMaxDL);
    pool.release(bufSumExpDL);
    pool.release(bufLogitsStage);
    pool.release(bufTargetStage);
    pool.release(bufLossStage);
}

// ── Cross-entropy backward ───────────────────────────────────────────────

void crossEntropyBackward(CommandBatch& batch, BufferPool& pool,
                          PipelineCache& cache,
                          const float* logits, const uint32_t* targets,
                          float* gradLogits,
                          const CrossEntropyBackwardParams& p) {
    const size_t logitBytes = size_t(p.batchSize) * p.numClasses * sizeof(float);
    const size_t targetBytes = size_t(p.batchSize) * sizeof(uint32_t);

    // Staging pattern: 2 stage-in (logits, targets), 1 stage-out (gradLogits)
    GrillyBuffer bufLogitsDL = pool.acquireDeviceLocal(logitBytes);
    GrillyBuffer bufTargetDL = pool.acquireDeviceLocal(targetBytes);
    GrillyBuffer bufGradDL   = pool.acquireDeviceLocal(logitBytes);

    GrillyBuffer bufLogitsStage = pool.acquire(logitBytes);
    GrillyBuffer bufTargetStage = pool.acquire(targetBytes);
    GrillyBuffer bufGradStage   = pool.acquireReadback(logitBytes);

    pool.upload(bufLogitsStage, logits, logitBytes);
    pool.upload(bufTargetStage, reinterpret_cast<const float*>(targets), targetBytes);

    PipelineEntry pipe = cache.getOrCreate("cross-entropy-backward", 3,
                                           sizeof(CrossEntropyBackwardParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufLogitsDL.handle, 0, logitBytes},
        {bufTargetDL.handle, 0, targetBytes},
        {bufGradDL.handle,   0, logitBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("cross-entropy-backward",
                                                        bufInfos);

    uint32_t gx = (p.batchSize + 255) / 256;

    batch.begin();
    batch.copyBuffer(bufLogitsStage, bufLogitsDL, logitBytes);
    batch.copyBuffer(bufTargetStage, bufTargetDL, targetBytes);
    batch.transferComputeBarrier();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.transferComputeBarrier();
    batch.copyBuffer(bufGradDL, bufGradStage, logitBytes);
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufGradStage, gradLogits, logitBytes);

    pool.release(bufLogitsDL);
    pool.release(bufTargetDL);
    pool.release(bufGradDL);
    pool.release(bufLogitsStage);
    pool.release(bufTargetStage);
    pool.release(bufGradStage);
}

}  // namespace ops
}  // namespace grilly
