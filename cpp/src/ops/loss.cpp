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

// â”€â”€ Cross-entropy FUSED loss + gradient â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// ONE dispatch: per-row loss AND grad_logits = softmax - one_hot, sharing a
// single subgroup-reduced max + sum_exp pass per row. Workgroup-per-row, so
// gx = batchSize (NOT (batchSize+255)/256 like the tree-reduction backward).

void crossEntropyFused(CommandBatch& batch, BufferPool& pool,
                       PipelineCache& cache,
                       const float* logits, const uint32_t* targets,
                       float* losses, float* gradLogits,
                       const CrossEntropyFusedParams& p) {
    const size_t logitBytes  = size_t(p.batchSize) * p.numClasses * sizeof(float);
    const size_t targetBytes = size_t(p.batchSize) * sizeof(uint32_t);
    const size_t lossBytes   = size_t(p.batchSize) * sizeof(float);

    // 2 stage-in (logits, targets), 2 stage-out (losses, gradLogits).
    GrillyBuffer bufLogitsDL = pool.acquireDeviceLocal(logitBytes);
    GrillyBuffer bufTargetDL = pool.acquireDeviceLocal(targetBytes);
    GrillyBuffer bufLossDL   = pool.acquireDeviceLocal(lossBytes);
    GrillyBuffer bufGradDL   = pool.acquireDeviceLocal(logitBytes);

    GrillyBuffer bufLogitsStage = pool.acquire(logitBytes);
    GrillyBuffer bufTargetStage = pool.acquire(targetBytes);
    GrillyBuffer bufLossStage   = pool.acquireReadback(lossBytes);
    GrillyBuffer bufGradStage   = pool.acquireReadback(logitBytes);

    pool.upload(bufLogitsStage, logits, logitBytes);
    pool.upload(bufTargetStage, reinterpret_cast<const float*>(targets), targetBytes);

    PipelineEntry pipe = cache.getOrCreate("loss-ce-fused", 4,
                                           sizeof(CrossEntropyFusedParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufLogitsDL.handle, 0, logitBytes},
        {bufTargetDL.handle, 0, targetBytes},
        {bufLossDL.handle,   0, lossBytes},
        {bufGradDL.handle,   0, logitBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("loss-ce-fused", bufInfos);

    uint32_t gx = p.batchSize;  // one workgroup per row

    batch.begin();
    batch.copyBuffer(bufLogitsStage, bufLogitsDL, logitBytes);
    batch.copyBuffer(bufTargetStage, bufTargetDL, targetBytes);
    batch.transferComputeBarrier();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.transferComputeBarrier();
    batch.copyBuffer(bufLossDL, bufLossStage, lossBytes);
    batch.copyBuffer(bufGradDL, bufGradStage, logitBytes);
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufLossStage, losses, lossBytes);
    pool.download(bufGradStage, gradLogits, logitBytes);

    pool.release(bufLogitsDL);
    pool.release(bufTargetDL);
    pool.release(bufLossDL);
    pool.release(bufGradDL);
    pool.release(bufLogitsStage);
    pool.release(bufTargetStage);
    pool.release(bufLossStage);
    pool.release(bufGradStage);
}
// ── MSE Loss ─────────────────────────────────────────────────────────────

void mseLoss(CommandBatch& batch, BufferPool& pool,
             PipelineCache& cache,
             const float* preds, const float* targets,
             float* losses, const MSELossParams& p) {
    const size_t bytes = size_t(p.n) * sizeof(float);

    GrillyBuffer bPDL = pool.acquireDeviceLocal(bytes);
    GrillyBuffer bTDL = pool.acquireDeviceLocal(bytes);
    GrillyBuffer bLDL = pool.acquireDeviceLocal(bytes);

    GrillyBuffer bPStage = pool.acquire(bytes);
    GrillyBuffer bTStage = pool.acquire(bytes);
    GrillyBuffer bLStage = pool.acquireReadback(bytes);

    pool.upload(bPStage, preds, bytes);
    pool.upload(bTStage, targets, bytes);

    PipelineEntry pipe = cache.getOrCreate("loss-mse", 3, sizeof(MSELossParams));

    std::vector<VkDescriptorBufferInfo> bufs = {
        {bPDL.handle, 0, bytes},
        {bTDL.handle, 0, bytes},
        {bLDL.handle, 0, bytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("loss-mse", bufs);

    uint32_t gx = (p.n + 255u) / 256u;

    batch.begin();
    batch.copyBuffer(bPStage, bPDL, bytes);
    batch.copyBuffer(bTStage, bTDL, bytes);
    batch.transferComputeBarrier();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1, &p, sizeof(p));
    batch.transferComputeBarrier();
    batch.copyBuffer(bLDL, bLStage, bytes);
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bLStage, losses, bytes);

    pool.release(bPDL); pool.release(bTDL); pool.release(bLDL);
    pool.release(bPStage); pool.release(bTStage); pool.release(bLStage);
}

// ── Cosine Similarity Loss ───────────────────────────────────────────────

void cosineSimilarityLoss(CommandBatch& batch, BufferPool& pool,
                          PipelineCache& cache,
                          const float* preds, const float* targets,
                          float* losses, const CosineLossParams& p) {
    const size_t inBytes = size_t(p.batchSize) * p.dim * sizeof(float);
    const size_t outBytes = size_t(p.batchSize) * sizeof(float);

    GrillyBuffer bPDL = pool.acquireDeviceLocal(inBytes);
    GrillyBuffer bTDL = pool.acquireDeviceLocal(inBytes);
    GrillyBuffer bLDL = pool.acquireDeviceLocal(outBytes);

    GrillyBuffer bPStage = pool.acquire(inBytes);
    GrillyBuffer bTStage = pool.acquire(inBytes);
    GrillyBuffer bLStage = pool.acquireReadback(outBytes);

    pool.upload(bPStage, preds, inBytes);
    pool.upload(bTStage, targets, inBytes);

    PipelineEntry pipe = cache.getOrCreate("loss-cosine", 3, sizeof(CosineLossParams));

    std::vector<VkDescriptorBufferInfo> bufs = {
        {bPDL.handle, 0, inBytes},
        {bTDL.handle, 0, inBytes},
        {bLDL.handle, 0, outBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("loss-cosine", bufs);

    uint32_t gx = (p.batchSize + 63u) / 64u;

    batch.begin();
    batch.copyBuffer(bPStage, bPDL, inBytes);
    batch.copyBuffer(bTStage, bTDL, inBytes);
    batch.transferComputeBarrier();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1, &p, sizeof(p));
    batch.transferComputeBarrier();
    batch.copyBuffer(bLDL, bLStage, outBytes);
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bLStage, losses, outBytes);

    pool.release(bPDL); pool.release(bTDL); pool.release(bLDL);
    pool.release(bPStage); pool.release(bTStage); pool.release(bLStage);
}

}  // namespace ops
}  // namespace grilly
