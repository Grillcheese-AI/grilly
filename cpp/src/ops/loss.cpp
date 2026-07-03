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
// ── Sampled-BCE (NCE/SGNS) FUSED loss + dH + dW ──────────────────────────
// The softmax-free vocab head. Two dispatches, ONE submit (the fixed cost is
// per submit, not per dispatch): pass 0 is workgroup-per-token (subgroup/LDS
// loss reduction, plain stores); pass 1 accumulates dH and scatters dW into
// the (V, d) table via a CAS-loop float add on core uint atomics — NO
// GL_EXT_shader_atomic_float (float buffer atomicAdd measured broken on this
// stack; all accumulations landed at flat index 0). gradTable is fillZero'd
// on-GPU before pass 1; losses is fully written by pass 0.

void sampledBceFused(CommandBatch& batch, BufferPool& pool,
                     PipelineCache& cache,
                     const float* hidden, const float* table,
                     const uint32_t* ids,
                     float* losses, float* gradHidden, float* gradTable,
                     uint32_t vocabSize, const SampledBceParams& p) {
    const size_t hBytes  = size_t(p.nTokens) * p.dim * sizeof(float);
    const size_t wBytes  = size_t(vocabSize) * p.dim * sizeof(float);
    const size_t idBytes = size_t(p.nTokens) * p.nCand * sizeof(uint32_t);
    const size_t dsBytes = size_t(p.nTokens) * p.nCand * sizeof(float);
    const size_t lBytes  = size_t(p.nTokens) * sizeof(float);

    // 3 stage-in (hidden, table, ids), 3 stage-out (losses, dH, dW).
    // dscore is an intermediate DL-only buffer (CPU never sees it).
    GrillyBuffer bufHDL   = pool.acquireDeviceLocal(hBytes);
    GrillyBuffer bufWDL   = pool.acquireDeviceLocal(wBytes);
    GrillyBuffer bufIdDL  = pool.acquireDeviceLocal(idBytes);
    GrillyBuffer bufDsDL  = pool.acquireDeviceLocal(dsBytes);
    GrillyBuffer bufLDL   = pool.acquireDeviceLocal(lBytes);
    GrillyBuffer bufGHDL  = pool.acquireDeviceLocal(hBytes);
    GrillyBuffer bufGWDL  = pool.acquireDeviceLocal(wBytes);

    GrillyBuffer bufHStage  = pool.acquire(hBytes);
    GrillyBuffer bufWStage  = pool.acquire(wBytes);
    GrillyBuffer bufIdStage = pool.acquire(idBytes);
    GrillyBuffer bufLStage  = pool.acquireReadback(lBytes);
    GrillyBuffer bufGHStage = pool.acquireReadback(hBytes);
    GrillyBuffer bufGWStage = pool.acquireReadback(wBytes);

    pool.upload(bufHStage, hidden, hBytes);
    pool.upload(bufWStage, table, wBytes);
    pool.upload(bufIdStage, reinterpret_cast<const float*>(ids), idBytes);

    PipelineEntry pipe = cache.getOrCreate("loss-sampled-bce-fused", 7,
                                           sizeof(SampledBceParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufHDL.handle,  0, hBytes},
        {bufWDL.handle,  0, wBytes},
        {bufIdDL.handle, 0, idBytes},
        {bufDsDL.handle, 0, dsBytes},
        {bufLDL.handle,  0, lBytes},
        {bufGHDL.handle, 0, hBytes},
        {bufGWDL.handle, 0, wBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("loss-sampled-bce-fused",
                                                        bufInfos);

    const uint32_t gx0 = p.nTokens;                       // workgroup per token
    const uint32_t gx1 = (p.nTokens * p.dim + 255u) / 256u;

    batch.begin();
    batch.copyBuffer(bufHStage, bufHDL, hBytes);
    batch.copyBuffer(bufWStage, bufWDL, wBytes);
    batch.copyBuffer(bufIdStage, bufIdDL, idBytes);
    batch.fillZero(bufGWDL, wBytes);   // CAS-add target: must start as 0 bits
    batch.transferComputeBarrier();

    // Pass 0: scores + per-token loss + dscore
    SampledBceParams push0 = p;
    push0.passType = 0;
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx0, 1, 1,
                   &push0, sizeof(push0));
    batch.barrier();

    // Pass 1: dH accumulate + dW atomic scatter
    SampledBceParams push1 = p;
    push1.passType = 1;
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx1, 1, 1,
                   &push1, sizeof(push1));

    batch.transferComputeBarrier();
    batch.copyBuffer(bufLDL, bufLStage, lBytes);
    batch.copyBuffer(bufGHDL, bufGHStage, hBytes);
    batch.copyBuffer(bufGWDL, bufGWStage, wBytes);

    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufLStage, losses, lBytes);
    pool.download(bufGHStage, gradHidden, hBytes);
    pool.download(bufGWStage, gradTable, wBytes);

    pool.release(bufHDL);  pool.release(bufWDL);  pool.release(bufIdDL);
    pool.release(bufDsDL); pool.release(bufLDL);  pool.release(bufGHDL);
    pool.release(bufGWDL);
    pool.release(bufHStage);  pool.release(bufWStage); pool.release(bufIdStage);
    pool.release(bufLStage);  pool.release(bufGHStage); pool.release(bufGWStage);
}

// ── NCE FUSED loss + dH + dW + db (corrected sampled-BCE) ────────────────
// Adds the noise-distribution correction the SGNS head lacks. Same two-pass /
// one-submit structure and CAS-add machinery as sampledBceFused, plus two
// read-only (V,) inputs (logkq, bias) and a third CAS grad output (grad_bias).

void nceFused(CommandBatch& batch, BufferPool& pool,
              PipelineCache& cache,
              const float* hidden, const float* table, const uint32_t* ids,
              const float* logkq, const float* bias,
              float* losses, float* gradHidden, float* gradTable,
              float* gradBias, uint32_t vocabSize, const NceParams& p) {
    const size_t hBytes  = size_t(p.nTokens) * p.dim * sizeof(float);
    const size_t wBytes  = size_t(vocabSize) * p.dim * sizeof(float);
    const size_t idBytes = size_t(p.nTokens) * p.nCand * sizeof(uint32_t);
    const size_t dsBytes = size_t(p.nTokens) * p.nCand * sizeof(float);
    const size_t lBytes  = size_t(p.nTokens) * sizeof(float);
    const size_t vBytes  = size_t(vocabSize) * sizeof(float);  // logkq/bias/db

    GrillyBuffer bufHDL   = pool.acquireDeviceLocal(hBytes);
    GrillyBuffer bufWDL   = pool.acquireDeviceLocal(wBytes);
    GrillyBuffer bufIdDL  = pool.acquireDeviceLocal(idBytes);
    GrillyBuffer bufDsDL  = pool.acquireDeviceLocal(dsBytes);
    GrillyBuffer bufLDL   = pool.acquireDeviceLocal(lBytes);
    GrillyBuffer bufGHDL  = pool.acquireDeviceLocal(hBytes);
    GrillyBuffer bufGWDL  = pool.acquireDeviceLocal(wBytes);
    GrillyBuffer bufLKQDL = pool.acquireDeviceLocal(vBytes);
    GrillyBuffer bufBDL   = pool.acquireDeviceLocal(vBytes);
    GrillyBuffer bufGBDL  = pool.acquireDeviceLocal(vBytes);

    GrillyBuffer bufHStage   = pool.acquire(hBytes);
    GrillyBuffer bufWStage   = pool.acquire(wBytes);
    GrillyBuffer bufIdStage  = pool.acquire(idBytes);
    GrillyBuffer bufLKQStage = pool.acquire(vBytes);
    GrillyBuffer bufBStage   = pool.acquire(vBytes);
    GrillyBuffer bufLStage   = pool.acquireReadback(lBytes);
    GrillyBuffer bufGHStage  = pool.acquireReadback(hBytes);
    GrillyBuffer bufGWStage  = pool.acquireReadback(wBytes);
    GrillyBuffer bufGBStage  = pool.acquireReadback(vBytes);

    pool.upload(bufHStage, hidden, hBytes);
    pool.upload(bufWStage, table, wBytes);
    pool.upload(bufIdStage, reinterpret_cast<const float*>(ids), idBytes);
    pool.upload(bufLKQStage, logkq, vBytes);
    pool.upload(bufBStage, bias, vBytes);

    PipelineEntry pipe = cache.getOrCreate("loss-nce-fused", 10,
                                           sizeof(NceParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufHDL.handle,   0, hBytes},
        {bufWDL.handle,   0, wBytes},
        {bufIdDL.handle,  0, idBytes},
        {bufDsDL.handle,  0, dsBytes},
        {bufLDL.handle,   0, lBytes},
        {bufGHDL.handle,  0, hBytes},
        {bufGWDL.handle,  0, wBytes},
        {bufLKQDL.handle, 0, vBytes},
        {bufBDL.handle,   0, vBytes},
        {bufGBDL.handle,  0, vBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("loss-nce-fused",
                                                        bufInfos);

    const uint32_t gx0 = p.nTokens;
    const uint32_t gx1 = (p.nTokens * p.dim + 255u) / 256u;

    batch.begin();
    batch.copyBuffer(bufHStage, bufHDL, hBytes);
    batch.copyBuffer(bufWStage, bufWDL, wBytes);
    batch.copyBuffer(bufIdStage, bufIdDL, idBytes);
    batch.copyBuffer(bufLKQStage, bufLKQDL, vBytes);
    batch.copyBuffer(bufBStage, bufBDL, vBytes);
    batch.fillZero(bufGWDL, wBytes);
    batch.fillZero(bufGBDL, vBytes);
    batch.transferComputeBarrier();

    NceParams push0 = p; push0.passType = 0;
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx0, 1, 1,
                   &push0, sizeof(push0));
    batch.barrier();

    NceParams push1 = p; push1.passType = 1;
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx1, 1, 1,
                   &push1, sizeof(push1));

    batch.transferComputeBarrier();
    batch.copyBuffer(bufLDL, bufLStage, lBytes);
    batch.copyBuffer(bufGHDL, bufGHStage, hBytes);
    batch.copyBuffer(bufGWDL, bufGWStage, wBytes);
    batch.copyBuffer(bufGBDL, bufGBStage, vBytes);

    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufLStage, losses, lBytes);
    pool.download(bufGHStage, gradHidden, hBytes);
    pool.download(bufGWStage, gradTable, wBytes);
    pool.download(bufGBStage, gradBias, vBytes);

    pool.release(bufHDL);  pool.release(bufWDL);  pool.release(bufIdDL);
    pool.release(bufDsDL); pool.release(bufLDL);  pool.release(bufGHDL);
    pool.release(bufGWDL); pool.release(bufLKQDL); pool.release(bufBDL);
    pool.release(bufGBDL);
    pool.release(bufHStage);   pool.release(bufWStage);  pool.release(bufIdStage);
    pool.release(bufLKQStage); pool.release(bufBStage);  pool.release(bufLStage);
    pool.release(bufGHStage);  pool.release(bufGWStage); pool.release(bufGBStage);
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
