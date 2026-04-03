#include "grilly/ops/attention_ops.h"

#include "grilly/ops/activations.h"

#include <cstring>
#include <stdexcept>

namespace grilly {
namespace ops {

// ═══════════════════════════════════════════════════════════════════════════
// Decomposed attention ops
//
// These ops expose the individual steps of multi-head attention as separate
// GPU dispatches. The full pipeline is:
//   1. attentionScores: Q @ K^T scaled by 1/sqrt(d_h)
//   2. attentionMask: apply causal or custom mask
//   3. softmax (from activations module, applied to scores)
//   4. attentionOutput: softmax(scores) @ V
//   5. attentionConcatHeads: reshape (B,H,S,D) → (B,S,H*D)
//
// Each matches the standard acquire → upload → dispatch → download → release
// pattern. The attention-scores shader uses 2D (16×16) workgroups for the
// matmul, while mask/output/concat use 1D (256) element-parallel dispatch.
// ═══════════════════════════════════════════════════════════════════════════

// ── Attention scores ─────────────────────────────────────────────────────

void attentionScores(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                     const float* Q, const float* K,
                     float* scores, const AttentionScoresParams& p) {
    const size_t qkvBytes = size_t(p.batchSize) * p.numHeads * p.seqLen *
                            p.headDim * sizeof(float);
    const size_t scoreBytes = size_t(p.batchSize) * p.numHeads * p.seqLen *
                              p.seqLen * sizeof(float);

    GrillyBuffer bufQ      = pool.acquire(qkvBytes);
    GrillyBuffer bufK      = pool.acquire(qkvBytes);
    // V buffer is required by the shader binding but not used for scores
    GrillyBuffer bufV      = pool.acquire(sizeof(float));
    GrillyBuffer bufScores = pool.acquire(scoreBytes);

    pool.upload(bufQ, Q, qkvBytes);
    pool.upload(bufK, K, qkvBytes);

    PipelineEntry pipe = cache.getOrCreate("attention-scores", 4,
                                           sizeof(AttentionScoresParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufQ.handle,      0, qkvBytes},
        {bufK.handle,      0, qkvBytes},
        {bufV.handle,      0, sizeof(float)},
        {bufScores.handle, 0, scoreBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("attention-scores",
                                                        bufInfos);

    uint32_t gx = (p.seqLen + 15) / 16;
    uint32_t gy = (p.seqLen + 15) / 16;
    uint32_t gz = p.batchSize * p.numHeads;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, gy, gz,
                   &p, sizeof(p));
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufScores, scores, scoreBytes);

    pool.release(bufQ);
    pool.release(bufK);
    pool.release(bufV);
    pool.release(bufScores);
}

void attentionScoresSoftmaxOutput(CommandBatch& batch, BufferPool& pool,
                                  PipelineCache& cache, const float* Q,
                                  const float* K, const float* V, float* output,
                                  float* softmaxWeights, const AttentionScoresParams& sp,
                                  const AttentionOutputParams& op) {
    const uint32_t B = sp.batchSize;
    const uint32_t S = sp.seqLen;
    const uint32_t H = sp.numHeads;
    const uint32_t D = sp.headDim;

    if (op.batchSize != B || op.seqLen != S || op.numHeads != H || op.headDim != D) {
        throw std::runtime_error(
            "attentionScoresSoftmaxOutput: score and output params must match");
    }

    const size_t qkvBytes = size_t(B) * H * S * D * sizeof(float);
    const size_t scoreBytes = size_t(B) * H * S * S * sizeof(float);
    const size_t outBytes = qkvBytes;

    GrillyBuffer bufQ = pool.acquire(qkvBytes);
    GrillyBuffer bufK = pool.acquire(qkvBytes);
    GrillyBuffer bufVDummy = pool.acquire(sizeof(float));
    GrillyBuffer bufScores = pool.acquire(scoreBytes);
    GrillyBuffer bufWeights = pool.acquire(scoreBytes);
    GrillyBuffer bufV = pool.acquire(qkvBytes);
    GrillyBuffer bufOut = pool.acquire(outBytes);

    const uint32_t totalSoftmaxRows = B * H * S;
    const size_t auxBytes = size_t(totalSoftmaxRows) * sizeof(float);

    GrillyBuffer bufMax = pool.acquire(auxBytes);
    GrillyBuffer bufSumExp = pool.acquire(auxBytes);

    pool.upload(bufQ, Q, qkvBytes);
    pool.upload(bufK, K, qkvBytes);
    pool.upload(bufV, V, qkvBytes);

    PipelineEntry pipeScores = cache.getOrCreate("attention-scores", 4,
                                                 sizeof(AttentionScoresParams));
    std::vector<VkDescriptorBufferInfo> scoresInfos = {
        {bufQ.handle, 0, qkvBytes},
        {bufK.handle, 0, qkvBytes},
        {bufVDummy.handle, 0, sizeof(float)},
        {bufScores.handle, 0, scoreBytes},
    };
    VkDescriptorSet descScores = cache.allocDescriptorSet("attention-scores", scoresInfos);

    PipelineEntry pipeSoftmax = cache.getOrCreate("activation-softmax", 4,
                                                  sizeof(SoftmaxParams));
    std::vector<VkDescriptorBufferInfo> softmaxInfos = {
        {bufScores.handle, 0, scoreBytes},
        {bufWeights.handle, 0, scoreBytes},
        {bufMax.handle, 0, auxBytes},
        {bufSumExp.handle, 0, auxBytes},
    };
    VkDescriptorSet descSoftmax = cache.allocDescriptorSet("activation-softmax", softmaxInfos);

    PipelineEntry pipeOut = cache.getOrCreate("attention-output", 3,
                                              sizeof(AttentionOutputParams));
    std::vector<VkDescriptorBufferInfo> outInfos = {
        {bufWeights.handle, 0, scoreBytes},
        {bufV.handle, 0, qkvBytes},
        {bufOut.handle, 0, outBytes},
    };
    VkDescriptorSet descOut = cache.allocDescriptorSet("attention-output", outInfos);

    const uint32_t gxS = (S + 15) / 16;
    const uint32_t gyS = (S + 15) / 16;
    const uint32_t gzS = B * H;

    const uint32_t softmaxGx = (totalSoftmaxRows + 255) / 256;
    const uint32_t totalElements = totalSoftmaxRows * S;
    const uint32_t softmaxGx2 = (totalElements + 255) / 256;

    const uint32_t outTotal = B * H * S * D;
    const uint32_t gxOut = (outTotal + 255) / 256;

    batch.begin();

    batch.dispatch(pipeScores.pipeline, pipeScores.layout, descScores, gxS, gyS, gzS,
                   &sp, sizeof(sp));
    batch.barrier();

    SoftmaxParams push0{1, totalSoftmaxRows, S, 0, S};
    batch.dispatch(pipeSoftmax.pipeline, pipeSoftmax.layout, descSoftmax, softmaxGx, 1, 1,
                   &push0, sizeof(push0));
    batch.barrier();

    SoftmaxParams push1{1, totalSoftmaxRows, S, 1, S};
    batch.dispatch(pipeSoftmax.pipeline, pipeSoftmax.layout, descSoftmax, softmaxGx, 1, 1,
                   &push1, sizeof(push1));
    batch.barrier();

    SoftmaxParams push2{1, totalSoftmaxRows, S, 2, S};
    batch.dispatch(pipeSoftmax.pipeline, pipeSoftmax.layout, descSoftmax, softmaxGx2, 1, 1,
                   &push2, sizeof(push2));
    batch.barrier();

    batch.dispatch(pipeOut.pipeline, pipeOut.layout, descOut, gxOut, 1, 1, &op,
                   sizeof(op));

    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufOut, output, outBytes);
    pool.download(bufWeights, softmaxWeights, scoreBytes);

    pool.release(bufQ);
    pool.release(bufK);
    pool.release(bufVDummy);
    pool.release(bufScores);
    pool.release(bufWeights);
    pool.release(bufV);
    pool.release(bufOut);
    pool.release(bufMax);
    pool.release(bufSumExp);
}

// ── Attention mask ───────────────────────────────────────────────────────

void attentionMask(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                   float* scores, const float* mask,
                   const AttentionMaskParams& p) {
    const size_t scoreBytes = size_t(p.batchSize) * p.numHeads * p.seqLen *
                              p.seqLen * sizeof(float);
    // Mask may be (1, 1, S, S) for causal or (B, H, S, S) for custom
    const size_t maskBytes = p.useCausalMask ?
        sizeof(float) :  // causal mask is generated in-shader
        scoreBytes;

    GrillyBuffer bufScores = pool.acquire(scoreBytes);
    GrillyBuffer bufMask   = pool.acquire(maskBytes);

    pool.upload(bufScores, scores, scoreBytes);
    if (!p.useCausalMask && mask != nullptr) {
        pool.upload(bufMask, mask, maskBytes);
    }

    PipelineEntry pipe = cache.getOrCreate("attention-mask", 2,
                                           sizeof(AttentionMaskParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufScores.handle, 0, scoreBytes},
        {bufMask.handle,   0, maskBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("attention-mask",
                                                        bufInfos);

    uint32_t total = p.batchSize * p.numHeads * p.seqLen * p.seqLen;
    uint32_t gx = (total + 255) / 256;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufScores, scores, scoreBytes);

    pool.release(bufScores);
    pool.release(bufMask);
}

// ── Attention output ─────────────────────────────────────────────────────

void attentionOutput(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                     const float* weights, const float* V,
                     float* output, const AttentionOutputParams& p) {
    const size_t weightsBytes = size_t(p.batchSize) * p.numHeads * p.seqLen *
                                p.seqLen * sizeof(float);
    const size_t vBytes = size_t(p.batchSize) * p.numHeads * p.seqLen *
                          p.headDim * sizeof(float);
    const size_t outBytes = vBytes;  // same shape as V

    GrillyBuffer bufWeights = pool.acquire(weightsBytes);
    GrillyBuffer bufV       = pool.acquire(vBytes);
    GrillyBuffer bufOutput  = pool.acquire(outBytes);

    pool.upload(bufWeights, weights, weightsBytes);
    pool.upload(bufV, V, vBytes);

    PipelineEntry pipe = cache.getOrCreate("attention-output", 3,
                                           sizeof(AttentionOutputParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufWeights.handle, 0, weightsBytes},
        {bufV.handle,       0, vBytes},
        {bufOutput.handle,  0, outBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("attention-output",
                                                        bufInfos);

    uint32_t total = p.batchSize * p.numHeads * p.seqLen * p.headDim;
    uint32_t gx = (total + 255) / 256;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufOutput, output, outBytes);

    pool.release(bufWeights);
    pool.release(bufV);
    pool.release(bufOutput);
}

// ── Concat heads ─────────────────────────────────────────────────────────

void attentionConcatHeads(CommandBatch& batch, BufferPool& pool,
                          PipelineCache& cache,
                          const float* mhOutput, float* concatOutput,
                          const ConcatHeadsParams& p) {
    const size_t inBytes = size_t(p.batchSize) * p.numHeads * p.seqLen *
                           p.headDim * sizeof(float);
    const size_t outBytes = inBytes;  // same total elements, different layout

    GrillyBuffer bufIn  = pool.acquire(inBytes);
    GrillyBuffer bufOut = pool.acquire(outBytes);

    pool.upload(bufIn, mhOutput, inBytes);

    PipelineEntry pipe = cache.getOrCreate("attention-concat-heads", 2,
                                           sizeof(ConcatHeadsParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufIn.handle,  0, inBytes},
        {bufOut.handle, 0, outBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("attention-concat-heads",
                                                        bufInfos);

    uint32_t total = p.batchSize * p.numHeads * p.seqLen * p.headDim;
    uint32_t gx = (total + 255) / 256;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufOut, concatOutput, outBytes);

    pool.release(bufIn);
    pool.release(bufOut);
}

// ── RoPE ─────────────────────────────────────────────────────────────────

void applyRoPE(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
               const float* input, float* output,
               const float* cosTable, const float* sinTable,
               const RoPEParams& p) {
    const size_t dataBytes = size_t(p.batchSize) * p.numHeads * p.seqLen *
                             p.headDim * sizeof(float);
    // Tables are (seqLen, headDim/2) each
    const size_t tableBytes = size_t(p.seqLen) * (p.headDim / 2) *
                              sizeof(float);

    GrillyBuffer bufIn    = pool.acquire(dataBytes);
    GrillyBuffer bufOut   = pool.acquire(dataBytes);
    GrillyBuffer bufCos   = pool.acquire(tableBytes);
    GrillyBuffer bufSin   = pool.acquire(tableBytes);

    pool.upload(bufIn, input, dataBytes);
    if (p.usePrecomputed && cosTable && sinTable) {
        pool.upload(bufCos, cosTable, tableBytes);
        pool.upload(bufSin, sinTable, tableBytes);
    }

    PipelineEntry pipe = cache.getOrCreate("rope", 4, sizeof(RoPEParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufIn.handle,  0, dataBytes},
        {bufOut.handle, 0, dataBytes},
        {bufCos.handle, 0, tableBytes},
        {bufSin.handle, 0, tableBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("rope", bufInfos);

    uint32_t total = p.batchSize * p.numHeads * p.seqLen * p.headDim;
    uint32_t gx = (total + 255) / 256;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufOut, output, dataBytes);

    pool.release(bufIn);
    pool.release(bufOut);
    pool.release(bufCos);
    pool.release(bufSin);
}

}  // namespace ops
}  // namespace grilly
