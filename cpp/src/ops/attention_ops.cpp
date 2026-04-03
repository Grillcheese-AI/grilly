#include "grilly/ops/attention_ops.h"

#include <cstring>

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
