/// batched_ops.cpp — Buffer-handle ops that record into external CommandBatch.
///
/// These variants do NOT call batch.begin()/submit(). The caller chains
/// multiple dispatches in one batch, inserts barriers, then submits once.
/// Buffers must be pre-acquired and pre-uploaded by the caller.
///
/// This eliminates the per-op submit+fence overhead (the main bottleneck
/// when chaining many small ops like in a transformer).

#include "grilly/ops/batched_ops.h"
#include "grilly/ops/linear.h"
#include "grilly/ops/activations.h"

namespace grilly {
namespace ops {

void batchedLinear(CommandBatch& batch, PipelineCache& cache,
                   GrillyBuffer& input, GrillyBuffer& weight,
                   GrillyBuffer* bias, GrillyBuffer& output,
                   uint32_t M, uint32_t K, uint32_t N) {

    PipelineEntry pipe = cache.getOrCreate("fnn-linear", 4, 16);

    size_t inputBytes  = size_t(M) * K * sizeof(float);
    size_t weightBytes = size_t(N) * K * sizeof(float);
    size_t biasBytes   = bias ? size_t(N) * sizeof(float) : sizeof(float);
    size_t outputBytes = size_t(M) * N * sizeof(float);

    std::vector<VkDescriptorBufferInfo> bufs = {
        {input.handle,  0, inputBytes},
        {weight.handle, 0, weightBytes},
        {bias ? bias->handle : input.handle, 0, biasBytes},
        {output.handle, 0, outputBytes},
    };
    VkDescriptorSet desc = cache.allocDescriptorSet("fnn-linear", bufs);

    LinearParams push{M, K, N, bias ? 1u : 0u};
    uint32_t gx = (N + 15) / 16;
    uint32_t gy = (M + 15) / 16;

    // Record dispatch — does NOT submit
    batch.dispatch(pipe.pipeline, pipe.layout, desc, gx, gy, 1,
                   &push, sizeof(push));
}

void batchedGelu(CommandBatch& batch, PipelineCache& cache,
                 GrillyBuffer& input, GrillyBuffer& output,
                 uint32_t totalElements) {

    PipelineEntry pipe = cache.getOrCreate("activation-gelu", 2, sizeof(uint32_t));

    size_t bytes = size_t(totalElements) * sizeof(float);
    std::vector<VkDescriptorBufferInfo> bufs = {
        {input.handle,  0, bytes},
        {output.handle, 0, bytes},
    };
    VkDescriptorSet desc = cache.allocDescriptorSet("activation-gelu", bufs);

    uint32_t push = totalElements;
    uint32_t gx = (totalElements + 255) / 256;

    batch.dispatch(pipe.pipeline, pipe.layout, desc, gx, 1, 1,
                   &push, sizeof(push));
}

void batchedFusedMlpGelu(CommandBatch& batch, PipelineCache& cache,
                          GrillyBuffer& input,
                          GrillyBuffer& w1, GrillyBuffer& b1,
                          GrillyBuffer& w2, GrillyBuffer& b2,
                          GrillyBuffer& output,
                          uint32_t seqLen) {

    PipelineEntry pipe = cache.getOrCreate("fused-mlp-gelu", 6, 0);

    // Sizes derived from buffer.size fields
    std::vector<VkDescriptorBufferInfo> bufs = {
        {input.handle,  0, input.size},
        {w1.handle,     0, w1.size},
        {b1.handle,     0, b1.size},
        {w2.handle,     0, w2.size},
        {b2.handle,     0, b2.size},
        {output.handle, 0, output.size},
    };
    VkDescriptorSet desc = cache.allocDescriptorSet("fused-mlp-gelu", bufs);

    batch.dispatch(pipe.pipeline, pipe.layout, desc, seqLen, 1, 1,
                   nullptr, 0);
}

void batchedFusedLnLinear(CommandBatch& batch, PipelineCache& cache,
                           GrillyBuffer& input,
                           GrillyBuffer& lnW, GrillyBuffer& lnB,
                           GrillyBuffer& projW, GrillyBuffer& projB,
                           GrillyBuffer& output,
                           uint32_t seqLen) {

    PipelineEntry pipe = cache.getOrCreate("fused-layernorm-linear", 6, 0);

    std::vector<VkDescriptorBufferInfo> bufs = {
        {input.handle,  0, input.size},
        {lnW.handle,    0, lnW.size},
        {lnB.handle,    0, lnB.size},
        {projW.handle,  0, projW.size},
        {projB.handle,  0, projB.size},
        {output.handle, 0, output.size},
    };
    VkDescriptorSet desc = cache.allocDescriptorSet("fused-layernorm-linear", bufs);

    batch.dispatch(pipe.pipeline, pipe.layout, desc, seqLen, 1, 1,
                   nullptr, 0);
}

void batchedFlashAttention2(CommandBatch& batch, PipelineCache& cache,
                            GrillyBuffer& Q, GrillyBuffer& K, GrillyBuffer& V,
                            GrillyBuffer& output,
                            GrillyBuffer& runMax, GrillyBuffer& runSum,
                            GrillyBuffer& accum,
                            uint32_t batchSize, uint32_t numHeads,
                            uint32_t seqLen, uint32_t headDim, float scale) {

    if (scale == 0.0f)
        scale = 1.0f / sqrtf(float(headDim));

    uint32_t tileSizeQ = 64;
    uint32_t tileSizeK = 64;
    uint32_t numTilesQ = (seqLen + tileSizeQ - 1) / tileSizeQ;
    uint32_t numTilesK = (seqLen + tileSizeK - 1) / tileSizeK;
    uint32_t totalPositions = batchSize * numHeads * seqLen;

    PipelineEntry pipe = cache.getOrCreate("flash-attention2", 8, 44);

    size_t qkvBytes = size_t(batchSize) * numHeads * seqLen * headDim * sizeof(float);
    size_t runBytes = size_t(totalPositions) * sizeof(float);

    std::vector<VkDescriptorBufferInfo> bufs = {
        {Q.handle,      0, qkvBytes},
        {K.handle,      0, qkvBytes},
        {V.handle,      0, qkvBytes},
        {output.handle, 0, qkvBytes},
        {runMax.handle, 0, runBytes},
        {runSum.handle, 0, runBytes},
        {accum.handle,  0, qkvBytes},
        {Q.handle,      0, sizeof(float)},  // mask placeholder
    };
    VkDescriptorSet desc = cache.allocDescriptorSet("flash-attention2", bufs);

    // Push constants: 11 uint32s = 44 bytes
    struct {
        uint32_t batchSize, numHeads, seqLen, headDim;
        float scale;
        uint32_t tileSizeQ, tileSizeK, numTilesK, hasMask;
        uint32_t totalPositions, passIdx;
    } push;

    // Multi-pass tiled attention
    for (uint32_t tq = 0; tq < numTilesQ; ++tq) {
        for (uint32_t tk = 0; tk < numTilesK; ++tk) {
            push = {batchSize, numHeads, seqLen, headDim, scale,
                    tileSizeQ, tileSizeK, numTilesK, 0,
                    totalPositions, tq * numTilesK + tk};

            batch.dispatch(pipe.pipeline, pipe.layout, desc,
                          (tileSizeQ + 15) / 16,
                          totalPositions, 1,
                          &push, sizeof(push));
            batch.barrier();
        }
    }
}

}  // namespace ops
}  // namespace grilly
