#include "grilly/ops/attention.h"

#include <cmath>
#include <cstring>

namespace grilly {
namespace ops {

// ── Flash Attention 2 (port of backend/attention.py:365-633) ─────────────
//
// Flash Attention's key insight: instead of materializing the full N×N
// attention matrix (which is O(N^2) memory), process Q and K in tiles.
// Each tile computes a partial softmax using the "online softmax" trick
// (Milakov & Gimelshein 2018): maintain running max and running sum
// across tiles, then correct the accumulated output at the end.
//
// The 3-pass structure:
//
//   Pass 0 (init): Zero out accumulation buffers. running_max is set to
//     -infinity (so any real value becomes the new max), running_sum to 0,
//     and output_accum to 0. One dispatch covers all (batch * heads * seqLen)
//     positions.
//
//   Pass 1 (tile processing): For each (q_tile, k_tile) pair:
//     - Load Q_tile and K_tile into shared memory (RDNA 2: 128KB LDS)
//     - Compute S = Q_tile @ K_tile^T (tile-local attention scores)
//     - Apply mask if present
//     - Online softmax update:
//         new_max = max(running_max, max(S))
//         correction = exp(running_max - new_max)
//         running_sum = running_sum * correction + sum(exp(S - new_max))
//         output_accum = output_accum * correction + exp(S - new_max) @ V_tile
//     - Update running_max = new_max
//     This is the critical loop — number of dispatches = numQTiles * numKTiles.
//
//   Pass 2 (finalize): output = output_accum / running_sum. One dispatch.
//
// For RDNA 2 (Wave32, 128KB shared memory), tile sizes of 64 work well.
// The total shared memory per tile pair is:
//   Q_tile: 64 * head_dim * 4 bytes
//   K_tile: 64 * head_dim * 4 bytes
//   S_tile: 64 * 64 * 4 bytes = 16KB
// For head_dim=64: 16KB + 16KB + 16KB = 48KB — fits comfortably.

void flashAttention2(CommandBatch& batch, BufferPool& pool,
                     PipelineCache& cache,
                     const float* Q, const float* K, const float* V,
                     const float* mask, float* output,
                     uint32_t batchSize, uint32_t seqLen,
                     uint32_t numHeads, uint32_t headDim,
                     float scale, uint32_t tileSizeQ, uint32_t tileSizeK) {
    // Auto-scale: 1/sqrt(head_dim)
    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(headDim));
    }

    const uint32_t hasMask = (mask != nullptr) ? 1 : 0;
    const uint32_t totalPositions = batchSize * numHeads * seqLen;

    // Buffer sizes
    const size_t qkvBytes  = size_t(batchSize) * numHeads * seqLen * headDim *
                             sizeof(float);
    const size_t maskBytes = hasMask
                                 ? size_t(batchSize) * seqLen * seqLen *
                                       sizeof(float)
                                 : sizeof(float);  // dummy
    const size_t outBytes  = qkvBytes;
    const size_t accumBytes = qkvBytes;  // same shape as output
    const size_t runMaxBytes  = size_t(totalPositions) * sizeof(float);
    const size_t runSumBytes  = runMaxBytes;

    // Acquire 8 buffers
    GrillyBuffer bufQ       = pool.acquire(qkvBytes);
    GrillyBuffer bufK       = pool.acquire(qkvBytes);
    GrillyBuffer bufV       = pool.acquire(qkvBytes);
    GrillyBuffer bufMask    = pool.acquire(maskBytes);
    GrillyBuffer bufOutput  = pool.acquire(outBytes);
    GrillyBuffer bufRunMax  = pool.acquire(runMaxBytes);
    GrillyBuffer bufRunSum  = pool.acquire(runSumBytes);
    GrillyBuffer bufAccum   = pool.acquire(accumBytes);

    // Upload
    pool.upload(bufQ, Q, qkvBytes);
    pool.upload(bufK, K, qkvBytes);
    pool.upload(bufV, V, qkvBytes);
    if (hasMask) {
        pool.upload(bufMask, mask, maskBytes);
    }

    // Pipeline: 8 buffers, 44 bytes push constants (11 fields)
    PipelineEntry pipe = cache.getOrCreate("flash-attention2", 8,
                                           sizeof(FlashAttention2Params));

    // Descriptor set (shared across all passes — same buffers)
    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufQ.handle,      0, qkvBytes},
        {bufK.handle,      0, qkvBytes},
        {bufV.handle,      0, qkvBytes},
        {bufMask.handle,   0, maskBytes},
        {bufOutput.handle, 0, outBytes},
        {bufRunMax.handle, 0, runMaxBytes},
        {bufRunSum.handle, 0, runSumBytes},
        {bufAccum.handle,  0, accumBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("flash-attention2",
                                                        bufInfos);

    // Tile counts
    uint32_t numQTiles = (seqLen + tileSizeQ - 1) / tileSizeQ;
    uint32_t numKTiles = (seqLen + tileSizeK - 1) / tileSizeK;

    batch.begin();

    // ── Pass 0: Initialize accumulators ──
    FlashAttention2Params push0{
        batchSize, seqLen, numHeads, headDim, scale,
        tileSizeQ, tileSizeK, 0 /*init*/, hasMask, 0, 0
    };
    uint32_t initGroups = (totalPositions + 255) / 256;
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, initGroups, 1, 1,
                   &push0, sizeof(push0));
    batch.barrier();

    // ── Pass 1: Process all tile pairs ──
    for (uint32_t qt = 0; qt < numQTiles; ++qt) {
        for (uint32_t kt = 0; kt < numKTiles; ++kt) {
            FlashAttention2Params pushTile{
                batchSize, seqLen, numHeads, headDim, scale,
                tileSizeQ, tileSizeK, 1 /*tile*/, hasMask, qt, kt
            };

            // Workgroups: each workgroup processes one (batch, head, q_pos)
            // within the current Q-tile
            uint32_t qTileSize = std::min(tileSizeQ, seqLen - qt * tileSizeQ);
            uint32_t tileGroups = (batchSize * numHeads * qTileSize + 255) / 256;

            batch.dispatch(pipe.pipeline, pipe.layout, descSet,
                           tileGroups, 1, 1, &pushTile, sizeof(pushTile));
            batch.barrier();
        }
    }

    // ── Pass 2: Finalize (divide by running sum) ──
    FlashAttention2Params push2{
        batchSize, seqLen, numHeads, headDim, scale,
        tileSizeQ, tileSizeK, 2 /*finalize*/, hasMask, 0, 0
    };
    uint32_t finalGroups = (batchSize * numHeads * seqLen * headDim + 255) / 256;
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, finalGroups, 1, 1,
                   &push2, sizeof(push2));

    batch.submit();

    // Download result
    pool.download(bufOutput, output, outBytes);

    // Release all buffers
    pool.release(bufQ);
    pool.release(bufK);
    pool.release(bufV);
    pool.release(bufMask);
    pool.release(bufOutput);
    pool.release(bufRunMax);
    pool.release(bufRunSum);
    pool.release(bufAccum);
}

}  // namespace ops
}  // namespace grilly
