#pragma once

#include <cstdint>
#include <vector>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

/// Flash Attention 2 push constants — matches flash-attention2.glsl layout.
///   uint batch_size;     // offset  0
///   uint seq_len;        // offset  4
///   uint num_heads;      // offset  8
///   uint head_dim;       // offset 12
///   float scale;         // offset 16
///   uint tile_size_q;    // offset 20
///   uint tile_size_k;    // offset 24
///   uint pass_type;      // offset 28  (0=init, 1=tile, 2=finalize)
///   uint has_mask;       // offset 32
///   uint q_tile_idx;     // offset 36
///   uint k_tile_idx;     // offset 40
struct FlashAttention2Params {
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t numHeads;
    uint32_t headDim;
    float scale;
    uint32_t tileSizeQ;
    uint32_t tileSizeK;
    uint32_t passType;
    uint32_t hasMask;
    uint32_t qTileIdx;
    uint32_t kTileIdx;
};

/// GPU Flash Attention 2 forward pass.
///
/// Ports backend/attention.py:365-633 to C++. Uses the 3-pass tiled
/// approach with online softmax (numerically stable):
///
///   Pass 0 (init):     Zero out running_max, running_sum, output_accum
///   Pass 1 (tile):     Loop over Q-tiles × K-tiles, accumulating via
///                      online softmax (log-sum-exp trick)
///   Pass 2 (finalize): Divide accumulated output by running_sum
///
/// 8 buffer bindings: Q, K, V, mask, output, running_max, running_sum,
///                    output_accum.
/// 44 bytes push constants (11 uint/float fields).
///
/// The tiling eliminates O(N^2) memory — only O(tile_size) SRAM needed.
/// For RDNA 2 with 128KB shared memory, we use tile sizes of 64 or 128.
void flashAttention2(CommandBatch& batch, BufferPool& pool,
                     PipelineCache& cache,
                     const float* Q, const float* K, const float* V,
                     const float* mask, float* output,
                     uint32_t batchSize, uint32_t seqLen,
                     uint32_t numHeads, uint32_t headDim,
                     float scale = 0.0f,  // 0 = auto (1/sqrt(head_dim))
                     uint32_t tileSizeQ = 64, uint32_t tileSizeK = 64);

}  // namespace ops
}  // namespace grilly
