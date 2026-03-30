#pragma once

#include <cstdint>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

/// Push constants for perceiver-encode.spv
struct PerceiverEncodeParams {
    uint32_t seq_M;     // Input sequence length
    uint32_t seq_N;     // Latent sequence length
    uint32_t head_dim;  // D per head
    float scale;        // 1.0 / sqrt(D)
};

/// Perceiver IO cross-attention (encode step).
///
/// 1 thread = 1 latent token. Q pinned in VGPRs, K/V streamed from VRAM.
/// Online softmax — O(1) VRAM regardless of input length M.
/// Zero LDS, zero barriers.
///
/// Shader: perceiver-encode.spv
/// Bindings: 0=Q (latents), 1=K (inputs), 2=V (inputs), 3=Output
/// Dispatch: workgroups_x = ceil(N / 64)
///
/// @param Q        (N * D) row-major latent queries
/// @param K        (M * D) row-major input keys
/// @param V        (M * D) row-major input values
/// @param output   (N * D) row-major updated latents
/// @param seqN     Latent count N
/// @param seqM     Input count M
/// @param headDim  Head dimension D (must be multiple of 4, max 64)
void perceiverEncode(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                     const float* Q, const float* K, const float* V, float* output,
                     uint32_t seqN, uint32_t seqM, uint32_t headDim);

/// Buffer-handle version for batched dispatch (no upload/download).
/// All buffers must already be on GPU.
void batchedPerceiverEncode(CommandBatch& batch, PipelineCache& cache,
                            GrillyBuffer& bufQ, GrillyBuffer& bufK,
                            GrillyBuffer& bufV, GrillyBuffer& bufOut,
                            uint32_t seqN, uint32_t seqM, uint32_t headDim);

}  // namespace ops
}  // namespace grilly
