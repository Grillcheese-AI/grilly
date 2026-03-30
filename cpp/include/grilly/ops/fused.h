#pragma once

#include <cstdint>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

/// Push constants for fused-mlp-gelu.spv
struct FusedMLPParams {
    uint32_t d_in;
    uint32_t d_hidden;
    uint32_t d_out;
};

/// Fused MLP: fc1 → GELU → fc2 in one GPU dispatch.
/// Hidden activations stay in LDS (shared memory), never touch VRAM.
///
/// Shader: fused-mlp-gelu.spv
/// Bindings: 0=input, 1=w1, 2=b1, 3=w2, 4=b2, 5=output
/// Dispatch: workgroups_x = seq_len (one workgroup per token)
///
/// @param input   (seq_len * d_in) row-major
/// @param w1      (d_hidden * d_in) fc1 weights
/// @param b1      (d_hidden) fc1 bias
/// @param w2      (d_out * d_hidden) fc2 weights
/// @param b2      (d_out) fc2 bias
/// @param output  (seq_len * d_out) row-major
void fusedMlpGelu(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                  const float* input, const float* w1, const float* b1,
                  const float* w2, const float* b2, float* output,
                  uint32_t seqLen, uint32_t dIn, uint32_t dHidden, uint32_t dOut);

/// Push constants for fused-layernorm-linear.spv
struct FusedLNLinearParams {
    uint32_t d_in;
    uint32_t d_out;
};

/// Fused LayerNorm + Linear projection in one GPU dispatch.
/// Normalized values stay in LDS.
///
/// Shader: fused-layernorm-linear.spv
/// Bindings: 0=input, 1=ln_weight, 2=ln_bias, 3=proj_w, 4=proj_b, 5=output
/// Dispatch: workgroups_x = seq_len
///
/// @param input    (seq_len * d_in) row-major
/// @param lnWeight (d_in) LayerNorm weight
/// @param lnBias   (d_in) LayerNorm bias
/// @param projW    (d_out * d_in) projection weights
/// @param projB    (d_out) projection bias
/// @param output   (seq_len * d_out) row-major
void fusedLayernormLinear(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                          const float* input, const float* lnWeight, const float* lnBias,
                          const float* projW, const float* projB, float* output,
                          uint32_t seqLen, uint32_t dIn, uint32_t dOut);

}  // namespace ops
}  // namespace grilly
