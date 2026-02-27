#pragma once

#include <cstdint>
#include <vector>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

/// LayerNorm push constants — matches fnn-layernorm.glsl layout.
///   uint batch_size;   // offset 0
///   uint seq_len;      // offset 4
///   uint features;     // offset 8
///   float eps;         // offset 12
///   uint pass_type;    // offset 16  (0=mean, 1=variance, 2=normalize)
struct LayerNormParams {
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t features;
    float eps;
    uint32_t passType;
};

/// GPU LayerNorm forward pass.
///
/// Runs 3 dispatches via the same "fnn-layernorm" shader with different
/// pass_type values:
///   Pass 0: compute per-position mean
///   Pass 1: compute per-position variance
///   Pass 2: normalize + affine transform (gamma * (x-mean)/sqrt(var+eps) + beta)
///
/// Uses 6 buffer bindings: input, output, gamma, beta, mean, variance.
/// Workgroups are 1D at 256 threads.
///
/// When used in an OpGraph, the 3 passes are recorded with barriers between
/// them — the mean must be computed before variance, and both before normalize.
void layernorm(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
               const float* input, float* output,
               const float* gamma, const float* beta,
               uint32_t batchSize, uint32_t seqLen, uint32_t features,
               float eps = 1e-5f);

/// GPU LayerNorm backward pass.
/// 8 buffer bindings: grad_out, input, gamma, mean, var,
///                    grad_in, grad_gamma, grad_beta.
/// 3-pass dispatch (pass 0/1/2).
void layernormBackward(CommandBatch& batch, BufferPool& pool,
                       PipelineCache& cache,
                       const float* gradOutput, const float* input,
                       const float* gamma, const float* mean,
                       const float* var,
                       float* gradInput, float* gradGamma, float* gradBeta,
                       uint32_t batchSize, uint32_t seqLen, uint32_t features,
                       float eps = 1e-5f);

}  // namespace ops
}  // namespace grilly
