#pragma once

#include <cstdint>
#include <vector>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

/// RMSNorm push constants — matches rms-norm.glsl layout.
///   uint batch_size;   // offset 0
///   uint seq_len;      // offset 4
///   uint features;     // offset 8
///   float eps;         // offset 12
///   uint pass_type;    // offset 16  (0=compute RMS, 1=normalize)
struct RMSNormParams {
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t features;
    float eps;
    uint32_t passType;
};

/// GPU RMSNorm forward pass.
///
/// Runs 2 dispatches via the same "rms-norm" shader with different
/// pass_type values:
///   Pass 0: compute per-position mean(x^2)
///   Pass 1: normalize: out = weight * x * rsqrt(mean_sq + eps)
///
/// Uses 4 buffer bindings: input, output, weight, rms_vals.
/// Workgroups are 1D at 256 threads.
///
/// When used in an OpGraph, the 2 passes are recorded with a barrier
/// between them — the mean(x^2) must be computed before normalize.
void rmsnorm(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
             const float* input, float* output,
             const float* weight,
             uint32_t batchSize, uint32_t seqLen, uint32_t features,
             float eps = 1e-5f);

}  // namespace ops
}  // namespace grilly
