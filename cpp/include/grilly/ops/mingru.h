#pragma once

#include "grilly/command_batch.h"
#include "grilly/buffer_pool.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

struct MinGruParams {
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t hiddenDim;
};

/**
 * Fused MinGRU Forward (projections + activations + recurrence)
 * Logic:
 *   x_scan = sigmoid(g) * tanh(v)
 *   a      = 0.05 + 0.9 * sigmoid(d)
 *   h_t    = a_t * h_{t-1} + x_scan_t
 */
void minGruForward(CommandBatch& batch, BufferPool& pool,
                   PipelineCache& cache,
                   const float* G, const float* V, const float* D,
                   float* H, const MinGruParams& p);

/**
 * Fused MinGRU Backward
 */
void minGruBackward(CommandBatch& batch, BufferPool& pool,
                    PipelineCache& cache,
                    const float* gradH, const float* G, const float* V, const float* D,
                    const float* H,
                    float* gradG, float* gradV, float* gradD,
                    const MinGruParams& p);

}  // namespace ops
}  // namespace grilly
