#pragma once

#include "grilly/command_batch.h"
#include "grilly/buffer_pool.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

struct BanditParams {
    uint32_t nArms;
    uint32_t nInstances;
    uint32_t iters;
    float delta;
};

/**
 * Bandit Top-2 Solver & Stopping Criterion
 * 
 * Computes TargetW (K, nInstances) and StopFlags (nInstances).
 */
void banditSolve(CommandBatch& batch, BufferPool& pool,
                 PipelineCache& cache,
                 const float* muHat, const float* N,
                 float* targetW, uint32_t* stopFlags,
                 const BanditParams& p);

}  // namespace ops
}  // namespace grilly
