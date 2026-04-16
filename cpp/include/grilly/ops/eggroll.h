#pragma once

#include "grilly/command_batch.h"
#include "grilly/buffer_pool.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

struct EggrollGenParams {
    uint32_t dOut;
    uint32_t dIn;
    uint32_t nWorkers;
    uint32_t seed;
    float sigma;
};

struct EggrollUpdateParams {
    uint32_t dOut;
    uint32_t dIn;
    uint32_t topK;
    uint32_t nWorkers;
    float meritIncrease;
    float meritDecay;
};

/**
 * Generate perturbations (U, V)
 */
void eggrollGenerate(CommandBatch& batch, BufferPool& pool,
                    PipelineCache& cache,
                    float* U, float* V,
                    const EggrollGenParams& p);

/**
 * Apply fused update to weights and merit
 */
void eggrollUpdate(CommandBatch& batch, BufferPool& pool,
                  PipelineCache& cache,
                  float* W, float* Merit,
                  const float* U, const float* V,
                  const uint32_t* topIdx, const float* topWeights,
                  const EggrollUpdateParams& p);

}  // namespace ops
}  // namespace grilly
