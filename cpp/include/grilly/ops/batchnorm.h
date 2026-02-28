#pragma once

#include <cstdint>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

// ═══════════════════════════════════════════════════════════════════════════
// BatchNorm2d — forward and backward
//
// Unlike LayerNorm (normalizes over features per position), BatchNorm
// normalizes over the (batch, height, width) dimensions per channel.
// Both shaders dispatch at (256, 1, 1) — one thread per channel.
// ═══════════════════════════════════════════════════════════════════════════

// ── BatchNorm2d forward ──────────────────────────────────────────────────
// Shader: batchnorm2d-forward.spv
// Buffers: input(0), output(1), gamma(2), beta(3),
//          running_mean(4), running_var(5), batch_mean(6), batch_var(7)

struct BatchNorm2dForwardParams {
    uint32_t batchSize;
    uint32_t numFeatures;
    uint32_t height;
    uint32_t width;
    float eps;
    float momentum;
    uint32_t training;
    uint32_t affine;
};

void batchnorm2dForward(CommandBatch& batch, BufferPool& pool,
                        PipelineCache& cache,
                        const float* input, float* output,
                        const float* gamma, const float* beta,
                        float* runningMean, float* runningVar,
                        float* batchMean, float* batchVar,
                        const BatchNorm2dForwardParams& p);

// ── BatchNorm2d backward ─────────────────────────────────────────────────
// Shader: batchnorm2d-backward.spv
// Buffers: grad_out(0), input(1), grad_input(2),
//          batch_mean(3), batch_var(4), gamma(5),
//          grad_gamma(6), grad_beta(7)

struct BatchNorm2dBackwardParams {
    uint32_t batchSize;
    uint32_t numFeatures;
    uint32_t height;
    uint32_t width;
    float eps;
    uint32_t affine;
};

void batchnorm2dBackward(CommandBatch& batch, BufferPool& pool,
                         PipelineCache& cache,
                         const float* gradOutput, const float* input,
                         float* gradInput,
                         const float* batchMean, const float* batchVar,
                         const float* gamma,
                         float* gradGamma, float* gradBeta,
                         const BatchNorm2dBackwardParams& p);

}  // namespace ops
}  // namespace grilly
