#pragma once

#include <cstdint>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

// ═══════════════════════════════════════════════════════════════════════════
// GPU optimizer steps — dispatch single shader per parameter update
// ═══════════════════════════════════════════════════════════════════════════

// ── Adam ─────────────────────────────────────────────────────────────────
// Shader: adam-update.spv
// Buffers: W(0) rw, grad(1) rw, m(2) rw, v(3) rw

struct AdamParams {
    uint32_t totalWeights;
    float learningRate;
    float beta1;
    float beta2;
    float epsilon;
    float beta1T;   // beta1^t (precomputed power)
    float beta2T;   // beta2^t
    uint32_t clearGrad;  // 1 to zero gradients after update
};

void adamUpdate(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                float* weights, float* grad, float* m, float* v,
                const AdamParams& p);

// ── AdamW ────────────────────────────────────────────────────────────────
// Shader: adamw-update.spv
// Buffers: W(0) rw, grad(1) rw, m(2) rw, v(3) rw

struct AdamWParams {
    uint32_t totalWeights;
    float learningRate;
    float beta1;
    float beta2;
    float epsilon;
    float weightDecay;
    float beta1T;
    float beta2T;
    uint32_t clearGrad;
};

void adamwUpdate(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                 float* weights, float* grad, float* m, float* v,
                 const AdamWParams& p);

}  // namespace ops
}  // namespace grilly
