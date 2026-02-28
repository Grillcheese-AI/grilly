#pragma once

#include <cstdint>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

// ═══════════════════════════════════════════════════════════════════════════
// Loss functions — cross-entropy forward and backward
// ═══════════════════════════════════════════════════════════════════════════

// ── Cross-entropy loss forward ───────────────────────────────────────────
// Shader: loss-cross-entropy.spv
// 3-pass: max → sum_exp → loss
// Buffers: logits(0), targets(1), losses(2), max_logits(3), sum_exp(4)

struct CrossEntropyParams {
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t vocabSize;
    uint32_t passType;
    float labelSmoothing;
};

void crossEntropyLoss(CommandBatch& batch, BufferPool& pool,
                      PipelineCache& cache,
                      const float* logits, const uint32_t* targets,
                      float* losses, const CrossEntropyParams& p);

// ── Cross-entropy backward ──────────────────────────────────────────────
// Shader: cross-entropy-backward.spv
// Buffers: logits(0), targets(1), grad_logits(2)

struct CrossEntropyBackwardParams {
    uint32_t batchSize;
    uint32_t numClasses;
};

void crossEntropyBackward(CommandBatch& batch, BufferPool& pool,
                          PipelineCache& cache,
                          const float* logits, const uint32_t* targets,
                          float* gradLogits,
                          const CrossEntropyBackwardParams& p);

}  // namespace ops
}  // namespace grilly
