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

// â”€â”€ Cross-entropy FUSED loss + gradient â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Shader: loss-ce-fused.spv
// ONE dispatch computes BOTH per-row loss AND grad_logits, sharing a single
// subgroup-reduced max + sum_exp pass per row (workgroup-per-row).
// Buffers: logits(0), targets(1, as float), losses(2), grad_logits(3)

struct CrossEntropyFusedParams {
    uint32_t batchSize;
    uint32_t numClasses;
};

void crossEntropyFused(CommandBatch& batch, BufferPool& pool,
                       PipelineCache& cache,
                       const float* logits, const uint32_t* targets,
                       float* losses, float* gradLogits,
                       const CrossEntropyFusedParams& p);

// ── MSE Loss ────────────────────────────────────────────────────────────

struct MSELossParams {
    uint32_t n;
};

void mseLoss(CommandBatch& batch, BufferPool& pool,
             PipelineCache& cache,
             const float* preds, const float* targets,
             float* losses, const MSELossParams& p);

// ── Cosine Similarity Loss ──────────────────────────────────────────────

struct CosineLossParams {
    uint32_t batchSize;
    uint32_t dim;
};

void cosineSimilarityLoss(CommandBatch& batch, BufferPool& pool,
                          PipelineCache& cache,
                          const float* preds, const float* targets,
                          float* losses, const CosineLossParams& p);

}  // namespace ops
}  // namespace grilly
