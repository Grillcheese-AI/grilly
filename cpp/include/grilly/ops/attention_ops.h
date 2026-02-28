#pragma once

#include <cstdint>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

// ═══════════════════════════════════════════════════════════════════════════
// Decomposed attention ops
//
// These are the building blocks of multi-head attention, exposed as
// individual GPU dispatch functions. While flash_attention2 fuses the
// entire attention into one kernel, many models and tests use the
// decomposed pipeline: scores → mask → output → concat_heads.
// ═══════════════════════════════════════════════════════════════════════════

// ── Attention scores: Q @ K^T / sqrt(d_h) ───────────────────────────────
// Shader: attention-scores.spv
// local_size: (16, 16, 1)
// Buffers: Q(0), K(1), V(2), scores(3)

struct AttentionScoresParams {
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t numHeads;
    uint32_t headDim;
    float scale;
    uint32_t passType;  // 0 = compute scores
};

void attentionScores(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                     const float* Q, const float* K,
                     float* scores, const AttentionScoresParams& p);

// ── Attention mask ───────────────────────────────────────────────────────
// Shader: attention-mask.spv
// local_size: (256, 1, 1)
// Buffers: scores(0) rw, mask(1) readonly

struct AttentionMaskParams {
    uint32_t batchSize;
    uint32_t numHeads;
    uint32_t seqLen;
    uint32_t useCausalMask;
    float maskValue;
};

void attentionMask(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                   float* scores, const float* mask,
                   const AttentionMaskParams& p);

// ── Attention output: softmax(scores) @ V ────────────────────────────────
// Shader: attention-output.spv
// local_size: (256, 1, 1)
// Buffers: weights(0) readonly, V(1) readonly, output(2) write

struct AttentionOutputParams {
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t numHeads;
    uint32_t headDim;
};

void attentionOutput(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                     const float* weights, const float* V,
                     float* output, const AttentionOutputParams& p);

// ── Concat heads: (B, H, S, D) → (B, S, H*D) ───────────────────────────
// Shader: attention-concat-heads.spv
// local_size: (256, 1, 1)
// Buffers: mh_output(0) readonly, concat_output(1) write

struct ConcatHeadsParams {
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t numHeads;
    uint32_t headDim;
};

void attentionConcatHeads(CommandBatch& batch, BufferPool& pool,
                          PipelineCache& cache,
                          const float* mhOutput, float* concatOutput,
                          const ConcatHeadsParams& p);

// ── Rotary Position Embeddings ───────────────────────────────────────────
// Shader: rope.spv
// local_size: (256, 1, 1)
// Buffers: input(0), output(1), cos_table(2), sin_table(3)

struct RoPEParams {
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t numHeads;
    uint32_t headDim;
    float ropeBase;
    uint32_t usePrecomputed;
    float ropeScaling;
};

void applyRoPE(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
               const float* input, float* output,
               const float* cosTable, const float* sinTable,
               const RoPEParams& p);

}  // namespace ops
}  // namespace grilly
