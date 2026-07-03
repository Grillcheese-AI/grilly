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

// ── Sampled-BCE (NCE/SGNS) FUSED loss + dH + dW ─────────────────────────
// Shader: loss-sampled-bce-fused.spv (two passes, one submit)
// The softmax-free LM head: each token scores only target + K negatives.
// Pass 0 is workgroup-per-token (subgroup/LDS loss reduce, plain stores);
// pass 1 scatters dW via CAS-loop float add on CORE uint atomics — no
// GL_EXT_shader_atomic_float dependency (float atomicAdd measured broken).
// Buffers: hidden(0), table(1), ids(2 uint), dscore(3), losses(4),
//          grad_hidden(5), grad_table(6, uint-bits view, CAS scatter)
// grad_table is fillZero'd inside the batch before pass 1.

struct SampledBceParams {
    uint32_t nTokens;    // N = B*S flattened token positions
    uint32_t nCand;      // 1 + K (column 0 of ids is the true target)
    uint32_t dim;        // d_model
    uint32_t passType;   // 0 = scores/loss/dscore, 1 = dH + dW
    float invN;          // 1/nTokens (bakes the mean into the grads)
};

void sampledBceFused(CommandBatch& batch, BufferPool& pool,
                     PipelineCache& cache,
                     const float* hidden, const float* table,
                     const uint32_t* ids,
                     float* losses, float* gradHidden, float* gradTable,
                     uint32_t vocabSize, const SampledBceParams& p);

// ── NCE FUSED loss + dH + dW + db (corrected sampled-BCE) ────────────────
// Shader: loss-nce-fused.spv. Adds the noise-distribution correction that the
// plain SGNS head (sampledBceFused) lacks: G = dot + bias[id] - log(k*q_id),
// so the trained head converges to a calibrated log P(w|ctx) and full-softmax
// eval PPL tracks full CE. bias is a learned (V,) scalar; logkq is the fixed
// (V,) noise term. useCorrection=0 recovers the SGNS head for A/B.
// Buffers: hidden(0), table(1), ids(2 uint), dscore(3), losses(4),
//          grad_hidden(5), grad_table(6 CAS), logkq(7), bias(8),
//          grad_bias(9 CAS). grad_table + grad_bias fillZero'd before pass 1.

struct NceParams {
    uint32_t nTokens;
    uint32_t nCand;
    uint32_t dim;
    uint32_t passType;
    float invN;
    uint32_t useCorrection;   // 1 = NCE (bias - logkq), 0 = raw-dot SGNS
};

void nceFused(CommandBatch& batch, BufferPool& pool,
              PipelineCache& cache,
              const float* hidden, const float* table, const uint32_t* ids,
              const float* logkq, const float* bias,
              float* losses, float* gradHidden, float* gradTable,
              float* gradBias, uint32_t vocabSize, const NceParams& p);

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
