#pragma once

#include <cstdint>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

// ═══════════════════════════════════════════════════════════════════════════
// Pooling ops — max pool, avg pool, mean pool
// All spatial pooling shaders use (8,8,1) workgroups.
// ═══════════════════════════════════════════════════════════════════════════

// ── MaxPool2d forward ────────────────────────────────────────────────────
// Shader: maxpool2d-forward.spv
// Buffers: input(0), output(1), indices(2)

struct MaxPool2dParams {
    uint32_t batchSize;
    uint32_t channels;
    uint32_t inHeight;
    uint32_t inWidth;
    uint32_t outHeight;
    uint32_t outWidth;
    uint32_t kernelH;
    uint32_t kernelW;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t paddingH;
    uint32_t paddingW;
    uint32_t dilationH;
    uint32_t dilationW;
};

void maxpool2dForward(CommandBatch& batch, BufferPool& pool,
                      PipelineCache& cache,
                      const float* input, float* output, uint32_t* indices,
                      const MaxPool2dParams& p);

// ── MaxPool2d backward ───────────────────────────────────────────────────
// Shader: maxpool2d-backward.spv
// Buffers: grad_output(0), indices(1), grad_input(2)

struct MaxPool2dBackwardParams {
    uint32_t batchSize;
    uint32_t channels;
    uint32_t inHeight;
    uint32_t inWidth;
    uint32_t outHeight;
    uint32_t outWidth;
};

void maxpool2dBackward(CommandBatch& batch, BufferPool& pool,
                       PipelineCache& cache,
                       const float* gradOutput, const uint32_t* indices,
                       float* gradInput, const MaxPool2dBackwardParams& p);

// ── AvgPool2d forward ────────────────────────────────────────────────────
// Shader: avgpool2d-forward.spv
// Buffers: input(0), output(1)

struct AvgPool2dParams {
    uint32_t batchSize;
    uint32_t channels;
    uint32_t inHeight;
    uint32_t inWidth;
    uint32_t outHeight;
    uint32_t outWidth;
    uint32_t kernelH;
    uint32_t kernelW;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t paddingH;
    uint32_t paddingW;
    uint32_t countIncludePad;
};

void avgpool2dForward(CommandBatch& batch, BufferPool& pool,
                      PipelineCache& cache,
                      const float* input, float* output,
                      const AvgPool2dParams& p);

// ── AvgPool2d backward ──────────────────────────────────────────────────
// Shader: avgpool2d-backward.spv (reuses AvgPool2dParams)

void avgpool2dBackward(CommandBatch& batch, BufferPool& pool,
                       PipelineCache& cache,
                       const float* gradOutput, float* gradInput,
                       const AvgPool2dParams& p);

// ── Mean pooling (sequence-level) ────────────────────────────────────────
// Shader: mean-pooling.spv
// Averages over the sequence dimension: (B, S, D) → (B, D)

struct MeanPoolParams {
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t features;
};

void meanPool(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
              const float* input, float* output, const MeanPoolParams& p);

}  // namespace ops
}  // namespace grilly
