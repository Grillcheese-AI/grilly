#pragma once
/// Batched op variants — record dispatches into an EXTERNAL CommandBatch
/// without calling begin()/submit(). Caller manages batch lifecycle.
///
/// Usage:
///   batch.begin();
///   batchedLinear(batch, pool, cache, bufIn, bufW, bufB, bufOut, M, K, N);
///   batch.barrier();
///   batchedGelu(batch, pool, cache, bufOut, bufGelu, total);
///   batch.barrier();
///   batchedLinear(batch, pool, cache, bufGelu, bufW2, bufB2, bufFinal, M, K2, N2);
///   batch.submit();  // ONE fence wait for all three ops!

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

/// Record a linear dispatch into batch (does NOT begin/submit).
/// Buffers must already be uploaded.
void batchedLinear(CommandBatch& batch, PipelineCache& cache,
                   const GrillyBuffer& input, const GrillyBuffer& weight,
                   const GrillyBuffer* bias, GrillyBuffer& output,
                   uint32_t M, uint32_t K, uint32_t N);

/// Record a GELU dispatch into batch.
void batchedGelu(CommandBatch& batch, PipelineCache& cache,
                 const GrillyBuffer& input, GrillyBuffer& output,
                 uint32_t totalElements);

/// Record a fused MLP (fc1→GELU→fc2) dispatch into batch.
void batchedFusedMlpGelu(CommandBatch& batch, PipelineCache& cache,
                          GrillyBuffer& input,
                          const GrillyBuffer& w1, const GrillyBuffer& b1,
                          const GrillyBuffer& w2, const GrillyBuffer& b2,
                          GrillyBuffer& output,
                          uint32_t seqLen);

/// Record a fused LayerNorm+Linear dispatch into batch.
void batchedFusedLnLinear(CommandBatch& batch, PipelineCache& cache,
                           GrillyBuffer& input,
                           const GrillyBuffer& lnW, const GrillyBuffer& lnB,
                           const GrillyBuffer& projW, const GrillyBuffer& projB,
                           GrillyBuffer& output,
                           uint32_t seqLen);

/// Record a flash attention dispatch into batch.
void batchedFlashAttention2(CommandBatch& batch, PipelineCache& cache,
                            const GrillyBuffer& Q, const GrillyBuffer& K, const GrillyBuffer& V,
                            GrillyBuffer& output,
                            GrillyBuffer& runMax, GrillyBuffer& runSum,
                            GrillyBuffer& accum,
                            uint32_t batchSize, uint32_t numHeads,
                            uint32_t seqLen, uint32_t headDim, float scale);

/// Element-wise add: a[i] += b[i]. In-place on buffer A.
void batchedAdd(CommandBatch& batch, PipelineCache& cache,
                GrillyBuffer& a, const GrillyBuffer& b,
                uint32_t totalElements);

/// Tiled GEMM: C[M,N] = A[M,K] @ B[N,K]^T + bias[N]
/// Uses 32x32 shared memory tiles, 2x2 per-thread sub-tiles.
void batchedTiledLinear(CommandBatch& batch, PipelineCache& cache,
                        const GrillyBuffer& input, const GrillyBuffer& weight,
                        const GrillyBuffer* bias, GrillyBuffer& output,
                        uint32_t M, uint32_t K, uint32_t N);
}  // namespace ops
}  // namespace grilly

