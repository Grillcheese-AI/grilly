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
                   GrillyBuffer& input, GrillyBuffer& weight,
                   GrillyBuffer* bias, GrillyBuffer& output,
                   uint32_t M, uint32_t K, uint32_t N);

/// Record a GELU dispatch into batch.
void batchedGelu(CommandBatch& batch, PipelineCache& cache,
                 GrillyBuffer& input, GrillyBuffer& output,
                 uint32_t totalElements);

/// Record a fused MLP (fc1→GELU→fc2) dispatch into batch.
void batchedFusedMlpGelu(CommandBatch& batch, PipelineCache& cache,
                          GrillyBuffer& input,
                          GrillyBuffer& w1, GrillyBuffer& b1,
                          GrillyBuffer& w2, GrillyBuffer& b2,
                          GrillyBuffer& output,
                          uint32_t seqLen);

/// Record a fused LayerNorm+Linear dispatch into batch.
void batchedFusedLnLinear(CommandBatch& batch, PipelineCache& cache,
                           GrillyBuffer& input,
                           GrillyBuffer& lnW, GrillyBuffer& lnB,
                           GrillyBuffer& projW, GrillyBuffer& projB,
                           GrillyBuffer& output,
                           uint32_t seqLen);

/// Record a flash attention dispatch into batch.
void batchedFlashAttention2(CommandBatch& batch, PipelineCache& cache,
                            GrillyBuffer& Q, GrillyBuffer& K, GrillyBuffer& V,
                            GrillyBuffer& output,
                            GrillyBuffer& runMax, GrillyBuffer& runSum,
                            GrillyBuffer& accum,
                            uint32_t batchSize, uint32_t numHeads,
                            uint32_t seqLen, uint32_t headDim, float scale);

}  // namespace ops
}  // namespace grilly
