#pragma once

#include <cstdint>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

// ═══════════════════════════════════════════════════════════════════════════
// Embedding lookup
// Shader: embedding-lookup.spv
// Buffers: token_ids(0), embeddings(1), output(2)
// ═══════════════════════════════════════════════════════════════════════════

struct EmbeddingParams {
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t vocabSize;
    uint32_t embeddingDim;
};

void embeddingLookup(CommandBatch& batch, BufferPool& pool,
                     PipelineCache& cache,
                     const uint32_t* tokenIds, const float* embeddings,
                     float* output, const EmbeddingParams& p);

}  // namespace ops
}  // namespace grilly
