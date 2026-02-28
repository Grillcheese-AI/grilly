#include "grilly/ops/embedding.h"

#include <cstring>

namespace grilly {
namespace ops {

// ═══════════════════════════════════════════════════════════════════════════
// Embedding lookup
//
// Each thread looks up one token's embedding vector. The shader reads
// token_ids[i] and copies the corresponding row from the embedding table
// to the output. dispatch at (256,1,1) over total tokens.
// ═══════════════════════════════════════════════════════════════════════════

void embeddingLookup(CommandBatch& batch, BufferPool& pool,
                     PipelineCache& cache,
                     const uint32_t* tokenIds, const float* embeddings,
                     float* output, const EmbeddingParams& p) {
    const uint32_t totalTokens = p.batchSize * p.seqLen;
    const size_t idBytes    = size_t(totalTokens) * sizeof(uint32_t);
    const size_t embedBytes = size_t(p.vocabSize) * p.embeddingDim * sizeof(float);
    const size_t outBytes   = size_t(totalTokens) * p.embeddingDim * sizeof(float);

    GrillyBuffer bufIds   = pool.acquire(idBytes);
    GrillyBuffer bufEmbed = pool.acquire(embedBytes);
    GrillyBuffer bufOut   = pool.acquire(outBytes);

    pool.upload(bufIds, reinterpret_cast<const float*>(tokenIds), idBytes);
    pool.upload(bufEmbed, embeddings, embedBytes);

    PipelineEntry pipe = cache.getOrCreate("embedding-lookup", 3,
                                           sizeof(EmbeddingParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufIds.handle,   0, idBytes},
        {bufEmbed.handle, 0, embedBytes},
        {bufOut.handle,   0, outBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("embedding-lookup",
                                                        bufInfos);

    uint32_t gx = (totalTokens + 255) / 256;

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.submit();

    pool.download(bufOut, output, outBytes);

    pool.release(bufIds);
    pool.release(bufEmbed);
    pool.release(bufOut);
}

}  // namespace ops
}  // namespace grilly
