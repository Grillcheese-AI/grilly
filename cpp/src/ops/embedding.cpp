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

    // Staging pattern: 2 stage-in (token ids, embedding table), 1 stage-out
    // (output token vectors). Embedding table is the big one (vocab × dim);
    // for VSA-LM with vocab=8192, dim=384 it's 12 MB and re-uploaded every
    // call. A weight cache (TODO) would amortize this across training steps.
    GrillyBuffer bufIdsDL   = pool.acquireDeviceLocal(idBytes);
    GrillyBuffer bufEmbedDL = pool.acquireDeviceLocal(embedBytes);
    GrillyBuffer bufOutDL   = pool.acquireDeviceLocal(outBytes);

    GrillyBuffer bufIdsStage   = pool.acquire(idBytes);
    GrillyBuffer bufEmbedStage = pool.acquire(embedBytes);
    GrillyBuffer bufOutStage   = pool.acquireReadback(outBytes);

    pool.upload(bufIdsStage, reinterpret_cast<const float*>(tokenIds), idBytes);
    pool.upload(bufEmbedStage, embeddings, embedBytes);

    PipelineEntry pipe = cache.getOrCreate("embedding-lookup", 3,
                                           sizeof(EmbeddingParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufIdsDL.handle,   0, idBytes},
        {bufEmbedDL.handle, 0, embedBytes},
        {bufOutDL.handle,   0, outBytes},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("embedding-lookup",
                                                        bufInfos);

    uint32_t gx = (totalTokens + 255) / 256;

    batch.begin();
    batch.copyBuffer(bufIdsStage,   bufIdsDL,   idBytes);
    batch.copyBuffer(bufEmbedStage, bufEmbedDL, embedBytes);
    batch.transferComputeBarrier();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                   &p, sizeof(p));
    batch.transferComputeBarrier();
    batch.copyBuffer(bufOutDL, bufOutStage, outBytes);
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufOutStage, output, outBytes);

    pool.release(bufIdsDL);
    pool.release(bufEmbedDL);
    pool.release(bufOutDL);
    pool.release(bufIdsStage);
    pool.release(bufEmbedStage);
    pool.release(bufOutStage);
}

}  // namespace ops
}  // namespace grilly
