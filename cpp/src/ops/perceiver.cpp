/// perceiver.cpp — Perceiver IO cross-attention GPU dispatch.
///
/// Register-pinned Q, streaming K/V, online softmax.
/// See perceiver-encode.glsl for shader details.

#include "grilly/ops/perceiver.h"

#include <cmath>
#include <cstring>

namespace grilly {
namespace ops {

// ── Upload/download version (convenience for Python) ─────────────────

void perceiverEncode(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                     const float* Q, const float* K, const float* V, float* output,
                     uint32_t seqN, uint32_t seqM, uint32_t headDim) {

    const size_t qBytes   = size_t(seqN) * headDim * sizeof(float);
    const size_t kvBytes  = size_t(seqM) * headDim * sizeof(float);
    const size_t outBytes = qBytes;

    GrillyBuffer bufQ   = pool.acquire(qBytes);
    GrillyBuffer bufK   = pool.acquire(kvBytes);
    GrillyBuffer bufV   = pool.acquire(kvBytes);
    GrillyBuffer bufOut = pool.acquire(outBytes);

    pool.upload(bufQ, Q, qBytes);
    pool.upload(bufK, K, kvBytes);
    pool.upload(bufV, V, kvBytes);

    batch.begin();
    batchedPerceiverEncode(batch, cache, bufQ, bufK, bufV, bufOut,
                           seqN, seqM, headDim);
    batch.submit();

    pool.download(bufOut, output, outBytes);

    pool.release(bufQ);
    pool.release(bufK);
    pool.release(bufV);
    pool.release(bufOut);
}

// ── Buffer-handle version (for batched/fused dispatch chains) ────────

void batchedPerceiverEncode(CommandBatch& batch, PipelineCache& cache,
                            GrillyBuffer& bufQ, GrillyBuffer& bufK,
                            GrillyBuffer& bufV, GrillyBuffer& bufOut,
                            uint32_t seqN, uint32_t seqM, uint32_t headDim) {

    PipelineEntry pipe = cache.getOrCreate("perceiver-encode", 4, sizeof(PerceiverEncodeParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufQ.handle,   0, size_t(seqN) * headDim * sizeof(float)},
        {bufK.handle,   0, size_t(seqM) * headDim * sizeof(float)},
        {bufV.handle,   0, size_t(seqM) * headDim * sizeof(float)},
        {bufOut.handle, 0, size_t(seqN) * headDim * sizeof(float)},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("perceiver-encode", bufInfos);

    PerceiverEncodeParams pc;
    pc.seq_M    = seqM;
    pc.seq_N    = seqN;
    pc.head_dim = headDim;
    pc.scale    = 1.0f / std::sqrt(static_cast<float>(headDim));

    uint32_t workgroups_x = (seqN + 63) / 64;

    batch.dispatch(pipe.pipeline, pipe.layout, descSet,
                   workgroups_x, 1, 1,
                   &pc, sizeof(pc));
}

}  // namespace ops
}  // namespace grilly
