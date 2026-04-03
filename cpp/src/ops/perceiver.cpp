/// perceiver.cpp — Perceiver IO cross-attention GPU dispatch.
///
/// Supports both single-head (stride=head_dim) and multi-head
/// (stride=D_model, head_offset=h*head_dim) modes.

#include "grilly/ops/perceiver.h"

#include <cmath>
#include <cstring>

namespace grilly {
namespace ops {

// ── Single-head convenience (upload/download) ────────────────────────

void perceiverEncode(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                     const float* Q, const float* K, const float* V, float* output,
                     uint32_t seqN, uint32_t seqM, uint32_t headDim) {

    const size_t qBytes   = size_t(seqN) * headDim * sizeof(float);
    const size_t kvBytes  = size_t(seqM) * headDim * sizeof(float);

    GrillyBuffer bufQ   = pool.acquire(qBytes);
    GrillyBuffer bufK   = pool.acquire(kvBytes);
    GrillyBuffer bufV   = pool.acquire(kvBytes);
    GrillyBuffer bufOut = pool.acquire(qBytes);

    pool.upload(bufQ, Q, qBytes);
    pool.upload(bufK, K, kvBytes);
    pool.upload(bufV, V, kvBytes);

    batch.begin();
    batchedPerceiverEncode(batch, cache, bufQ, bufK, bufV, bufOut,
                           seqN, seqM, headDim);
    batch.submitDeferred();
    batch.waitForCompletion();

    pool.download(bufOut, output, qBytes);

    pool.release(bufQ);
    pool.release(bufK);
    pool.release(bufV);
    pool.release(bufOut);
}

// ── Buffer-handle single dispatch (stride = headDim) ─────────────────

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
    pc.seq_M       = seqM;
    pc.seq_N       = seqN;
    pc.head_dim    = headDim;
    pc.scale       = 1.0f / std::sqrt(static_cast<float>(headDim));
    pc.stride      = headDim;  // Contiguous layout
    pc.head_offset = 0;

    uint32_t workgroups_x = (seqN + 63) / 64;
    batch.dispatch(pipe.pipeline, pipe.layout, descSet,
                   workgroups_x, 1, 1, &pc, sizeof(pc));
}

// ── Multi-head dispatch: nHeads dispatches, barrier-free ─────────────
// All heads read from the same Q/K/V buffers with (seq, D_model) layout.
// Each dispatch handles one head via stride + head_offset.
// No barriers between heads — GPU overlaps them on dual compute pipe.

void batchedPerceiverEncodeMultiHead(CommandBatch& batch, PipelineCache& cache,
                                      GrillyBuffer& bufQ, GrillyBuffer& bufK,
                                      GrillyBuffer& bufV, GrillyBuffer& bufOut,
                                      uint32_t seqN, uint32_t seqM,
                                      uint32_t dModel, uint32_t nHeads) {

    uint32_t headDim = dModel / nHeads;
    float invSqrtD = 1.0f / std::sqrt(static_cast<float>(headDim));

    PipelineEntry pipe = cache.getOrCreate("perceiver-encode", 4, sizeof(PerceiverEncodeParams));

    // All heads share the same descriptor set (same buffers, different offsets via push constants)
    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufQ.handle,   0, size_t(seqN) * dModel * sizeof(float)},
        {bufK.handle,   0, size_t(seqM) * dModel * sizeof(float)},
        {bufV.handle,   0, size_t(seqM) * dModel * sizeof(float)},
        {bufOut.handle, 0, size_t(seqN) * dModel * sizeof(float)},
    };
    VkDescriptorSet descSet = cache.allocDescriptorSet("perceiver-encode", bufInfos);

    uint32_t workgroups_x = (seqN + 63) / 64;

    // Dispatch all heads barrier-free — GPU hardware overlaps them
    for (uint32_t h = 0; h < nHeads; ++h) {
        PerceiverEncodeParams pc;
        pc.seq_M       = seqM;
        pc.seq_N       = seqN;
        pc.head_dim    = headDim;
        pc.scale       = invSqrtD;
        pc.stride      = dModel;
        pc.head_offset = h * headDim;

        batch.dispatch(pipe.pipeline, pipe.layout, descSet,
                       workgroups_x, 1, 1, &pc, sizeof(pc));
    }
    // Caller adds barrier after all heads complete
}

}  // namespace ops
}  // namespace grilly
