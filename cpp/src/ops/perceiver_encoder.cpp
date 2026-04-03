/// perceiver_encoder.cpp — Native Batched Perceiver IO Encoder with IndexCache.
///
/// IndexCache optimization: cross-attention K/V projections are pre-computed
/// for ALL layers in a single GPU submit before the main loop starts.
/// Since inputBuf (image patches) doesn't change across layers, we batch
/// 2*nLayers GEMMs into one submission, then the main loop only needs:
///   Q_cross (1 GEMM) + cached K/V + perceiver attention
///
/// Per-layer dispatch: 10 → 7 dispatches (saved 2 K/V GEMMs per layer)
/// For 6 layers: 12 GEMMs eliminated, replaced by 1 upfront batch.
///
/// Also: barrier-free parallel dispatch within each attention block.

#include "grilly/ops/perceiver_encoder.h"
#include "grilly/ops/perceiver.h"
#include "grilly/ops/batched_ops.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace grilly {
namespace ops {

static std::unordered_map<int, PerceiverCache> g_perceiver_caches;
static int g_next_perceiver_handle = 1;

PerceiverCache& perceiver_get_cache(int handle) {
    auto it = g_perceiver_caches.find(handle);
    if (it == g_perceiver_caches.end())
        throw std::runtime_error("Invalid perceiver cache handle");
    return it->second;
}

static GrillyBuffer uploadVec(BufferPool& pool, const std::vector<float>& data) {
    size_t bytes = data.size() * sizeof(float);
    GrillyBuffer buf = pool.acquire(bytes);
    pool.upload(buf, data.data(), bytes);
    return buf;
}

int perceiver_upload_weights(BufferPool& pool,
                             const std::vector<std::vector<float>>& weights,
                             uint32_t nLatents, uint32_t dModel,
                             uint32_t nHeads, uint32_t nLayers,
                             uint32_t maxPatches) {

    PerceiverCache pc;
    pc.numLayers = nLayers;
    pc.nLatents = nLatents;
    pc.dModel = dModel;
    pc.nHeads = nHeads;
    pc.headDim = dModel / nHeads;
    pc.maxPatches = maxPatches;

    size_t idx = 0;
    pc.layers.resize(nLayers);
    for (uint32_t l = 0; l < nLayers; ++l) {
        auto& lw = pc.layers[l];
        lw.W_Q_cross = uploadVec(pool, weights[idx++]);
        lw.W_K_cross = uploadVec(pool, weights[idx++]);
        lw.W_V_cross = uploadVec(pool, weights[idx++]);
        lw.W_Q_self  = uploadVec(pool, weights[idx++]);
        lw.W_K_self  = uploadVec(pool, weights[idx++]);
        lw.W_V_self  = uploadVec(pool, weights[idx++]);
    }

    pc.baseLatents = uploadVec(pool, weights[idx++]);
    pc.posEmbeddings = uploadVec(pool, weights[idx++]);

    // Pre-allocate working VRAM
    size_t latentBytes = size_t(nLatents) * dModel * sizeof(float);
    size_t inputBytes  = size_t(maxPatches) * dModel * sizeof(float);

    pc.latentsPing = pool.acquire(latentBytes);
    pc.latentsPong = pool.acquire(latentBytes);
    pc.crossQ      = pool.acquire(latentBytes);
    pc.crossK      = pool.acquire(inputBytes);   // Working buffer for single layer
    pc.crossV      = pool.acquire(inputBytes);
    pc.crossOut    = pool.acquire(latentBytes);
    pc.selfQ       = pool.acquire(latentBytes);
    pc.selfK       = pool.acquire(latentBytes);
    pc.selfV       = pool.acquire(latentBytes);
    pc.selfOut     = pool.acquire(latentBytes);
    pc.inputBuf    = pool.acquire(inputBytes);

    // IndexCache: pre-allocate per-layer K/V buffers
    pc.allCrossK.resize(nLayers);
    pc.allCrossV.resize(nLayers);
    for (uint32_t l = 0; l < nLayers; ++l) {
        pc.allCrossK[l] = pool.acquire(inputBytes);
        pc.allCrossV[l] = pool.acquire(inputBytes);
    }

    int handle = g_next_perceiver_handle++;
    g_perceiver_caches[handle] = std::move(pc);
    return handle;
}

std::vector<float> perceiver_encode_native(
    CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
    PerceiverCache& pc,
    const float* patches, uint32_t nPatches) {

    const uint32_t N = pc.nLatents;
    const uint32_t D = pc.dModel;
    const size_t latentBytes = size_t(N) * D * sizeof(float);
    const size_t inputBytes  = size_t(nPatches) * D * sizeof(float);

    // Upload input patches and initialize latents
    pool.upload(pc.inputBuf, patches, inputBytes);

    std::vector<float> latent_data(N * D);
    pool.download(pc.baseLatents, latent_data.data(), latentBytes);
    pool.upload(pc.latentsPing, latent_data.data(), latentBytes);

    // ══════════════════════════════════════════════════════════════════
    // PHASE 1: IndexCache — pre-project ALL cross-attention K/V
    //
    // inputBuf doesn't change across layers, so we compute ALL K and V
    // projections in one batch. This replaces 2*nLayers serial GEMMs
    // in the main loop with one parallel burst.
    //
    // For 6 layers: 12 GEMMs (6×K + 6×V) dispatched barrier-free,
    // all reading the same inputBuf, writing to separate allCrossK/V.
    // ══════════════════════════════════════════════════════════════════
    batch.begin();

    for (uint32_t l = 0; l < pc.numLayers; ++l) {
        const auto& lw = pc.layers[l];
        // All K/V projections are independent — barrier-free
        batchedTiledLinear(batch, cache,
                           pc.inputBuf, lw.W_K_cross, nullptr, pc.allCrossK[l],
                           nPatches, D, D);
        batchedTiledLinear(batch, cache,
                           pc.inputBuf, lw.W_V_cross, nullptr, pc.allCrossV[l],
                           nPatches, D, D);
    }
    // ONE barrier after ALL K/V projections complete
    batch.submitDeferred();
    batch.waitForCompletion();

    // ══════════════════════════════════════════════════════════════════
    // PHASE 2: Main layer loop — only Q_cross + cached K/V + self-attn
    //
    // Per layer: 1 Q GEMM + perceiver(Q, cachedK, cachedV) + self-attn
    // Saves 2 GEMMs per layer vs the non-cached version.
    // ══════════════════════════════════════════════════════════════════
    batch.begin();

    GrillyBuffer* current = &pc.latentsPing;
    GrillyBuffer* scratch = &pc.latentsPong;

    for (uint32_t l = 0; l < pc.numLayers; ++l) {
        const auto& lw = pc.layers[l];

        // ── Cross-attention: only Q needs computing (K/V pre-cached) ──
        batchedTiledLinear(batch, cache,
                           *current, lw.W_Q_cross, nullptr, pc.crossQ,
                           N, D, D);
        batch.barrier();

        // Cross-attention using pre-cached K/V
        batchedPerceiverEncodeMultiHead(batch, cache,
                                         pc.crossQ, pc.allCrossK[l],
                                         pc.allCrossV[l], pc.crossOut,
                                         N, nPatches, D, pc.nHeads);
        batch.barrier();

        // Residual
        batchedAdd(batch, cache, pc.crossOut, *current, N * D);
        batch.barrier();

        // ── Self-attention: 3 parallel GEMMs (no barriers between) ───
        batchedTiledLinear(batch, cache,
                           pc.crossOut, lw.W_Q_self, nullptr, pc.selfQ,
                           N, D, D);
        batchedTiledLinear(batch, cache,
                           pc.crossOut, lw.W_K_self, nullptr, pc.selfK,
                           N, D, D);
        batchedTiledLinear(batch, cache,
                           pc.crossOut, lw.W_V_self, nullptr, pc.selfV,
                           N, D, D);
        batch.barrier();

        batchedPerceiverEncodeMultiHead(batch, cache,
                                         pc.selfQ, pc.selfK, pc.selfV, pc.selfOut,
                                         N, N, D, pc.nHeads);
        batch.barrier();

        // Residual + ping-pong
        batchedAdd(batch, cache, pc.selfOut, pc.crossOut, N * D);
        batch.barrier();

        batch.copyBuffer(pc.selfOut, *scratch, latentBytes);
        batch.barrier();

        std::swap(current, scratch);
    }

    batch.submitDeferred();
    batch.waitForCompletion();

    // Download and mean-pool
    std::vector<float> finalLatents(N * D);
    pool.download(*current, finalLatents.data(), latentBytes);

    std::vector<float> output(D, 0.0f);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t d = 0; d < D; ++d) {
            output[d] += finalLatents[n * D + d];
        }
    }
    float inv_n = 1.0f / static_cast<float>(N);
    for (uint32_t d = 0; d < D; ++d) {
        output[d] *= inv_n;
    }

    return output;
}

}  // namespace ops
}  // namespace grilly
