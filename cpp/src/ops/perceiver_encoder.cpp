/// perceiver_encoder.cpp — Native Batched Perceiver IO Encoder.
///
/// Single command buffer for the entire N-layer pipeline.
/// Barrier-free parallel GEMM dispatch: Q/K/V projections are independent
/// (read same input, write different outputs), so the GPU hardware
/// scheduler can overlap them on RDNA2's dual compute pipe.
///
/// Per layer dispatch pattern:
///   [Q_cross | K_cross | V_cross] → barrier → perceiver_cross → barrier
///   → residual_add → barrier
///   → [Q_self | K_self | V_self] → barrier → perceiver_self → barrier
///   → residual_add → barrier → copyBuffer (ping-pong)
///
/// Dispatches per layer: 10 (6 GEMMs + 2 perceiver + 2 adds + 1 copy)
/// Barriers per layer: 6 (vs 10 if every dispatch had its own barrier)

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

    // Pre-allocate all working VRAM
    size_t latentBytes = size_t(nLatents) * dModel * sizeof(float);
    size_t inputBytes  = size_t(maxPatches) * dModel * sizeof(float);

    pc.latentsPing = pool.acquire(latentBytes);
    pc.latentsPong = pool.acquire(latentBytes);
    pc.crossQ      = pool.acquire(latentBytes);
    pc.crossK      = pool.acquire(inputBytes);
    pc.crossV      = pool.acquire(inputBytes);
    pc.crossOut    = pool.acquire(latentBytes);
    pc.selfQ       = pool.acquire(latentBytes);
    pc.selfK       = pool.acquire(latentBytes);
    pc.selfV       = pool.acquire(latentBytes);
    pc.selfOut     = pool.acquire(latentBytes);
    pc.inputBuf    = pool.acquire(inputBytes);

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

    // Upload input patches and initialize latents (before batch)
    pool.upload(pc.inputBuf, patches, inputBytes);

    std::vector<float> latent_data(N * D);
    pool.download(pc.baseLatents, latent_data.data(), latentBytes);
    pool.upload(pc.latentsPing, latent_data.data(), latentBytes);

    // ══════════════════════════════════════════════════════════════════
    // Record entire perceiver pipeline into ONE command buffer
    // ══════════════════════════════════════════════════════════════════
    batch.begin();

    GrillyBuffer* current = &pc.latentsPing;
    GrillyBuffer* scratch = &pc.latentsPong;

    for (uint32_t l = 0; l < pc.numLayers; ++l) {
        const auto& lw = pc.layers[l];

        // ── Cross-attention QKV: 3 parallel GEMMs (no barriers) ──────
        // Q reads *current, K/V read inputBuf. All write different buffers.
        // GPU hardware scheduler can overlap these on dual compute pipe.
        batchedTiledLinear(batch, cache,
                           *current, lw.W_Q_cross, nullptr, pc.crossQ,
                           N, D, D);
        batchedTiledLinear(batch, cache,
                           pc.inputBuf, lw.W_K_cross, nullptr, pc.crossK,
                           nPatches, D, D);
        batchedTiledLinear(batch, cache,
                           pc.inputBuf, lw.W_V_cross, nullptr, pc.crossV,
                           nPatches, D, D);
        batch.barrier();  // Wait for ALL three QKV projections

        // Cross-attention: nHeads barrier-free dispatches at head_dim=D/nHeads
        // Each head uses 16 VGPRs (D=64) → full occupancy on RDNA2
        batchedPerceiverEncodeMultiHead(batch, cache,
                                         pc.crossQ, pc.crossK, pc.crossV, pc.crossOut,
                                         N, nPatches, D, pc.nHeads);
        batch.barrier();

        // Residual: crossOut += current
        batchedAdd(batch, cache, pc.crossOut, *current, N * D);
        batch.barrier();

        // ── Self-attention QKV: 3 parallel GEMMs (no barriers) ───────
        // All read crossOut, write different buffers.
        batchedTiledLinear(batch, cache,
                           pc.crossOut, lw.W_Q_self, nullptr, pc.selfQ,
                           N, D, D);
        batchedTiledLinear(batch, cache,
                           pc.crossOut, lw.W_K_self, nullptr, pc.selfK,
                           N, D, D);
        batchedTiledLinear(batch, cache,
                           pc.crossOut, lw.W_V_self, nullptr, pc.selfV,
                           N, D, D);
        batch.barrier();  // Wait for ALL three QKV projections

        // Self-attention: nHeads barrier-free dispatches at head_dim
        batchedPerceiverEncodeMultiHead(batch, cache,
                                         pc.selfQ, pc.selfK, pc.selfV, pc.selfOut,
                                         N, N, D, pc.nHeads);
        batch.barrier();

        // Residual: selfOut += crossOut
        batchedAdd(batch, cache, pc.selfOut, pc.crossOut, N * D);
        batch.barrier();

        // Ping-pong: copy result to scratch, swap for next layer
        batch.copyBuffer(pc.selfOut, *scratch, latentBytes);
        batch.barrier();

        std::swap(current, scratch);
    }

    // ══════════════════════════════════════════════════════════════════
    // SUBMIT: One fence wait for the entire pipeline!
    // ══════════════════════════════════════════════════════════════════
    batch.submit();

    // Download final latents and mean-pool on CPU
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
