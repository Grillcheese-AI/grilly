#pragma once
/// Native Batched Perceiver IO Encoder — single command buffer submission.
///
/// Pre-allocates all VRAM working buffers at weight upload time.
/// The encode() call records the entire N-layer dispatch graph into ONE
/// Vulkan command buffer, submits, and waits on a single fence.
///
/// Optimization: barrier-free parallel GEMM dispatch.
/// Q/K/V projections within each attention block are independent (read
/// same input, write different outputs). By omitting barriers between them,
/// the RDNA2 dual compute pipe can overlap them concurrently.
///   Cross-attn: 3 GEMMs barrier-free → 1 barrier → perceiver dispatch
///   Self-attn:  3 GEMMs barrier-free → 1 barrier → perceiver dispatch
///   Per layer: 6 serial dispatches → 2 parallel groups + 2 perceiver

#include <cstdint>
#include <vector>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

/// Per-layer weights for the Perceiver encoder.
struct PerceiverLayerGPU {
    // Cross-attention: Q from latents, K/V from patches
    GrillyBuffer W_Q_cross, W_K_cross, W_V_cross;
    // Self-attention: Q/K/V from latent state
    GrillyBuffer W_Q_self, W_K_self, W_V_self;
};

/// Pre-allocated Perceiver cache: weights + working VRAM.
struct PerceiverCache {
    uint32_t numLayers = 0;
    uint32_t nLatents = 0;   // N
    uint32_t dModel = 0;     // D
    uint32_t nHeads = 0;
    uint32_t headDim = 0;    // D / nHeads

    // Persistent weights
    GrillyBuffer baseLatents;    // (N, D)
    GrillyBuffer posEmbeddings;  // (maxPatches, D)
    std::vector<PerceiverLayerGPU> layers;

    // Pre-allocated working buffers
    GrillyBuffer latentsPing;    // (N, D) — ping-pong
    GrillyBuffer latentsPong;    // (N, D) — ping-pong
    GrillyBuffer crossQ;        // (N, D)
    GrillyBuffer crossK;        // (M, D)
    GrillyBuffer crossV;        // (M, D)
    GrillyBuffer crossOut;      // (N, D)
    GrillyBuffer selfQ;         // (N, D)
    GrillyBuffer selfK;         // (N, D)
    GrillyBuffer selfV;         // (N, D)
    GrillyBuffer selfOut;       // (N, D)
    GrillyBuffer inputBuf;      // (M, D)

    // IndexCache: pre-projected K/V per layer (computed once per encode)
    // Eliminates 2 * nLayers GEMM dispatches from the main loop by
    // batching all K_cross and V_cross projections upfront.
    std::vector<GrillyBuffer> allCrossK;  // [nLayers] × (M, D)
    std::vector<GrillyBuffer> allCrossV;  // [nLayers] × (M, D)

    uint32_t maxPatches = 0;
};

int perceiver_upload_weights(BufferPool& pool,
                             const std::vector<std::vector<float>>& weights,
                             uint32_t nLatents, uint32_t dModel,
                             uint32_t nHeads, uint32_t nLayers,
                             uint32_t maxPatches);

PerceiverCache& perceiver_get_cache(int handle);

std::vector<float> perceiver_encode_native(
    CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
    PerceiverCache& pc,
    const float* patches, uint32_t nPatches);

}  // namespace ops
}  // namespace grilly
