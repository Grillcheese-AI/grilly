#pragma once
/// Native Batched Perceiver IO Encoder — single command buffer submission.
///
/// Pre-allocates all VRAM working buffers at weight upload time.
/// The encode() call records the entire N-layer dispatch graph into ONE
/// Vulkan command buffer, submits, and waits on a single fence.
///
/// Architecture:
///   1. Cross-attention: latents (N) query inputs (M) via perceiver-encode.spv
///   2. Self-attention + FFN layers with ping-pong working buffers
///   3. Final LayerNorm + mean pool → dense vector
///
/// Zero Python round-trips during inference. GIL released during GPU work.

#include <cstdint>
#include <vector>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

/// Per-layer weights for the Perceiver encoder.
struct PerceiverLayerGPU {
    // Cross-attention (only layer 0 uses these if share_cross=false)
    GrillyBuffer W_Q_cross, W_K_cross, W_V_cross;
    // Self-attention
    GrillyBuffer W_Q_self, W_K_self, W_V_self;
};

/// Pre-allocated Perceiver cache: weights + working VRAM.
struct PerceiverCache {
    // Config
    uint32_t numLayers = 0;
    uint32_t nLatents = 0;   // N (bottleneck size, e.g. 64-256)
    uint32_t dModel = 0;     // Full model dim (e.g. 256)
    uint32_t nHeads = 0;
    uint32_t headDim = 0;    // dModel / nHeads

    // Persistent weights (uploaded once, never released)
    GrillyBuffer baseLatents;           // (N, dModel)
    GrillyBuffer posEmbeddings;         // (maxPatches, dModel)
    std::vector<PerceiverLayerGPU> layers;

    // Pre-allocated working buffers (ping-pong)
    GrillyBuffer latentsPing;           // (N, dModel)
    GrillyBuffer latentsPong;           // (N, dModel)
    GrillyBuffer crossQ;               // (N, dModel) — projected queries
    GrillyBuffer crossK;               // (maxPatches, dModel)
    GrillyBuffer crossV;               // (maxPatches, dModel)
    GrillyBuffer crossOut;             // (N, headDim) per head
    GrillyBuffer selfQ, selfK, selfV;  // (N, dModel)
    GrillyBuffer selfOut;              // (N, dModel)
    GrillyBuffer inputBuf;             // (maxPatches, dModel)

    uint32_t maxPatches = 0;
};

/// Upload Perceiver weights to GPU and pre-allocate all working VRAM.
/// Returns a handle for use with perceiver_encode_native().
///
/// @param weights  Flat list of numpy arrays in layer order:
///   For each layer: W_Q_cross, W_K_cross, W_V_cross, W_Q_self, W_K_self, W_V_self
///   Then: base_latents, position_embeddings
/// @param nLatents   Bottleneck size N
/// @param dModel     Model dimension
/// @param nHeads     Number of attention heads
/// @param nLayers    Number of cross+self attention layers
/// @param maxPatches Maximum input sequence length
int perceiver_upload_weights(BufferPool& pool,
                             const std::vector<std::vector<float>>& weights,
                             uint32_t nLatents, uint32_t dModel,
                             uint32_t nHeads, uint32_t nLayers,
                             uint32_t maxPatches);

/// Get a reference to a cached PerceiverCache by handle.
PerceiverCache& perceiver_get_cache(int handle);

/// Encode input patches through the full Perceiver pipeline.
/// Single command buffer submission — zero Python round-trips.
///
/// @param patches  (M, dModel) float32 input patches (will be uploaded)
/// @param nPatches Actual number of patches in this call (≤ maxPatches)
/// @return         (dModel,) float32 mean-pooled output vector
std::vector<float> perceiver_encode_native(
    CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
    PerceiverCache& pc,
    const float* patches, uint32_t nPatches);

}  // namespace ops
}  // namespace grilly
