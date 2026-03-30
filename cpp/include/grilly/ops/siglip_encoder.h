#pragma once
/// SigLIP2 full encoder — all 12 transformer layers in one CommandBatch.
///
/// Pre-uploads weights to device-local VRAM once. Subsequent encodes only
/// upload the input patches (1024×768) and download the final embedding (768).
/// All intermediate buffers stay on GPU — zero PCIe round-trips between layers.

#include <cstdint>
#include <vector>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

/// Pre-uploaded weight set for one transformer layer.
struct SigLIPLayerWeights {
    GrillyBuffer qkv_w;       // (3*768, 768)
    GrillyBuffer qkv_b;       // (3*768,)
    GrillyBuffer out_proj_w;   // (768, 768)
    GrillyBuffer out_proj_b;   // (768,)
    GrillyBuffer ln1_w;        // (768,)
    GrillyBuffer ln1_b;        // (768,)
    GrillyBuffer ln2_w;        // (768,)
    GrillyBuffer ln2_b;        // (768,)
    GrillyBuffer mlp_w1;       // (3072, 768)
    GrillyBuffer mlp_b1;       // (3072,)
    GrillyBuffer mlp_w2;       // (768, 3072)
    GrillyBuffer mlp_b2;       // (768,)
};

/// Full SigLIP2 encoder state — weights on GPU, ready for batched inference.
struct SigLIPEncoderState {
    std::vector<SigLIPLayerWeights> layers;
    GrillyBuffer post_ln_w;    // (768,)
    GrillyBuffer post_ln_b;    // (768,)
    // Pooling head weights
    GrillyBuffer pool_in_proj_w;  // (2304, 768)
    GrillyBuffer pool_in_proj_b;  // (2304,)
    GrillyBuffer pool_out_proj_w; // (768, 768)
    GrillyBuffer pool_out_proj_b; // (768,)
    GrillyBuffer pool_ln_w;       // (768,)
    GrillyBuffer pool_ln_b;       // (768,)
    GrillyBuffer pool_mlp_w1;     // (3072, 768)
    GrillyBuffer pool_mlp_b1;     // (3072,)
    GrillyBuffer pool_mlp_w2;     // (768, 3072)
    GrillyBuffer pool_mlp_b2;     // (768,)
    GrillyBuffer pool_probe;      // (1, 768)

    uint32_t numLayers = 0;
    uint32_t seqLen = 0;     // 1024
    uint32_t hiddenSize = 0; // 768
    uint32_t numHeads = 0;   // 12
    uint32_t headDim = 0;    // 64
    uint32_t mlpDim = 0;     // 3072
    bool initialized = false;
};

/// Upload all SigLIP2 weights to device-local VRAM (call once at init).
void siglipUploadWeights(
    BufferPool& pool, SigLIPEncoderState& state,
    const float* const* layerWeights,  // array of weight pointers per layer
    uint32_t numLayers, uint32_t seqLen, uint32_t hiddenSize,
    uint32_t numHeads, uint32_t mlpDim);

/// Run the full SigLIP2 encoder in ONE CommandBatch.
/// Input: patches (seqLen × hiddenSize) on CPU.
/// Output: embedding (hiddenSize) on CPU, L2-normalized.
/// All intermediates stay on GPU — only one upload + one download.
void siglipEncode(
    CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
    const SigLIPEncoderState& state,
    const float* patches, float* embedding);

}  // namespace ops
}  // namespace grilly
