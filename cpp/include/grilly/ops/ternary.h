#pragma once

#include <cstdint>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

// ── Ternary weight-only GEMM (BitNet b1.58), multiply-free ───────────────
// Shader: ternary-gemm.spv
//   output[m,n] = alpha * sum_k activation[m,k] * trit[n,k],  trit in {-1,0,+1}
// One per-tensor scale (alpha = mean|W|). Weights packed 16 trits/uint32,
// 2 bits each (v1); the packing is a host-side seam for 1.58-bit later.
//
// Uses the DEVICE_LOCAL + staging pattern (see linear.cpp): compute buffers
// live in cached VRAM (~432 GB/s), I/O moves via staging copies over the DMA
// engine. On RX 6750 XT the naive HOST_VISIBLE path runs compute reads at
// ~0.05 GB/s, so this is mandatory, not an optimization.
//
// Buffers: activations(0, fp32 M*K), weights_packed(1, uint32 N*ceil(K/16)),
//          output(2, fp32 M*N).

struct TernaryGemmParams {
    uint32_t M;        // batch*seq (activation rows)
    uint32_t K;        // input dim
    uint32_t N;        // output dim (weight rows)
    float alpha;       // per-tensor scale = mean|W|
};

// wordsPerRow = ceil(K/16); weightsPacked has N*wordsPerRow uint32 elements.
void ternaryGemm(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                 const float* activations, const uint32_t* weightsPacked,
                 float* output, uint32_t wordsPerRow,
                 const TernaryGemmParams& p);

}  // namespace ops
}  // namespace grilly
