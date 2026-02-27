#pragma once

#include <cstdint>
#include <vector>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

/// Conv2d forward push constants — matches conv2d-forward.glsl layout.
/// "17I" = 17 unsigned ints, 68 bytes.
struct Conv2dParams {
    uint32_t batchSize;
    uint32_t inChannels;
    uint32_t inHeight;
    uint32_t inWidth;
    uint32_t outChannels;
    uint32_t outHeight;
    uint32_t outWidth;
    uint32_t kernelH;
    uint32_t kernelW;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t paddingH;
    uint32_t paddingW;
    uint32_t dilationH;
    uint32_t dilationW;
    uint32_t groups;
    uint32_t hasBias;
};

/// Im2col push constants — matches convd_im2col.glsl layout.
/// "14I" = 14 unsigned ints, 56 bytes.
struct Im2colParams {
    uint32_t batchSize;
    uint32_t inChannels;
    uint32_t inH;
    uint32_t inW;
    uint32_t outH;
    uint32_t outW;
    uint32_t kernelH;
    uint32_t kernelW;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t paddingH;
    uint32_t paddingW;
    uint32_t dilationH;
    uint32_t dilationW;
};

/// GEMM push constants — matches gemm_mnk.glsl layout.
/// "3I" = 3 unsigned ints, 12 bytes.
struct GemmParams {
    uint32_t M;
    uint32_t K;
    uint32_t N;
};

/// Bias addition + reshape push constants — matches bias-add.glsl layout.
/// Used by the conv2d GEMM path to add bias on GPU instead of downloading
/// to CPU. The shader reads the GEMM output (M x N), adds per-channel bias,
/// and writes the result. M = outChannels, N = batch * outH * outW.
///
/// "3I" = 3 unsigned ints, 12 bytes.
struct BiasAddParams {
    uint32_t M;         // outChannels (rows)
    uint32_t N;         // batch * outH * outW (columns)
    uint32_t hasBias;   // 1 if bias is present
};

/// Compute output spatial dimensions for conv2d.
inline uint32_t convOutputSize(uint32_t input, uint32_t kernel,
                                uint32_t stride, uint32_t padding,
                                uint32_t dilation) {
    return (input + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
}

/// GPU Conv2d forward pass.
///
/// Supports two paths:
///   1. Direct conv2d shader (groups > 1 or dilation > 1)
///   2. GEMM path: im2col + gemm_mnk (preferred for groups==1, dilation==1)
///
/// The GEMM path uses two dispatches with a barrier between them:
///   Step 1: im2col — unfold input patches into column matrix
///   Step 2: gemm_mnk — output_channels × patch_columns matmul
///
/// 4 buffer bindings for direct path: input, weight, bias, output.
/// GEMM path uses 2 + 3 = 5 total dispatches across 2 shaders.
void conv2d(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
            const float* input, const float* weight, const float* bias,
            float* output,
            uint32_t batchSize, uint32_t inChannels,
            uint32_t inHeight, uint32_t inWidth,
            uint32_t outChannels, uint32_t kernelH, uint32_t kernelW,
            uint32_t strideH = 1, uint32_t strideW = 1,
            uint32_t paddingH = 0, uint32_t paddingW = 0,
            uint32_t dilationH = 1, uint32_t dilationW = 1,
            uint32_t groups = 1);

/// Conv1d is a thin wrapper around Conv2d.
/// Reshapes (N, C, L) -> (N, C, 1, L), runs conv2d with kernel (1, K),
/// then squeezes dim 2. Same as nn/conv.py:Conv1d.
void conv1d(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
            const float* input, const float* weight, const float* bias,
            float* output,
            uint32_t batchSize, uint32_t inChannels, uint32_t length,
            uint32_t outChannels, uint32_t kernelSize,
            uint32_t stride = 1, uint32_t padding = 0,
            uint32_t dilation = 1, uint32_t groups = 1);

}  // namespace ops
}  // namespace grilly
