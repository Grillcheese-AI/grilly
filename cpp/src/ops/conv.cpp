#include "grilly/ops/conv.h"

#include <cstring>
#include <stdexcept>

namespace grilly {
namespace ops {

// ── Conv2d forward (port of backend/conv.py) ─────────────────────────────
//
// Two dispatch paths, matching the Python backend's logic:
//
// 1. Direct path (groups > 1 or dilation > 1):
//    Single "conv2d-forward" shader dispatch. 4 buffers, 17I push constants.
//    Workgroups: (out_w+7)/8 × (out_h+7)/8 × (batch * out_channels).
//    Each 8×8 workgroup computes one spatial tile for one (batch, channel).
//
// 2. GEMM path (groups == 1, dilation == 1) — PREFERRED:
//    Two-step dispatch with a barrier between them:
//
//    Step 1: im2col — unfold input patches into a column matrix.
//    The "convd_im2col" shader converts spatial convolution into
//    matrix multiplication by extracting all receptive-field patches
//    into rows: cols[k][n] where k indexes into (C_in, kH, kW) and
//    n indexes into (batch, out_h, out_w). This is a memory-bandwidth
//    operation with no arithmetic.
//
//    Step 2: GEMM — output = weight_reshaped @ cols.
//    The "gemm_mnk" shader does the matmul at 16×16 tile size.
//    M = out_channels, K = C_in * kH * kW, N = batch * out_h * out_w.
//
// The GEMM path is faster because GPU matrix multiply is highly optimized
// (better occupancy, memory coalescing) compared to the direct conv which
// has irregular memory access patterns.

void conv2d(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
            const float* input, const float* weight, const float* bias,
            float* output,
            uint32_t batchSize, uint32_t inChannels,
            uint32_t inHeight, uint32_t inWidth,
            uint32_t outChannels, uint32_t kernelH, uint32_t kernelW,
            uint32_t strideH, uint32_t strideW,
            uint32_t paddingH, uint32_t paddingW,
            uint32_t dilationH, uint32_t dilationW,
            uint32_t groups) {
    // Compute output dimensions
    uint32_t outH = convOutputSize(inHeight, kernelH, strideH, paddingH, dilationH);
    uint32_t outW = convOutputSize(inWidth, kernelW, strideW, paddingW, dilationW);

    const uint32_t hasBias = (bias != nullptr) ? 1 : 0;

    // Buffer sizes
    const size_t inputBytes  = size_t(batchSize) * inChannels * inHeight *
                               inWidth * sizeof(float);
    const size_t weightBytes = size_t(outChannels) * (inChannels / groups) *
                               kernelH * kernelW * sizeof(float);
    const size_t biasBytes   = hasBias ? size_t(outChannels) * sizeof(float)
                                       : sizeof(float);
    const size_t outputBytes = size_t(batchSize) * outChannels * outH * outW *
                               sizeof(float);

    // Prefer GEMM path when groups == 1 and dilation == (1,1)
    bool useGEMM = (groups == 1) && (dilationH == 1) && (dilationW == 1);

    if (useGEMM) {
        // ── GEMM path: im2col + gemm_mnk + GPU bias ──
        //
        // Three-step GPU pipeline (all in one CommandBatch):
        //   Step 1: im2col — unfold input patches into column matrix
        //   Step 2: GEMM — output = weight @ columns
        //   Step 3: bias-add — output += bias (broadcast per channel)
        //
        // Step 3 replaces the old CPU-side bias loop, eliminating the
        // download→CPU bias→upload round-trip that was the bottleneck.

        uint32_t K_dim = inChannels * kernelH * kernelW;
        uint32_t N_cols = batchSize * outH * outW;
        uint32_t M = outChannels;

        const size_t colsBytes = size_t(K_dim) * N_cols * sizeof(float);
        const size_t gemmOutBytes = size_t(M) * N_cols * sizeof(float);

        GrillyBuffer bufInput  = pool.acquire(inputBytes);
        GrillyBuffer bufCols   = pool.acquire(colsBytes);
        GrillyBuffer bufWeight = pool.acquire(weightBytes);
        GrillyBuffer bufGemm   = pool.acquire(gemmOutBytes);

        pool.upload(bufInput, input, inputBytes);
        pool.upload(bufWeight, weight, weightBytes);

        // ── Step 1: im2col ──
        PipelineEntry pipeIm2col = cache.getOrCreate("convd_im2col", 2,
                                                      sizeof(Im2colParams));

        std::vector<VkDescriptorBufferInfo> im2colBufs = {
            {bufInput.handle, 0, inputBytes},
            {bufCols.handle,  0, colsBytes},
        };
        VkDescriptorSet descIm2col = cache.allocDescriptorSet("convd_im2col",
                                                               im2colBufs);

        Im2colParams im2colPush{
            batchSize, inChannels, inHeight, inWidth,
            outH, outW, kernelH, kernelW,
            strideH, strideW, paddingH, paddingW, dilationH, dilationW
        };

        uint32_t im2colGX = (K_dim  + 15) / 16;
        uint32_t im2colGY = (N_cols + 15) / 16;

        // ── Step 2: GEMM ──
        PipelineEntry pipeGemm = cache.getOrCreate("gemm_mnk", 3,
                                                    sizeof(GemmParams));

        std::vector<VkDescriptorBufferInfo> gemmBufs = {
            {bufWeight.handle, 0, weightBytes},
            {bufCols.handle,   0, colsBytes},
            {bufGemm.handle,   0, gemmOutBytes},
        };
        VkDescriptorSet descGemm = cache.allocDescriptorSet("gemm_mnk",
                                                             gemmBufs);

        GemmParams gemmPush{M, K_dim, N_cols};

        uint32_t gemmGX = (N_cols + 15) / 16;
        uint32_t gemmGY = (M + 15) / 16;

        // ── Step 3: GPU-side bias addition ──
        // Upload bias to GPU and dispatch bias-add shader as a third step,
        // eliminating the CPU download→bias→upload bottleneck.
        GrillyBuffer bufBias = pool.acquire(biasBytes);
        GrillyBuffer bufOutput = pool.acquire(gemmOutBytes);
        if (hasBias && bias) {
            pool.upload(bufBias, bias, outChannels * sizeof(float));
        }

        // Record all dispatches in one CommandBatch
        batch.begin();

        // Step 1: im2col
        batch.dispatch(pipeIm2col.pipeline, pipeIm2col.layout, descIm2col,
                       im2colGX, im2colGY, 1,
                       &im2colPush, sizeof(im2colPush));
        batch.barrier();

        // Step 2: GEMM
        batch.dispatch(pipeGemm.pipeline, pipeGemm.layout, descGemm,
                       gemmGX, gemmGY, 1,
                       &gemmPush, sizeof(gemmPush));

        if (hasBias && bias && cache.hasShader("bias-add")) {
            // Step 3: GPU bias-add — the bias-add shader reads GEMM output
            // (binding 0), bias vector (binding 1), and writes biased output
            // (binding 2). Each thread handles one element: output[row][col]
            // = gemm[row][col] + bias[row].
            batch.barrier();

            PipelineEntry pipeBias = cache.getOrCreate("bias-add", 3,
                                                        sizeof(BiasAddParams));

            std::vector<VkDescriptorBufferInfo> biasBufs = {
                {bufGemm.handle,   0, gemmOutBytes},
                {bufBias.handle,   0, biasBytes},
                {bufOutput.handle, 0, gemmOutBytes},
            };
            VkDescriptorSet descBias = cache.allocDescriptorSet("bias-add",
                                                                 biasBufs);

            BiasAddParams biasPush{M, N_cols, hasBias};
            uint32_t biasGX = (N_cols + 255) / 256;
            uint32_t biasGY = M;

            batch.dispatch(pipeBias.pipeline, pipeBias.layout, descBias,
                           biasGX, biasGY, 1,
                           &biasPush, sizeof(biasPush));
        }

        batch.submit();

        // Download from the correct buffer (biased output if shader was
        // available, raw GEMM output otherwise)
        if (hasBias && bias && cache.hasShader("bias-add")) {
            pool.download(bufOutput, output, gemmOutBytes);
        } else {
            pool.download(bufGemm, output, gemmOutBytes);

            // Fallback: CPU bias addition when bias-add shader isn't loaded
            if (hasBias && bias) {
                for (uint32_t b = 0; b < batchSize; ++b) {
                    for (uint32_t oc = 0; oc < outChannels; ++oc) {
                        float biasVal = bias[oc];
                        size_t offset = (size_t(oc) * N_cols + b * outH * outW);
                        for (uint32_t hw = 0; hw < outH * outW; ++hw) {
                            output[offset + hw] += biasVal;
                        }
                    }
                }
            }
        }

        pool.release(bufInput);
        pool.release(bufCols);
        pool.release(bufWeight);
        pool.release(bufGemm);
        pool.release(bufBias);
        pool.release(bufOutput);

    } else {
        // ── Direct conv2d path ──

        GrillyBuffer bufInput  = pool.acquire(inputBytes);
        GrillyBuffer bufWeight = pool.acquire(weightBytes);
        GrillyBuffer bufBias   = pool.acquire(biasBytes);
        GrillyBuffer bufOutput = pool.acquire(outputBytes);

        pool.upload(bufInput, input, inputBytes);
        pool.upload(bufWeight, weight, weightBytes);
        if (hasBias && bias) {
            pool.upload(bufBias, bias, outChannels * sizeof(float));
        }

        PipelineEntry pipe = cache.getOrCreate("conv2d-forward", 4,
                                                sizeof(Conv2dParams));

        std::vector<VkDescriptorBufferInfo> bufInfos = {
            {bufInput.handle,  0, inputBytes},
            {bufWeight.handle, 0, weightBytes},
            {bufBias.handle,   0, biasBytes},
            {bufOutput.handle, 0, outputBytes},
        };
        VkDescriptorSet descSet = cache.allocDescriptorSet("conv2d-forward",
                                                            bufInfos);

        Conv2dParams push{
            batchSize, inChannels, inHeight, inWidth,
            outChannels, outH, outW,
            kernelH, kernelW, strideH, strideW,
            paddingH, paddingW, dilationH, dilationW,
            groups, hasBias
        };

        uint32_t gx = (outW + 7) / 8;
        uint32_t gy = (outH + 7) / 8;
        uint32_t gz = batchSize * outChannels;

        batch.begin();
        batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, gy, gz,
                       &push, sizeof(push));
        batch.submit();

        pool.download(bufOutput, output, outputBytes);

        pool.release(bufInput);
        pool.release(bufWeight);
        pool.release(bufBias);
        pool.release(bufOutput);
    }
}

// ── Conv1d: thin wrapper around Conv2d ──────────────────────────────────
//
// Exactly mirrors nn/conv.py:Conv1d — reshape (N, C, L) to (N, C, 1, L),
// run conv2d with kernel (1, K), then squeeze. This avoids duplicating
// the conv dispatch logic.

void conv1d(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
            const float* input, const float* weight, const float* bias,
            float* output,
            uint32_t batchSize, uint32_t inChannels, uint32_t length,
            uint32_t outChannels, uint32_t kernelSize,
            uint32_t stride, uint32_t padding,
            uint32_t dilation, uint32_t groups) {
    // Conv1d -> Conv2d with height=1, kernelH=1
    conv2d(batch, pool, cache,
           input, weight, bias, output,
           batchSize, inChannels,
           1, length,           // inHeight=1, inWidth=length
           outChannels,
           1, kernelSize,       // kernelH=1, kernelW=kernelSize
           1, stride,           // strideH=1
           0, padding,          // paddingH=0
           1, dilation,         // dilationH=1
           groups);
}

}  // namespace ops
}  // namespace grilly
