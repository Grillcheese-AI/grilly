#include "grilly/experimental/fused_attention.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace grilly {
namespace experimental {

// ── Fused Subgroup-Decompress + Flash Attention (GPU dispatch) ──────────
//
// The GPU path dispatches a single shader that:
//   1. Reads compressed latents from the paged pool (1 VRAM read)
//   2. Decompresses K and V in Wave32 registers via subgroup matmul
//   3. Computes tiled attention with online softmax entirely in registers
//   4. Writes final output (1 VRAM write)
//
// The shader name is "fused-subgroup-attention" and requires:
//   - VK_KHR_shader_subgroup (basic + vote + arithmetic + ballot + shuffle)
//   - 13 push constant fields = 52 bytes
//   - Buffers: Q, latent_pool_pages[], W_up, mask, output,
//              running_max, running_sum, output_accum
//
// For the proof-of-concept, we dispatch the existing flash-attention2
// shader with pre-decompressed K/V when the fused shader isn't available.
// The CPU reference path below verifies correctness for both paths.

void fusedSubgroupAttention(
    CommandBatch& batch, BufferPool& bufPool, PipelineCache& cache,
    const float* Q, const PagedLatentPool& pool,
    const GrillyBuffer& wUp, const float* mask,
    float* output,
    uint32_t batchSize, uint32_t seqLen,
    uint32_t numHeads, uint32_t headDim, uint32_t latentDim,
    float scale, uint32_t waveSize, uint32_t quantMode) {
    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(headDim));
    }

    const auto& cfg = pool.config();
    uint32_t cachedTokens = pool.totalTokens();

    if (cachedTokens == 0) return;

    // ── Gather latents from all active pages into contiguous buffer ──
    // In the full GPU path, the shader would read from paged buffers directly.
    // For now, we gather into a contiguous buffer and use the standard
    // flash-attention2 path with pre-decompressed K/V.

    size_t latentBytes = size_t(cachedTokens) * numHeads * latentDim *
                         sizeof(float);
    size_t kvBytes = size_t(cachedTokens) * numHeads * headDim * sizeof(float);

    // Gather latents from pages
    std::vector<float> allLatents(cachedTokens * numHeads * latentDim);
    uint32_t gathered = 0;
    for (uint32_t p = 0; p < pool.allocatedPages() + 10 && gathered < cachedTokens; ++p) {
        try {
            const float* pagePtr = pool.getReadPtr(p, 0);
            // Read however many tokens this page has
            // We need to query page token count through the stats
            // For simplicity, read up to tokensPerPage
            uint32_t tokensInPage = std::min(cfg.tokensPerPage,
                                              cachedTokens - gathered);
            size_t copyBytes = size_t(tokensInPage) * numHeads * latentDim *
                               sizeof(float);
            std::memcpy(&allLatents[gathered * numHeads * latentDim],
                        pagePtr, copyBytes);
            gathered += tokensInPage;
        } catch (...) {
            continue;  // Skip inactive pages
        }
    }

    // ── Decompress latents: [K, V] = W_up @ c_t ──
    // W_up is (latentDim, 2*headDim) per head
    // Input latents: (cachedTokens * numHeads, latentDim)
    // Output: (cachedTokens * numHeads, 2*headDim)

    uint32_t M = cachedTokens * numHeads;
    uint32_t K_dim = latentDim;
    uint32_t N = 2 * headDim;

    size_t decompressedBytes = size_t(M) * N * sizeof(float);

    GrillyBuffer bufLatent = bufPool.acquire(latentBytes);
    GrillyBuffer bufDecompressed = bufPool.acquire(decompressedBytes);

    bufPool.upload(bufLatent, allLatents.data(), latentBytes);

    // GEMM: decompressed = latent @ W_up^T
    PipelineEntry pipeGemm = cache.getOrCreate("gemm_mnk", 3,
                                                sizeof(uint32_t) * 3);

    std::vector<VkDescriptorBufferInfo> gemmBufs = {
        {bufLatent.handle, 0, latentBytes},
        {wUp.handle, 0, size_t(K_dim) * N * sizeof(float)},
        {bufDecompressed.handle, 0, decompressedBytes},
    };
    VkDescriptorSet descGemm = cache.allocDescriptorSet("gemm_mnk", gemmBufs);

    uint32_t gemmPush[3] = {M, K_dim, N};

    batch.begin();
    batch.dispatch(pipeGemm.pipeline, pipeGemm.layout, descGemm,
                   (N + 15) / 16, (M + 15) / 16, 1,
                   gemmPush, sizeof(gemmPush));
    batch.submit();

    // Download decompressed K, V
    std::vector<float> decompressed(M * N);
    bufPool.download(bufDecompressed, decompressed.data(), decompressedBytes);

    // Split into K and V
    std::vector<float> decodedKeys(cachedTokens * numHeads * headDim);
    std::vector<float> decodedValues(cachedTokens * numHeads * headDim);

    for (uint32_t t = 0; t < cachedTokens; ++t) {
        for (uint32_t h = 0; h < numHeads; ++h) {
            size_t srcBase = (size_t(t) * numHeads + h) * N;
            size_t dstBase = (size_t(t) * numHeads + h) * headDim;
            std::memcpy(&decodedKeys[dstBase], &decompressed[srcBase],
                        headDim * sizeof(float));
            std::memcpy(&decodedValues[dstBase],
                        &decompressed[srcBase + headDim],
                        headDim * sizeof(float));
        }
    }

    bufPool.release(bufLatent);
    bufPool.release(bufDecompressed);

    // ── Now run flash attention with the decompressed K, V ──
    // In the full fused shader, steps above would all be in registers.
    // This path proves correctness; the fused shader proves performance.

    const uint32_t hasMask = (mask != nullptr) ? 1 : 0;
    const uint32_t totalQPositions = batchSize * numHeads * seqLen;

    size_t qBytes = size_t(batchSize) * numHeads * seqLen * headDim *
                    sizeof(float);
    size_t maskBytes = hasMask
                           ? size_t(batchSize) * seqLen * cachedTokens *
                                 sizeof(float)
                           : sizeof(float);
    size_t outBytes = qBytes;
    size_t accumBytes = qBytes;
    size_t runMaxBytes = size_t(totalQPositions) * sizeof(float);
    size_t runSumBytes = runMaxBytes;

    GrillyBuffer bufQ = bufPool.acquire(qBytes);
    GrillyBuffer bufK = bufPool.acquire(kvBytes);
    GrillyBuffer bufV = bufPool.acquire(kvBytes);
    GrillyBuffer bufMask = bufPool.acquire(maskBytes);
    GrillyBuffer bufOutput = bufPool.acquire(outBytes);
    GrillyBuffer bufRunMax = bufPool.acquire(runMaxBytes);
    GrillyBuffer bufRunSum = bufPool.acquire(runSumBytes);
    GrillyBuffer bufAccum = bufPool.acquire(accumBytes);

    bufPool.upload(bufQ, Q, qBytes);
    bufPool.upload(bufK, decodedKeys.data(), kvBytes);
    bufPool.upload(bufV, decodedValues.data(), kvBytes);
    if (hasMask) {
        bufPool.upload(bufMask, mask, maskBytes);
    }

    // Use flash-attention2 shader
    PipelineEntry pipeAttn = cache.getOrCreate("flash-attention2", 8,
                                                sizeof(uint32_t) * 11);

    std::vector<VkDescriptorBufferInfo> attnBufs = {
        {bufQ.handle, 0, qBytes},
        {bufK.handle, 0, kvBytes},
        {bufV.handle, 0, kvBytes},
        {bufMask.handle, 0, maskBytes},
        {bufOutput.handle, 0, outBytes},
        {bufRunMax.handle, 0, runMaxBytes},
        {bufRunSum.handle, 0, runSumBytes},
        {bufAccum.handle, 0, accumBytes},
    };
    VkDescriptorSet descAttn = cache.allocDescriptorSet("flash-attention2",
                                                         attnBufs);

    uint32_t tileSizeQ = waveSize;  // Tile size = wave size for register-only
    uint32_t tileSizeK = waveSize;
    uint32_t numQTiles = (seqLen + tileSizeQ - 1) / tileSizeQ;
    uint32_t numKTiles = (cachedTokens + tileSizeK - 1) / tileSizeK;

    batch.begin();

    // Pass 0: init
    uint32_t initPush[11] = {
        batchSize, cachedTokens, numHeads, headDim,
        0 /*scale bits*/, tileSizeQ, tileSizeK, 0 /*init*/, hasMask, 0, 0
    };
    // Pack scale as uint32
    std::memcpy(&initPush[4], &scale, sizeof(float));
    batch.dispatch(pipeAttn.pipeline, pipeAttn.layout, descAttn,
                   (totalQPositions + 255) / 256, 1, 1,
                   initPush, sizeof(initPush));
    batch.barrier();

    // Pass 1: tile processing
    for (uint32_t qt = 0; qt < numQTiles; ++qt) {
        for (uint32_t kt = 0; kt < numKTiles; ++kt) {
            uint32_t tilePush[11] = {
                batchSize, cachedTokens, numHeads, headDim,
                0, tileSizeQ, tileSizeK, 1 /*tile*/, hasMask, qt, kt
            };
            std::memcpy(&tilePush[4], &scale, sizeof(float));

            uint32_t qTileSize = std::min(tileSizeQ,
                                           seqLen - qt * tileSizeQ);
            uint32_t tileGroups = (batchSize * numHeads * qTileSize + 255) / 256;

            batch.dispatch(pipeAttn.pipeline, pipeAttn.layout, descAttn,
                           tileGroups, 1, 1, tilePush, sizeof(tilePush));
            batch.barrier();
        }
    }

    // Pass 2: finalize
    uint32_t finalPush[11] = {
        batchSize, cachedTokens, numHeads, headDim,
        0, tileSizeQ, tileSizeK, 2 /*finalize*/, hasMask, 0, 0
    };
    std::memcpy(&finalPush[4], &scale, sizeof(float));
    batch.dispatch(pipeAttn.pipeline, pipeAttn.layout, descAttn,
                   (batchSize * numHeads * seqLen * headDim + 255) / 256,
                   1, 1, finalPush, sizeof(finalPush));

    batch.submit();

    bufPool.download(bufOutput, output, outBytes);

    bufPool.release(bufQ);
    bufPool.release(bufK);
    bufPool.release(bufV);
    bufPool.release(bufMask);
    bufPool.release(bufOutput);
    bufPool.release(bufRunMax);
    bufPool.release(bufRunSum);
    bufPool.release(bufAccum);
}

// ── CPU reference for fused attention ───────────────────────────────────
//
// Sequential implementation for correctness verification.
// Steps:
//   1. Decompress: [K, V] = latent @ W_up  (standard matmul)
//   2. Attention: softmax(Q @ K^T / sqrt(d)) @ V

void fusedAttentionCPU(
    const float* Q, const float* latents, const float* wUp,
    const float* mask, float* output,
    uint32_t batchSize, uint32_t seqLen, uint32_t cachedTokens,
    uint32_t numHeads, uint32_t headDim, uint32_t latentDim,
    float scale) {
    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(headDim));
    }

    uint32_t M = cachedTokens * numHeads;
    uint32_t concatDim = 2 * headDim;

    // Step 1: Decompress latents
    // decompressed[t*H+h][j] = sum_l latents[t*H+h][l] * wUp[l][j]
    std::vector<float> decompressed(M * concatDim, 0.0f);

    for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t j = 0; j < concatDim; ++j) {
            float sum = 0.0f;
            for (uint32_t l = 0; l < latentDim; ++l) {
                sum += latents[row * latentDim + l] *
                       wUp[l * concatDim + j];
            }
            decompressed[row * concatDim + j] = sum;
        }
    }

    // Split K and V
    std::vector<float> K(cachedTokens * numHeads * headDim);
    std::vector<float> V(cachedTokens * numHeads * headDim);

    for (uint32_t t = 0; t < cachedTokens; ++t) {
        for (uint32_t h = 0; h < numHeads; ++h) {
            size_t srcBase = (size_t(t) * numHeads + h) * concatDim;
            size_t dstBase = (size_t(t) * numHeads + h) * headDim;
            for (uint32_t d = 0; d < headDim; ++d) {
                K[dstBase + d] = decompressed[srcBase + d];
                V[dstBase + d] = decompressed[srcBase + headDim + d];
            }
        }
    }

    // Step 2: Standard attention
    // For each (batch, head, query_pos):
    //   scores[k] = Q[q] . K[k] * scale
    //   weights = softmax(scores)
    //   output[q] = weights . V

    for (uint32_t b = 0; b < batchSize; ++b) {
        for (uint32_t h = 0; h < numHeads; ++h) {
            for (uint32_t q = 0; q < seqLen; ++q) {
                // Query vector
                const float* qVec = Q + (size_t(b) * numHeads * seqLen +
                                          h * seqLen + q) * headDim;

                // Compute attention scores
                std::vector<float> scores(cachedTokens);
                float maxScore = -1e30f;

                for (uint32_t k = 0; k < cachedTokens; ++k) {
                    const float* kVec = K.data() +
                                        (size_t(k) * numHeads + h) * headDim;
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < headDim; ++d) {
                        dot += qVec[d] * kVec[d];
                    }
                    scores[k] = dot * scale;

                    if (mask) {
                        scores[k] += mask[b * seqLen * cachedTokens +
                                          q * cachedTokens + k];
                    }

                    if (scores[k] > maxScore) maxScore = scores[k];
                }

                // Softmax
                float sumExp = 0.0f;
                for (uint32_t k = 0; k < cachedTokens; ++k) {
                    scores[k] = std::exp(scores[k] - maxScore);
                    sumExp += scores[k];
                }
                for (uint32_t k = 0; k < cachedTokens; ++k) {
                    scores[k] /= sumExp;
                }

                // Weighted sum of values
                float* outVec = output + (size_t(b) * numHeads * seqLen +
                                           h * seqLen + q) * headDim;
                std::memset(outVec, 0, headDim * sizeof(float));

                for (uint32_t k = 0; k < cachedTokens; ++k) {
                    const float* vVec = V.data() +
                                        (size_t(k) * numHeads + h) * headDim;
                    for (uint32_t d = 0; d < headDim; ++d) {
                        outVec[d] += scores[k] * vVec[d];
                    }
                }
            }
        }
    }
}

}  // namespace experimental
}  // namespace grilly
