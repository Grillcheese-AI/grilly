#include "grilly/ops/kv_cache.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace grilly {
namespace ops {

// ── KV Cache Implementation ──────────────────────────────────────────────
//
// This is the most architecturally complex component in grilly-cpp. It
// combines 5 techniques into a unified KV cache system:
//
// Memory layout (per layer, or per layer-pair if cross-layer sharing):
//
//   ┌──────────────────────────────────────────────────────┐
//   │ MLA Latent Buffer                                    │
//   │  c_t[i] for each cached token i                     │
//   │  Shape: (maxTokens, numHeads, latentDim)            │
//   │  latentDim = headDim / compressionRatio             │
//   ├──────────────────────────────────────────────────────┤
//   │ Quantized Keys (FP8 E4M3)                           │
//   │  Only used when MLA is disabled and quant is on     │
//   │  Shape: (maxTokens, numHeads, headDim) in uint8     │
//   ├──────────────────────────────────────────────────────┤
//   │ Quantized Values (INT4)                             │
//   │  2 values packed per byte (4 bits each)             │
//   │  Shape: (maxTokens, numHeads, headDim/2) in uint8   │
//   ├──────────────────────────────────────────────────────┤
//   │ Scale Factors                                       │
//   │  Per-group scales for dequantization                │
//   │  Group size = Wave32 subgroup (32 elements)         │
//   │  Shape: (maxTokens, numHeads, headDim/32) float16   │
//   ├──────────────────────────────────────────────────────┤
//   │ Token Metadata (eviction)                           │
//   │  cumScore, predictedRelevance, tokenIdx, nextActive │
//   │  Shape: (maxTokens,) of TokenMeta                   │
//   ├──────────────────────────────────────────────────────┤
//   │ W_down / W_up projection matrices (MLA)             │
//   │  Static — loaded once at init                       │
//   └──────────────────────────────────────────────────────┘
//
// Wave32 Subgroup Quantization:
//   On AMD RDNA 2, a Wave32 consists of 32 threads executing in lockstep.
//   We quantize headDim elements in groups of 32, where each group shares
//   one scale factor. During dequantization, the scale is broadcast across
//   the subgroup using subgroupBroadcastFirst() — this means the scale
//   lives in a scalar register, not memory. Zero extra memory fetches.

KVCache createKVCache(BufferPool& pool, const KVCacheConfig& config) {
    KVCache kv;
    kv.config = config;
    kv.currentLen = 0;
    kv.evictionHead.initialized = false;
    kv.evictionHead.totalUpdates = 0;

    const uint32_t effectiveLayers = config.crossLayerSharing
        ? (config.numLayers + 1) / 2
        : config.numLayers;

    // MLA projection matrices
    if (config.compressionRatio > 1) {
        uint32_t latentDim = config.headDim / config.compressionRatio;
        // W_down: (numHeads, headDim, latentDim) per effective layer
        size_t wDownBytes = size_t(effectiveLayers) * config.numHeads *
                            config.headDim * latentDim * sizeof(float);
        // W_up: (numHeads, latentDim, 2*headDim) — decodes to both K and V
        size_t wUpBytes = size_t(effectiveLayers) * config.numHeads *
                          latentDim * 2 * config.headDim * sizeof(float);

        kv.wDown = pool.acquire(wDownBytes);
        kv.wUp = pool.acquire(wUpBytes);

        // Latent storage
        size_t latentBytes = size_t(config.maxCacheTokens) * config.numHeads *
                             latentDim * sizeof(float);
        kv.latents = pool.acquire(latentBytes);
    }

    // Quantized storage (when not using MLA, or as a secondary compression)
    if (config.useAsymmetricQuant) {
        // Keys: FP8 (1 byte per element)
        size_t keyBytes = size_t(config.maxCacheTokens) * config.numHeads *
                          config.headDim * sizeof(uint8_t);
        kv.keysQuant = pool.acquire(keyBytes);

        // Values: INT4 (packed, 2 per byte)
        uint32_t valBytesPerToken = (config.numHeads * config.headDim + 1) / 2;
        size_t valBytes = size_t(config.maxCacheTokens) * valBytesPerToken;
        kv.valuesQuant = pool.acquire(valBytes);

        // Scale factors: one per group of 32 (Wave32 subgroup size)
        uint32_t numGroups = (config.headDim + 31) / 32;
        size_t scaleBytes = size_t(config.maxCacheTokens) * config.numHeads *
                            numGroups * sizeof(uint16_t);  // float16
        kv.scaleFactors = pool.acquire(scaleBytes);
    }

    // Token metadata for eviction
    size_t metaBytes = size_t(config.maxCacheTokens) * sizeof(TokenMeta);
    kv.tokenMeta = pool.acquire(metaBytes);

    // Active list (linked list head pointers, one per head)
    size_t activeBytes = size_t(config.numHeads) * sizeof(uint32_t);
    kv.activeList = pool.acquire(activeBytes);

    // Initialize metadata to zero
    std::vector<uint8_t> zeros(metaBytes, 0);
    pool.upload(kv.tokenMeta, reinterpret_cast<const float*>(zeros.data()),
                metaBytes);

    return kv;
}

void destroyKVCache(BufferPool& pool, KVCache& kv) {
    if (kv.config.compressionRatio > 1) {
        pool.release(kv.wDown);
        pool.release(kv.wUp);
        pool.release(kv.latents);
    }
    if (kv.config.useAsymmetricQuant) {
        pool.release(kv.keysQuant);
        pool.release(kv.valuesQuant);
        pool.release(kv.scaleFactors);
    }
    kvCacheDestroyEvictionHead(pool, kv);
    pool.release(kv.tokenMeta);
    pool.release(kv.activeList);
    kv.currentLen = 0;
}

// ── KV Cache Append ──────────────────────────────────────────────────────
//
// Appending new tokens involves:
//
// 1. If MLA: compress [K; V] -> c_t via GPU matmul (W_down projection)
//    This is the bottleneck-breaker — instead of storing 2 * headDim floats
//    per token per head, we store headDim/compressionRatio floats.
//    For compression_ratio=4 and headDim=64: 16 floats instead of 128.
//
// 2. If asymmetric quant (without MLA): quantize K to FP8, V to INT4
//    FP8 E4M3: 4 exponent bits, 3 mantissa bits. Preserves the attention
//    distribution shape (which is what K controls) while saving 4x memory.
//    INT4: uniform quantization with per-group scales. V is robust to this
//    because attention-weighted averaging smooths out quantization noise.
//
// 3. Update token metadata (position, initial scores)
//
// 4. If cache is full: trigger eviction before appending

void kvCacheAppend(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                   KVCache& kvCache,
                   const float* newKeys, const float* newValues,
                   uint32_t numNewTokens) {
    const auto& cfg = kvCache.config;

    // Check if eviction needed
    if (kvCache.currentLen + numNewTokens > cfg.maxCacheTokens) {
        uint32_t numEvict = (kvCache.currentLen + numNewTokens) -
                            cfg.maxCacheTokens;

        if (cfg.useH2O) {
            // H2O eviction: remove lowest-scoring tokens
            kvCacheEvictH2O(batch, pool, cache, kvCache, nullptr, numEvict);
        }
        // Compact after eviction
        kvCacheCompact(batch, pool, cache, kvCache);
    }

    if (cfg.compressionRatio > 1) {
        // ── MLA compression: c_t = W_down @ [k_t; v_t] ──
        //
        // We use the linear op to do the projection. Input is the
        // concatenated [K; V] for new tokens, output is the compressed
        // latent c_t stored in the cache.

        uint32_t latentDim = cfg.headDim / cfg.compressionRatio;
        uint32_t concatDim = 2 * cfg.headDim;  // [K; V] concatenated

        // Concatenate K and V
        size_t concatBytes = size_t(numNewTokens) * cfg.numHeads *
                             concatDim * sizeof(float);
        GrillyBuffer bufConcat = pool.acquire(concatBytes);

        // Interleave K and V: for each token, [k_h0, v_h0, k_h1, v_h1, ...]
        std::vector<float> concat(numNewTokens * cfg.numHeads * concatDim);
        for (uint32_t t = 0; t < numNewTokens; ++t) {
            for (uint32_t h = 0; h < cfg.numHeads; ++h) {
                size_t srcK = (size_t(t) * cfg.numHeads + h) * cfg.headDim;
                size_t srcV = srcK;  // same layout for V
                size_t dst = (size_t(t) * cfg.numHeads + h) * concatDim;

                std::memcpy(&concat[dst], &newKeys[srcK],
                            cfg.headDim * sizeof(float));
                std::memcpy(&concat[dst + cfg.headDim], &newValues[srcV],
                            cfg.headDim * sizeof(float));
            }
        }
        pool.upload(bufConcat, concat.data(), concatBytes);

        // Project: compressed = concat @ W_down^T
        // W_down is (latentDim, concatDim) so the matmul gives (tokens, latentDim)
        size_t latentOutBytes = size_t(numNewTokens) * cfg.numHeads *
                                latentDim * sizeof(float);
        GrillyBuffer bufLatentOut = pool.acquire(latentOutBytes);

        // Use the GEMM shader for the projection
        uint32_t M_proj = numNewTokens * cfg.numHeads;
        uint32_t K_proj = concatDim;
        uint32_t N_proj = latentDim;

        PipelineEntry pipeGemm = cache.getOrCreate("gemm_mnk", 3,
                                                    sizeof(uint32_t) * 3);

        std::vector<VkDescriptorBufferInfo> gemmBufs = {
            {bufConcat.handle,    0, concatBytes},
            {kvCache.wDown.handle, 0, size_t(K_proj) * N_proj * sizeof(float)},
            {bufLatentOut.handle, 0, latentOutBytes},
        };
        VkDescriptorSet descGemm = cache.allocDescriptorSet("gemm_mnk",
                                                             gemmBufs);

        uint32_t gemmPush[3] = {M_proj, K_proj, N_proj};
        uint32_t gemmGX = (N_proj + 15) / 16;
        uint32_t gemmGY = (M_proj + 15) / 16;

        batch.begin();
        batch.dispatch(pipeGemm.pipeline, pipeGemm.layout, descGemm,
                       gemmGX, gemmGY, 1, gemmPush, sizeof(gemmPush));
        batch.submit();

        // Copy compressed latents into cache at currentLen offset
        std::vector<float> latentData(numNewTokens * cfg.numHeads * latentDim);
        pool.download(bufLatentOut,latentData.data(), latentOutBytes);

        // Upload to the latent cache at the correct offset
        // (We upload to the mapped pointer with an offset)
        size_t offset = size_t(kvCache.currentLen) * cfg.numHeads *
                        latentDim * sizeof(float);
        auto* dst = static_cast<uint8_t*>(kvCache.latents.mappedPtr) + offset;
        std::memcpy(dst, latentData.data(), latentOutBytes);

        pool.release(bufConcat);
        pool.release(bufLatentOut);

    } else if (cfg.useAsymmetricQuant) {
        // ── Asymmetric quantization: FP8 keys + INT4 values ──
        //
        // FP8 E4M3 encoding for keys:
        //   sign(1) | exponent(4) | mantissa(3)
        //   Range: [-448, 448], precision: ~0.0625 for values near 1.0
        //   We compute per-group (32 elements) scale factors that map the
        //   float range to E4M3 range, then encode.
        //
        // INT4 encoding for values:
        //   4 bits per value = 16 levels, stored 2-per-byte
        //   Per-group (32 elements) scale + zero-point
        //   Dequant: val = (int4_val - 8) * scale
        //
        // Wave32 subgroup trick:
        //   The group size of 32 matches AMD RDNA 2's Wave32 wavefront.
        //   During shader dequantization, each thread in the wave loads
        //   one quantized value. The scale factor is loaded by thread 0
        //   and broadcast via subgroupBroadcastFirst(). This means the
        //   scale lives in a SGPR (scalar register), not VGPR or memory.
        //   Zero extra memory bandwidth for dequantization.

        const uint32_t groupSize = 32;  // Wave32
        const uint32_t numGroups = (cfg.headDim + groupSize - 1) / groupSize;

        // Quantize keys to FP8
        std::vector<uint8_t> keysQ(numNewTokens * cfg.numHeads * cfg.headDim);
        std::vector<uint16_t> keyScales(numNewTokens * cfg.numHeads * numGroups);

        for (uint32_t t = 0; t < numNewTokens; ++t) {
            for (uint32_t h = 0; h < cfg.numHeads; ++h) {
                size_t base = (size_t(t) * cfg.numHeads + h) * cfg.headDim;
                for (uint32_t g = 0; g < numGroups; ++g) {
                    // Find max absolute value in this group
                    float maxAbs = 0.0f;
                    uint32_t groupStart = g * groupSize;
                    uint32_t groupEnd = std::min(groupStart + groupSize,
                                                  cfg.headDim);
                    for (uint32_t i = groupStart; i < groupEnd; ++i) {
                        float val = std::abs(newKeys[base + i]);
                        if (val > maxAbs) maxAbs = val;
                    }

                    // Scale to FP8 range: max representable is ~448
                    float scale = (maxAbs > 0.0f) ? 448.0f / maxAbs : 1.0f;
                    size_t scaleIdx = (size_t(t) * cfg.numHeads + h) *
                                     numGroups + g;
                    // Store scale as float16 (just truncate for now)
                    uint32_t scaleBits;
                    std::memcpy(&scaleBits, &scale, sizeof(float));
                    keyScales[scaleIdx] = static_cast<uint16_t>(
                        ((scaleBits >> 16) & 0x8000) |     // sign
                        (((scaleBits >> 23) - 112) << 10) | // exponent
                        ((scaleBits >> 13) & 0x03FF));      // mantissa

                    // Quantize each element
                    for (uint32_t i = groupStart; i < groupEnd; ++i) {
                        float scaled = newKeys[base + i] * scale;
                        // Clamp to E4M3 range and round
                        scaled = std::max(-448.0f, std::min(448.0f, scaled));
                        // Simple E4M3 encoding (truncate mantissa)
                        int8_t sign = (scaled < 0) ? 1 : 0;
                        float absVal = std::abs(scaled);
                        uint8_t encoded;
                        if (absVal < 1.0f / 512.0f) {
                            encoded = 0;  // denormal/zero
                        } else {
                            int exp = static_cast<int>(std::log2(absVal));
                            exp = std::max(0, std::min(15, exp + 7));
                            float mantissa = absVal / std::pow(2.0f, exp - 7) -
                                             1.0f;
                            uint8_t mant = static_cast<uint8_t>(mantissa * 8) &
                                           0x07;
                            encoded = static_cast<uint8_t>(
                                (sign << 7) | (exp << 3) | mant);
                        }
                        keysQ[base + i] = encoded;
                    }
                }
            }
        }

        // Quantize values to INT4
        uint32_t valBytesPerToken = (cfg.numHeads * cfg.headDim + 1) / 2;
        std::vector<uint8_t> valsQ(numNewTokens * valBytesPerToken);
        std::vector<uint16_t> valScales(numNewTokens * cfg.numHeads * numGroups);

        for (uint32_t t = 0; t < numNewTokens; ++t) {
            for (uint32_t h = 0; h < cfg.numHeads; ++h) {
                size_t base = (size_t(t) * cfg.numHeads + h) * cfg.headDim;
                for (uint32_t g = 0; g < numGroups; ++g) {
                    float maxAbs = 0.0f;
                    uint32_t groupStart = g * groupSize;
                    uint32_t groupEnd = std::min(groupStart + groupSize,
                                                  cfg.headDim);
                    for (uint32_t i = groupStart; i < groupEnd; ++i) {
                        float val = std::abs(newValues[base + i]);
                        if (val > maxAbs) maxAbs = val;
                    }

                    float scale = (maxAbs > 0.0f) ? 7.0f / maxAbs : 1.0f;
                    size_t scaleIdx = (size_t(t) * cfg.numHeads + h) *
                                     numGroups + g;
                    uint32_t scaleBits;
                    std::memcpy(&scaleBits, &scale, sizeof(float));
                    valScales[scaleIdx] = static_cast<uint16_t>(
                        ((scaleBits >> 16) & 0x8000) |
                        (((scaleBits >> 23) - 112) << 10) |
                        ((scaleBits >> 13) & 0x03FF));

                    for (uint32_t i = groupStart; i < groupEnd; ++i) {
                        float scaled = newValues[base + i] * scale;
                        // INT4: range [-8, 7], store as unsigned [0, 15]
                        int val4 = static_cast<int>(std::round(scaled)) + 8;
                        val4 = std::max(0, std::min(15, val4));

                        size_t flatIdx = (size_t(t) * cfg.numHeads + h) *
                                         cfg.headDim + i;
                        size_t byteIdx = flatIdx / 2;
                        if (flatIdx % 2 == 0) {
                            valsQ[byteIdx] = static_cast<uint8_t>(val4 & 0x0F);
                        } else {
                            valsQ[byteIdx] |= static_cast<uint8_t>(
                                (val4 & 0x0F) << 4);
                        }
                    }
                }
            }
        }

        // Upload quantized data to cache at offset
        size_t keyOffset = size_t(kvCache.currentLen) * cfg.numHeads *
                           cfg.headDim;
        auto* keyDst = static_cast<uint8_t*>(kvCache.keysQuant.mappedPtr) +
                       keyOffset;
        std::memcpy(keyDst, keysQ.data(), keysQ.size());

        size_t valOffset = size_t(kvCache.currentLen) * valBytesPerToken;
        auto* valDst = static_cast<uint8_t*>(kvCache.valuesQuant.mappedPtr) +
                       valOffset;
        std::memcpy(valDst, valsQ.data(), valsQ.size());

        // Upload scale factors
        size_t scaleOffset = size_t(kvCache.currentLen) * cfg.numHeads *
                             numGroups * sizeof(uint16_t);
        auto* scaleDst = static_cast<uint8_t*>(
            kvCache.scaleFactors.mappedPtr) + scaleOffset;
        // Interleave key and value scales
        std::memcpy(scaleDst, keyScales.data(),
                    keyScales.size() * sizeof(uint16_t));
    }

    // Update token metadata
    for (uint32_t t = 0; t < numNewTokens; ++t) {
        TokenMeta meta;
        meta.cumulativeScore = 0.0f;
        meta.predictedRelevance = 1.0f;  // new tokens start with full relevance
        meta.tokenIdx = kvCache.currentLen + t;
        meta.nextActive = (t + 1 < numNewTokens)
            ? kvCache.currentLen + t + 1
            : UINT32_MAX;  // end of list

        size_t metaOffset = (kvCache.currentLen + t) * sizeof(TokenMeta);
        auto* metaDst = static_cast<uint8_t*>(kvCache.tokenMeta.mappedPtr) +
                        metaOffset;
        std::memcpy(metaDst, &meta, sizeof(TokenMeta));
    }

    kvCache.currentLen += numNewTokens;
}

// ── KV Cache Decode ──────────────────────────────────────────────────────
//
// Reconstruct full-precision K, V from the compressed cache.
//
// If MLA: [K, V] = W_up @ c_t
//   The decompression is a GPU matmul — the latent c_t is multiplied by
//   W_up to recover the full K and V. This happens on-the-fly during
//   attention computation, so the large K/V never need to exist in memory
//   all at once. In practice, we decode tile-by-tile.
//
// If quantized: dequantize using stored scale factors
//   On GPU, this would use subgroupBroadcastFirst() for zero-cost scale
//   distribution. Here we provide the CPU reference path.

void kvCacheDecode(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                   const KVCache& kvCache,
                   float* decodedKeys, float* decodedValues) {
    const auto& cfg = kvCache.config;
    const uint32_t cachedTokens = kvCache.currentLen;

    if (cfg.compressionRatio > 1) {
        // MLA decode: [K, V] = W_up @ c_t
        uint32_t latentDim = cfg.headDim / cfg.compressionRatio;
        uint32_t concatDim = 2 * cfg.headDim;

        uint32_t M_proj = cachedTokens * cfg.numHeads;
        uint32_t K_proj = latentDim;
        uint32_t N_proj = concatDim;

        size_t latentBytes = size_t(M_proj) * K_proj * sizeof(float);
        size_t outBytes = size_t(M_proj) * N_proj * sizeof(float);

        GrillyBuffer bufLatent = pool.acquire(latentBytes);
        GrillyBuffer bufDecoded = pool.acquire(outBytes);

        // Copy latents from cache
        pool.upload(bufLatent,
                    static_cast<const float*>(kvCache.latents.mappedPtr),
                    latentBytes);

        PipelineEntry pipeGemm = cache.getOrCreate("gemm_mnk", 3,
                                                    sizeof(uint32_t) * 3);

        std::vector<VkDescriptorBufferInfo> gemmBufs = {
            {bufLatent.handle,     0, latentBytes},
            {kvCache.wUp.handle,   0, size_t(K_proj) * N_proj * sizeof(float)},
            {bufDecoded.handle,    0, outBytes},
        };
        VkDescriptorSet descGemm = cache.allocDescriptorSet("gemm_mnk",
                                                             gemmBufs);

        uint32_t gemmPush[3] = {M_proj, K_proj, N_proj};

        batch.begin();
        batch.dispatch(pipeGemm.pipeline, pipeGemm.layout, descGemm,
                       (N_proj + 15) / 16, (M_proj + 15) / 16, 1,
                       gemmPush, sizeof(gemmPush));
        batch.submit();

        // Download and split into K and V
        std::vector<float> decoded(M_proj * N_proj);
        pool.download(bufDecoded, decoded.data(), outBytes);

        for (uint32_t t = 0; t < cachedTokens; ++t) {
            for (uint32_t h = 0; h < cfg.numHeads; ++h) {
                size_t srcBase = (size_t(t) * cfg.numHeads + h) * concatDim;
                size_t dstBase = (size_t(t) * cfg.numHeads + h) * cfg.headDim;

                std::memcpy(&decodedKeys[dstBase], &decoded[srcBase],
                            cfg.headDim * sizeof(float));
                std::memcpy(&decodedValues[dstBase],
                            &decoded[srcBase + cfg.headDim],
                            cfg.headDim * sizeof(float));
            }
        }

        pool.release(bufLatent);
        pool.release(bufDecoded);

    } else if (cfg.useAsymmetricQuant) {
        // Dequantize from FP8/INT4 (CPU reference path)
        const uint32_t groupSize = 32;
        const uint32_t numGroups = (cfg.headDim + groupSize - 1) / groupSize;

        auto* keysData = static_cast<const uint8_t*>(
            kvCache.keysQuant.mappedPtr);
        auto* valsData = static_cast<const uint8_t*>(
            kvCache.valuesQuant.mappedPtr);
        auto* scalesData = static_cast<const uint16_t*>(
            kvCache.scaleFactors.mappedPtr);

        for (uint32_t t = 0; t < cachedTokens; ++t) {
            for (uint32_t h = 0; h < cfg.numHeads; ++h) {
                size_t base = (size_t(t) * cfg.numHeads + h) * cfg.headDim;

                for (uint32_t g = 0; g < numGroups; ++g) {
                    // Decode scale factor from float16
                    size_t scaleIdx = (size_t(t) * cfg.numHeads + h) *
                                     numGroups + g;
                    uint16_t fp16 = scalesData[scaleIdx];
                    // Float16 to float32 conversion
                    uint32_t sign = (fp16 >> 15) & 1;
                    uint32_t exp16 = (fp16 >> 10) & 0x1F;
                    uint32_t mant16 = fp16 & 0x03FF;
                    float scale;
                    if (exp16 == 0) {
                        scale = std::ldexp(static_cast<float>(mant16),
                                           -24);
                    } else {
                        uint32_t fp32 = (sign << 31) |
                                        ((exp16 + 112) << 23) |
                                        (mant16 << 13);
                        std::memcpy(&scale, &fp32, sizeof(float));
                    }
                    float invScale = (scale > 0.0f) ? 1.0f / scale : 1.0f;

                    uint32_t groupStart = g * groupSize;
                    uint32_t groupEnd = std::min(groupStart + groupSize,
                                                  cfg.headDim);

                    // Dequantize keys (FP8 E4M3)
                    for (uint32_t i = groupStart; i < groupEnd; ++i) {
                        uint8_t encoded = keysData[base + i];
                        int8_t s = (encoded >> 7) & 1;
                        uint8_t exp8 = (encoded >> 3) & 0x0F;
                        uint8_t mant = encoded & 0x07;

                        float val;
                        if (exp8 == 0) {
                            val = std::ldexp(static_cast<float>(mant), -9);
                        } else {
                            val = std::ldexp(1.0f + mant / 8.0f, exp8 - 7);
                        }
                        if (s) val = -val;

                        decodedKeys[base + i] = val * invScale;
                    }

                    // Dequantize values (INT4)
                    for (uint32_t i = groupStart; i < groupEnd; ++i) {
                        size_t flatIdx = base + i;
                        size_t byteIdx = flatIdx / 2;
                        uint8_t packed = valsData[byteIdx];
                        int val4;
                        if (flatIdx % 2 == 0) {
                            val4 = packed & 0x0F;
                        } else {
                            val4 = (packed >> 4) & 0x0F;
                        }
                        // Undo offset: stored as [0,15], real is [-8,7]
                        float val = static_cast<float>(val4 - 8) * invScale;
                        decodedValues[base + i] = val;
                    }
                }
            }
        }
    }
}

// ── H2O (Heavy Hitter Oracle) Eviction ──────────────────────────────────
//
// Heavy Hitter Oracle (Zhang et al., 2023) observation: in transformer
// attention, a small subset of tokens consistently receive high attention
// scores across layers and heads. These "heavy hitters" are critical for
// output quality. All other tokens can be safely evicted.
//
// Algorithm:
//   1. After each attention forward pass, accumulate the attention scores
//      each cached token received: cumScore[t] += sum_over_queries(attn[q,t])
//   2. When eviction is needed, sort by cumulative score and evict the
//      bottom-k tokens.
//   3. Always keep the most recent tokens (they haven't had a chance to
//      accumulate scores yet) — configurable "protection window".

void kvCacheEvictH2O(CommandBatch& batch, BufferPool& pool,
                     PipelineCache& cache, KVCache& kvCache,
                     const float* attentionScores, uint32_t numEvict) {
    if (numEvict == 0) {
        numEvict = kvCache.currentLen / 4;  // default: evict 25%
    }
    if (numEvict >= kvCache.currentLen) return;

    // Update cumulative scores if attention scores provided
    if (attentionScores != nullptr) {
        auto* metaPtr = static_cast<TokenMeta*>(kvCache.tokenMeta.mappedPtr);
        for (uint32_t t = 0; t < kvCache.currentLen; ++t) {
            // Sum attention received by token t across all query positions
            float totalAttn = 0.0f;
            for (uint32_t q = 0; q < kvCache.currentLen; ++q) {
                totalAttn += attentionScores[q * kvCache.currentLen + t];
            }
            metaPtr[t].cumulativeScore += totalAttn;
        }
    }

    // Find tokens to evict (lowest cumulative scores)
    auto* metaPtr = static_cast<TokenMeta*>(kvCache.tokenMeta.mappedPtr);

    // Build sorted index by cumulative score
    std::vector<uint32_t> indices(kvCache.currentLen);
    for (uint32_t i = 0; i < kvCache.currentLen; ++i) indices[i] = i;

    std::partial_sort(indices.begin(), indices.begin() + numEvict,
                      indices.end(),
                      [metaPtr](uint32_t a, uint32_t b) {
                          return metaPtr[a].cumulativeScore <
                                 metaPtr[b].cumulativeScore;
                      });

    // Mark evicted tokens by setting nextActive = UINT32_MAX - 1 (tombstone)
    for (uint32_t i = 0; i < numEvict; ++i) {
        metaPtr[indices[i]].nextActive = UINT32_MAX - 1;  // tombstone
    }
}

// ── Trainable Eviction Head ──────────────────────────────────────────────
//
// The eviction head is a tiny 2-layer MLP:
//
//   hidden = ReLU(W1 @ features + b1)     W1: (inputDim, hiddenDim)
//   score  = sigmoid(W2 @ hidden + b2)    W2: (hiddenDim, 1)
//
// It predicts the probability that a token will receive high attention in
// future forward passes. This replaces the hand-tuned heuristic (0.7 *
// normalized_score + 0.3 * recency_bonus) with a learned predictor.
//
// Training uses online SGD with binary cross-entropy loss:
//   label[t] = 1 if attention_received[t] > median, 0 otherwise
//   loss = -label * log(score) - (1-label) * log(1-score)
//
// Xavier initialization: W ~ U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))

void kvCacheInitEvictionHead(BufferPool& pool, KVCache& kvCache,
                              uint32_t inputDim, uint32_t hiddenDim,
                              float lr) {
    auto& head = kvCache.evictionHead;
    head.inputDim = inputDim;
    head.hiddenDim = hiddenDim;
    head.learningRate = lr;
    head.totalUpdates = 0;
    head.initialized = true;

    // Allocate weight and bias buffers
    size_t w1Bytes = size_t(inputDim) * hiddenDim * sizeof(float);
    size_t b1Bytes = size_t(hiddenDim) * sizeof(float);
    size_t w2Bytes = size_t(hiddenDim) * sizeof(float);  // hiddenDim × 1
    size_t b2Bytes = sizeof(float);

    head.w1 = pool.acquire(w1Bytes);
    head.b1 = pool.acquire(b1Bytes);
    head.w2 = pool.acquire(w2Bytes);
    head.b2 = pool.acquire(b2Bytes);

    // Gradient buffers (same sizes)
    head.gradW1 = pool.acquire(w1Bytes);
    head.gradB1 = pool.acquire(b1Bytes);
    head.gradW2 = pool.acquire(w2Bytes);
    head.gradB2 = pool.acquire(b2Bytes);

    // Xavier initialization for W1
    float xavierW1 = std::sqrt(6.0f / (inputDim + hiddenDim));
    std::vector<float> w1Init(inputDim * hiddenDim);
    for (auto& v : w1Init) {
        v = (static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f) * xavierW1;
    }
    pool.upload(head.w1, w1Init.data(), w1Bytes);

    // Xavier initialization for W2
    float xavierW2 = std::sqrt(6.0f / (hiddenDim + 1));
    std::vector<float> w2Init(hiddenDim);
    for (auto& v : w2Init) {
        v = (static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f) * xavierW2;
    }
    pool.upload(head.w2, w2Init.data(), w2Bytes);

    // Zero-initialize biases
    std::vector<float> zerosB1(hiddenDim, 0.0f);
    pool.upload(head.b1, zerosB1.data(), b1Bytes);
    float zeroB2 = 0.0f;
    pool.upload(head.b2, &zeroB2, b2Bytes);
}

void kvCacheDestroyEvictionHead(BufferPool& pool, KVCache& kvCache) {
    auto& head = kvCache.evictionHead;
    if (!head.initialized) return;

    pool.release(head.w1);
    pool.release(head.b1);
    pool.release(head.w2);
    pool.release(head.b2);
    pool.release(head.gradW1);
    pool.release(head.gradB1);
    pool.release(head.gradW2);
    pool.release(head.gradB2);
    head.initialized = false;
}

// ── Eviction Head Training ──────────────────────────────────────────────
//
// Forward pass (CPU reference — GPU dispatch would use linear op):
//   1. hidden[h] = sum_i(W1[i][h] * features[i]) + b1[h]
//   2. hidden[h] = max(0, hidden[h])   // ReLU
//   3. logit = sum_h(W2[h] * hidden[h]) + b2
//   4. score = sigmoid(logit)           // [0, 1]
//
// Backward pass (SGD update):
//   dL/dlogit = score - label           // BCE gradient
//   dL/dW2[h] = dL/dlogit * hidden[h]
//   dL/db2 = dL/dlogit
//   dL/dhidden[h] = dL/dlogit * W2[h] * (hidden[h] > 0 ? 1 : 0)  // ReLU grad
//   dL/dW1[i][h] = dL/dhidden[h] * features[i]
//   dL/db1[h] = dL/dhidden[h]

void kvCacheTrainEvictionHead(CommandBatch& /*batch*/, BufferPool& pool,
                               PipelineCache& /*cache*/, KVCache& kvCache,
                               const float* tokenFeatures,
                               const float* attentionScores,
                               uint32_t seqLen) {
    auto& head = kvCache.evictionHead;
    if (!head.initialized) return;

    const uint32_t numTokens = kvCache.currentLen;
    if (numTokens == 0) return;

    const uint32_t inDim = head.inputDim;
    const uint32_t hDim = head.hiddenDim;

    // Read current weights from mapped pointers
    auto* w1 = static_cast<float*>(head.w1.mappedPtr);
    auto* b1 = static_cast<float*>(head.b1.mappedPtr);
    auto* w2 = static_cast<float*>(head.w2.mappedPtr);
    auto* b2 = static_cast<float*>(head.b2.mappedPtr);

    // Step 1: Compute ground truth labels from attention scores
    // Sum attention each token received across all query positions
    std::vector<float> attnReceived(numTokens, 0.0f);
    for (uint32_t q = 0; q < seqLen; ++q) {
        for (uint32_t t = 0; t < numTokens; ++t) {
            attnReceived[t] += attentionScores[q * numTokens + t];
        }
    }

    // Find median attention to create binary labels
    std::vector<float> sorted = attnReceived;
    std::nth_element(sorted.begin(), sorted.begin() + numTokens / 2,
                     sorted.end());
    float median = sorted[numTokens / 2];

    std::vector<float> labels(numTokens);
    for (uint32_t t = 0; t < numTokens; ++t) {
        labels[t] = (attnReceived[t] > median) ? 1.0f : 0.0f;
    }

    // Step 2: Forward + Backward pass for each token (mini-batch SGD)
    // Accumulate gradients across all tokens, then apply update.
    std::vector<float> gradW1Acc(inDim * hDim, 0.0f);
    std::vector<float> gradB1Acc(hDim, 0.0f);
    std::vector<float> gradW2Acc(hDim, 0.0f);
    float gradB2Acc = 0.0f;

    auto* metaPtr = static_cast<TokenMeta*>(kvCache.tokenMeta.mappedPtr);

    for (uint32_t t = 0; t < numTokens; ++t) {
        if (metaPtr[t].nextActive == UINT32_MAX - 1) continue;

        const float* feat = tokenFeatures + size_t(t) * inDim;

        // Forward: hidden = ReLU(W1^T @ feat + b1)
        std::vector<float> hidden(hDim);
        for (uint32_t h = 0; h < hDim; ++h) {
            float sum = b1[h];
            for (uint32_t i = 0; i < inDim; ++i) {
                sum += w1[i * hDim + h] * feat[i];
            }
            hidden[h] = std::max(0.0f, sum);  // ReLU
        }

        // Forward: logit = W2^T @ hidden + b2, score = sigmoid(logit)
        float logit = *b2;
        for (uint32_t h = 0; h < hDim; ++h) {
            logit += w2[h] * hidden[h];
        }
        float score = 1.0f / (1.0f + std::exp(-logit));  // sigmoid

        // Store prediction in metadata for eviction decisions
        metaPtr[t].predictedRelevance = score;

        // Backward: BCE gradient
        float dLogit = score - labels[t];

        // Grad W2, b2
        for (uint32_t h = 0; h < hDim; ++h) {
            gradW2Acc[h] += dLogit * hidden[h];
        }
        gradB2Acc += dLogit;

        // Grad hidden (through ReLU)
        std::vector<float> dHidden(hDim);
        for (uint32_t h = 0; h < hDim; ++h) {
            dHidden[h] = dLogit * w2[h] * (hidden[h] > 0.0f ? 1.0f : 0.0f);
        }

        // Grad W1, b1
        for (uint32_t h = 0; h < hDim; ++h) {
            gradB1Acc[h] += dHidden[h];
            for (uint32_t i = 0; i < inDim; ++i) {
                gradW1Acc[i * hDim + h] += dHidden[h] * feat[i];
            }
        }
    }

    // Step 3: SGD update (scale by 1/numTokens for mean gradient)
    float scale = head.learningRate / static_cast<float>(numTokens);

    for (uint32_t i = 0; i < inDim * hDim; ++i) {
        w1[i] -= scale * gradW1Acc[i];
    }
    for (uint32_t h = 0; h < hDim; ++h) {
        b1[h] -= scale * gradB1Acc[h];
        w2[h] -= scale * gradW2Acc[h];
    }
    *b2 -= scale * gradB2Acc;

    head.totalUpdates++;
}

// ── Speculative Eviction via Trained Auxiliary Head ──────────────────────
//
// Uses the trained eviction head to predict which tokens will become
// irrelevant. If the head hasn't been initialized or trained yet, falls
// back to the heuristic (normalized score + recency bonus).
//
// After prediction, tokens with predicted relevance below the threshold
// are tombstoned for later compaction.

void kvCacheEvictSpeculative(CommandBatch& batch, BufferPool& pool,
                             PipelineCache& cache, KVCache& kvCache,
                             const float* hiddenStates, uint32_t hiddenDim) {
    const auto& cfg = kvCache.config;
    auto* metaPtr = static_cast<TokenMeta*>(kvCache.tokenMeta.mappedPtr);

    // If the eviction head is trained, use it for prediction
    if (kvCache.evictionHead.initialized &&
        kvCache.evictionHead.totalUpdates > 0 &&
        hiddenStates != nullptr) {

        const auto& head = kvCache.evictionHead;
        auto* w1 = static_cast<const float*>(head.w1.mappedPtr);
        auto* b1 = static_cast<const float*>(head.b1.mappedPtr);
        auto* w2 = static_cast<const float*>(head.w2.mappedPtr);
        auto* b2 = static_cast<const float*>(head.b2.mappedPtr);
        const uint32_t inDim = head.inputDim;
        const uint32_t hDim = head.hiddenDim;

        for (uint32_t t = 0; t < kvCache.currentLen; ++t) {
            if (metaPtr[t].nextActive == UINT32_MAX - 1) continue;

            const float* feat = hiddenStates + size_t(t) * inDim;

            // Forward: hidden = ReLU(W1^T @ feat + b1)
            std::vector<float> hidden(hDim);
            for (uint32_t h = 0; h < hDim; ++h) {
                float sum = b1[h];
                for (uint32_t i = 0; i < inDim; ++i) {
                    sum += w1[i * hDim + h] * feat[i];
                }
                hidden[h] = std::max(0.0f, sum);
            }

            // Forward: score = sigmoid(W2^T @ hidden + b2)
            float logit = *b2;
            for (uint32_t h = 0; h < hDim; ++h) {
                logit += w2[h] * hidden[h];
            }
            float score = 1.0f / (1.0f + std::exp(-logit));

            metaPtr[t].predictedRelevance = score;

            if (score < cfg.evictionThreshold) {
                metaPtr[t].nextActive = UINT32_MAX - 1;  // tombstone
            }
        }
        return;
    }

    // Fallback: heuristic when head isn't trained yet
    float maxScore = 0.0f;
    for (uint32_t t = 0; t < kvCache.currentLen; ++t) {
        if (metaPtr[t].cumulativeScore > maxScore) {
            maxScore = metaPtr[t].cumulativeScore;
        }
    }

    float invMax = (maxScore > 0.0f) ? 1.0f / maxScore : 1.0f;

    for (uint32_t t = 0; t < kvCache.currentLen; ++t) {
        if (metaPtr[t].nextActive == UINT32_MAX - 1) continue;

        float normalizedScore = metaPtr[t].cumulativeScore * invMax;
        float recencyBonus = static_cast<float>(t) /
                             static_cast<float>(kvCache.currentLen);
        metaPtr[t].predictedRelevance = 0.7f * normalizedScore +
                                         0.3f * recencyBonus;

        if (metaPtr[t].predictedRelevance < cfg.evictionThreshold) {
            metaPtr[t].nextActive = UINT32_MAX - 1;
        }
    }
}

// ── Cache Compaction ─────────────────────────────────────────────────────
//
// After eviction, the cache has "holes" (tombstoned entries). Compaction
// moves all active tokens to contiguous positions, updating the linked
// list and reducing cache length.
//
// In a GPU shader, this would be a parallel prefix-sum (scan) to compute
// new positions, followed by a scatter. Here we do it on CPU since the
// metadata is small and persistently mapped.

void kvCacheCompact(CommandBatch& batch, BufferPool& pool,
                    PipelineCache& cache, KVCache& kvCache) {
    const auto& cfg = kvCache.config;
    auto* metaPtr = static_cast<TokenMeta*>(kvCache.tokenMeta.mappedPtr);

    // Collect active token indices
    std::vector<uint32_t> activeIndices;
    for (uint32_t t = 0; t < kvCache.currentLen; ++t) {
        if (metaPtr[t].nextActive != UINT32_MAX - 1) {
            activeIndices.push_back(t);
        }
    }

    if (activeIndices.size() == kvCache.currentLen) return;  // nothing to compact

    // Move active tokens to front
    for (uint32_t newIdx = 0; newIdx < activeIndices.size(); ++newIdx) {
        uint32_t oldIdx = activeIndices[newIdx];
        if (newIdx == oldIdx) continue;

        // Move metadata
        metaPtr[newIdx] = metaPtr[oldIdx];
        metaPtr[newIdx].nextActive = (newIdx + 1 < activeIndices.size())
            ? newIdx + 1
            : UINT32_MAX;

        // Move compressed data (latents or quantized)
        if (cfg.compressionRatio > 1) {
            uint32_t latentDim = cfg.headDim / cfg.compressionRatio;
            size_t entryBytes = size_t(cfg.numHeads) * latentDim * sizeof(float);
            auto* base = static_cast<uint8_t*>(kvCache.latents.mappedPtr);
            std::memmove(base + newIdx * entryBytes,
                         base + oldIdx * entryBytes, entryBytes);
        }

        if (cfg.useAsymmetricQuant) {
            // Move quantized keys
            size_t keyEntry = size_t(cfg.numHeads) * cfg.headDim;
            auto* keyBase = static_cast<uint8_t*>(kvCache.keysQuant.mappedPtr);
            std::memmove(keyBase + newIdx * keyEntry,
                         keyBase + oldIdx * keyEntry, keyEntry);

            // Move quantized values
            uint32_t valEntry = (cfg.numHeads * cfg.headDim + 1) / 2;
            auto* valBase = static_cast<uint8_t*>(kvCache.valuesQuant.mappedPtr);
            std::memmove(valBase + newIdx * valEntry,
                         valBase + oldIdx * valEntry, valEntry);
        }
    }

    kvCache.currentLen = static_cast<uint32_t>(activeIndices.size());
}

// ── Cache Statistics ────────────────────────────────────────────────────

KVCacheStats kvCacheGetStats(const KVCache& kvCache) {
    const auto& cfg = kvCache.config;
    auto* metaPtr = static_cast<const TokenMeta*>(kvCache.tokenMeta.mappedPtr);

    KVCacheStats stats{};
    stats.currentTokens = kvCache.currentLen;
    stats.maxTokens = cfg.maxCacheTokens;

    // Count active tokens and average score
    float totalScore = 0.0f;
    uint32_t activeCount = 0;
    for (uint32_t t = 0; t < kvCache.currentLen; ++t) {
        if (metaPtr[t].nextActive != UINT32_MAX - 1) {
            totalScore += metaPtr[t].cumulativeScore;
            activeCount++;
        }
    }
    stats.avgAttentionScore = (activeCount > 0)
        ? totalScore / activeCount
        : 0.0f;

    // Compression ratio
    float uncompressedBytes = static_cast<float>(
        cfg.maxCacheTokens * cfg.numHeads * cfg.headDim * 2 * sizeof(float));
    float compressedBytes;
    if (cfg.compressionRatio > 1) {
        uint32_t latentDim = cfg.headDim / cfg.compressionRatio;
        compressedBytes = static_cast<float>(
            cfg.maxCacheTokens * cfg.numHeads * latentDim * sizeof(float));
    } else if (cfg.useAsymmetricQuant) {
        // FP8 keys (1 byte) + INT4 values (0.5 bytes) = 1.5 bytes per element
        compressedBytes = static_cast<float>(
            cfg.maxCacheTokens * cfg.numHeads * cfg.headDim) * 1.5f;
    } else {
        compressedBytes = uncompressedBytes;
    }
    stats.compressionRatio = uncompressedBytes / compressedBytes;

    return stats;
}

}  // namespace ops
}  // namespace grilly
