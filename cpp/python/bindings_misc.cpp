/// bindings_misc.cpp — Miscellaneous op bindings (dropout, embedding,
/// KV cache, swizzle, fused attention, CubeMind/VSA, Hamming search,
/// cube state, temporal, cognitive, autograd, training pipeline, etc.).
///
/// Migrated from monolithic bindings.cpp to use Tensor-based I/O where
/// applicable. Non-float ops (VSA int8, cube uint8) keep their native
/// numpy return types since grilly::nn::Tensor is float32-only.

#include "bindings_core.h"
#include "grilly/ops/activations.h"
#include "grilly/ops/linear.h"
#include "grilly/ops/embedding.h"
#include "grilly/ops/kv_cache.h"
#include "grilly/ops/swizzle.h"
#include "grilly/experimental/paged_latent_pool.h"
#include "grilly/experimental/fused_attention.h"
#include "grilly/cubemind/types.h"
#include "grilly/cubemind/vsa.h"
#include "grilly/cubemind/block_ops.h"
#include "grilly/cubemind/hmm_ops.h"
#include "grilly/cubemind/tensor_ops.h"
#include "grilly/cubemind/cube.h"
#include "grilly/cubemind/cache.h"
#include "grilly/cubemind/text_encoder.h"
#include "grilly/cubemind/semantic_assigner.h"
#include "grilly/cubemind/resonator.h"
#include "grilly/training/pipeline.h"
#include "grilly/cognitive/world_model.h"
#include "grilly/temporal/temporal_encoder.h"
#include "grilly/temporal/counterfactual.h"
#include "grilly/temporal/vulkan_temporal.h"
#include "grilly/temporal/hippocampus.h"
#include "grilly/autograd/autograd.h"
#include "grilly/system_profile.h"

/// Persistent GPU cache for Hamming search benchmarking.
struct HammingSearchBench {
    GrillyCoreContext* ctx = nullptr;
    std::vector<uint32_t> cachePacked;
    grilly::GrillyBuffer gpuCache{};
    grilly::GrillyBuffer gpuQuery{};
    grilly::GrillyBuffer gpuDist{};
    VkDescriptorSet descSet = VK_NULL_HANDLE;
    grilly::PipelineEntry pipe{};
    uint32_t numEntries = 0;
    uint32_t wordsPerVec = 0;
    uint32_t dim = 0;
    bool gpuReady = false;

    std::vector<uint32_t> cachePackedSoA;
    grilly::GrillyBuffer gpuCacheSoA{};
    grilly::GrillyBuffer gpuResult{};
    VkDescriptorSet descSetTop1 = VK_NULL_HANDLE;
    grilly::PipelineEntry pipeTop1{};
    bool top1Ready = false;

    uint32_t subgroupSize = 64;
    uint32_t entriesPerWG = 4;

    VkQueryPool tsPool = VK_NULL_HANDLE;
    float timestampPeriod = 0.0f;
    double lastGpuMs = 0.0;

    ~HammingSearchBench() {
        if (!ctx) return;
        if (tsPool != VK_NULL_HANDLE)
            vkDestroyQueryPool(ctx->device.device(), tsPool, nullptr);
        auto alloc = ctx->pool.allocator();
        if (gpuCache.handle != VK_NULL_HANDLE)
            vmaDestroyBuffer(alloc, gpuCache.handle, gpuCache.allocation);
        if (gpuQuery.handle != VK_NULL_HANDLE)
            vmaDestroyBuffer(alloc, gpuQuery.handle, gpuQuery.allocation);
        if (gpuDist.handle != VK_NULL_HANDLE)
            vmaDestroyBuffer(alloc, gpuDist.handle, gpuDist.allocation);
        if (gpuResult.handle != VK_NULL_HANDLE)
            vmaDestroyBuffer(alloc, gpuResult.handle, gpuResult.allocation);
        if (gpuCacheSoA.handle != VK_NULL_HANDLE)
            vmaDestroyBuffer(alloc, gpuCacheSoA.handle,
                             gpuCacheSoA.allocation);
    }
};

void register_misc_ops(py::module_& m) {
    using namespace grilly::nn;

    // ── Dropout ──────────────────────────────────────────────────────────
    m.def(
        "dropout",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input, py::array_t<float> random_mask,
           float p, bool training) -> Tensor {
            auto inBuf = input.request();
            uint32_t total = 1;
            for (int i = 0; i < inBuf.ndim; ++i)
                total *= static_cast<uint32_t>(inBuf.shape[i]);

            py::array_t<float> result(inBuf.shape);
            grilly::ops::dropout(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(inBuf.ptr),
                static_cast<const float*>(random_mask.request().ptr),
                static_cast<float*>(result.request().ptr),
                total, p, training);
            return Tensor::from_numpy(result);
        },
        py::arg("device"), py::arg("input"), py::arg("random_mask"),
        py::arg("p") = 0.5f, py::arg("training") = true,
        "GPU dropout with inverted scaling");

    // ── Embedding lookup ─────────────────────────────────────────────────
    m.def(
        "embedding_lookup",
        [](GrillyCoreContext& ctx,
           py::array_t<uint32_t> token_ids,
           py::array_t<float> embeddings) -> Tensor {
            auto idBuf = token_ids.request();
            auto eBuf = embeddings.request();

            uint32_t batchSize = 1, seqLen;
            if (idBuf.ndim == 1) {
                seqLen = static_cast<uint32_t>(idBuf.shape[0]);
            } else {
                batchSize = static_cast<uint32_t>(idBuf.shape[0]);
                seqLen = static_cast<uint32_t>(idBuf.shape[1]);
            }
            uint32_t vocabSize = static_cast<uint32_t>(eBuf.shape[0]);
            uint32_t embDim = static_cast<uint32_t>(eBuf.shape[1]);

            py::array_t<float> result({
                static_cast<py::ssize_t>(batchSize),
                static_cast<py::ssize_t>(seqLen),
                static_cast<py::ssize_t>(embDim)});

            grilly::ops::EmbeddingParams p{
                batchSize, seqLen, vocabSize, embDim};
            grilly::ops::embeddingLookup(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const uint32_t*>(idBuf.ptr),
                static_cast<const float*>(eBuf.ptr),
                static_cast<float*>(result.request().ptr), p);

            if (idBuf.ndim == 1)
                result = result.reshape({
                    static_cast<py::ssize_t>(seqLen),
                    static_cast<py::ssize_t>(embDim)});
            return Tensor::from_numpy(result);
        },
        py::arg("device"), py::arg("token_ids"), py::arg("embeddings"),
        "GPU embedding table lookup");

    // ── KV Cache ─────────────────────────────────────────────────────────
    py::class_<grilly::ops::KVCache>(m, "KVCache")
        .def_readonly("current_len",
                      &grilly::ops::KVCache::currentLen)
        .def("stats", [](const grilly::ops::KVCache& kv) {
            auto s = grilly::ops::kvCacheGetStats(kv);
            py::dict d;
            d["current_tokens"] = s.currentTokens;
            d["max_tokens"] = s.maxTokens;
            d["total_evicted"] = s.totalEvicted;
            d["total_appended"] = s.totalAppended;
            d["compression_ratio"] = s.compressionRatio;
            d["avg_attention_score"] = s.avgAttentionScore;
            return d;
        });

    m.def(
        "create_kv_cache",
        [](GrillyCoreContext& ctx,
           uint32_t maxSeqLen, uint32_t numHeads, uint32_t headDim,
           uint32_t numLayers, uint32_t compressionRatio,
           uint32_t maxCacheTokens, bool useAsymmetricQuant,
           uint32_t valueBits, bool crossLayerSharing,
           bool useH2O, bool useSpeculativeEviction,
           float evictionThreshold) -> grilly::ops::KVCache {
            grilly::ops::KVCacheConfig cfg;
            cfg.maxSeqLen = maxSeqLen;
            cfg.numHeads = numHeads;
            cfg.headDim = headDim;
            cfg.numLayers = numLayers;
            cfg.compressionRatio = compressionRatio;
            cfg.maxCacheTokens = maxCacheTokens;
            cfg.useAsymmetricQuant = useAsymmetricQuant;
            cfg.valueBits = valueBits;
            cfg.crossLayerSharing = crossLayerSharing;
            cfg.useH2O = useH2O;
            cfg.useSpeculativeEviction = useSpeculativeEviction;
            cfg.evictionThreshold = evictionThreshold;
            return grilly::ops::createKVCache(ctx.pool, cfg);
        },
        py::arg("device"),
        py::arg("max_seq_len") = 2048,
        py::arg("num_heads") = 8,
        py::arg("head_dim") = 64,
        py::arg("num_layers") = 12,
        py::arg("compression_ratio") = 4,
        py::arg("max_cache_tokens") = 2048,
        py::arg("use_asymmetric_quant") = false,
        py::arg("value_bits") = 4,
        py::arg("cross_layer_sharing") = false,
        py::arg("use_h2o") = true,
        py::arg("use_speculative_eviction") = false,
        py::arg("eviction_threshold") = 0.1f,
        "Create a KV cache with MLA compression and H2O eviction");

    m.def(
        "kv_cache_append",
        [](GrillyCoreContext& ctx, grilly::ops::KVCache& kvCache,
           py::array_t<float> newKeys, py::array_t<float> newValues) {
            auto kBuf = newKeys.request();
            uint32_t numNew = static_cast<uint32_t>(kBuf.shape[0]);
            grilly::ops::kvCacheAppend(
                ctx.batch, ctx.pool, ctx.cache, kvCache,
                static_cast<const float*>(kBuf.ptr),
                static_cast<const float*>(
                    newValues.request().ptr),
                numNew);
        },
        py::arg("device"), py::arg("kv_cache"),
        py::arg("new_keys"), py::arg("new_values"),
        "Append new KV pairs to cache");

    m.def(
        "kv_cache_decode",
        [](GrillyCoreContext& ctx,
           const grilly::ops::KVCache& kvCache) -> py::dict {
            const auto& cfg = kvCache.config;
            uint32_t tokens = kvCache.currentLen;
            py::array_t<float> keys({
                static_cast<py::ssize_t>(tokens),
                static_cast<py::ssize_t>(cfg.numHeads),
                static_cast<py::ssize_t>(cfg.headDim)});
            py::array_t<float> values({
                static_cast<py::ssize_t>(tokens),
                static_cast<py::ssize_t>(cfg.numHeads),
                static_cast<py::ssize_t>(cfg.headDim)});
            grilly::ops::kvCacheDecode(
                ctx.batch, ctx.pool, ctx.cache, kvCache,
                static_cast<float*>(keys.request().ptr),
                static_cast<float*>(values.request().ptr));
            py::dict result;
            result["keys"] = Tensor::from_numpy(keys);
            result["values"] = Tensor::from_numpy(values);
            return result;
        },
        py::arg("device"), py::arg("kv_cache"),
        "Decode KV from compressed cache");

    m.def(
        "kv_cache_evict_h2o",
        [](GrillyCoreContext& ctx, grilly::ops::KVCache& kvCache,
           std::optional<py::array_t<float>> attentionScores,
           uint32_t numEvict) {
            const float* scoresPtr = nullptr;
            if (attentionScores.has_value())
                scoresPtr = static_cast<const float*>(
                    attentionScores->request().ptr);
            grilly::ops::kvCacheEvictH2O(
                ctx.batch, ctx.pool, ctx.cache, kvCache,
                scoresPtr, numEvict);
        },
        py::arg("device"), py::arg("kv_cache"),
        py::arg("attention_scores") = py::none(),
        py::arg("num_evict") = 0,
        "Run H2O eviction on KV cache");

    m.def(
        "kv_cache_compact",
        [](GrillyCoreContext& ctx, grilly::ops::KVCache& kvCache) {
            grilly::ops::kvCacheCompact(
                ctx.batch, ctx.pool, ctx.cache, kvCache);
        },
        py::arg("device"), py::arg("kv_cache"),
        "Compact KV cache after eviction");

    m.def(
        "destroy_kv_cache",
        [](GrillyCoreContext& ctx, grilly::ops::KVCache& kvCache) {
            grilly::ops::destroyKVCache(ctx.pool, kvCache);
        },
        py::arg("device"), py::arg("kv_cache"),
        "Destroy KV cache and release GPU buffers");

    m.def(
        "kv_cache_init_eviction_head",
        [](GrillyCoreContext& ctx, grilly::ops::KVCache& kvCache,
           uint32_t inputDim, uint32_t hiddenDim, float lr) {
            grilly::ops::kvCacheInitEvictionHead(
                ctx.pool, kvCache, inputDim, hiddenDim, lr);
        },
        py::arg("device"), py::arg("kv_cache"),
        py::arg("input_dim"), py::arg("hidden_dim") = 32,
        py::arg("lr") = 1e-3f,
        "Initialize trainable eviction head");

    m.def(
        "kv_cache_train_eviction_head",
        [](GrillyCoreContext& ctx, grilly::ops::KVCache& kvCache,
           py::array_t<float> tokenFeatures,
           py::array_t<float> attentionScores, uint32_t seqLen) {
            grilly::ops::kvCacheTrainEvictionHead(
                ctx.batch, ctx.pool, ctx.cache, kvCache,
                static_cast<const float*>(
                    tokenFeatures.request().ptr),
                static_cast<const float*>(
                    attentionScores.request().ptr),
                seqLen);
        },
        py::arg("device"), py::arg("kv_cache"),
        py::arg("token_features"), py::arg("attention_scores"),
        py::arg("seq_len"),
        "Train the eviction head on attention patterns");

    m.def(
        "kv_cache_evict_speculative",
        [](GrillyCoreContext& ctx, grilly::ops::KVCache& kvCache,
           std::optional<py::array_t<float>> hiddenStates,
           uint32_t hiddenDim) {
            const float* hsPtr = nullptr;
            if (hiddenStates.has_value())
                hsPtr = static_cast<const float*>(
                    hiddenStates->request().ptr);
            grilly::ops::kvCacheEvictSpeculative(
                ctx.batch, ctx.pool, ctx.cache, kvCache,
                hsPtr, hiddenDim);
        },
        py::arg("device"), py::arg("kv_cache"),
        py::arg("hidden_states") = py::none(),
        py::arg("hidden_dim") = 64,
        "Run speculative eviction");

    // ── Swizzle KV ───────────────────────────────────────────────────────
    m.def(
        "swizzle_kv",
        [](GrillyCoreContext& ctx, py::array_t<float> input,
           uint32_t waveSize, bool reverse) -> py::array_t<float> {
            auto inBuf = input.request();
            if (inBuf.ndim != 4)
                throw std::runtime_error("input must be 4D");
            uint32_t batchSize = static_cast<uint32_t>(inBuf.shape[0]);
            uint32_t numHeads  = static_cast<uint32_t>(inBuf.shape[1]);
            uint32_t seqLen    = static_cast<uint32_t>(inBuf.shape[2]);
            uint32_t headDim   = static_cast<uint32_t>(inBuf.shape[3]);

            py::array_t<float> result;
            if (reverse) {
                result = py::array_t<float>({
                    static_cast<py::ssize_t>(batchSize),
                    static_cast<py::ssize_t>(numHeads),
                    static_cast<py::ssize_t>(seqLen),
                    static_cast<py::ssize_t>(headDim)});
            } else {
                size_t outSize = grilly::ops::swizzledBufferSize(
                    batchSize, numHeads, seqLen, headDim, waveSize);
                result = py::array_t<float>(outSize / sizeof(float));
            }
            auto rBuf = result.request();
            grilly::ops::swizzle(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(inBuf.ptr),
                static_cast<float*>(rBuf.ptr),
                batchSize, numHeads, seqLen, headDim,
                waveSize, reverse);
            return result;
        },
        py::arg("device"), py::arg("input"),
        py::arg("wave_size") = 32, py::arg("reverse") = false,
        "Swizzle/unswizzle KV tensor for Wave32 memory alignment");

    // ── Fused attention CPU reference ────────────────────────────────────
    m.def(
        "fused_attention_cpu",
        [](py::array_t<float> Q, py::array_t<float> latents,
           py::array_t<float> wUp,
           std::optional<py::array_t<float>> mask,
           uint32_t cachedTokens, uint32_t numHeads,
           uint32_t headDim, uint32_t latentDim,
           float scale) -> Tensor {
            auto qBuf = Q.request();
            if (qBuf.ndim < 2)
                throw std::runtime_error("Q must be at least 2D");
            uint32_t batchSize = 1, seqLen = 1;
            if (qBuf.ndim == 4) {
                batchSize = static_cast<uint32_t>(qBuf.shape[0]);
                seqLen = static_cast<uint32_t>(qBuf.shape[2]);
            } else if (qBuf.ndim == 3) {
                batchSize = static_cast<uint32_t>(qBuf.shape[0]);
                seqLen = static_cast<uint32_t>(qBuf.shape[1]);
            } else {
                seqLen = static_cast<uint32_t>(qBuf.shape[0]);
            }
            const float* maskPtr = nullptr;
            if (mask.has_value())
                maskPtr = static_cast<const float*>(
                    mask->request().ptr);
            size_t outSize = size_t(batchSize) * numHeads *
                             seqLen * headDim;
            py::array_t<float> result(outSize);
            grilly::experimental::fusedAttentionCPU(
                static_cast<const float*>(qBuf.ptr),
                static_cast<const float*>(latents.request().ptr),
                static_cast<const float*>(wUp.request().ptr),
                maskPtr, static_cast<float*>(result.request().ptr),
                batchSize, seqLen, cachedTokens,
                numHeads, headDim, latentDim, scale);
            result = result.reshape({
                static_cast<py::ssize_t>(batchSize),
                static_cast<py::ssize_t>(numHeads),
                static_cast<py::ssize_t>(seqLen),
                static_cast<py::ssize_t>(headDim)});
            return Tensor::from_numpy(result);
        },
        py::arg("Q"), py::arg("latents"), py::arg("w_up"),
        py::arg("mask") = py::none(),
        py::arg("cached_tokens") = 0,
        py::arg("num_heads") = 8, py::arg("head_dim") = 64,
        py::arg("latent_dim") = 16, py::arg("scale") = 0.0f,
        "CPU reference for fused MLA decompression + attention");

    // ── CubeMind: VSA encoding ───────────────────────────────────────────
    m.def("blake3_role",
        [](const std::string& key, uint32_t dim,
           const std::string& domain) -> py::array_t<int8_t> {
            auto result = grilly::cubemind::blake3Role(key, dim, domain);
            py::array_t<int8_t> arr(dim);
            std::memcpy(arr.request().ptr, result.data(),
                        dim * sizeof(int8_t));
            return arr;
        },
        py::arg("key"), py::arg("dim"),
        py::arg("domain") = "grilly.cubemind",
        "Generate deterministic bipolar role vector via BLAKE3");

    m.def("vsa_bind",
        [](py::array_t<int8_t> a,
           py::array_t<int8_t> b) -> py::array_t<int8_t> {
            auto aBuf = a.request(); auto bBuf = b.request();
            uint32_t dim = static_cast<uint32_t>(aBuf.shape[0]);
            if (static_cast<uint32_t>(bBuf.shape[0]) != dim)
                throw std::runtime_error("VSA bind: dimension mismatch");
            auto result = grilly::cubemind::vsaBind(
                static_cast<const int8_t*>(aBuf.ptr),
                static_cast<const int8_t*>(bBuf.ptr), dim);
            py::array_t<int8_t> arr(dim);
            std::memcpy(arr.request().ptr, result.data(),
                        dim * sizeof(int8_t));
            return arr;
        },
        py::arg("a"), py::arg("b"),
        "Bipolar binding: element-wise multiply");

    m.def("vsa_bundle",
        [](std::vector<py::array_t<int8_t>> vectors)
           -> py::array_t<int8_t> {
            if (vectors.empty())
                throw std::runtime_error("VSA bundle: empty");
            uint32_t dim = static_cast<uint32_t>(
                vectors[0].request().shape[0]);
            std::vector<const int8_t*> ptrs;
            ptrs.reserve(vectors.size());
            for (auto& v : vectors)
                ptrs.push_back(
                    static_cast<const int8_t*>(v.request().ptr));
            auto result = grilly::cubemind::vsaBundle(ptrs, dim);
            py::array_t<int8_t> arr(dim);
            std::memcpy(arr.request().ptr, result.data(),
                        dim * sizeof(int8_t));
            return arr;
        },
        py::arg("vectors"),
        "Bipolar bundling: majority vote superposition");

    m.def("vsa_bitpack",
        [](py::array_t<int8_t> bipolar) -> py::array_t<uint32_t> {
            auto buf = bipolar.request();
            uint32_t dim = static_cast<uint32_t>(buf.shape[0]);
            auto packed = grilly::cubemind::vsaBitpack(
                static_cast<const int8_t*>(buf.ptr), dim);
            py::array_t<uint32_t> arr(packed.numWords());
            std::memcpy(arr.request().ptr, packed.data.data(),
                        packed.numWords() * sizeof(uint32_t));
            return arr;
        },
        py::arg("bipolar"),
        "Bitpack bipolar int8 to packed uint32 bits");

    m.def("vsa_encode",
        [](std::vector<std::string> roles,
           std::vector<py::array_t<int8_t>> fillers,
           uint32_t dim) -> py::array_t<uint32_t> {
            if (roles.size() != fillers.size())
                throw std::runtime_error("VSA encode: size mismatch");
            std::vector<const int8_t*> fillerPtrs;
            fillerPtrs.reserve(fillers.size());
            for (auto& f : fillers)
                fillerPtrs.push_back(
                    static_cast<const int8_t*>(f.request().ptr));
            auto packed = grilly::cubemind::vsaEncode(
                roles, fillerPtrs, dim);
            py::array_t<uint32_t> arr(packed.numWords());
            std::memcpy(arr.request().ptr, packed.data.data(),
                        packed.numWords() * sizeof(uint32_t));
            return arr;
        },
        py::arg("roles"), py::arg("fillers"), py::arg("dim"),
        "Full VSA encode: BLAKE3 roles + bind + bundle + bitpack");

    // ── CubeMind: block/hmm/tensor/explore ops (from .inc files) ─────────
#include "block_ops_bindings.inc"
#include "hmm_ops_bindings.inc"
#include "vsa_explore_bindings.inc"
#include "tensor_ops_bindings.inc"

    // ── Hamming search ───────────────────────────────────────────────────
    m.def("hamming_search",
        [](GrillyCoreContext& ctx, py::array_t<int8_t> query,
           py::array_t<int8_t> cache_data) -> py::array_t<uint32_t> {
            auto qBuf = query.request();
            auto cBuf = cache_data.request();
            uint32_t dim = static_cast<uint32_t>(qBuf.shape[0]);
            uint32_t numEntries;
            if (cBuf.ndim == 2) {
                numEntries = static_cast<uint32_t>(cBuf.shape[0]);
                if (static_cast<uint32_t>(cBuf.shape[1]) != dim)
                    throw std::runtime_error("Cache dim mismatch");
            } else {
                numEntries = static_cast<uint32_t>(cBuf.size) / dim;
            }
            auto queryPacked = grilly::cubemind::vsaBitpack(
                static_cast<const int8_t*>(qBuf.ptr), dim);
            uint32_t wordsPerVec = queryPacked.numWords();
            std::vector<uint32_t> cachePacked(
                size_t(numEntries) * wordsPerVec);
            for (uint32_t i = 0; i < numEntries; ++i) {
                auto packed = grilly::cubemind::vsaBitpack(
                    static_cast<const int8_t*>(cBuf.ptr) + i * dim,
                    dim);
                std::memcpy(
                    cachePacked.data() + size_t(i) * wordsPerVec,
                    packed.data.data(),
                    wordsPerVec * sizeof(uint32_t));
            }
            py::array_t<uint32_t> result(numEntries);
            grilly::cubemind::hammingSearch(
                ctx.batch, ctx.pool, ctx.cache,
                queryPacked.data.data(), cachePacked.data(),
                static_cast<uint32_t*>(result.request().ptr),
                numEntries, wordsPerVec);
            return result;
        },
        py::arg("device"), py::arg("query"), py::arg("cache"),
        "GPU Hamming search: distances from query to all entries");

    m.def("hamming_search_cpu",
        [](py::array_t<int8_t> query,
           py::array_t<int8_t> cache_data) -> py::array_t<uint32_t> {
            auto qBuf = query.request();
            auto cBuf = cache_data.request();
            uint32_t dim = static_cast<uint32_t>(qBuf.shape[0]);
            uint32_t numEntries = (cBuf.ndim == 2)
                ? static_cast<uint32_t>(cBuf.shape[0])
                : static_cast<uint32_t>(cBuf.size) / dim;
            auto queryPacked = grilly::cubemind::vsaBitpack(
                static_cast<const int8_t*>(qBuf.ptr), dim);
            uint32_t wordsPerVec = queryPacked.numWords();
            std::vector<uint32_t> cachePacked(
                size_t(numEntries) * wordsPerVec);
            for (uint32_t i = 0; i < numEntries; ++i) {
                auto packed = grilly::cubemind::vsaBitpack(
                    static_cast<const int8_t*>(cBuf.ptr) + i * dim,
                    dim);
                std::memcpy(
                    cachePacked.data() + size_t(i) * wordsPerVec,
                    packed.data.data(),
                    wordsPerVec * sizeof(uint32_t));
            }
            auto distances = grilly::cubemind::hammingSearchCPU(
                queryPacked.data.data(), cachePacked.data(),
                numEntries, wordsPerVec);
            py::array_t<uint32_t> result(numEntries);
            std::memcpy(result.request().ptr, distances.data(),
                        numEntries * sizeof(uint32_t));
            return result;
        },
        py::arg("query"), py::arg("cache"),
        "CPU reference for Hamming search");

    // ── Elementwise math (.inc) ──────────────────────────────────────────
#include "elementwise_bindings.inc"

    // ── Cube state ops ───────────────────────────────────────────────────
    m.def("cube_solved",
        [](uint32_t size) -> py::array_t<uint8_t> {
            auto cs = (size == 2) ? grilly::cubemind::CubeSize::Cube2x2
                                  : grilly::cubemind::CubeSize::Cube3x3;
            auto state = grilly::cubemind::cubeSolved(cs);
            py::array_t<uint8_t> arr(state.facelets.size());
            std::memcpy(arr.request().ptr, state.facelets.data(),
                        state.facelets.size());
            return arr;
        },
        py::arg("size") = 3, "Create solved cube state");

    m.def("cube_apply_move",
        [](py::array_t<uint8_t> state, uint32_t size,
           uint8_t move) -> py::array_t<uint8_t> {
            auto buf = state.request();
            grilly::cubemind::CubeState cs;
            cs.size = (size == 2) ? grilly::cubemind::CubeSize::Cube2x2
                                  : grilly::cubemind::CubeSize::Cube3x3;
            cs.facelets.assign(static_cast<uint8_t*>(buf.ptr),
                               static_cast<uint8_t*>(buf.ptr) + buf.size);
            auto result = grilly::cubemind::cubeApplyMove(
                cs, static_cast<grilly::cubemind::CubeMove>(move));
            py::array_t<uint8_t> arr(result.facelets.size());
            std::memcpy(arr.request().ptr, result.facelets.data(),
                        result.facelets.size());
            return arr;
        },
        py::arg("state"), py::arg("size") = 3, py::arg("move") = 0,
        "Apply a move to a cube state");

    m.def("cube_random_walk",
        [](uint32_t size, uint32_t numMoves,
           uint32_t seed) -> py::array_t<uint8_t> {
            auto cs = (size == 2) ? grilly::cubemind::CubeSize::Cube2x2
                                  : grilly::cubemind::CubeSize::Cube3x3;
            auto state = grilly::cubemind::cubeRandomWalk(
                cs, numMoves, seed);
            py::array_t<uint8_t> arr(state.facelets.size());
            std::memcpy(arr.request().ptr, state.facelets.data(),
                        state.facelets.size());
            return arr;
        },
        py::arg("size") = 3, py::arg("num_moves") = 20,
        py::arg("seed") = 0, "Random walk from solved state");

    m.def("cube_estimate_distance",
        [](py::array_t<uint8_t> state, uint32_t size) -> uint32_t {
            auto buf = state.request();
            grilly::cubemind::CubeState cs;
            cs.size = (size == 2) ? grilly::cubemind::CubeSize::Cube2x2
                                  : grilly::cubemind::CubeSize::Cube3x3;
            cs.facelets.assign(static_cast<uint8_t*>(buf.ptr),
                               static_cast<uint8_t*>(buf.ptr) + buf.size);
            return grilly::cubemind::cubeEstimateDistance(cs);
        },
        py::arg("state"), py::arg("size") = 3,
        "Estimate distance from solved");

    m.def("cube_to_vsa",
        [](py::array_t<uint8_t> state, uint32_t size,
           uint32_t dim) -> py::array_t<int8_t> {
            auto buf = state.request();
            grilly::cubemind::CubeState cs;
            cs.size = (size == 2) ? grilly::cubemind::CubeSize::Cube2x2
                                  : grilly::cubemind::CubeSize::Cube3x3;
            cs.facelets.assign(static_cast<uint8_t*>(buf.ptr),
                               static_cast<uint8_t*>(buf.ptr) + buf.size);
            auto result = grilly::cubemind::cubeToVSA(cs, dim);
            py::array_t<int8_t> arr(dim);
            std::memcpy(arr.request().ptr, result.data(),
                        dim * sizeof(int8_t));
            return arr;
        },
        py::arg("state"), py::arg("size") = 3,
        py::arg("dim") = 10240,
        "Encode cube state as bipolar VSA hypervector");

    // ── BufferUsage enum ─────────────────────────────────────────────────
    py::enum_<grilly::BufferDesc::Usage>(m, "BufferUsage")
        .value("HostVisible", grilly::BufferDesc::HostVisible)
        .value("DeviceLocal", grilly::BufferDesc::DeviceLocal)
        .value("Readback", grilly::BufferDesc::Readback)
        .value("PreferDevice", grilly::BufferDesc::PreferDevice);

    // ── SystemProfile ────────────────────────────────────────────────────
    py::class_<grilly::SystemProfile>(m, "SystemProfile")
        .def_readonly("device_name",
                      &grilly::SystemProfile::deviceName)
        .def_readonly("subgroup_size",
                      &grilly::SystemProfile::subgroupSize)
        .def_readonly("arena_size_bytes",
                      &grilly::SystemProfile::arenaSizeBytes)
        .def_readonly("vsa_dim", &grilly::SystemProfile::vsaDim)
        .def_readonly("max_cache_capacity",
                      &grilly::SystemProfile::maxCacheCapacity)
        .def_readonly("max_constraint_capacity",
                      &grilly::SystemProfile::maxConstraintCapacity)
        .def_readonly("surprise_threshold",
                      &grilly::SystemProfile::surpriseThreshold)
        .def_readonly("coherence_threshold",
                      &grilly::SystemProfile::coherenceThreshold)
        .def_readonly("thinking_steps",
                      &grilly::SystemProfile::thinkingSteps)
        .def_readonly("batch_size",
                      &grilly::SystemProfile::batchSize)
        .def_readonly("workgroup_size",
                      &grilly::SystemProfile::workgroupSize)
        .def_property_readonly("entries_per_wg",
             &grilly::SystemProfile::entriesPerWG)
        .def_static("load", &grilly::SystemProfile::load,
             py::arg("path"), py::arg("profile_name"),
             "Load a hardware profile from profiles.json");
}
