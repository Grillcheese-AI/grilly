#include "grilly/cubemind/dream_cycle.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace grilly {
namespace cubemind {

// ── CPU Oja Update ──────────────────────────────────────────────────

void ojaUpdate(float* m, const float* x, uint32_t dim, float eta) {
    // Phase 1: Compute activation y = m · x
    float y = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        y += m[i] * x[i];
    }

    // Phase 2: Oja update: m += η·y·(x - y·m)
    float eta_y = eta * y;
    for (uint32_t i = 0; i < dim; ++i) {
        m[i] += eta_y * (x[i] - y * m[i]);
    }
}

void ojaUpdateBatch(float* memories, const float* inputs,
                     uint32_t n, uint32_t dim, float eta) {
    for (uint32_t v = 0; v < n; ++v) {
        ojaUpdate(memories + v * dim, inputs + v * dim, dim, eta);
    }
}

// ── GPU Oja Update ──────────────────────────────────────────────────

void ojaUpdateGPU(CommandBatch& batch, BufferPool& pool,
                   PipelineCache& pipeCache,
                   float* memories, const float* inputs,
                   uint32_t n, uint32_t dim, float eta) {
    // Upload memory and input buffers to GPU
    uint32_t totalFloats = n * dim;
    uint32_t bufSize = totalFloats * sizeof(float);

    auto memBuf = pool.acquire(bufSize);
    auto inBuf  = pool.acquire(bufSize);

    memBuf.upload(memories, bufSize);
    inBuf.upload(inputs, bufSize);

    // Push constants: {num_vectors, dim, eta}
    struct PushConsts {
        uint32_t numVectors;
        uint32_t dim;
        float eta;
    } pc = {n, dim, eta};

    // Dispatch oja-learning.spv
    auto pipeline = pipeCache.get("oja-learning");
    batch.bindPipeline(pipeline);
    batch.bindBuffers({memBuf.buffer(), inBuf.buffer()});
    batch.pushConstants(&pc, sizeof(pc));
    batch.dispatch((n + 255) / 256, 1, 1);
    batch.barrier();  // Wait for shader completion

    // Download updated memories
    memBuf.download(memories, bufSize);

    pool.release(memBuf);
    pool.release(inBuf);
}

// ── Sleep Cycle ─────────────────────────────────────────────────────

SleepCycle::SleepCycle(BufferPool& pool, VSACache& cache,
                        float* codebook, uint32_t nCodebook, uint32_t dim)
    : pool_(pool), cache_(cache),
      codebook_(codebook), nCodebook_(nCodebook), dim_(dim) {}

uint32_t SleepCycle::findNearest(const float* query) const {
    uint32_t bestIdx = 0;
    float bestDot = -1e30f;

    for (uint32_t i = 0; i < nCodebook_; ++i) {
        float dot = 0.0f;
        const float* entry = codebook_ + i * dim_;
        for (uint32_t d = 0; d < dim_; ++d) {
            dot += query[d] * entry[d];
        }
        if (dot > bestDot) {
            bestDot = dot;
            bestIdx = i;
        }
    }
    return bestIdx;
}

SleepStats SleepCycle::dream(CommandBatch& batch, PipelineCache& pipeCache,
                              const SleepConfig& config) {
    auto t0 = std::chrono::high_resolution_clock::now();

    SleepStats stats = {};
    uint32_t cacheSize = cache_.size();
    if (cacheSize == 0) return stats;

    uint32_t batchSize = std::min(config.batchSize, cacheSize);

    // ── 1. REPLAY: Sample high-utility episodic memories ────────────
    //
    // Sort by utility descending, take top batchSize.
    // Each memory is a bitpacked VSA key — convert to continuous float.
    //
    // For now, we sample the first batchSize entries (utilities are
    // maintained by the cache's updateUtility mechanism). A full
    // implementation would use reservoir sampling weighted by utility.

    std::vector<float> X_buffer(batchSize * dim_, 0.0f);
    std::vector<uint32_t> sampleIndices(batchSize);
    std::iota(sampleIndices.begin(), sampleIndices.end(), 0);

    // Convert bitpacked keys to continuous floats for the Oja shader
    uint32_t wordsPerVec = cache_.wordsPerVec();
    const uint32_t* keyData = cache_.keyData();

    for (uint32_t i = 0; i < batchSize; ++i) {
        uint32_t idx = sampleIndices[i];
        const uint32_t* packed = keyData + idx * wordsPerVec;

        // Unpack bits to bipolar floats {-1.0, +1.0}
        for (uint32_t w = 0; w < wordsPerVec; ++w) {
            uint32_t word = packed[w];
            for (uint32_t b = 0; b < 32 && (w * 32 + b) < dim_; ++b) {
                float val = (word & (1u << (31 - b))) ? 1.0f : -1.0f;
                X_buffer[i * dim_ + w * 32 + b] = val;
            }
        }
    }
    stats.memoriesReplayed = batchSize;

    // ── 2. ALIGN: Find closest semantic prototypes ──────────────────
    //
    // For each sampled memory, find the nearest codebook entry.
    // W_buffer holds copies of the matched prototypes.

    std::vector<float> W_buffer(batchSize * dim_);
    std::vector<uint32_t> matchedEntries(batchSize);
    float totalSimBefore = 0.0f;

    for (uint32_t i = 0; i < batchSize; ++i) {
        const float* query = X_buffer.data() + i * dim_;
        uint32_t nearest = findNearest(query);
        matchedEntries[i] = nearest;

        // Copy prototype into contiguous W buffer
        const float* proto = codebook_ + nearest * dim_;
        std::copy(proto, proto + dim_, W_buffer.begin() + i * dim_);

        // Track pre-consolidation similarity
        float dot = 0.0f;
        for (uint32_t d = 0; d < dim_; ++d) {
            dot += query[d] * proto[d];
        }
        totalSimBefore += dot / static_cast<float>(dim_);
    }
    stats.avgSimilarityBefore = totalSimBefore / batchSize;

    // ── 3. PLASTICITY: Apply Oja's rule via GPU shader ──────────────

    ojaUpdateGPU(batch, pool_, pipeCache,
                  W_buffer.data(), X_buffer.data(),
                  batchSize, dim_, config.learningRate);

    // ── 4. UPDATE: Write sharpened prototypes back to codebook ───────

    float totalSimAfter = 0.0f;
    for (uint32_t i = 0; i < batchSize; ++i) {
        uint32_t entryIdx = matchedEntries[i];
        float* proto = codebook_ + entryIdx * dim_;
        const float* updated = W_buffer.data() + i * dim_;

        // Compute post-consolidation similarity
        const float* query = X_buffer.data() + i * dim_;
        float dot = 0.0f;
        for (uint32_t d = 0; d < dim_; ++d) {
            dot += query[d] * updated[d];
        }
        totalSimAfter += dot / static_cast<float>(dim_);

        // Write back
        std::copy(updated, updated + dim_, proto);
    }
    stats.avgSimilarityAfter = totalSimAfter / batchSize;
    stats.prototypesUpdated = batchSize;

    // ── 5. FORGET: Decay episodic utilities and prune ────────────────
    //
    // Now that semantic truth has been extracted into the codebook,
    // decay the raw episodic traces. Low-utility memories are evicted.

    // Decay all utilities
    cache_.updateUtility({});  // Triggers decay without boosting

    // Count entries below threshold and evict
    auto cacheStats = cache_.stats();
    uint32_t toEvict = 0;
    if (cacheStats.avgUtility < config.evictionThreshold && cacheSize > 100) {
        toEvict = std::min(batchSize / 4, cacheSize / 10);
        if (toEvict > 0) {
            cache_.evict(toEvict);
        }
    }
    stats.memoriesPruned = toEvict;

    auto t1 = std::chrono::high_resolution_clock::now();
    stats.wallClockMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return stats;
}

}  // namespace cubemind
}  // namespace grilly
