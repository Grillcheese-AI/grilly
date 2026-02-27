#pragma once

#include <cstdint>
#include <vector>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/cubemind/types.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace cubemind {

// ── Hippocampal VSA Cache ─────────────────────────────────────────────
//
// GPU-accelerated associative memory using bitpacked Hamming distance.
//
// Design principles (from the CubeMind paper):
//   - Surprise-driven insertion: only novel entries (high surprise) enter
//   - Utility-based eviction: low-utility entries evicted first
//   - Lazy capacity doubling: starts small, grows 66x via expansion
//   - GPU Hamming search: XOR + popcount via hamming-search.glsl
//
// The cache stores bitpacked VSA keys on the host and mirrors them to
// a GPU buffer for accelerated search. The GPU buffer is re-uploaded
// whenever keys change (insert/evict sets a dirty flag).

struct CacheConfig {
    uint32_t initialCapacity;    // Starting size (default: 1024)
    uint32_t maxCapacity;        // Maximum entries (default: 500000)
    uint32_t dim;                // VSA dimension (default: 10240)
    float surpriseThreshold;     // Insert if surprise > tau (default: 0.3)
    float utilityDecay;          // Per-step decay (default: 0.99)
};

struct CacheLookupResult {
    std::vector<uint32_t> indices;      // Top-k indices
    std::vector<uint32_t> distances;    // Hamming distances
    std::vector<EmotionState> emotions; // Emotions of retrieved entries
    float querySurprise;                // Min distance / dim (normalized)
};

struct CacheStats {
    uint32_t size;               // Current entries
    uint32_t capacity;           // Allocated capacity
    uint32_t totalInserts;
    uint32_t totalEvictions;
    uint32_t totalLookups;
    float avgSurprise;
    float avgUtility;
    double lastLookupMs;         // Latency of last lookup
};

class VSACache {
public:
    VSACache(BufferPool& pool, const CacheConfig& config);
    ~VSACache();

    VSACache(const VSACache&) = delete;
    VSACache& operator=(const VSACache&) = delete;

    /// GPU-accelerated lookup: Hamming search + top-k selection.
    /// Returns top-k nearest neighbors by Hamming distance.
    CacheLookupResult lookup(CommandBatch& batch, PipelineCache& pipeCache,
                              const BitpackedVec& query, uint32_t topK = 5);

    /// CPU reference lookup (for verification).
    CacheLookupResult lookupCPU(const BitpackedVec& query, uint32_t topK = 5);

    /// Insert entry if surprise > threshold. Returns true if inserted.
    /// Surprise = 1.0 - (minHammingDistance / dim).
    bool insert(const BitpackedVec& key, const EmotionState& emotion);

    /// Insert batch of entries (surprise-filtered).
    /// Returns number actually inserted.
    uint32_t insertBatch(const std::vector<BitpackedVec>& keys,
                          const std::vector<EmotionState>& emotions);

    /// Evict n lowest-utility entries.
    void evict(uint32_t count);

    /// Update utilities: decay all by utilityDecay, boost accessed indices.
    void updateUtility(const std::vector<uint32_t>& accessedIndices);

    /// Get current stats.
    CacheStats stats() const;

    /// Get current size.
    uint32_t size() const { return size_; }

    /// Get words per vector.
    uint32_t wordsPerVec() const { return wordsPerVec_; }

    /// Direct access to contiguous key data (for GPU upload in benchmarks).
    const uint32_t* keyData() const { return keys_.data(); }

private:
    BufferPool& pool_;
    CacheConfig config_;
    uint32_t wordsPerVec_;

    // Host-side storage (mirrored to GPU for search)
    std::vector<uint32_t> keys_;        // Contiguous: [N * wordsPerVec]
    std::vector<EmotionState> emotions_;
    std::vector<float> utilities_;
    uint32_t size_;
    uint32_t capacity_;

    // GPU buffer for cache keys (uploaded on search if dirty)
    GrillyBuffer gpuKeys_;
    bool gpuDirty_;

    // Stats
    CacheStats stats_;

    void ensureGPUSync();
    void growCapacity();

    // Top-k selection on CPU (from Hamming distances)
    void topKSelect(const uint32_t* distances, uint32_t n,
                     uint32_t k, std::vector<uint32_t>& indices,
                     std::vector<uint32_t>& topDists) const;
};

}  // namespace cubemind
}  // namespace grilly
