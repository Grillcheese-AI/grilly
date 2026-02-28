#include "grilly/cubemind/cache.h"
#include "grilly/cubemind/vsa.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>

namespace grilly {
namespace cubemind {

// ── VSACache Constructor/Destructor ──────────────────────────────────

VSACache::VSACache(BufferPool& pool, const CacheConfig& config)
    : pool_(pool)
    , config_(config)
    , wordsPerVec_((config.dim + 31) / 32)
    , size_(0)
    , capacity_(config.initialCapacity)
    , gpuKeys_{}
    , gpuDirty_(true)
    , stats_{} {
    // Pre-allocate host storage
    keys_.resize(size_t(capacity_) * wordsPerVec_, 0);
    emotions_.resize(capacity_);
    utilities_.resize(capacity_, 0.0f);
}

VSACache::~VSACache() {
    if (gpuKeys_.handle != VK_NULL_HANDLE) {
        pool_.release(gpuKeys_);
    }
}

// ── GPU Sync ─────────────────────────────────────────────────────────
//
// Upload host-side keys to GPU buffer when dirty. This is called
// before every GPU Hamming search. The GPU buffer is sized to the
// current key array (may grow on insert).

void VSACache::ensureGPUSync() {
    if (!gpuDirty_ || size_ == 0) return;

    size_t keyBytes = size_t(size_) * wordsPerVec_ * sizeof(uint32_t);

    // Release old GPU buffer if it exists and is too small
    if (gpuKeys_.handle != VK_NULL_HANDLE) {
        if (gpuKeys_.size < keyBytes) {
            pool_.release(gpuKeys_);
            gpuKeys_ = {};
        }
    }

    // Acquire host-visible buffer from pool
    if (gpuKeys_.handle == VK_NULL_HANDLE) {
        gpuKeys_ = pool_.acquire(keyBytes);
    }

    pool_.upload(gpuKeys_, reinterpret_cast<const float*>(keys_.data()),
                 keyBytes);
    gpuDirty_ = false;
}

// ── Capacity Growth ──────────────────────────────────────────────────
//
// Double capacity when full, up to maxCapacity. Mirrors the paper's
// "66x growth via surprise-driven expansion" pattern.

void VSACache::growCapacity() {
    uint32_t newCapacity = std::min(capacity_ * 2, config_.maxCapacity);
    if (newCapacity == capacity_) return;  // At max

    keys_.resize(size_t(newCapacity) * wordsPerVec_, 0);
    emotions_.resize(newCapacity);
    utilities_.resize(newCapacity, 0.0f);
    capacity_ = newCapacity;
}

// ── Top-K Selection ──────────────────────────────────────────────────
//
// Simple partial sort to find k smallest distances. For the cache
// sizes we target (<500K), this is fast enough on CPU. The GPU does
// the expensive Hamming distance computation; top-k is O(n*k).

void VSACache::topKSelect(const uint32_t* distances, uint32_t n,
                            uint32_t k, std::vector<uint32_t>& indices,
                            std::vector<uint32_t>& topDists) const {
    k = std::min(k, n);
    indices.resize(k);
    topDists.resize(k);

    // Create index array and partial sort by distance
    std::vector<uint32_t> allIndices(n);
    std::iota(allIndices.begin(), allIndices.end(), 0);

    std::partial_sort(allIndices.begin(), allIndices.begin() + k,
                       allIndices.end(),
                       [distances](uint32_t a, uint32_t b) {
                           return distances[a] < distances[b];
                       });

    for (uint32_t i = 0; i < k; ++i) {
        indices[i] = allIndices[i];
        topDists[i] = distances[allIndices[i]];
    }
}

// ── GPU Lookup ──────────────────────────────────────────────────────
//
// Pipeline:
//   1. Ensure GPU keys are synced
//   2. Dispatch hamming-search.glsl (or CPU fallback)
//   3. Download distances
//   4. CPU top-k selection
//   5. Return results with emotions and surprise

CacheLookupResult VSACache::lookup(CommandBatch& batch,
                                     PipelineCache& pipeCache,
                                     const BitpackedVec& query,
                                     uint32_t topK) {
    CacheLookupResult result;

    if (size_ == 0) {
        result.querySurprise = 1.0f;  // Everything is novel to empty cache
        return result;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    ensureGPUSync();

    // Compute Hamming distances using persistent GPU cache buffer
    std::vector<uint32_t> distances(size_);
    size_t cacheBytes = size_t(size_) * wordsPerVec_ * sizeof(uint32_t);
    hammingSearchPersistent(batch, pool_, pipeCache,
                             query.data.data(),
                             gpuKeys_, cacheBytes,
                             distances.data(), size_, wordsPerVec_);

    // Top-k selection
    topKSelect(distances.data(), size_, topK,
               result.indices, result.distances);

    // Gather emotions for top-k
    result.emotions.resize(result.indices.size());
    for (size_t i = 0; i < result.indices.size(); ++i) {
        result.emotions[i] = emotions_[result.indices[i]];
    }

    // Surprise = normalized minimum distance
    uint32_t minDist = result.distances.empty() ? config_.dim
                                                 : result.distances[0];
    result.querySurprise = static_cast<float>(minDist) / config_.dim;

    auto t1 = std::chrono::high_resolution_clock::now();
    stats_.lastLookupMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    stats_.totalLookups++;

    return result;
}

CacheLookupResult VSACache::lookupCPU(const BitpackedVec& query,
                                        uint32_t topK) {
    CacheLookupResult result;

    if (size_ == 0) {
        result.querySurprise = 1.0f;
        return result;
    }

    auto distances = hammingSearchCPU(query.data.data(), keys_.data(),
                                       size_, wordsPerVec_);

    topKSelect(distances.data(), size_, topK,
               result.indices, result.distances);

    result.emotions.resize(result.indices.size());
    for (size_t i = 0; i < result.indices.size(); ++i) {
        result.emotions[i] = emotions_[result.indices[i]];
    }

    uint32_t minDist = result.distances.empty() ? config_.dim
                                                 : result.distances[0];
    result.querySurprise = static_cast<float>(minDist) / config_.dim;
    stats_.totalLookups++;

    return result;
}

// ── Insert ──────────────────────────────────────────────────────────
//
// Surprise-driven insertion: compute normalized Hamming distance to
// nearest existing entry. If surprise > threshold, insert.
// If cache is full, grow capacity (up to max) or evict.

bool VSACache::insert(const BitpackedVec& key, const EmotionState& emotion) {
    // Check surprise against existing entries
    if (size_ > 0) {
        auto distances = hammingSearchCPU(key.data.data(), keys_.data(),
                                           size_, wordsPerVec_);
        uint32_t minDist = *std::min_element(distances.begin(), distances.end());
        float surprise = static_cast<float>(minDist) / config_.dim;

        if (surprise < config_.surpriseThreshold) {
            return false;  // Not novel enough
        }
    }

    // Grow if needed
    if (size_ >= capacity_) {
        if (capacity_ < config_.maxCapacity) {
            growCapacity();
        } else {
            // At max capacity — evict 1 entry first
            evict(1);
        }
    }

    // Insert at position size_
    std::memcpy(keys_.data() + size_t(size_) * wordsPerVec_,
                key.data.data(),
                wordsPerVec_ * sizeof(uint32_t));
    emotions_[size_] = emotion;
    // Initial utility proportional to surprise — novel entries survive eviction.
    // surprise=1.0 → utility=1.0, surprise=0.1 → utility=0.1
    utilities_[size_] = std::max(emotion.surprise, 0.01f);

    size_++;
    gpuDirty_ = true;
    stats_.totalInserts++;
    stats_.size = size_;

    return true;
}

uint32_t VSACache::insertBatch(const std::vector<BitpackedVec>& keys,
                                const std::vector<EmotionState>& emotions) {
    uint32_t inserted = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (insert(keys[i], emotions[i])) {
            inserted++;
        }
    }
    return inserted;
}

// ── Eviction ────────────────────────────────────────────────────────
//
// Remove the n entries with lowest utility. Compact the arrays by
// moving the last entry into each gap (swap-and-pop).

void VSACache::evict(uint32_t count) {
    if (count == 0 || size_ == 0) return;
    count = std::min(count, size_);

    // Find indices of lowest-utility entries
    std::vector<uint32_t> indices(size_);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + count,
                       indices.end(),
                       [this](uint32_t a, uint32_t b) {
                           return utilities_[a] < utilities_[b];
                       });

    // Sort eviction indices in descending order to avoid invalidation
    std::vector<uint32_t> toEvict(indices.begin(), indices.begin() + count);
    std::sort(toEvict.rbegin(), toEvict.rend());

    for (uint32_t idx : toEvict) {
        // Swap with last entry
        uint32_t last = size_ - 1;
        if (idx != last) {
            std::memcpy(keys_.data() + size_t(idx) * wordsPerVec_,
                        keys_.data() + size_t(last) * wordsPerVec_,
                        wordsPerVec_ * sizeof(uint32_t));
            emotions_[idx] = emotions_[last];
            utilities_[idx] = utilities_[last];
        }
        size_--;
    }

    gpuDirty_ = true;
    stats_.totalEvictions += count;
    stats_.size = size_;
}

// ── Utility Update ──────────────────────────────────────────────────
//
// Decay all utilities by the decay factor (recency penalty).
// Boost accessed entries (frequency reward).

void VSACache::updateUtility(const std::vector<uint32_t>& accessedIndices) {
    // Decay all
    for (uint32_t i = 0; i < size_; ++i) {
        utilities_[i] *= config_.utilityDecay;
    }

    // Boost accessed
    for (uint32_t idx : accessedIndices) {
        if (idx < size_) {
            utilities_[idx] += 1.0f;
        }
    }
}

CacheStats VSACache::stats() const {
    CacheStats s = stats_;
    s.size = size_;
    s.capacity = capacity_;

    // Compute averages
    if (size_ > 0) {
        float sumSurprise = 0.0f, sumUtility = 0.0f;
        for (uint32_t i = 0; i < size_; ++i) {
            sumSurprise += emotions_[i].surprise;
            sumUtility += utilities_[i];
        }
        s.avgSurprise = sumSurprise / size_;
        s.avgUtility = sumUtility / size_;
    }

    return s;
}

}  // namespace cubemind
}  // namespace grilly
