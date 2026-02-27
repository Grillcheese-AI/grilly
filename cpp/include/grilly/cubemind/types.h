#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace grilly {
namespace cubemind {

// ── CubeMind Shared Types ─────────────────────────────────────────────
//
// Core data structures for Vector Symbolic Architecture (VSA) with
// bitpacked Hamming distance search on Vulkan compute.
//
// VSA encoding pipeline:
//   1. Generate BLAKE3 role vectors (structural positions)
//   2. Bind each role with its filler value (element-wise multiply)
//   3. Bundle all bound pairs (majority vote)
//   4. Sign-snap to strict bipolar {-1,+1}
//   5. Bitpack for Hamming distance: +1 -> bit 1, -1 -> bit 0
//
// At d_vsa=10240: each vector = 320 uint32 words = 1280 bytes.
// Hamming distance = popcount(XOR(a, b)) — single-cycle on RDNA 2.

/// Bitpacked bipolar vector. d_vsa bits packed into uint32 words.
/// Encoding: +1 maps to bit 1, -1 maps to bit 0.
/// The Hamming distance between two bitpacked vectors equals the
/// number of positions where the original bipolar vectors differ.
struct BitpackedVec {
    std::vector<uint32_t> data;
    uint32_t dim;   // Original bipolar dimension (e.g., 10240)

    uint32_t numWords() const { return (dim + 31) / 32; }

    /// Element-wise XOR (transition delta for hippocampal consolidation).
    BitpackedVec operator^(const BitpackedVec& other) const {
        BitpackedVec result;
        result.dim = dim;
        result.data.resize(data.size());
        for (size_t i = 0; i < data.size(); ++i)
            result.data[i] = data[i] ^ other.data[i];
        return result;
    }

    bool operator==(const BitpackedVec& other) const {
        return dim == other.dim && data == other.data;
    }
};

/// FNV-1a hash for BitpackedVec (unordered_map key support).
struct BitpackedVecHash {
    size_t operator()(const BitpackedVec& v) const {
        size_t h = 14695981039346656037ULL;  // FNV offset basis
        for (uint32_t w : v.data) {
            h ^= static_cast<size_t>(w);
            h *= 1099511628211ULL;           // FNV prime
        }
        return h;
    }
};

/// Per-entry emotional state for hippocampal cache.
/// Surprise drives insertion (novel entries have high surprise).
/// Stress tracks distance from reference/solved states.
struct EmotionState {
    float surprise;   // Cosine distance to nearest cache neighbor
    float stress;     // Normalized Hamming distance to reference state
};

/// Cache entry: bitpacked key + emotions + utility score.
/// Utility decays over time and is boosted on access, implementing
/// a recency+frequency heuristic for eviction decisions.
struct CacheEntry {
    BitpackedVec key;
    EmotionState emotion;
    float utility;
    uint32_t insertionStep;
};

// ── Shader Push Constants ─────────────────────────────────────────────
//
// These structs are passed as Vulkan push constants and must match
// the GLSL layout exactly. They are tightly packed (no padding).

/// Push constants for hamming-search.glsl (16 bytes).
/// Each thread handles one cache entry, computing the full Hamming
/// distance by iterating over wordsPerVec uint32 XOR+popcount ops.
struct HammingSearchParams {
    uint32_t numEntries;     // Number of cache entries to search
    uint32_t wordsPerVec;    // uint32 words per bitpacked vector (dim/32)
    uint32_t numQueries;     // Batch query count
    uint32_t queryOffset;    // Which query to process (for batched dispatch)
};

/// Push constants for hamming-topk.glsl (20 bytes).
/// Fused Hamming distance + top-k selection in a single dispatch.
struct HammingTopKParams {
    uint32_t numEntries;
    uint32_t wordsPerVec;
    uint32_t topK;
    uint32_t numQueries;
    uint32_t queryOffset;
};

/// Push constants for vsa-bitpack.glsl (8 bytes).
/// Converts float bipolar {-1.0, +1.0} to packed uint32 bits on GPU.
struct BitpackParams {
    uint32_t totalElements;  // Number of bipolar elements (= dim)
    uint32_t numWords;       // Output words (= (dim+31)/32)
};

}  // namespace cubemind
}  // namespace grilly
