#pragma once

#include <atomic>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "grilly/cubemind/types.h"
#include "grilly/cubemind/vsa.h"

#ifdef GRILLY_HAS_FASTTEXT
#include <fasttext/fasttext.h>
#endif

namespace grilly {
namespace cubemind {

/// SemanticAssigner: Lazy-evaluated memoization cache for LSH-projected
/// bipolar word vectors.
///
/// In LLM pretraining, projecting a 300D FastText vector through a Gaussian
/// random matrix to get a 10240D bipolar vector costs ~3M FLOPs per word.
/// At 10K tokens/sec, this saturates the CPU before data reaches the GPU.
///
/// Because natural language follows a Zipfian distribution (the top 1000
/// words cover ~80% of all tokens), caching the projection result for each
/// unique word eliminates nearly all redundant computation.
///
/// Architecture:
///   FAST PATH: shared_lock -> unordered_map lookup -> return BitpackedVec
///   SLOW PATH: project_to_bipolar() -> vsaBitpack() -> unique_lock -> insert
///
/// Thread safety: std::shared_mutex allows N concurrent readers on the fast
/// path, only serializing when a genuinely new word needs to be cached.
///
class SemanticAssigner {
public:
    /// @param dim   VSA bipolar dimension (default 10240 = 320 uint32 words)
    /// @param ft_dim Dense embedding dimension (FastText=300, MiniLM=384)
    explicit SemanticAssigner(uint32_t dim = 10240, uint32_t ft_dim = 300);

    // ── Fast-Path API ────────────────────────────────────────────────────

    /// The primary function called by the data loader on every token.
    ///
    /// 1. Checks bitpacked cache (shared_lock, O(1) hash lookup)
    /// 2. On miss: checks float vector store -> project -> bitpack -> cache
    /// 3. On double miss: BLAKE3 fallback (deterministic, non-semantic)
    ///
    /// Returns a bitpacked VSA vector ready for XOR binding with BLAKE3 roles.
    BitpackedVec get_semantic_filler(const std::string& token);

    // ── Loading / Pre-computation ────────────────────────────────────────

    /// Register a dense float embedding for a token.
    /// The projection is NOT done here — it's deferred to first access
    /// (lazy evaluation). This keeps vocabulary loading fast.
    void add_float_vector(const std::string& token, const float* vec);

    /// Batch-register float embeddings from a contiguous buffer.
    /// @param tokens   Vector of token strings
    /// @param vectors  Flat buffer: tokens.size() * ft_dim_ floats, row-major
    void add_float_vectors_batch(const std::vector<std::string>& tokens,
                                  const float* vectors);

    /// Pre-warm the cache by projecting all registered float vectors now.
    /// Useful for benchmarking or when you want deterministic first-query timing.
    /// Returns the number of entries projected and cached.
    size_t prewarm();

    /// Register a pre-computed bipolar filler directly (skips projection).
    void add_bipolar_filler(const std::string& token,
                             const std::vector<int8_t>& bipolar);

    /// Load pre-computed bipolar fillers from binary file.
    /// Format: [uint32 token_len][char[token_len]][int8[dim]] repeated.
    void load_fillers(const std::string& path);

#ifdef GRILLY_HAS_FASTTEXT
    /// Load a FastText model (e.g., cc.en.300.bin) for automatic OOV handling.
    /// When loaded, cache misses query FastText instead of falling back to BLAKE3.
    void load_fasttext_model(const std::string& path);
#endif

    // ── Raw Projection ───────────────────────────────────────────────────

    /// Project a dense float vector to bipolar {-1,+1} via LSH Gaussian matrix.
    /// This is the "slow path" — 300 * 10240 = 3M multiply-adds.
    std::vector<int8_t> project_to_bipolar(const float* vec) const;

    // ── Introspection ────────────────────────────────────────────────────

    size_t cache_size() const;
    size_t float_vocab_size() const;
    uint64_t cache_hits() const { return cache_hits_.load(std::memory_order_relaxed); }
    uint64_t cache_misses() const { return cache_misses_.load(std::memory_order_relaxed); }
    double hit_rate() const;
    void reset_stats();

    uint32_t dim() const { return dim_; }
    uint32_t ft_dim() const { return ft_dim_; }

private:
    uint32_t dim_;
    uint32_t ft_dim_;

    // LSH Gaussian Projection Matrix: R^{ft_dim} -> R^{dim}
    // Flat row-major: projection_[i * dim_ + j] for input dim i, output dim j
    // Deterministic seed 42 ensures reproducibility across sessions.
    std::vector<float> projection_;

    // ── Two-Level Cache Architecture ─────────────────────────────────────
    //
    // Level 1: Float vector store (dense embeddings, not yet projected)
    //          Populated by add_float_vector() / load_fasttext_model()
    //
    // Level 2: Bitpacked cache (projected + bitpacked, ready for GPU)
    //          Populated lazily on first access via get_semantic_filler()

    // Level 1: token -> dense float embedding (pre-projection)
    std::unordered_map<std::string, std::vector<float>> float_vectors_;
    mutable std::shared_mutex float_mutex_;

    // Level 2: token -> bitpacked VSA vector (post-projection, GPU-ready)
    std::unordered_map<std::string, BitpackedVec> bitpacked_cache_;
    mutable std::shared_mutex cache_mutex_;

    // Cache performance counters
    std::atomic<uint64_t> cache_hits_{0};
    std::atomic<uint64_t> cache_misses_{0};

    // Slow-path computation: project float vector -> bipolar -> bitpack
    BitpackedVec compute_and_cache(const std::string& token, const float* vec);

    // Fallback for tokens with no float vector and no FastText model
    BitpackedVec compute_fallback(const std::string& token);

#ifdef GRILLY_HAS_FASTTEXT
    fasttext::FastText ft_model_;
    bool ft_loaded_ = false;
    BitpackedVec compute_from_fasttext(const std::string& token);
#endif
};

}  // namespace cubemind
}  // namespace grilly
