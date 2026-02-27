#include "grilly/cubemind/semantic_assigner.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <random>

namespace grilly {
namespace cubemind {

SemanticAssigner::SemanticAssigner(uint32_t dim, uint32_t ft_dim)
    : dim_(dim), ft_dim_(ft_dim) {
    // Initialize the LSH Gaussian random projection matrix.
    //
    // By the Johnson-Lindenstrauss lemma, for M ~ N(0,1)^{d x D}:
    //   cos(a,b) ≈ 1 - 2*hamming(sign(Ma), sign(Mb))/D
    //
    // At D=10240, the approximation error is extremely small.
    // Seed 42 ensures the same projection across all sessions and machines.
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    projection_.resize(static_cast<size_t>(ft_dim_) * dim_);
    for (size_t i = 0; i < projection_.size(); ++i) {
        projection_[i] = dist(gen);
    }
}

// ── Fast-Path API ────────────────────────────────────────────────────────

BitpackedVec SemanticAssigner::get_semantic_filler(const std::string& token) {
    // FAST PATH: shared (read) lock on the bitpacked cache.
    // Multiple threads can blast through this simultaneously.
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        auto it = bitpacked_cache_.find(token);
        if (it != bitpacked_cache_.end()) {
            cache_hits_.fetch_add(1, std::memory_order_relaxed);
            return it->second;
        }
    }

    // Cache miss — we need to compute the projection.
    cache_misses_.fetch_add(1, std::memory_order_relaxed);

    // Check if we have a dense float vector registered for this token.
    {
        std::shared_lock<std::shared_mutex> flock(float_mutex_);
        auto it = float_vectors_.find(token);
        if (it != float_vectors_.end()) {
            return compute_and_cache(token, it->second.data());
        }
    }

#ifdef GRILLY_HAS_FASTTEXT
    // If FastText is loaded, query it for OOV handling via subword n-grams.
    if (ft_loaded_) {
        return compute_from_fasttext(token);
    }
#endif

    // Final fallback: deterministic BLAKE3 hash (non-semantic but consistent).
    return compute_fallback(token);
}

// ── Loading / Pre-computation ────────────────────────────────────────────

void SemanticAssigner::add_float_vector(const std::string& token,
                                         const float* vec) {
    std::vector<float> v(vec, vec + ft_dim_);
    std::unique_lock<std::shared_mutex> lock(float_mutex_);
    float_vectors_[token] = std::move(v);
}

void SemanticAssigner::add_float_vectors_batch(
    const std::vector<std::string>& tokens, const float* vectors) {
    std::unique_lock<std::shared_mutex> lock(float_mutex_);
    for (size_t i = 0; i < tokens.size(); ++i) {
        const float* row = vectors + i * ft_dim_;
        float_vectors_[tokens[i]] = std::vector<float>(row, row + ft_dim_);
    }
}

size_t SemanticAssigner::prewarm() {
    // Project all registered float vectors and populate the bitpacked cache.
    // This trades startup time for guaranteed zero-miss first queries.
    std::shared_lock<std::shared_mutex> flock(float_mutex_);
    size_t count = 0;

    for (const auto& [token, vec] : float_vectors_) {
        // Check if already cached
        {
            std::shared_lock<std::shared_mutex> clock(cache_mutex_);
            if (bitpacked_cache_.count(token) > 0) continue;
        }

        // Project and cache
        std::vector<int8_t> bipolar = project_to_bipolar(vec.data());
        BitpackedVec packed = vsaBitpack(bipolar.data(), dim_);

        {
            std::unique_lock<std::shared_mutex> clock(cache_mutex_);
            bitpacked_cache_[token] = std::move(packed);
        }
        ++count;
    }

    return count;
}

void SemanticAssigner::add_bipolar_filler(const std::string& token,
                                           const std::vector<int8_t>& bipolar) {
    BitpackedVec packed = vsaBitpack(bipolar.data(), dim_);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    bitpacked_cache_[token] = std::move(packed);
}

void SemanticAssigner::load_fillers(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open filler file: " + path);
    }

    // Same binary format as TextEncoder::load_fillers:
    //   [uint32 token_len][char[token_len]][int8[dim_]] repeated
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);

    while (file.good()) {
        uint32_t token_len = 0;
        file.read(reinterpret_cast<char*>(&token_len), sizeof(uint32_t));
        if (!file.good() || token_len == 0 || token_len > 1024) break;

        std::string token(token_len, '\0');
        file.read(token.data(), token_len);
        if (!file.good()) break;

        std::vector<int8_t> bipolar(dim_);
        file.read(reinterpret_cast<char*>(bipolar.data()), dim_);
        if (!file.good()) break;

        bitpacked_cache_[std::move(token)] = vsaBitpack(bipolar.data(), dim_);
    }
}

#ifdef GRILLY_HAS_FASTTEXT
void SemanticAssigner::load_fasttext_model(const std::string& path) {
    ft_model_.loadModel(path);
    ft_loaded_ = true;
}
#endif

// ── Raw Projection ───────────────────────────────────────────────────────

std::vector<int8_t> SemanticAssigner::project_to_bipolar(
    const float* vec) const {
    // LSH random projection: bipolar[j] = sign( sum_i(vec[i] * M[i*dim+j]) )
    //
    // This is a [1 x ft_dim] * [ft_dim x dim] matrix-vector multiply
    // followed by element-wise sign. Cost: ft_dim * dim multiply-adds.
    // For ft_dim=300, dim=10240: 3,072,000 FLOPs.
    std::vector<int8_t> bipolar(dim_);

    for (uint32_t j = 0; j < dim_; ++j) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < ft_dim_; ++i) {
            sum += vec[i] * projection_[static_cast<size_t>(i) * dim_ + j];
        }
        bipolar[j] = (sum > 0.0f) ? 1 : -1;
    }

    return bipolar;
}

// ── Introspection ────────────────────────────────────────────────────────

size_t SemanticAssigner::cache_size() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return bitpacked_cache_.size();
}

size_t SemanticAssigner::float_vocab_size() const {
    std::shared_lock<std::shared_mutex> lock(float_mutex_);
    return float_vectors_.size();
}

double SemanticAssigner::hit_rate() const {
    uint64_t h = cache_hits_.load(std::memory_order_relaxed);
    uint64_t m = cache_misses_.load(std::memory_order_relaxed);
    uint64_t total = h + m;
    return (total > 0) ? static_cast<double>(h) / total : 0.0;
}

void SemanticAssigner::reset_stats() {
    cache_hits_.store(0, std::memory_order_relaxed);
    cache_misses_.store(0, std::memory_order_relaxed);
}

// ── Private: Slow-Path Computation ───────────────────────────────────────

BitpackedVec SemanticAssigner::compute_and_cache(const std::string& token,
                                                  const float* vec) {
    // 1. Project 300D float -> 10240D bipolar (the expensive step)
    std::vector<int8_t> bipolar = project_to_bipolar(vec);

    // 2. Bitpack: 10240 int8 -> 320 uint32 (1.28 KB, GPU-ready)
    BitpackedVec packed = vsaBitpack(bipolar.data(), dim_);

    // 3. Cache the result (exclusive write lock)
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        // Double-check: another thread may have cached this while we computed
        auto it = bitpacked_cache_.find(token);
        if (it != bitpacked_cache_.end()) {
            return it->second;  // Someone beat us to it
        }
        bitpacked_cache_[token] = packed;
    }

    return packed;
}

BitpackedVec SemanticAssigner::compute_fallback(const std::string& token) {
    // Deterministic BLAKE3 hash for tokens with no float embedding.
    // Not semantically meaningful, but consistent across sessions.
    std::vector<int8_t> bipolar = blake3Role("filler_" + token, dim_);
    BitpackedVec packed = vsaBitpack(bipolar.data(), dim_);

    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        bitpacked_cache_[token] = packed;
    }

    return packed;
}

#ifdef GRILLY_HAS_FASTTEXT
BitpackedVec SemanticAssigner::compute_from_fasttext(const std::string& token) {
    // Ask FastText for the dense vector. FastText handles OOV words natively
    // via character n-gram decomposition — no <UNK> token needed.
    fasttext::Vector ft_vec(ft_model_.getDimension());
    ft_model_.getWordVector(ft_vec, token);

    return compute_and_cache(token, ft_vec.data());
}
#endif

}  // namespace cubemind
}  // namespace grilly
