#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "grilly/cubemind/types.h"
#include "grilly/cubemind/vsa.h"

namespace grilly {
namespace cubemind {

// ── TextEncoder: Fast Bipolar Sentence Encoding ─────────────────────────
//
// Replaces the Python InstantLanguage encoder which used:
//   - Character n-gram loops for word vector generation
//   - HolographicOps.convolve (FFT→modify→IFFT) for binding
//   - Heuristic POS tagger (_auto_assign_roles)
//
// All of those are CPU killers at LLM pretraining scale.
//
// This C++ encoder uses:
//   - Pre-computed semantic fillers (quantized FastText via LSH projection)
//   - BLAKE3 deterministic role/position vectors (no heuristic tagging)
//   - vsaBind (element-wise multiply on int8) instead of circular convolution
//   - vsaBundle (majority vote + sign snap) for sentence composition
//   - vsaBitpack for Vulkan Hamming search readiness
//
// The pipeline:
//   1. Token → look up int8 bipolar filler from semantic_fillers_ map
//   2. Role string → BLAKE3 hash → bipolar role vector
//   3. Position index → BLAKE3 hash → bipolar position vector
//   4. Bind: word ⊗ role ⊗ position (element-wise multiply, stays in {-1,+1})
//   5. Bundle: majority vote over all bound components → sentence vector
//   6. Bitpack: {-1,+1} → packed uint32 bits for GPU Hamming search
//
// ── Locality Sensitive Hashing (LSH) for FastText ───────────────────────
//
// Standard FastText outputs 300D continuous float vectors. CubeMind needs
// 10240D strict bipolar {-1,+1}. The bridge is a random projection matrix:
//
//   M ∈ R^{300 × 10240}, drawn from N(0,1) with seed 42
//   bipolar[j] = sign( sum_i( ft_vec[i] * M[i][j] ) )
//
// By the Johnson-Lindenstrauss lemma, this preserves cosine similarity
// as Hamming distance: tokens that are semantically close in FastText's
// 300D space remain close in the 10240D bipolar space.
//
// The projection matrix is deterministic (fixed seed) so the same FastText
// model always produces the same bipolar vectors. It can be pre-computed
// once and serialized to disk.

class TextEncoder {
public:
    /// @param dim  VSA hypervector dimension (default 10240)
    /// @param ft_dim  FastText source dimension (default 300)
    explicit TextEncoder(uint32_t dim = 10240, uint32_t ft_dim = 300);

    /// Encode a pre-tokenized sentence into a bitpacked VSA vector.
    ///
    /// @param tokens           Subword tokens (from BPE/tiktoken)
    /// @param dependency_roles Dependency parse roles (from spaCy offline)
    /// @param positions        Token positions in the sentence
    /// @return Bitpacked sentence vector ready for Hamming search
    BitpackedVec encode_sentence(
        const std::vector<std::string>& tokens,
        const std::vector<std::string>& dependency_roles,
        const std::vector<uint32_t>& positions);

    /// Load pre-quantized bipolar fillers from a binary file.
    /// File format: for each entry:
    ///   uint32_t token_len, char[token_len], int8_t[dim]
    /// These are pre-projected from FastText via the LSH random projection.
    void load_fillers(const std::string& path);

    /// Project a single FastText float vector to bipolar via LSH.
    /// Used during offline preprocessing to build the filler vocabulary.
    ///
    /// @param ft_vec  Float vector of length ft_dim (e.g., 300)
    /// @return Bipolar vector of length dim with values in {-1,+1}
    std::vector<int8_t> project_to_bipolar(const float* ft_vec) const;

    /// Add a single token → bipolar filler mapping.
    void add_filler(const std::string& token, const std::vector<int8_t>& bipolar);

    /// Check if a token has a semantic filler loaded.
    bool has_filler(const std::string& token) const;

    /// Number of loaded semantic fillers.
    size_t vocab_size() const { return semantic_fillers_.size(); }

    uint32_t dim() const { return dim_; }
    uint32_t ft_dim() const { return ft_dim_; }

private:
    uint32_t dim_;
    uint32_t ft_dim_;

    // Semantic Memory: token → strict bipolar {-1, +1} vector of length dim_
    // Replaces the float32 word_vectors dictionary from Python.
    std::unordered_map<std::string, std::vector<int8_t>> semantic_fillers_;

    // LSH Random Projection Matrix: ft_dim_ × dim_
    // Stored as flat row-major: projection_[i * dim_ + j]
    // Generated deterministically from seed 42.
    std::vector<float> projection_;

    // Generate a fallback filler for unknown tokens via BLAKE3 hashing.
    // This gives unknown tokens a random-but-deterministic representation.
    std::vector<int8_t> fallback_filler(const std::string& token) const;
};

}  // namespace cubemind
}  // namespace grilly
