#pragma once

#include <cstdint>
#include <vector>

namespace grilly {
namespace cubemind {

// ── Block-Code VSA Operations ────────────────────────────────────────
//
// Sparse block codes represent hypervectors as k blocks of length l,
// where each block is typically one-hot (discrete) or a probability
// simplex (continuous). This is the NVSA (Neural VSA) representation
// from Hersche et al. (IBM Research).
//
// Key properties:
//   - Binding: per-block circular convolution (preserves L1 norm)
//   - Unbinding: per-block circular correlation (inverse of binding)
//   - Bundling: element-wise sum with optional L1 normalization
//   - Similarity: average dot product across blocks
//
// Memory layout: k * l floats, row-major. Block j occupies
// indices [j*l, (j+1)*l). Total dimensionality d = k * l.
//
// Typical parameters: k=16, l=128 (d=2048) or k=64, l=256 (d=16384).

/// Per-block circular convolution binding for sparse block codes.
/// Each block is independently convolved: c_j[i] = sum_m a_j[m] * b_j[(i-m) mod l]
/// This preserves L1 norm per block (Theorem 1 in the CubeMind papers).
///
/// Complexity: O(k * l^2). For k=16, l=128 this is ~262K multiply-adds,
/// trivially fitting in L1 cache and completing in <0.1ms on any modern CPU.
///
/// @param a  First block-code vector (k * l floats, row-major k blocks of l)
/// @param b  Second block-code vector (k * l floats)
/// @param k  Number of blocks
/// @param l  Block length
/// @return Bound vector (k * l floats)
std::vector<float> blockCodeBind(const float* a, const float* b,
                                  uint32_t k, uint32_t l);

/// Per-block circular correlation (unbinding).
/// Inverse of blockCodeBind: unbind(bind(a, b), b) ~ a
/// Implemented as convolution with time-reversed b:
///   b_rev[i] = b[(l - i) mod l]
///
/// @param composite Bound vector to decompose (k * l floats)
/// @param known     Known component to unbind (k * l floats)
/// @param k         Number of blocks
/// @param l         Block length
/// @return Unbound vector (k * l floats)
std::vector<float> blockCodeUnbind(const float* composite, const float* known,
                                    uint32_t k, uint32_t l);

/// Bundle multiple block-code vectors via element-wise sum.
/// Optionally L1-normalize each block so entries sum to 1,
/// restoring the probability simplex interpretation.
///
/// @param vectors   Vector of pointers to block-code vectors (each k * l floats)
/// @param k         Number of blocks
/// @param l         Block length
/// @param normalize If true, L1-normalize each block after summing
/// @return Bundled vector (k * l floats)
std::vector<float> blockCodeBundle(const std::vector<const float*>& vectors,
                                    uint32_t k, uint32_t l,
                                    bool normalize = true);

/// Compute similarity between a query and all codebook entries.
/// Similarity = (1/k) * sum_{j=0}^{k-1} dot(query_j, entry_j)
/// where query_j and entry_j are the j-th blocks.
///
/// For one-hot discrete codes, this reduces to (1/k) * number of matching blocks.
///
/// @param query     Query vector (k * l floats)
/// @param codebook  Codebook matrix (n * k * l floats, row-major)
/// @param n         Number of codebook entries
/// @param k         Number of blocks
/// @param l         Block length
/// @return Similarities (n floats), each in [0, 1] for one-hot codes
std::vector<float> blockCodeSimilarityBatch(const float* query,
                                             const float* codebook,
                                             uint32_t n, uint32_t k, uint32_t l);

/// Generate a codebook of n discrete block-code vectors via successive binding.
/// Entry 0 = identity (position-0 hot per block).
/// Entry i+1 = bind(entry_i, seed_vector).
/// This produces algebraically closed, collision-free codes (IBM NVSA method).
///
/// The seed vector has a random (non-zero) one-hot position per block,
/// generated from a deterministic PRNG seeded with the given seed.
///
/// @param k    Number of blocks
/// @param l    Block length
/// @param n    Number of codebook entries to generate (must be <= l)
/// @param seed Random seed for the seed vector generation
/// @return Codebook (n * k * l floats, row-major)
std::vector<float> blockCodeCodebook(uint32_t k, uint32_t l, uint32_t n,
                                      uint32_t seed = 42);

/// Discretize a continuous block-code to one-hot (argmax per block).
/// For each block j, find i* = argmax_i v[j*l + i], then set
/// result[j*l + i*] = 1 and all others to 0.
///
/// @param v  Continuous block-code vector (k * l floats)
/// @param k  Number of blocks
/// @param l  Block length
/// @return Discretized one-hot vector (k * l floats)
std::vector<float> blockCodeDiscretize(const float* v, uint32_t k, uint32_t l);

/// Convert similarity scores to a probability mass function via softmax.
/// pmf_i = exp(temperature * sim_i) / sum_j exp(temperature * sim_j)
///
/// Higher temperature sharpens the distribution (more peaked).
/// At temperature -> infinity, this becomes argmax (greedy).
/// At temperature -> 0, this becomes uniform.
///
/// @param similarities  Similarity scores (n floats)
/// @param n             Number of entries
/// @param temperature   Softmax temperature (default 40.0)
/// @return Probability mass function (n floats summing to 1)
std::vector<float> blockCodeCosineTopmf(const float* similarities, uint32_t n,
                                         float temperature = 40.0f);

// ── FFT-Accelerated Binding/Unbinding ────────────────────────────────
//
// Uses the Convolution Theorem: conv(a, b) = IFFT(FFT(a) * FFT(b))
// and correlation: corr(a, b) = IFFT(FFT(a) * conj(FFT(b)))
//
// Complexity: O(k * l * log(l)) vs O(k * l^2) for direct methods.
// For k=16, l=128: FFT = ~18K ops vs direct = ~262K ops (14x speedup).

/// FFT-accelerated per-block circular convolution binding.
/// Uses tensor_ops::fft_1d / ifft_1d for O(k * l * log(l)) complexity.
std::vector<float> blockCodeBindFFT(const float* a, const float* b,
                                     uint32_t k, uint32_t l);

/// FFT-accelerated per-block circular correlation unbinding.
std::vector<float> blockCodeUnbindFFT(const float* composite, const float* known,
                                       uint32_t k, uint32_t l);

// ── Block-Wise Softmax (Perception Bridge) ──────────────────────────
//
// Applies softmax independently to each of the k blocks, constraining
// the output to k independent probability simplexes. This is the
// neural-to-VSA bridge: CNN logits → block-wise softmax → valid block-code.

/// Apply softmax independently to each of k blocks of length l.
/// Output: k blocks where each block sums to 1.0.
///
/// @param logits  Raw neural network output (k * l floats)
/// @param k       Number of blocks
/// @param l       Block length
/// @return Probability simplexes (k * l floats, each block sums to 1)
std::vector<float> blockSoftmax(const float* logits, uint32_t k, uint32_t l);

// ── GPU-Accelerated FFT Bind/Unbind (Vulkan Compute) ────────────────
//
// Dispatches the fft-bind.glsl shader: entire FFT + complex multiply +
// IFFT pipeline runs in GPU shared memory. One workgroup per block.
// Requires CommandBatch/BufferPool/PipelineCache (same pattern as linear ops).

struct FFTBindParams {
    uint32_t k;
    uint32_t l;
    uint32_t is_unbind;
};

}  // namespace cubemind
}  // namespace grilly
