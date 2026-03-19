#include "grilly/cubemind/block_ops.h"
#include "grilly/cubemind/tensor_ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>

namespace grilly {
namespace cubemind {

// ── Per-Block Circular Convolution (Binding) ─────────────────────────
//
// Circular convolution of two length-l sequences a, b:
//   c[i] = sum_{m=0}^{l-1} a[m] * b[(i - m) mod l]
//
// This is the binding operation for block codes (NVSA). Applied
// independently to each of the k blocks, it preserves the L1 norm
// per block when inputs are on the probability simplex.
//
// For one-hot inputs (discrete block codes), circular convolution
// reduces to a circular shift: if a = e_p and b = e_q, then
// c = e_{(p+q) mod l}. This is why successive binding produces
// algebraically closed codebooks (cyclic group Z_l).
//
// Complexity: O(k * l^2). For typical parameters (k=16, l=128),
// the inner loop is 128*128 = 16K multiply-adds per block, all
// fitting comfortably in L1 cache.

std::vector<float> blockCodeBind(const float* a, const float* b,
                                  uint32_t k, uint32_t l) {
    if (k == 0 || l == 0)
        throw std::invalid_argument("blockCodeBind: k and l must be > 0");

    const uint32_t d = k * l;
    std::vector<float> result(d, 0.0f);

    for (uint32_t j = 0; j < k; ++j) {
        const float* aj = a + j * l;
        const float* bj = b + j * l;
        float* cj = result.data() + j * l;

        // Direct circular convolution: c[i] = sum_m a[m] * b[(i-m) mod l]
        for (uint32_t i = 0; i < l; ++i) {
            float sum = 0.0f;
            for (uint32_t m = 0; m < l; ++m) {
                // (i - m) mod l, handling unsigned underflow
                uint32_t idx = (i >= m) ? (i - m) : (l - m + i);
                sum += aj[m] * bj[idx];
            }
            cj[i] = sum;
        }
    }

    return result;
}

// ── Per-Block Circular Correlation (Unbinding) ───────────────────────
//
// Circular correlation is the inverse of circular convolution.
// It is equivalent to convolving with the time-reversed signal:
//   unbind(composite, known)[i] = sum_m composite[m] * known[(m - i) mod l]
//
// Equivalently: reverse known to get known_rev[i] = known[(l-i) mod l],
// then convolve composite with known_rev.
//
// For one-hot codes: if composite = e_{(p+q) mod l} and known = e_q,
// then unbind = e_p. This recovers the original position exactly.

std::vector<float> blockCodeUnbind(const float* composite, const float* known,
                                    uint32_t k, uint32_t l) {
    if (k == 0 || l == 0)
        throw std::invalid_argument("blockCodeUnbind: k and l must be > 0");

    const uint32_t d = k * l;
    std::vector<float> result(d, 0.0f);

    for (uint32_t j = 0; j < k; ++j) {
        const float* cj = composite + j * l;
        const float* kj = known + j * l;
        float* rj = result.data() + j * l;

        // Circular correlation: r[i] = sum_m composite[m] * known[(m - i) mod l]
        // Equivalently: convolve composite with reversed known.
        // known_rev[i] = known[(l - i) mod l], then r = conv(composite, known_rev)
        // But we compute it directly to avoid an extra buffer.
        for (uint32_t i = 0; i < l; ++i) {
            float sum = 0.0f;
            for (uint32_t m = 0; m < l; ++m) {
                // (m - i) mod l
                uint32_t idx = (m >= i) ? (m - i) : (l - i + m);
                sum += cj[m] * kj[idx];
            }
            rj[i] = sum;
        }
    }

    return result;
}

// ── Block-Code Bundling ──────────────────────────────────────────────
//
// Bundling creates a superposition of multiple block-code vectors
// via element-wise summation. Unlike bipolar bundling (majority vote),
// block-code bundling produces a continuous vector that can represent
// weighted mixtures.
//
// Optional L1 normalization restores each block to the probability
// simplex (entries sum to 1), which is important for maintaining
// the statistical properties of similarity computation.

std::vector<float> blockCodeBundle(const std::vector<const float*>& vectors,
                                    uint32_t k, uint32_t l,
                                    bool normalize) {
    if (vectors.empty())
        throw std::invalid_argument("blockCodeBundle: empty vector list");
    if (k == 0 || l == 0)
        throw std::invalid_argument("blockCodeBundle: k and l must be > 0");

    const uint32_t d = k * l;
    std::vector<float> result(d, 0.0f);

    // Element-wise sum across all input vectors
    for (const float* v : vectors) {
        for (uint32_t i = 0; i < d; ++i) {
            result[i] += v[i];
        }
    }

    // Optional per-block L1 normalization
    if (normalize) {
        for (uint32_t j = 0; j < k; ++j) {
            float* bj = result.data() + j * l;

            // Compute L1 norm of this block
            float l1 = 0.0f;
            for (uint32_t i = 0; i < l; ++i) {
                l1 += std::fabs(bj[i]);
            }

            // Normalize (guard against zero-norm blocks)
            if (l1 > 1e-12f) {
                float inv = 1.0f / l1;
                for (uint32_t i = 0; i < l; ++i) {
                    bj[i] *= inv;
                }
            }
        }
    }

    return result;
}

// ── Block-Code Similarity (Batch) ────────────────────────────────────
//
// Computes the average per-block dot product between a query and each
// entry in a codebook:
//   sim(query, entry) = (1/k) * sum_{j=0}^{k-1} dot(query_j, entry_j)
//
// For one-hot discrete codes, dot(query_j, entry_j) is 1 if the hot
// positions match, 0 otherwise. So sim = (matching blocks) / k.
//
// The loop order is codebook-entry-major for cache friendliness:
// each entry is k*l contiguous floats, streamed once.

std::vector<float> blockCodeSimilarityBatch(const float* query,
                                             const float* codebook,
                                             uint32_t n, uint32_t k, uint32_t l) {
    if (k == 0 || l == 0)
        throw std::invalid_argument("blockCodeSimilarityBatch: k and l must be > 0");
    if (n == 0) return {};

    const uint32_t d = k * l;
    const float invK = 1.0f / static_cast<float>(k);
    std::vector<float> sims(n);

    for (uint32_t e = 0; e < n; ++e) {
        const float* entry = codebook + static_cast<size_t>(e) * d;
        float total = 0.0f;

        // Sum dot products across all k blocks
        for (uint32_t j = 0; j < k; ++j) {
            const float* qj = query + j * l;
            const float* ej = entry + j * l;
            float dot = 0.0f;
            for (uint32_t i = 0; i < l; ++i) {
                dot += qj[i] * ej[i];
            }
            total += dot;
        }

        sims[e] = total * invK;
    }

    return sims;
}

// ── Codebook Generation via Successive Binding ───────────────────────
//
// Generates n algebraically closed block-code vectors using the
// IBM NVSA method:
//   1. Create an identity vector: position 0 hot in every block.
//   2. Create a seed vector: random non-zero one-hot position per block.
//   3. entry[0] = identity
//   4. entry[i+1] = bind(entry[i], seed)
//
// Because circular convolution of one-hot vectors is a circular shift,
// and the seed picks a fixed shift per block, the resulting codes are
// elements of the cyclic group Z_l in each block. Collision-free as
// long as n <= l (the shift wraps at l).
//
// The PRNG uses a simple LCG (Linear Congruential Generator) for
// deterministic, portable generation across platforms.

/// Simple LCG PRNG matching the Numerical Recipes constants.
/// Period 2^32, sufficient for codebook generation.
static inline uint32_t lcgNext(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

std::vector<float> blockCodeCodebook(uint32_t k, uint32_t l, uint32_t n,
                                      uint32_t seed) {
    if (k == 0 || l == 0)
        throw std::invalid_argument("blockCodeCodebook: k and l must be > 0");
    if (n == 0) return {};
    if (n > l)
        throw std::invalid_argument(
            "blockCodeCodebook: n must be <= l to avoid collisions");

    const uint32_t d = k * l;

    // Generate seed vector: one-hot with random non-zero position per block
    std::vector<float> seedVec(d, 0.0f);
    uint32_t rngState = seed;
    for (uint32_t j = 0; j < k; ++j) {
        // Pick a random position in [1, l-1] to ensure it's not the identity shift
        uint32_t pos = (lcgNext(rngState) % (l - 1)) + 1;
        seedVec[j * l + pos] = 1.0f;
    }

    // Allocate codebook: n entries, each k*l floats
    std::vector<float> codebook(static_cast<size_t>(n) * d, 0.0f);

    // Entry 0 = identity: position 0 hot in every block
    for (uint32_t j = 0; j < k; ++j) {
        codebook[j * l] = 1.0f;
    }

    // Entry i+1 = bind(entry_i, seed)
    // For one-hot codes, binding = circular shift, so we can optimize:
    // just track the hot position per block and shift it.
    // But for generality and correctness, we use the full bind operation.
    for (uint32_t i = 1; i < n; ++i) {
        const float* prev = codebook.data() + static_cast<size_t>(i - 1) * d;
        std::vector<float> bound = blockCodeBind(prev, seedVec.data(), k, l);
        std::memcpy(codebook.data() + static_cast<size_t>(i) * d,
                    bound.data(), d * sizeof(float));
    }

    return codebook;
}

// ── Discretization (Argmax per Block) ────────────────────────────────
//
// Converts a continuous block-code vector to discrete one-hot form
// by taking the argmax within each block. This is the quantization
// step used after bundling or unbinding to recover a clean code.
//
// Ties are broken by picking the first (lowest index) maximum.

std::vector<float> blockCodeDiscretize(const float* v, uint32_t k, uint32_t l) {
    if (k == 0 || l == 0)
        throw std::invalid_argument("blockCodeDiscretize: k and l must be > 0");

    const uint32_t d = k * l;
    std::vector<float> result(d, 0.0f);

    for (uint32_t j = 0; j < k; ++j) {
        const float* bj = v + j * l;

        // Find argmax within this block
        uint32_t maxIdx = 0;
        float maxVal = bj[0];
        for (uint32_t i = 1; i < l; ++i) {
            if (bj[i] > maxVal) {
                maxVal = bj[i];
                maxIdx = i;
            }
        }

        result[j * l + maxIdx] = 1.0f;
    }

    return result;
}

// ── Softmax: Similarity to Probability Mass Function ─────────────────
//
// Converts raw similarity scores to a proper probability distribution
// using the softmax function with a temperature parameter:
//   pmf_i = exp(T * sim_i) / sum_j exp(T * sim_j)
//
// The temperature T controls sharpness:
//   T -> inf: argmax (greedy decoding)
//   T = 1:    standard softmax
//   T -> 0:   uniform distribution
//
// For numerical stability, we subtract the maximum value before
// exponentiating (log-sum-exp trick).

std::vector<float> blockCodeCosineTopmf(const float* similarities, uint32_t n,
                                         float temperature) {
    if (n == 0) return {};

    std::vector<float> pmf(n);

    // Find max for numerical stability
    float maxSim = similarities[0];
    for (uint32_t i = 1; i < n; ++i) {
        if (similarities[i] > maxSim) maxSim = similarities[i];
    }

    // Compute exp(temperature * (sim_i - max)) and accumulate sum
    float sumExp = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        pmf[i] = std::exp(temperature * (similarities[i] - maxSim));
        sumExp += pmf[i];
    }

    // Normalize to probability distribution
    if (sumExp > 0.0f) {
        float invSum = 1.0f / sumExp;
        for (uint32_t i = 0; i < n; ++i) {
            pmf[i] *= invSum;
        }
    }

    return pmf;
}

// ── FFT-Accelerated Binding (Circular Convolution) ──────────────────────
//
// Convolution Theorem: conv(a, b) = IFFT(FFT(a) ⊙ FFT(b))
// where ⊙ is complex element-wise multiplication.
// Complexity: O(k * l * log(l)) vs O(k * l^2) for direct.

std::vector<float> blockCodeBindFFT(const float* a, const float* b,
                                     uint32_t k, uint32_t l) {
    if (k == 0 || l == 0)
        throw std::invalid_argument("blockCodeBindFFT: k and l must be > 0");

    const uint32_t d = k * l;
    std::vector<float> result(d, 0.0f);

    // Temp buffers for FFT
    std::vector<float> a_re(l), a_im(l);
    std::vector<float> b_re(l), b_im(l);
    std::vector<float> c_re(l), c_im(l);

    for (uint32_t j = 0; j < k; ++j) {
        const float* aj = a + j * l;
        const float* bj = b + j * l;
        float* cj = result.data() + j * l;

        // Forward FFT both blocks
        fft_1d(aj, a_re.data(), a_im.data(), l);
        fft_1d(bj, b_re.data(), b_im.data(), l);

        // Complex multiply: C = A * B
        for (uint32_t i = 0; i < l; ++i) {
            c_re[i] = a_re[i] * b_re[i] - a_im[i] * b_im[i];
            c_im[i] = a_re[i] * b_im[i] + a_im[i] * b_re[i];
        }

        // Inverse FFT → result block
        ifft_1d(c_re.data(), c_im.data(), cj, l);
    }

    return result;
}

// ── FFT-Accelerated Unbinding (Circular Correlation) ────────────────────
//
// Correlation Theorem: corr(a, b) = IFFT(FFT(a) ⊙ conj(FFT(b)))
// where conj flips the imaginary part.

std::vector<float> blockCodeUnbindFFT(const float* composite, const float* known,
                                       uint32_t k, uint32_t l) {
    if (k == 0 || l == 0)
        throw std::invalid_argument("blockCodeUnbindFFT: k and l must be > 0");

    const uint32_t d = k * l;
    std::vector<float> result(d, 0.0f);

    std::vector<float> c_re(l), c_im(l);
    std::vector<float> k_re(l), k_im(l);
    std::vector<float> r_re(l), r_im(l);

    for (uint32_t j = 0; j < k; ++j) {
        const float* cj = composite + j * l;
        const float* kj = known + j * l;
        float* rj = result.data() + j * l;

        // Forward FFT both blocks
        fft_1d(cj, c_re.data(), c_im.data(), l);
        fft_1d(kj, k_re.data(), k_im.data(), l);

        // Complex conjugate of known (negate imaginary)
        for (uint32_t i = 0; i < l; ++i)
            k_im[i] = -k_im[i];

        // Complex multiply: R = C * conj(K)
        for (uint32_t i = 0; i < l; ++i) {
            r_re[i] = c_re[i] * k_re[i] - c_im[i] * k_im[i];
            r_im[i] = c_re[i] * k_im[i] + c_im[i] * k_re[i];
        }

        // Inverse FFT → result block
        ifft_1d(r_re.data(), r_im.data(), rj, l);
    }

    return result;
}

// ── Block-Wise Softmax (Perception Bridge) ──────────────────────────────
//
// Applies softmax independently to each of k blocks of length l.
// Each block becomes a valid probability simplex (sums to 1.0).
// This is the neural-to-VSA bridge for the PerceptionBridge.

std::vector<float> blockSoftmax(const float* logits, uint32_t k, uint32_t l) {
    if (k == 0 || l == 0)
        throw std::invalid_argument("blockSoftmax: k and l must be > 0");

    std::vector<float> probs(k * l, 0.0f);

    for (uint32_t j = 0; j < k; ++j) {
        const float* block_in = logits + j * l;
        float* block_out = probs.data() + j * l;

        // Max for numerical stability
        float max_val = block_in[0];
        for (uint32_t i = 1; i < l; ++i)
            max_val = std::max(max_val, block_in[i]);

        // exp(x - max) and sum
        float sum_exp = 0.0f;
        for (uint32_t i = 0; i < l; ++i) {
            block_out[i] = std::exp(block_in[i] - max_val);
            sum_exp += block_out[i];
        }

        // Normalize
        float inv_sum = 1.0f / sum_exp;
        for (uint32_t i = 0; i < l; ++i)
            block_out[i] *= inv_sum;
    }

    return probs;
}

}  // namespace cubemind
}  // namespace grilly
