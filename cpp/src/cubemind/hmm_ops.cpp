// HMM forward-backward and Baum-Welch operations for CubeMind.
//
// All computations in log-space for numerical stability.
// The forward algorithm is: log_alpha[t][j] = log_B[t][j] + logsumexp_i(log_alpha[t-1][i] + log_A[i][j])
// The backward algorithm is: log_beta[t][i] = logsumexp_j(log_A[i][j] + log_B[t+1][j] + log_beta[t+1][j])
//
// These are pure CPU implementations. For GPU dispatch, wrap the inner
// matmul (A^T @ alpha) as a Vulkan compute shader call.

#include "grilly/cubemind/hmm_ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace grilly {
namespace cubemind {

namespace {

/// Numerically stable log-sum-exp over a contiguous array of n values.
float logsumexp(const float* vals, uint32_t n) {
    float max_val = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < n; ++i) {
        if (vals[i] > max_val) max_val = vals[i];
    }
    if (std::isinf(max_val) && max_val < 0) return max_val;

    float sum = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        sum += std::exp(vals[i] - max_val);
    }
    return max_val + std::log(sum);
}

/// Log-sum-exp of (a[i] + b[i]) for i in [0, n).
float logsumexp_add(const float* a, const float* b, uint32_t n) {
    float max_val = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < n; ++i) {
        float v = a[i] + b[i];
        if (v > max_val) max_val = v;
    }
    if (std::isinf(max_val) && max_val < 0) return max_val;

    float sum = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        sum += std::exp(a[i] + b[i] - max_val);
    }
    return max_val + std::log(sum);
}

}  // namespace

// ── Forward algorithm ─────────────────────────────────────────────────

HMMForwardBackwardResult hmmForward(const float* log_A, const float* log_pi,
                                     const float* log_B,
                                     uint32_t T, uint32_t n) {
    if (T == 0 || n == 0)
        throw std::invalid_argument("hmmForward: T and n must be > 0");

    HMMForwardBackwardResult result;
    result.T = T;
    result.n = n;
    result.log_alpha.resize(T * n);

    // t = 0: log_alpha[0][i] = log_pi[i] + log_B[0][i]
    for (uint32_t i = 0; i < n; ++i) {
        result.log_alpha[i] = log_pi[i] + log_B[i];
    }

    // Temp buffer for the column of log_A[:, j] + log_alpha[t-1][:]
    std::vector<float> tmp(n);

    // t = 1 .. T-1
    for (uint32_t t = 1; t < T; ++t) {
        const float* prev_alpha = &result.log_alpha[(t - 1) * n];
        float* curr_alpha = &result.log_alpha[t * n];

        for (uint32_t j = 0; j < n; ++j) {
            // tmp[i] = log_alpha[t-1][i] + log_A[i][j]
            for (uint32_t i = 0; i < n; ++i) {
                tmp[i] = prev_alpha[i] + log_A[i * n + j];
            }
            curr_alpha[j] = log_B[t * n + j] + logsumexp(tmp.data(), n);
        }
    }

    // Log-likelihood = logsumexp(log_alpha[T-1][:])
    result.log_likelihood = logsumexp(&result.log_alpha[(T - 1) * n], n);

    return result;
}

// ── Forward-backward algorithm ────────────────────────────────────────

HMMForwardBackwardResult hmmForwardBackward(const float* log_A, const float* log_pi,
                                             const float* log_B,
                                             uint32_t T, uint32_t n) {
    // Forward pass
    auto result = hmmForward(log_A, log_pi, log_B, T, n);

    // Backward pass
    result.log_beta.resize(T * n);
    std::vector<float> tmp(n);

    // t = T-1: log_beta[T-1][i] = 0
    for (uint32_t i = 0; i < n; ++i) {
        result.log_beta[(T - 1) * n + i] = 0.0f;
    }

    for (uint32_t t_rev = 1; t_rev < T; ++t_rev) {
        uint32_t t = T - 1 - t_rev;
        const float* next_beta = &result.log_beta[(t + 1) * n];
        float* curr_beta = &result.log_beta[t * n];

        for (uint32_t i = 0; i < n; ++i) {
            // tmp[j] = log_A[i][j] + log_B[t+1][j] + log_beta[t+1][j]
            for (uint32_t j = 0; j < n; ++j) {
                tmp[j] = log_A[i * n + j] + log_B[(t + 1) * n + j] + next_beta[j];
            }
            curr_beta[i] = logsumexp(tmp.data(), n);
        }
    }

    // Gamma: log_gamma[t][i] = log_alpha[t][i] + log_beta[t][i] - log P(O)
    result.log_gamma.resize(T * n);
    for (uint32_t t = 0; t < T; ++t) {
        // Compute normalizer for this timestep
        std::vector<float> ab(n);
        for (uint32_t i = 0; i < n; ++i) {
            ab[i] = result.log_alpha[t * n + i] + result.log_beta[t * n + i];
        }
        float log_norm = logsumexp(ab.data(), n);
        for (uint32_t i = 0; i < n; ++i) {
            result.log_gamma[t * n + i] = ab[i] - log_norm;
        }
    }

    return result;
}

// ── Baum-Welch batch E-step + M-step accumulation ─────────────────────

HMMBaumWelchResult hmmBaumWelchBatch(const float* log_A, const float* log_pi,
                                      const float* batch_log_B,
                                      const uint32_t* seq_lengths,
                                      uint32_t n_seqs, uint32_t n,
                                      float smoothing) {
    HMMBaumWelchResult result;
    result.n = n;
    result.A_num.assign(n * n, smoothing);
    result.pi_num.assign(n, smoothing);
    result.avg_log_likelihood = 0.0f;

    uint32_t offset = 0;  // Offset into batch_log_B

    for (uint32_t s = 0; s < n_seqs; ++s) {
        uint32_t T = seq_lengths[s];
        if (T < 2) {
            offset += T * n;
            continue;
        }

        const float* log_B = &batch_log_B[offset];

        // Forward-backward for this sequence
        auto fb = hmmForwardBackward(log_A, log_pi, log_B, T, n);

        result.avg_log_likelihood += fb.log_likelihood;

        // Accumulate initial state counts from gamma[0]
        for (uint32_t i = 0; i < n; ++i) {
            result.pi_num[i] += std::exp(fb.log_gamma[i]);
        }

        // Accumulate transition counts from xi
        // xi[t][i][j] = alpha[t][i] + log_A[i][j] + log_B[t+1][j] + beta[t+1][j] - log_likelihood
        for (uint32_t t = 0; t < T - 1; ++t) {
            for (uint32_t i = 0; i < n; ++i) {
                for (uint32_t j = 0; j < n; ++j) {
                    float log_xi = fb.log_alpha[t * n + i]
                                 + log_A[i * n + j]
                                 + log_B[(t + 1) * n + j]
                                 + fb.log_beta[(t + 1) * n + j]
                                 - fb.log_likelihood;
                    result.A_num[i * n + j] += std::exp(log_xi);
                }
            }
        }

        offset += T * n;
    }

    if (n_seqs > 0) {
        result.avg_log_likelihood /= static_cast<float>(n_seqs);
    }

    return result;
}

}  // namespace cubemind
}  // namespace grilly
