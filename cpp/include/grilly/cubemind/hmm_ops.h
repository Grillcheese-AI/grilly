#pragma once

#include <cstdint>
#include <vector>

namespace grilly {
namespace cubemind {

/// Result of the HMM forward-backward algorithm.
struct HMMForwardBackwardResult {
    /// Log-alpha matrix (T * n floats, row-major: T rows of n states).
    std::vector<float> log_alpha;
    /// Log-beta matrix (T * n floats, row-major).
    std::vector<float> log_beta;
    /// Log-gamma matrix (T * n floats, row-major). P(s_t = i | O).
    std::vector<float> log_gamma;
    /// Sequence log-likelihood: log P(O | lambda).
    float log_likelihood;
    /// Dimensions.
    uint32_t T;
    uint32_t n;
};

/// Result of Baum-Welch EM M-step accumulators.
struct HMMBaumWelchResult {
    /// Accumulated transition counts A_num (n * n floats, row-major).
    std::vector<float> A_num;
    /// Accumulated initial state counts pi_num (n floats).
    std::vector<float> pi_num;
    /// Average log-likelihood across the batch.
    float avg_log_likelihood;
    uint32_t n;
};

/// Run the HMM forward algorithm in log-space.
///
/// @param log_A     Log transition matrix (n * n, row-major). log_A[i*n+j] = log A[i][j].
/// @param log_pi    Log initial distribution (n,).
/// @param log_B     Log emission matrix (T * n, row-major). log_B[t*n+i] = log P(o_t | s_i).
/// @param T         Sequence length.
/// @param n         Number of hidden states.
/// @return Log-alpha matrix (T * n) and log-likelihood.
HMMForwardBackwardResult hmmForward(const float* log_A, const float* log_pi,
                                     const float* log_B,
                                     uint32_t T, uint32_t n);

/// Run the HMM backward algorithm in log-space.
///
/// @param log_A     Log transition matrix (n * n, row-major).
/// @param log_B     Log emission matrix (T * n, row-major).
/// @param T         Sequence length.
/// @param n         Number of hidden states.
/// @param log_alpha Pre-computed forward log-alphas (T * n) — used for gamma.
/// @return Full forward-backward result including log_beta, log_gamma, log_likelihood.
HMMForwardBackwardResult hmmForwardBackward(const float* log_A, const float* log_pi,
                                             const float* log_B,
                                             uint32_t T, uint32_t n);

/// Run Baum-Welch E-step + accumulate M-step counts over a batch of sequences.
///
/// Each sequence has its own emission matrix but shares the same A and pi.
/// This is the core of EM training — computes xi and gamma counts for all
/// sequences in one call.
///
/// @param log_A         Log transition matrix (n * n).
/// @param log_pi        Log initial distribution (n,).
/// @param batch_log_B   Concatenated log emission matrices. batch_log_B[seq_offset + t*n + i].
/// @param seq_lengths   Length of each sequence in the batch.
/// @param n_seqs        Number of sequences.
/// @param n             Number of hidden states.
/// @param smoothing     Laplace smoothing added to counts.
/// @return Accumulated A_num, pi_num counts and average log-likelihood.
HMMBaumWelchResult hmmBaumWelchBatch(const float* log_A, const float* log_pi,
                                      const float* batch_log_B,
                                      const uint32_t* seq_lengths,
                                      uint32_t n_seqs, uint32_t n,
                                      float smoothing = 1e-3f);

}  // namespace cubemind
}  // namespace grilly
