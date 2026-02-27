#pragma once

#include <cstdint>
#include <cstring>

#include "grilly/cubemind/types.h"

namespace grilly {
namespace temporal {

/// Temporal binding via circular bit shifting.
///
/// In a strict bipolar bitpacked VSA, time binding is a circular
/// permutation of the bit array. This replaces FFT-based HRR binding
/// with an O(dim/32) bitwise operation.
///
/// Properties:
///   - Self-inverse:   unbind_time(bind_time(v, t), t) == v
///   - Distance-preserving: hamming(bind(a,t), bind(b,t)) == hamming(a, b)
///   - Preserves weight: popcount stays constant (bit permutation)
///
/// Implementation:
///   For shift_amount S across N uint32 words (N*32 total bits):
///     word_shift  = S / 32   (how many whole words to skip)
///     bit_shift   = S % 32   (remaining bits within a word)
///
///   Each output word[i] combines:
///     src_word[(i - word_shift) mod N] >> bit_shift
///     src_word[(i - word_shift - 1) mod N] << (32 - bit_shift)
///
/// Cost: ~320 shifts + ORs for 10240-bit vector ≈ nanoseconds on CPU,
///       parallelizable to single-cycle per warp on GPU.
///
class TemporalEncoder {
public:
    /// Bind a VSA state to time step t (circular right shift by t bits).
    ///
    /// This positions the state at time step t in the temporal sequence.
    /// Successive calls with t=0, t=1, t=2... create orthogonal versions
    /// of the same state, enabling temporal superposition via bundling.
    ///
    /// @param state  The bitpacked VSA vector to bind
    /// @param t      Time step (shift amount in bits)
    /// @return       New BitpackedVec shifted right by t bits (circular)
    static cubemind::BitpackedVec bind_time(
            const cubemind::BitpackedVec& state, uint32_t t) {
        const uint32_t n = state.numWords();
        if (n == 0) return state;

        const uint32_t total_bits = n * 32;
        const uint32_t shift = t % total_bits;
        if (shift == 0) return state;  // Identity

        cubemind::BitpackedVec result;
        result.dim = state.dim;
        result.data.resize(n);

        circular_shift_right(state.data.data(), result.data.data(), n, shift);
        return result;
    }

    /// Unbind time step t (circular left shift = inverse of right shift).
    ///
    /// unbind_time(bind_time(state, t), t) == state  (exact recovery)
    ///
    /// @param bound_state  A previously time-bound vector
    /// @param t            Time step that was bound
    /// @return             Original unbound state
    static cubemind::BitpackedVec unbind_time(
            const cubemind::BitpackedVec& bound_state, uint32_t t) {
        const uint32_t n = bound_state.numWords();
        if (n == 0) return bound_state;

        const uint32_t total_bits = n * 32;
        const uint32_t shift = t % total_bits;
        if (shift == 0) return bound_state;

        // Left shift by S == Right shift by (total - S)
        const uint32_t inv_shift = total_bits - shift;

        cubemind::BitpackedVec result;
        result.dim = bound_state.dim;
        result.data.resize(n);

        circular_shift_right(bound_state.data.data(), result.data.data(),
                             n, inv_shift);
        return result;
    }

    /// XOR-bind two bitpacked vectors (for fact binding, counterfactuals).
    /// Returns a ^ b element-wise.
    static cubemind::BitpackedVec xor_bind(
            const cubemind::BitpackedVec& a,
            const cubemind::BitpackedVec& b) {
        cubemind::BitpackedVec result;
        result.dim = a.dim;
        const uint32_t n = a.numWords();
        result.data.resize(n);
        for (uint32_t i = 0; i < n; ++i) {
            result.data[i] = a.data[i] ^ b.data[i];
        }
        return result;
    }

private:
    /// Circular right shift of a bit array stored as uint32 words.
    ///
    /// For N words with shift S:
    ///   word_shift = S / 32
    ///   bit_shift  = S % 32
    ///   dst[i] = src[(i - word_shift) mod N] >> bit_shift
    ///          | src[(i - word_shift - 1) mod N] << (32 - bit_shift)
    ///
    /// When bit_shift == 0, only whole-word rotation is needed.
    static void circular_shift_right(const uint32_t* src, uint32_t* dst,
                                     uint32_t n, uint32_t shift) {
        const uint32_t word_shift = shift / 32;
        const uint32_t bit_shift  = shift % 32;

        if (bit_shift == 0) {
            // Pure word rotation: dst[i] = src[(i - word_shift) mod n]
            for (uint32_t i = 0; i < n; ++i) {
                uint32_t src_idx = (i + n - word_shift) % n;
                dst[i] = src[src_idx];
            }
        } else {
            const uint32_t complement = 32 - bit_shift;
            for (uint32_t i = 0; i < n; ++i) {
                uint32_t hi_idx = (i + n - word_shift) % n;
                uint32_t lo_idx = (i + n - word_shift - 1) % n;
                dst[i] = (src[hi_idx] >> bit_shift)
                       | (src[lo_idx] << complement);
            }
        }
    }
};

}  // namespace temporal
}  // namespace grilly
