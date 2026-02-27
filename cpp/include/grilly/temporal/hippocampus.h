#pragma once

// ── Hippocampal Consolidator ─────────────────────────────────────────
//
// Offline "dream cycle" that generalizes repeated patterns into permanent
// WorldModel rules. The consolidation pipeline:
//
//   1. Record episode pairs (state_t, state_t+1) during online learning
//   2. Periodically run dream():
//      a. XOR all episode pairs → transition deltas
//      b. Count delta frequency via unordered_map
//      c. Burn high-frequency deltas (>5% of episodes) as WorldModel facts
//      d. Generate synthetic mutations (future: GPU dispatch)
//      e. Clear episodic buffer
//
// This compresses experience: instead of storing N similar episodes in
// the VSA cache, the model learns the underlying transition rule once.

#include <cstdint>
#include <deque>
#include <mutex>
#include <random>
#include <unordered_map>

#include "grilly/cognitive/world_model.h"
#include "grilly/cubemind/types.h"

namespace grilly {

/// Summary returned after a dream consolidation cycle.
struct DreamReport {
    uint32_t episodes_replayed;
    uint32_t synthetic_dreams;
    uint32_t new_rules_extracted;
};

/// Hippocampal consolidator: records episodes and dreams to extract rules.
class HippocampalConsolidator {
public:
    struct Episode {
        cubemind::BitpackedVec state_t;
        cubemind::BitpackedVec state_t1;
    };

    explicit HippocampalConsolidator(uint32_t max_capacity = 10000)
        : max_capacity_(max_capacity) {}

    /// Record a state transition for later consolidation.
    void record_episode(const cubemind::BitpackedVec& state_t,
                        const cubemind::BitpackedVec& state_t1) {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        if (episodic_buffer_.size() >= max_capacity_)
            episodic_buffer_.pop_front();
        episodic_buffer_.push_back({state_t, state_t1});
    }

    /// Run offline dream consolidation cycle.
    ///
    /// Phase 1: XOR all episode pairs to compute transition deltas, count freq.
    /// Phase 2: Burn deltas appearing in >5% of episodes as WorldModel facts.
    /// Phase 3: Generate synthetic mutations (counted only, no GPU dispatch yet).
    /// Phase 4: Clear episodic buffer.
    DreamReport dream(cognitive::WorldModel& wm, uint32_t cycles = 128) {
        std::lock_guard<std::mutex> lock(buffer_mutex_);

        DreamReport report{};
        if (episodic_buffer_.empty())
            return report;

        const uint32_t num_episodes =
            static_cast<uint32_t>(episodic_buffer_.size());
        report.episodes_replayed = num_episodes;

        // Phase 1: compute transition deltas and count frequencies
        std::unordered_map<cubemind::BitpackedVec, uint32_t,
                           cubemind::BitpackedVecHash>
            delta_counts;

        for (const auto& ep : episodic_buffer_) {
            cubemind::BitpackedVec delta = ep.state_t ^ ep.state_t1;
            delta_counts[delta]++;
        }

        // Phase 2: burn high-frequency deltas (>5% threshold) as permanent rules
        const uint32_t threshold =
            std::max(1u, static_cast<uint32_t>(num_episodes * 0.05));

        for (const auto& [delta, count] : delta_counts) {
            if (count >= threshold) {
                wm.add_fact_vec(delta);
                report.new_rules_extracted++;
            }
        }

        // Phase 3: synthetic mutations (count only — GPU dispatch deferred)
        report.synthetic_dreams = 0;
        if (!episodic_buffer_.empty()) {
            std::mt19937 rng(std::random_device{}());
            for (uint32_t c = 0; c < cycles; ++c) {
                std::uniform_int_distribution<size_t> idx_dist(
                    0, episodic_buffer_.size() - 1);
                const auto& base = episodic_buffer_[idx_dist(rng)].state_t1;
                auto mutated = mutate_random_bits(base, 3, rng);
                (void)mutated;  // Future: dispatch to GPU
                report.synthetic_dreams++;
            }
        }

        // Phase 4: clear episodic buffer
        episodic_buffer_.clear();

        return report;
    }

    /// Current number of episodes in the buffer.
    uint32_t buffer_size() const {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        return static_cast<uint32_t>(episodic_buffer_.size());
    }

private:
    uint32_t max_capacity_;
    std::deque<Episode> episodic_buffer_;
    mutable std::mutex buffer_mutex_;

    /// Flip num_flips random bits in a bitpacked vector.
    static cubemind::BitpackedVec mutate_random_bits(
        const cubemind::BitpackedVec& base, uint32_t num_flips,
        std::mt19937& rng) {
        cubemind::BitpackedVec result = base;
        if (result.data.empty())
            return result;

        std::uniform_int_distribution<uint32_t> word_dist(
            0, static_cast<uint32_t>(result.data.size()) - 1);
        std::uniform_int_distribution<uint32_t> bit_dist(0, 31);

        for (uint32_t i = 0; i < num_flips; ++i) {
            uint32_t w = word_dist(rng);
            uint32_t b = bit_dist(rng);
            result.data[w] ^= (1u << b);
        }
        return result;
    }
};

}  // namespace grilly
