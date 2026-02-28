#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/cubemind/cache.h"
#include "grilly/cubemind/types.h"
#include "grilly/cubemind/vsa.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace cognitive {

// ── WorldModel: GPU-Accelerated Fact Engine ──────────────────────────────
//
// Stores knowledge as bitpacked triples and checks coherence in ~58us
// using two parallel VSACache lookups (one for facts, one for constraints).
//
// A "fact" is a (Subject, Relation, Object) triple encoded as:
//
//   fact_vec = BLAKE3("filler_" + S) XOR BLAKE3("filler_" + R)
//                                    XOR BLAKE3("filler_" + O)
//
// When a fact is inserted, its negation is auto-generated:
//
//   constraint_vec = BLAKE3("filler_" + S) XOR BLAKE3("filler_is_not")
//                                          XOR BLAKE3("filler_" + O)
//
// Coherence scoring:
//
//   support   = 1.0 - known_facts.querySurprise     (how close to a known truth)
//   violation = 1.0 - constraints.querySurprise      (how close to a known falsehood)
//   score     = support - violation                   (range: [-1, +1])
//
//   score >  0.3  →  statement aligns with known facts
//   score < -0.2  →  statement contradicts known facts
//
// Both lookups use the same 29us Hamming shader, so a full coherence
// check costs ~58us total regardless of how many facts are stored.

struct CoherenceResult {
    float support;       // 0..1: similarity to nearest known fact
    float violation;     // 0..1: similarity to nearest constraint
    float score;         // support - violation, range [-1, +1]
    bool coherent;       // score > coherenceThreshold
    uint32_t nearestFactIdx;       // Index of most similar fact
    uint32_t nearestConstraintIdx; // Index of most similar constraint
};

struct WorldModelConfig {
    uint32_t dim = 10240;
    uint32_t factCapacity = 500000;       // Max facts
    uint32_t constraintCapacity = 500000; // Max constraints
    float coherenceThreshold = 0.3f;      // score > this → coherent
    float surpriseThreshold = 0.3f;       // For VSACache insertion filtering
};

class WorldModel {
public:
    WorldModel(BufferPool& pool, const WorldModelConfig& config = {});
    ~WorldModel() = default;

    WorldModel(const WorldModel&) = delete;
    WorldModel& operator=(const WorldModel&) = delete;

    /// Encode a (subject, relation, object) triple into a bitpacked vector.
    /// Uses BLAKE3 deterministic hashing with "filler_" prefix for each term.
    cubemind::BitpackedVec encode_triple(
        const std::string& subject,
        const std::string& relation,
        const std::string& object) const;

    /// Add a fact to the world model.
    /// Also generates and stores the corresponding negation constraint.
    ///
    /// Example: add_fact("dog", "is", "animal")
    ///   → known_facts gets:  BLAKE3("filler_dog") ^ BLAKE3("filler_is") ^ BLAKE3("filler_animal")
    ///   → constraints gets:  BLAKE3("filler_dog") ^ BLAKE3("filler_is_not") ^ BLAKE3("filler_animal")
    void add_fact(const std::string& subject,
                  const std::string& relation,
                  const std::string& object);

    /// Add a pre-encoded fact vector directly (no negation auto-generation).
    void add_fact_vec(const cubemind::BitpackedVec& fact_vec);

    /// Add a pre-encoded constraint vector directly.
    void add_constraint_vec(const cubemind::BitpackedVec& constraint_vec);

    /// Check coherence of a statement against known facts and constraints.
    /// Uses GPU Hamming search (~58us for both lookups).
    CoherenceResult check_coherence(
        CommandBatch& batch, PipelineCache& pipeCache,
        const cubemind::BitpackedVec& statement);

    /// Check coherence of a (S, R, O) triple.
    CoherenceResult check_coherence(
        CommandBatch& batch, PipelineCache& pipeCache,
        const std::string& subject,
        const std::string& relation,
        const std::string& object);

    /// CPU-only coherence check (for testing without GPU).
    CoherenceResult check_coherence_cpu(
        const cubemind::BitpackedVec& statement);

    /// Number of facts stored.
    uint32_t fact_count() const { return known_facts_.size(); }

    /// Number of constraints stored.
    uint32_t constraint_count() const { return constraints_.size(); }

    /// Access to underlying caches (for advanced usage).
    cubemind::VSACache& facts_cache() { return known_facts_; }
    cubemind::VSACache& constraints_cache() { return constraints_; }

    uint32_t dim() const { return config_.dim; }

private:
    WorldModelConfig config_;
    cubemind::VSACache known_facts_;
    cubemind::VSACache constraints_;

    /// Generate the negation relation key from a relation.
    /// "is" → "is_not", "causes" → "causes_not", etc.
    static std::string negate_relation(const std::string& relation);
};

}  // namespace cognitive
}  // namespace grilly
