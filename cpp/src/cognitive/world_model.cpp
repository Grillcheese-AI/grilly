#include "grilly/cognitive/world_model.h"

namespace grilly {
namespace cognitive {

// ── Helper: build CacheConfig from WorldModelConfig ──────────────────────

static cubemind::CacheConfig make_cache_config(
    uint32_t dim, uint32_t capacity, float surpriseThreshold) {
    cubemind::CacheConfig cfg;
    cfg.initialCapacity = 1024;
    cfg.maxCapacity = capacity;
    cfg.dim = dim;
    cfg.surpriseThreshold = surpriseThreshold;
    cfg.utilityDecay = 0.99f;
    return cfg;
}

// ── WorldModel ───────────────────────────────────────────────────────────

WorldModel::WorldModel(BufferPool& pool, const WorldModelConfig& config)
    : config_(config),
      known_facts_(pool,
                   make_cache_config(config.dim, config.factCapacity,
                                     config.surpriseThreshold)),
      constraints_(pool,
                   make_cache_config(config.dim, config.constraintCapacity,
                                     config.surpriseThreshold)) {}

cubemind::BitpackedVec WorldModel::encode_triple(
    const std::string& subject,
    const std::string& relation,
    const std::string& object) const {

    // Generate BLAKE3 bipolar vectors for each term.
    // Uses "filler_" prefix to match TextEncoder's BLAKE3 key scheme.
    auto s_bipolar = cubemind::blake3Role("filler_" + subject, config_.dim);
    auto r_bipolar = cubemind::blake3Role("filler_" + relation, config_.dim);
    auto o_bipolar = cubemind::blake3Role("filler_" + object, config_.dim);

    // Three-way binding: S XOR R XOR O
    // In bipolar {-1,+1}, binding = element-wise multiply.
    auto sr = cubemind::vsaBind(s_bipolar.data(), r_bipolar.data(),
                                 config_.dim);
    auto sro = cubemind::vsaBind(sr.data(), o_bipolar.data(), config_.dim);

    // Bitpack for GPU Hamming search
    return cubemind::vsaBitpack(sro.data(), config_.dim);
}

void WorldModel::add_fact(
    const std::string& subject,
    const std::string& relation,
    const std::string& object) {

    // 1. Encode and store the positive fact
    auto fact_vec = encode_triple(subject, relation, object);
    cubemind::EmotionState emo{1.0f, 0.0f};  // High confidence
    known_facts_.insert(fact_vec, emo);

    // 2. Auto-generate and store the negation constraint.
    // If "dog is animal" is true, then "dog is_not animal" is a violation.
    std::string neg_rel = negate_relation(relation);
    auto constraint_vec = encode_triple(subject, neg_rel, object);
    constraints_.insert(constraint_vec, emo);
}

void WorldModel::add_fact_vec(const cubemind::BitpackedVec& fact_vec) {
    cubemind::EmotionState emo{1.0f, 0.0f};
    known_facts_.insert(fact_vec, emo);
}

void WorldModel::add_constraint_vec(
    const cubemind::BitpackedVec& constraint_vec) {
    cubemind::EmotionState emo{1.0f, 0.0f};
    constraints_.insert(constraint_vec, emo);
}

CoherenceResult WorldModel::check_coherence(
    CommandBatch& batch, PipelineCache& pipeCache,
    const cubemind::BitpackedVec& statement) {

    CoherenceResult result{};

    // If no facts loaded, everything is "unknown" — default to neutral
    if (known_facts_.size() == 0) {
        result.support = 0.0f;
        result.violation = 0.0f;
        result.score = 0.0f;
        result.coherent = true;  // No evidence of contradiction
        return result;
    }

    // 1. Check support: how close is this to a known truth?
    auto support_res = known_facts_.lookup(batch, pipeCache, statement, 1);

    // 2. Check violation: how close is this to a known falsehood?
    cubemind::CacheLookupResult violation_res;
    if (constraints_.size() > 0) {
        violation_res = constraints_.lookup(batch, pipeCache, statement, 1);
    }

    // 3. Compute coherence score
    //
    // querySurprise = minDist / dim, range [0, 1]
    //   0.0 = exact match (surprise = 0, very familiar)
    //   1.0 = maximally distant (surprise = 1, never seen)
    //
    // support  = 1.0 - querySurprise  (high = close to a known fact)
    // violation = 1.0 - querySurprise  (high = close to a known constraint)
    //
    result.support = 1.0f - support_res.querySurprise;
    result.violation = (constraints_.size() > 0)
                           ? (1.0f - violation_res.querySurprise)
                           : 0.0f;
    result.score = result.support - result.violation;
    result.coherent = result.score > config_.coherenceThreshold;

    if (!support_res.indices.empty()) {
        result.nearestFactIdx = support_res.indices[0];
    }
    if (!violation_res.indices.empty()) {
        result.nearestConstraintIdx = violation_res.indices[0];
    }

    return result;
}

CoherenceResult WorldModel::check_coherence(
    CommandBatch& batch, PipelineCache& pipeCache,
    const std::string& subject,
    const std::string& relation,
    const std::string& object) {

    auto statement = encode_triple(subject, relation, object);
    return check_coherence(batch, pipeCache, statement);
}

CoherenceResult WorldModel::check_coherence_cpu(
    const cubemind::BitpackedVec& statement) {

    CoherenceResult result{};

    if (known_facts_.size() == 0) {
        result.support = 0.0f;
        result.violation = 0.0f;
        result.score = 0.0f;
        result.coherent = true;
        return result;
    }

    auto support_res = known_facts_.lookupCPU(statement, 1);

    cubemind::CacheLookupResult violation_res;
    if (constraints_.size() > 0) {
        violation_res = constraints_.lookupCPU(statement, 1);
    }

    result.support = 1.0f - support_res.querySurprise;
    result.violation = (constraints_.size() > 0)
                           ? (1.0f - violation_res.querySurprise)
                           : 0.0f;
    result.score = result.support - result.violation;
    result.coherent = result.score > config_.coherenceThreshold;

    if (!support_res.indices.empty()) {
        result.nearestFactIdx = support_res.indices[0];
    }
    if (!violation_res.indices.empty()) {
        result.nearestConstraintIdx = violation_res.indices[0];
    }

    return result;
}

std::string WorldModel::negate_relation(const std::string& relation) {
    // Simple negation: append "_not" to the relation.
    // "is"      → "is_not"
    // "causes"  → "causes_not"
    // "has"     → "has_not"
    return relation + "_not";
}

}  // namespace cognitive
}  // namespace grilly
