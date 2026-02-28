#include "grilly/cubemind/resonator.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace grilly {
namespace cubemind {

// ── Portable timer ──────────────────────────────────────────────────────

static double now_ms() {
    using Clock = std::chrono::high_resolution_clock;
    auto tp = Clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(tp).count();
}

// ── ResonatorNetwork ────────────────────────────────────────────────────

ResonatorNetwork::ResonatorNetwork(BufferPool& pool, CommandBatch& batch,
                                   PipelineCache& pipeCache, uint32_t dim)
    : pool_(pool), batch_(batch), pipeCache_(pipeCache),
      dim_(dim), words_per_vec_((dim + 31) / 32) {}

ResonatorNetwork::~ResonatorNetwork() {
    if (codebook_loaded_ && codebook_buf_.handle != VK_NULL_HANDLE) {
        pool_.release(codebook_buf_);
    }
}

// ── Codebook Management ─────────────────────────────────────────────────

void ResonatorNetwork::load_codebook(const std::vector<std::string>& words,
                                      const uint32_t* vectors) {
    // Release old codebook if any
    if (codebook_loaded_ && codebook_buf_.handle != VK_NULL_HANDLE) {
        pool_.release(codebook_buf_);
    }

    words_ = words;
    size_t total_words = words.size() * words_per_vec_;
    size_t bytes = total_words * sizeof(uint32_t);

    // Host copy for CPU fallback
    codebook_host_.assign(vectors, vectors + total_words);

    // Upload to VRAM (persistent — stays until codebook is replaced)
    codebook_buf_ = pool_.acquire(bytes);
    pool_.upload(codebook_buf_, reinterpret_cast<const float*>(vectors), bytes);
    codebook_loaded_ = true;
}

void ResonatorNetwork::load_codebook_bipolar(
    const std::vector<std::string>& words,
    const int8_t* bipolar_vectors) {
    // Bitpack each word's bipolar vector and load as packed codebook
    size_t total_words_packed = words.size() * words_per_vec_;
    std::vector<uint32_t> packed(total_words_packed, 0);

    for (size_t w = 0; w < words.size(); ++w) {
        const int8_t* bip = bipolar_vectors + w * dim_;
        uint32_t* out = packed.data() + w * words_per_vec_;

        for (uint32_t j = 0; j < dim_; ++j) {
            if (bip[j] > 0) {
                out[j / 32] |= (1u << (j % 32));
            }
        }
    }

    load_codebook(words, packed.data());
}

const std::string& ResonatorNetwork::get_word(uint32_t index) const {
    static const std::string kUnknown = "<UNK>";
    if (index < words_.size()) return words_[index];
    return kUnknown;
}

// ── GPU Resonation ──────────────────────────────────────────────────────

void ResonatorNetwork::dispatch_resonator(const uint32_t* query_packed,
                                           float* similarities_out) {
    if (!codebook_loaded_) {
        throw std::runtime_error("Codebook not loaded");
    }

    uint32_t cb_size = static_cast<uint32_t>(words_.size());
    size_t queryBytes = size_t(words_per_vec_) * sizeof(uint32_t);
    size_t simBytes = size_t(cb_size) * sizeof(float);
    size_t cbBytes = size_t(cb_size) * words_per_vec_ * sizeof(uint32_t);

    GrillyBuffer bufQuery = pool_.acquire(queryBytes);
    GrillyBuffer bufSim = pool_.acquire(simBytes);

    pool_.upload(bufQuery, reinterpret_cast<const float*>(query_packed),
                 queryBytes);

    PipelineEntry pipe = pipeCache_.getOrCreate(
        "resonator-bitpacked", 3, sizeof(ResonatorParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufQuery.handle, 0, queryBytes},
        {codebook_buf_.handle, 0, cbBytes},
        {bufSim.handle, 0, simBytes},
    };
    VkDescriptorSet descSet = pipeCache_.allocDescriptorSet(
        "resonator-bitpacked", bufInfos);

    ResonatorParams push{dim_, words_per_vec_, cb_size, 0};

    // One workgroup per codebook entry (each WG has 256 threads)
    uint32_t gx = cb_size;

    batch_.begin();
    batch_.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                    &push, sizeof(push));
    batch_.submit();

    pool_.download(bufSim, similarities_out, simBytes);

    pool_.release(bufQuery);
    pool_.release(bufSim);
}

ResonateResult ResonatorNetwork::resonate(const BitpackedVec& query,
                                           bool return_all_similarities) {
    double t0 = now_ms();

    uint32_t cb_size = static_cast<uint32_t>(words_.size());
    std::vector<float> sims(cb_size);

    if (pipeCache_.hasShader("resonator-bitpacked")) {
        dispatch_resonator(query.data.data(), sims.data());
    } else {
        // CPU fallback
        auto cpu_result = resonate_cpu(query);
        last_resonate_ms_ = now_ms() - t0;
        total_resonations_++;
        return cpu_result;
    }

    // Find argmax
    uint32_t best_idx = 0;
    float best_sim = -2.0f;
    for (uint32_t i = 0; i < cb_size; ++i) {
        if (sims[i] > best_sim) {
            best_sim = sims[i];
            best_idx = i;
        }
    }

    last_resonate_ms_ = now_ms() - t0;
    total_resonations_++;

    ResonateResult result;
    result.best_index = best_idx;
    result.best_similarity = best_sim;
    if (return_all_similarities) {
        result.all_similarities = std::move(sims);
    }
    return result;
}

ResonateResult ResonatorNetwork::resonate_cpu(const BitpackedVec& query) const {
    uint32_t cb_size = static_cast<uint32_t>(words_.size());

    uint32_t best_idx = 0;
    float best_sim = -2.0f;
    std::vector<float> all_sims(cb_size);

    for (uint32_t w = 0; w < cb_size; ++w) {
        uint32_t hamming = 0;
        const uint32_t* cb_entry = codebook_host_.data() + w * words_per_vec_;
        for (uint32_t j = 0; j < words_per_vec_; ++j) {
#ifdef _MSC_VER
            hamming += __popcnt(query.data[j] ^ cb_entry[j]);
#else
            hamming += __builtin_popcount(query.data[j] ^ cb_entry[j]);
#endif
        }
        float sim = (float(dim_) - 2.0f * float(hamming)) / float(dim_);
        all_sims[w] = sim;
        if (sim > best_sim) {
            best_sim = sim;
            best_idx = w;
        }
    }

    ResonateResult result;
    result.best_index = best_idx;
    result.best_similarity = best_sim;
    result.all_similarities = std::move(all_sims);
    return result;
}

// ── Sentence Generation (Explaining Away) ────────────────────────────

std::vector<std::pair<std::string, float>> ResonatorNetwork::generate_sentence(
    const BitpackedVec& sentence_bundle,
    const std::vector<std::string>& dependency_roles,
    const std::vector<uint32_t>& positions,
    bool explain_away) {

    uint32_t length = static_cast<uint32_t>(dependency_roles.size());
    std::vector<std::pair<std::string, float>> generated;
    generated.reserve(length);

    // ── Explaining-Away Accumulator ──────────────────────────────────
    //
    // The bundle is a majority-vote superposition of N bound components.
    // Each component was encoded as: word ⊗ role ⊗ position (three-way).
    // After thresholding to {-1,+1}, we lose the vote counts. We use an
    // int16 accumulator to preserve analog magnitudes during subtraction.
    //
    //   accumulator[d] = unpack(bundle)[d]   (initially ±1)
    //   After finding word_i: accumulator -= bipolar(word_i ⊗ role_i ⊗ pos_i)
    //   Query for next position: threshold(accumulator) → BitpackedVec

    // Unpack bundle to int16 accumulator
    std::vector<int16_t> accumulator(dim_);
    for (uint32_t d = 0; d < dim_; ++d) {
        uint32_t word_idx = d / 32;
        uint32_t bit_idx = d % 32;
        bool bit_set = (sentence_bundle.data[word_idx] >> bit_idx) & 1u;
        accumulator[d] = bit_set ? 1 : -1;
    }

    for (uint32_t i = 0; i < length; ++i) {
        // ── 1. Threshold accumulator to get current bundle ──────────
        BitpackedVec current;
        current.dim = dim_;
        current.data.resize(words_per_vec_, 0);

        for (uint32_t d = 0; d < dim_; ++d) {
            if (accumulator[d] > 0) {
                current.data[d / 32] |= (1u << (d % 32));
            }
        }

        // ── 2. Unbind BOTH role AND position (three-way decode) ─────
        //
        // Encoding was: bound = word ⊗ role ⊗ pos
        // Since XOR binding is self-inverse:
        //   probe = current ⊗ role ⊗ pos
        //         = (word ⊗ role ⊗ pos + noise) ⊗ role ⊗ pos
        //         = word + noise'   (role and pos cancel out)
        //
        // We must use the SAME key prefixes as TextEncoder:
        //   role: blake3Role("role_" + dependency_role)
        //   pos:  blake3Role("pos_"  + position_index)

        std::string dep_role = (i < dependency_roles.size())
                                   ? dependency_roles[i] : "UNK";
        uint32_t pos = (i < positions.size())
                           ? positions[i] : i;

        std::vector<int8_t> role_bipolar = blake3Role("role_" + dep_role, dim_);
        std::vector<int8_t> pos_bipolar  = blake3Role("pos_" + std::to_string(pos), dim_);

        BitpackedVec role_packed = vsaBitpack(role_bipolar.data(), dim_);
        BitpackedVec pos_packed  = vsaBitpack(pos_bipolar.data(), dim_);

        BitpackedVec probe;
        probe.dim = dim_;
        probe.data.resize(words_per_vec_);
        for (uint32_t j = 0; j < words_per_vec_; ++j) {
            // XOR both role and position to unbind the three-way binding
            probe.data[j] = current.data[j] ^ role_packed.data[j] ^ pos_packed.data[j];
        }

        // ── 3. Resonate against codebook (29μs GPU shader) ──────────
        ResonateResult result = resonate(probe);

        std::string word = get_word(result.best_index);
        generated.push_back({word, result.best_similarity});

        // ── 4. Explaining Away ──────────────────────────────────────
        //
        // Reconstruct the THREE-WAY bound component: word ⊗ role ⊗ pos
        // Subtract it from the accumulator to clean the bundle for
        // the next position.
        //
        // In bipolar algebra: bound[d] = word[d] * role[d] * pos[d]
        // Subtracting: accumulator[d] -= bound[d]
        //
        // This prevents the winning word from "echoing" and causing
        // repetition or interference in subsequent positions.
        if (explain_away && i + 1 < length) {
            const uint32_t* winner_packed =
                codebook_host_.data() + result.best_index * words_per_vec_;

            for (uint32_t d = 0; d < dim_; ++d) {
                uint32_t w = d / 32;
                uint32_t b = d % 32;

                // Unpack winner word bit → bipolar
                int8_t word_val = ((winner_packed[w] >> b) & 1u) ? 1 : -1;

                // Three-way bound component = word * role * pos
                int8_t bound_val = word_val * role_bipolar[d] * pos_bipolar[d];

                // Subtract from accumulator
                accumulator[d] -= bound_val;
            }
        }
    }

    return generated;
}

std::vector<std::pair<std::string, float>> ResonatorNetwork::generate_positional(
    const BitpackedVec& sentence_bundle,
    uint32_t length,
    bool explain_away) {
    // Legacy: position-only unbinding (no dependency roles).
    // Uses "UNK" as the dependency role for all positions.
    std::vector<std::string> dep_roles(length, "UNK");
    std::vector<uint32_t> positions(length);
    for (uint32_t i = 0; i < length; ++i) {
        positions[i] = i;
    }
    return generate_sentence(sentence_bundle, dep_roles, positions, explain_away);
}

}  // namespace cubemind
}  // namespace grilly
