#include "grilly/cubemind/text_encoder.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <random>

namespace grilly {
namespace cubemind {

TextEncoder::TextEncoder(uint32_t dim, uint32_t ft_dim)
    : dim_(dim), ft_dim_(ft_dim) {
    // Initialize the LSH random projection matrix with deterministic seed.
    // This ensures the same FastText model always yields the same bipolar vectors
    // regardless of when or where the projection is computed.
    //
    // By the Johnson-Lindenstrauss lemma, for random Gaussian M ∈ R^{d×D}:
    //   Pr[|cos(a,b) - (1 - 2*hamming(sign(Ma), sign(Mb))/D)| > ε] < δ
    // where D = dim_ = 10240 gives excellent preservation.
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    projection_.resize(static_cast<size_t>(ft_dim_) * dim_);
    for (size_t i = 0; i < projection_.size(); ++i) {
        projection_[i] = dist(gen);
    }
}

BitpackedVec TextEncoder::encode_sentence(
    const std::vector<std::string>& tokens,
    const std::vector<std::string>& dependency_roles,
    const std::vector<uint32_t>& positions) {

    if (tokens.empty()) {
        // Return zero vector
        BitpackedVec empty;
        empty.dim = dim_;
        empty.data.resize((dim_ + 31) / 32, 0);
        return empty;
    }

    // Collect bound components for bundling.
    // Each component = word ⊗ role ⊗ position (element-wise multiply).
    std::vector<std::vector<int8_t>> bound_buffers;
    std::vector<const int8_t*> bundle_ptrs;
    bound_buffers.reserve(tokens.size());
    bundle_ptrs.reserve(tokens.size());

    for (size_t i = 0; i < tokens.size(); ++i) {
        // 1. Get semantic filler (pre-projected FastText → bipolar)
        //    Lazy caching: if token isn't in the map, generate BLAKE3
        //    fallback and cache it for all future lookups.
        const std::vector<int8_t>* word_vec = nullptr;
        auto it = semantic_fillers_.find(tokens[i]);
        if (it != semantic_fillers_.end()) {
            word_vec = &it->second;
        } else {
            // Unknown token → deterministic BLAKE3 fallback, cached on first miss
            auto [inserted_it, _] = semantic_fillers_.emplace(
                tokens[i], fallback_filler(tokens[i]));
            word_vec = &inserted_it->second;
        }

        // 2. Get structural role vector (BLAKE3 deterministic hash)
        std::string role_key = (i < dependency_roles.size())
                                   ? dependency_roles[i]
                                   : "UNK";
        std::vector<int8_t> role_vec = blake3Role("role_" + role_key, dim_);

        // 3. Get position vector (BLAKE3 deterministic hash)
        uint32_t pos = (i < positions.size()) ? positions[i]
                                               : static_cast<uint32_t>(i);
        std::vector<int8_t> pos_vec =
            blake3Role("pos_" + std::to_string(pos), dim_);

        // 4. Bind: word ⊗ role ⊗ position
        //    In bipolar {-1,+1}, binding = element-wise multiply.
        //    Result stays in {-1,+1} because (-1)*(-1)=1, (-1)*(1)=-1, etc.
        std::vector<int8_t> bound = vsaBind(word_vec->data(), role_vec.data(), dim_);
        bound = vsaBind(bound.data(), pos_vec.data(), dim_);

        bound_buffers.push_back(std::move(bound));
        bundle_ptrs.push_back(bound_buffers.back().data());
    }

    // 5. Bundle: majority vote superposition → sentence vector
    //    The bundled vector is similar to all of its components,
    //    encoding the semantic content + structural relations.
    std::vector<int8_t> sentence = vsaBundle(bundle_ptrs, dim_);

    // 6. Bitpack for GPU Hamming distance search
    return vsaBitpack(sentence.data(), dim_);
}

void TextEncoder::load_fillers(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open filler file: " + path);
    }

    // File format: repeated entries of:
    //   uint32_t token_len
    //   char[token_len]  (token string, no null terminator)
    //   int8_t[dim_]     (bipolar filler)
    while (file.good()) {
        uint32_t token_len = 0;
        file.read(reinterpret_cast<char*>(&token_len), sizeof(uint32_t));
        if (!file.good() || token_len == 0 || token_len > 1024) break;

        std::string token(token_len, '\0');
        file.read(token.data(), token_len);
        if (!file.good()) break;

        std::vector<int8_t> bipolar(dim_);
        file.read(reinterpret_cast<char*>(bipolar.data()), dim_);
        if (!file.good()) break;

        semantic_fillers_[std::move(token)] = std::move(bipolar);
    }
}

std::vector<int8_t> TextEncoder::project_to_bipolar(const float* ft_vec) const {
    // LSH random projection: bipolar[j] = sign( sum_i(ft[i] * M[i][j]) )
    //
    // This is a matrix-vector multiply [1 × ft_dim_] * [ft_dim_ × dim_]
    // followed by element-wise sign.
    std::vector<int8_t> bipolar(dim_);

    for (uint32_t j = 0; j < dim_; ++j) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < ft_dim_; ++i) {
            sum += ft_vec[i] * projection_[static_cast<size_t>(i) * dim_ + j];
        }
        bipolar[j] = (sum > 0.0f) ? 1 : -1;
    }

    return bipolar;
}

void TextEncoder::add_filler(const std::string& token,
                              const std::vector<int8_t>& bipolar) {
    semantic_fillers_[token] = bipolar;
}

bool TextEncoder::has_filler(const std::string& token) const {
    return semantic_fillers_.count(token) > 0;
}

std::vector<int8_t> TextEncoder::fallback_filler(const std::string& token) const {
    // Generate a deterministic random bipolar vector for unknown tokens.
    // Uses the existing BLAKE3 role generation with a "filler_" prefix.
    // This gives OOV tokens a consistent representation across sessions.
    return blake3Role("filler_" + token, dim_);
}

}  // namespace cubemind
}  // namespace grilly
