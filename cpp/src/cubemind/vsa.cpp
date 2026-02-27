#include "grilly/cubemind/vsa.h"

#include <blake3.h>

#include <cstring>

#ifdef _MSC_VER
#include <intrin.h>  // __popcnt
#else
#include <x86intrin.h>
#endif

namespace grilly {
namespace cubemind {

// ── BLAKE3 Role Generation ────────────────────────────────────────────
//
// Matches Python's grilly/utils/stable_hash.py:bipolar_from_key() exactly.
//
// Python algorithm:
//   1. nbytes = (dim + 7) // 8
//   2. For ctr = 0, 1, 2, ...:
//        message = join_with_0x1f(domain, key, str(ctr))
//        out.extend(BLAKE3(message, out_len=32))
//      until len(out) >= nbytes
//   3. Unpack bytes to bits (little-endian bit order)
//   4. Map: bit 0 -> -1, bit 1 -> +1
//
// The join function concatenates parts with ASCII Unit Separator (0x1F):
//   _join_parts(["grilly.cubemind", "facelet_0", "0"])
//   -> b"grilly.cubemind\x1ffacelet_0\x1f0"
//
// BLAKE3's variable-length output (finalize with out_len) is used to get
// exactly 32 bytes per hash call, matching the Python digest() function.

/// Join domain + key + counter with 0x1F separator, matching Python's _join_parts.
static std::vector<uint8_t> joinParts(const std::string& domain,
                                       const std::string& key,
                                       uint32_t counter) {
    std::string counterStr = std::to_string(counter);

    // domain \x1f key \x1f counterStr
    size_t totalLen = domain.size() + 1 + key.size() + 1 + counterStr.size();
    std::vector<uint8_t> msg(totalLen);

    size_t offset = 0;
    std::memcpy(msg.data() + offset, domain.data(), domain.size());
    offset += domain.size();
    msg[offset++] = 0x1F;  // ASCII Unit Separator
    std::memcpy(msg.data() + offset, key.data(), key.size());
    offset += key.size();
    msg[offset++] = 0x1F;
    std::memcpy(msg.data() + offset, counterStr.data(), counterStr.size());

    return msg;
}

std::vector<int8_t> blake3Role(const std::string& key, uint32_t dim,
                                const std::string& domain) {
    if (dim == 0) return {};

    uint32_t nbytes = (dim + 7) / 8;

    // Stream 32-byte BLAKE3 digests with incrementing counter
    std::vector<uint8_t> hashStream;
    hashStream.reserve(nbytes + 32);
    uint32_t ctr = 0;

    while (hashStream.size() < nbytes) {
        std::vector<uint8_t> msg = joinParts(domain, key, ctr);

        blake3_hasher hasher;
        blake3_hasher_init(&hasher);
        blake3_hasher_update(&hasher, msg.data(), msg.size());

        uint8_t digest[32];
        blake3_hasher_finalize(&hasher, digest, 32);
        hashStream.insert(hashStream.end(), digest, digest + 32);

        ctr++;
    }

    // Unpack bytes to bits (little-endian bit order) and map to bipolar
    // Little-endian bit order: bit 0 of byte 0 is the first element.
    // This matches numpy's np.unpackbits(..., bitorder="little").
    std::vector<int8_t> bipolar(dim);
    for (uint32_t i = 0; i < dim; ++i) {
        uint32_t byteIdx = i / 8;
        uint32_t bitIdx  = i % 8;
        uint8_t bit = (hashStream[byteIdx] >> bitIdx) & 1;
        bipolar[i] = bit ? 1 : -1;  // 1 -> +1, 0 -> -1
    }

    return bipolar;
}

// ── VSA Binding ──────────────────────────────────────────────────────
//
// Bipolar binding is element-wise multiplication:
//   bind(a, b)[i] = a[i] * b[i]
//
// Since a[i], b[i] ∈ {-1, +1}, the result is also in {-1, +1}.
// This is equivalent to XOR in the bitpacked domain.
//
// Key property: binding is self-inverse.
//   bind(a, bind(a, b)) = b  (because a[i]*a[i] = 1)

std::vector<int8_t> vsaBind(const int8_t* a, const int8_t* b, uint32_t dim) {
    std::vector<int8_t> result(dim);
    for (uint32_t i = 0; i < dim; ++i) {
        result[i] = a[i] * b[i];  // {-1,+1} * {-1,+1} = {-1,+1}
    }
    return result;
}

// ── VSA Bundling ─────────────────────────────────────────────────────
//
// Bundling creates a superposition of multiple vectors via majority vote:
//   1. Sum element-wise across all input vectors
//   2. Apply sign function: positive -> +1, zero or negative -> -1
//
// Ties (sum == 0) are resolved to +1 to maintain strict bipolarity.
// This matches the Python implementation's behavior.
//
// The resulting vector is approximately equidistant from all inputs,
// which is the core property that makes VSA bundling useful for
// representing sets of items.

std::vector<int8_t> vsaBundle(const std::vector<const int8_t*>& vectors,
                               uint32_t dim) {
    std::vector<int32_t> sums(dim, 0);
    for (const int8_t* v : vectors) {
        for (uint32_t i = 0; i < dim; ++i) {
            sums[i] += v[i];
        }
    }

    std::vector<int8_t> result(dim);
    for (uint32_t i = 0; i < dim; ++i) {
        result[i] = (sums[i] >= 0) ? 1 : -1;
    }
    return result;
}

// ── Bitpacking ──────────────────────────────────────────────────────
//
// Converts bipolar {-1, +1} int8 vectors to packed uint32 bit arrays.
//
// Encoding: +1 maps to bit 1, -1 maps to bit 0.
// Little-endian bit order within each uint32 word:
//   - bipolar[0]  -> bit 0 of word 0
//   - bipolar[31] -> bit 31 of word 0
//   - bipolar[32] -> bit 0 of word 1
//
// At d=10240: 320 uint32 words = 1280 bytes per vector.
// This is what the hamming-search.glsl shader operates on.

BitpackedVec vsaBitpack(const int8_t* bipolar, uint32_t dim) {
    BitpackedVec result;
    result.dim = dim;
    uint32_t numWords = (dim + 31) / 32;
    result.data.resize(numWords, 0);

    for (uint32_t i = 0; i < dim; ++i) {
        if (bipolar[i] > 0) {
            uint32_t wordIdx = i / 32;
            uint32_t bitIdx  = i % 32;
            result.data[wordIdx] |= (1u << bitIdx);
        }
        // -1 stays as 0 bit (default)
    }

    return result;
}

std::vector<int8_t> vsaUnpack(const BitpackedVec& packed) {
    std::vector<int8_t> result(packed.dim);
    for (uint32_t i = 0; i < packed.dim; ++i) {
        uint32_t wordIdx = i / 32;
        uint32_t bitIdx  = i % 32;
        uint8_t bit = (packed.data[wordIdx] >> bitIdx) & 1;
        result[i] = bit ? 1 : -1;
    }
    return result;
}

// ── Full Encoding Pipeline ──────────────────────────────────────────
//
// Combines all VSA operations into a single encoding step:
//   1. For each (role_name, filler): bind(BLAKE3(role_name), filler)
//   2. Bundle all bound pairs via majority vote
//   3. Bitpack the result for Hamming search

BitpackedVec vsaEncode(const std::vector<std::string>& roles,
                        const std::vector<const int8_t*>& fillers,
                        uint32_t dim) {
    std::vector<std::vector<int8_t>> boundPairs;
    boundPairs.reserve(roles.size());

    for (size_t i = 0; i < roles.size(); ++i) {
        std::vector<int8_t> role = blake3Role(roles[i], dim);
        boundPairs.push_back(vsaBind(role.data(), fillers[i], dim));
    }

    // Build pointer vector for bundling
    std::vector<const int8_t*> ptrs;
    ptrs.reserve(boundPairs.size());
    for (auto& bp : boundPairs) {
        ptrs.push_back(bp.data());
    }

    std::vector<int8_t> bundled = vsaBundle(ptrs, dim);
    return vsaBitpack(bundled.data(), dim);
}

// ── Hamming Distance ────────────────────────────────────────────────
//
// popcount(XOR(a, b)) counts the number of bit positions where
// a and b differ. Since each bit represents a bipolar element
// (+1 or -1), the Hamming distance equals the number of positions
// where the original bipolar vectors have opposite signs.
//
// On x86: _mm_popcnt_u32 compiles to a single POPCNT instruction.
// On GPU (RDNA 2): bitCount() compiles to s_bcnt1_i32 (scalar) or
// v_bcnt_u32_b32 (vector) — both single-cycle.

/// Portable popcount for uint32.
static inline uint32_t popcount32(uint32_t x) {
#ifdef _MSC_VER
    return __popcnt(x);
#else
    return __builtin_popcount(x);
#endif
}

uint32_t hammingDistance(const uint32_t* a, const uint32_t* b,
                          uint32_t wordsPerVec) {
    uint32_t dist = 0;
    for (uint32_t w = 0; w < wordsPerVec; ++w) {
        dist += popcount32(a[w] ^ b[w]);
    }
    return dist;
}

std::vector<uint32_t> hammingSearchCPU(const uint32_t* queryPacked,
                                        const uint32_t* cachePacked,
                                        uint32_t numEntries,
                                        uint32_t wordsPerVec) {
    std::vector<uint32_t> distances(numEntries);
    for (uint32_t i = 0; i < numEntries; ++i) {
        distances[i] = hammingDistance(queryPacked,
                                       cachePacked + i * wordsPerVec,
                                       wordsPerVec);
    }
    return distances;
}

// ── GPU Hamming Search ──────────────────────────────────────────────
//
// Dispatches hamming-search.glsl when available (loaded in PipelineCache),
// falls back to CPU reference otherwise.
//
// GPU dispatch pattern:
//   1. Acquire input/output GPU buffers
//   2. Upload query + cache to GPU
//   3. Create pipeline with 3 buffer bindings + push constants
//   4. Dispatch: each workgroup (256 threads / 32 per wave = 8 waves)
//      processes 8 cache entries. workgroups = ceil(numEntries / 8).
//   5. Download distances
//   6. Release buffers

void hammingSearch(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                   const uint32_t* queryPacked, const uint32_t* cachePacked,
                   uint32_t* distances,
                   uint32_t numEntries, uint32_t wordsPerVec) {
    if (cache.hasShader("hamming-search")) {
        // GPU path
        const size_t queryBytes = size_t(wordsPerVec) * sizeof(uint32_t);
        const size_t cacheBytes = size_t(numEntries) * wordsPerVec * sizeof(uint32_t);
        const size_t distBytes  = size_t(numEntries) * sizeof(uint32_t);

        GrillyBuffer bufQuery = pool.acquire(queryBytes);
        GrillyBuffer bufCache = pool.acquire(cacheBytes);
        GrillyBuffer bufDist  = pool.acquire(distBytes);

        pool.upload(bufQuery, reinterpret_cast<const float*>(queryPacked),
                    queryBytes);
        pool.upload(bufCache, reinterpret_cast<const float*>(cachePacked),
                    cacheBytes);

        PipelineEntry pipe = cache.getOrCreate("hamming-search", 3,
                                                sizeof(HammingSearchParams));

        std::vector<VkDescriptorBufferInfo> bufInfos = {
            {bufQuery.handle, 0, queryBytes},
            {bufCache.handle, 0, cacheBytes},
            {bufDist.handle,  0, distBytes},
        };
        VkDescriptorSet descSet = cache.allocDescriptorSet("hamming-search",
                                                            bufInfos);

        HammingSearchParams push{numEntries, wordsPerVec, 1, 0};
        // Each workgroup has 256 threads / 32 per wave = 8 subgroups.
        // Each subgroup handles one cache entry cooperatively.
        constexpr uint32_t kEntriesPerWorkgroup = 4;  // Wave64 safe  // 256 / wave_size(32)
        uint32_t gx = (numEntries + kEntriesPerWorkgroup - 1) / kEntriesPerWorkgroup;

        batch.begin();
        batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                       &push, sizeof(push));
        batch.submit();

        pool.download(bufDist, reinterpret_cast<float*>(distances), distBytes);

        pool.release(bufQuery);
        pool.release(bufCache);
        pool.release(bufDist);
    } else {
        // CPU fallback
        auto result = hammingSearchCPU(queryPacked, cachePacked,
                                        numEntries, wordsPerVec);
        std::memcpy(distances, result.data(),
                    numEntries * sizeof(uint32_t));
    }
}

// ── GPU Hamming Search (Persistent Cache) ─────────────────────────────
//
// Same as hammingSearch() but the cache buffer is pre-uploaded.
// Only the query vector is uploaded per call (~1.3 KB at d=10240),
// eliminating the 627 MB PCIe transfer that was the bottleneck.

void hammingSearchPersistent(CommandBatch& batch, BufferPool& pool,
                              PipelineCache& cache,
                              const uint32_t* queryPacked,
                              GrillyBuffer& bufCache, size_t cacheBytes,
                              uint32_t* distances,
                              uint32_t numEntries, uint32_t wordsPerVec) {
    if (cache.hasShader("hamming-search")) {
        const size_t queryBytes = size_t(wordsPerVec) * sizeof(uint32_t);
        const size_t distBytes  = size_t(numEntries) * sizeof(uint32_t);

        GrillyBuffer bufQuery = pool.acquire(queryBytes);
        GrillyBuffer bufDist  = pool.acquire(distBytes);

        pool.upload(bufQuery, reinterpret_cast<const float*>(queryPacked),
                    queryBytes);

        PipelineEntry pipe = cache.getOrCreate("hamming-search", 3,
                                                sizeof(HammingSearchParams));

        std::vector<VkDescriptorBufferInfo> bufInfos = {
            {bufQuery.handle, 0, queryBytes},
            {bufCache.handle, 0, cacheBytes},
            {bufDist.handle,  0, distBytes},
        };
        VkDescriptorSet descSet = cache.allocDescriptorSet("hamming-search",
                                                            bufInfos);

        HammingSearchParams push{numEntries, wordsPerVec, 1, 0};
        constexpr uint32_t kEntriesPerWorkgroup = 4;  // Wave64 safe
        uint32_t gx = (numEntries + kEntriesPerWorkgroup - 1) / kEntriesPerWorkgroup;

        batch.begin();
        batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                       &push, sizeof(push));
        batch.submit();

        pool.download(bufDist, reinterpret_cast<float*>(distances), distBytes);

        pool.release(bufQuery);
        pool.release(bufDist);
    } else {
        // CPU fallback — no persistent cache concept, just use the host data
        // Caller must provide access to host-side cache data separately
        auto result = hammingSearchCPU(queryPacked,
                                        reinterpret_cast<const uint32_t*>(bufCache.mappedPtr),
                                        numEntries, wordsPerVec);
        std::memcpy(distances, result.data(),
                    numEntries * sizeof(uint32_t));
    }
}

}  // namespace cubemind
}  // namespace grilly
