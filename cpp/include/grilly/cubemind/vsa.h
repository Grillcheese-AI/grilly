#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/cubemind/types.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace cubemind {

// ── VSA Encoding Core ─────────────────────────────────────────────────
//
// Vector Symbolic Architecture operations for CubeMind:
//
//   1. BLAKE3 Role Generation: Deterministic bipolar {-1,+1} vectors
//      from string keys. Matches Python's BinaryOps.hash_to_bipolar().
//
//   2. Binding (XOR / element-wise multiply): Self-inverse composition.
//      bind(a, bind(a, b)) = b. In bipolar domain: a[i] * b[i].
//      In bitpacked domain: XOR.
//
//   3. Bundling (majority vote): Superposition of multiple vectors.
//      sum + sign snap. Preserves similarity to all components.
//
//   4. Bitpacking: {-1,+1} int8 -> packed uint32 bits for Hamming search.
//      +1 maps to bit 1, -1 maps to bit 0.
//
// The BLAKE3 role generation algorithm:
//   For each 32-byte block needed:
//     message = domain + "\x1f" + key + "\x1f" + counter_as_string
//     hash = BLAKE3(message, out_len=32)
//   Unpack hash bytes to bits (little-endian bit order).
//   Map: bit 0 -> -1, bit 1 -> +1.
//
// This matches grilly/utils/stable_hash.py:bipolar_from_key() exactly.

/// Generate a deterministic bipolar role vector from a string key using BLAKE3.
///
/// Algorithm (matching Python BinaryOps.hash_to_bipolar):
///   1. Compute nbytes = (dim + 7) / 8
///   2. Stream 32-byte BLAKE3 digests with incrementing counter:
///        message = join_with_0x1f(domain, key, str(counter))
///        digest = BLAKE3(message, 32)
///   3. Unpack bytes to bits (little-endian bit order)
///   4. Map: bit 0 -> -1, bit 1 -> +1
///
/// @param key    String identifier (e.g., "facelet_0", "color_3")
/// @param dim    Hypervector dimension (e.g., 10240)
/// @param domain Domain prefix for hash isolation (default: "grilly.cubemind")
/// @return Bipolar vector of length dim with values in {-1, +1}
std::vector<int8_t> blake3Role(const std::string& key, uint32_t dim,
                                const std::string& domain = "grilly.cubemind");

/// Bipolar binding: element-wise multiply. Self-inverse: bind(a, bind(a, b)) = b.
/// In bipolar {-1,+1}: a[i] * b[i] stays in {-1,+1}.
std::vector<int8_t> vsaBind(const int8_t* a, const int8_t* b, uint32_t dim);

/// Bipolar bundling: element-wise sum + sign snap.
/// Majority vote superposition — the result is similar to all inputs.
/// Ties (sum == 0) are resolved to +1.
std::vector<int8_t> vsaBundle(const std::vector<const int8_t*>& vectors,
                               uint32_t dim);

/// Bitpack bipolar {-1,+1} int8 -> packed uint32 bits.
/// +1 maps to bit 1, -1 maps to bit 0.
/// Little-endian bit order within each uint32 word.
BitpackedVec vsaBitpack(const int8_t* bipolar, uint32_t dim);

/// Unpack bitpacked uint32 -> bipolar int8 {-1,+1}.
std::vector<int8_t> vsaUnpack(const BitpackedVec& packed);

/// Full VSA encoding pipeline:
///   1. Generate BLAKE3 role vectors for each structural position
///   2. Bind each role with its filler value
///   3. Bundle all bound pairs
///   4. Sign-snap to strict bipolar {-1,+1}
///   5. Bitpack for Hamming search
///
/// @param roles   Vector of role identifiers (e.g., facelet position names)
/// @param fillers Vector of bipolar filler vectors (same count as roles)
/// @param dim     Hypervector dimension
BitpackedVec vsaEncode(const std::vector<std::string>& roles,
                        const std::vector<const int8_t*>& fillers,
                        uint32_t dim);

// ── Hamming Distance Search ───────────────────────────────────────────
//
// GPU-accelerated Hamming distance via XOR + popcount.
// Dispatches hamming-search.glsl when available, falls back to CPU.
//
// On AMD RDNA 2, bitCount() is a single-cycle hardware instruction.
// At d=10240, wordsPerVec=320. Each thread does 320 XOR+popcount ops.
// With 256 threads/workgroup, 490K entries = ~1914 workgroups.
// Expected: <2ms on RX 6750 XT.

/// GPU Hamming distance search: find distances from query to all cache entries.
/// Dispatches hamming-search.glsl shader with hasShader() fallback to CPU.
///
/// @param queryPacked  Bitpacked query vector (wordsPerVec uint32s)
/// @param cachePacked  Contiguous cache (numEntries * wordsPerVec uint32s)
/// @param distances    Output: Hamming distance per entry (numEntries uint32s)
/// @param numEntries   Number of cache entries
/// @param wordsPerVec  uint32 words per bitpacked vector (dim / 32)
void hammingSearch(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                   const uint32_t* queryPacked, const uint32_t* cachePacked,
                   uint32_t* distances,
                   uint32_t numEntries, uint32_t wordsPerVec);

/// GPU Hamming search with persistent GPU cache buffer.
/// The cache buffer (bufCache) must be pre-uploaded and remain valid.
/// Only the query is uploaded per call. This avoids the per-call PCIe
/// transfer of the entire cache, reducing latency from ~40ms to <1ms at 490K.
///
/// @param bufCache   Pre-uploaded GPU buffer containing bitpacked cache
/// @param cacheBytes Total byte size of cache data in bufCache
void hammingSearchPersistent(CommandBatch& batch, BufferPool& pool,
                              PipelineCache& cache,
                              const uint32_t* queryPacked,
                              GrillyBuffer& bufCache, size_t cacheBytes,
                              uint32_t* distances,
                              uint32_t numEntries, uint32_t wordsPerVec);

/// CPU reference for Hamming search (always available, for verification).
std::vector<uint32_t> hammingSearchCPU(const uint32_t* queryPacked,
                                        const uint32_t* cachePacked,
                                        uint32_t numEntries,
                                        uint32_t wordsPerVec);

/// Hamming distance between two single bitpacked vectors.
uint32_t hammingDistance(const uint32_t* a, const uint32_t* b,
                          uint32_t wordsPerVec);

}  // namespace cubemind
}  // namespace grilly
