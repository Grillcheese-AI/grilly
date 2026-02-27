#pragma once

#include <cstdint>
#include <vector>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"
#include "grilly/experimental/paged_latent_pool.h"

namespace grilly {
namespace experimental {

// ── Fused Subgroup-Decompress + Flash Attention ─────────────────────────
//
// This is the endgame: MLA decompression and flash attention fused into
// a single shader dispatch. The full-precision K/V never exists in VRAM.
//
// Standard pipeline (what we had before):
//   1. Read compressed latent c_t from VRAM        → L2 cache hit
//   2. Decompress: [K, V] = W_up @ c_t             → Write to VRAM
//   3. Read K from VRAM for attention               → L2 cache miss (evicted)
//   4. Read V from VRAM for attention               → L2 cache miss
//   5. Write attention output to VRAM               → VRAM write
//   Total: 2 VRAM reads + 2 VRAM writes + 1 VRAM read = 5 memory round-trips
//
// Fused pipeline (what this does):
//   1. Read compressed latent c_t from VRAM         → L2 cache hit
//   2. Decompress in registers: subgroup matmul     → stays in VGPR/SGPR
//   3. Dequant K (FP8) + V (INT4) in registers     → subgroupBroadcastFirst for scales
//   4. Dot product Q @ K^T in registers             → subgroupAdd for reduction
//   5. Softmax correction in registers              → subgroupMax + subgroupAdd
//   6. Accumulate V weighted sum in registers       → stays in VGPR
//   7. Write final output to VRAM                   → 1 VRAM write
//   Total: 1 VRAM read + 1 VRAM write = 2 memory round-trips
//
// This is a 2.5x reduction in memory traffic, which directly translates
// to throughput improvement for memory-bound attention operations.
//
// The shader structure (conceptual GLSL):
//
//   layout(local_size_x = 32) in;  // Wave32
//
//   void main() {
//       uint lane = gl_SubgroupInvocationID;
//       uint head = gl_WorkGroupID.x;
//       uint q_pos = gl_WorkGroupID.y;
//
//       // Load query for this position (from VRAM, but it's hot in L2)
//       vec4 q = load_query(q_pos, head);
//
//       float running_max = -1e30;
//       float running_sum = 0.0;
//       vec4 accum = vec4(0.0);
//
//       // Tile loop over cached tokens
//       for (uint tile = 0; tile < num_tiles; tile++) {
//           uint token_id = tile * 32 + lane;  // Each lane handles one token
//
//           // ── STEP 1: Load compressed latent from paged pool ──
//           // Each lane loads its token's latent vector (small: latent_dim floats)
//           // This is the ONLY VRAM read in the inner loop
//           vec4 latent = load_latent(pool, token_id);
//
//           // ── STEP 2: Decompress K in registers ──
//           // W_up_k is loaded once per head (shared across tiles)
//           // Subgroup matmul: each lane contributes one element of the dot product
//           float k_element = 0.0;
//           for (uint l = 0; l < latent_dim / 4; l++) {
//               k_element += dot(latent, w_up_k[l]);
//           }
//           // FP8 quantize in register (just bit manipulation, no memory)
//           uint8_t k_fp8 = float_to_fp8_e4m3(k_element);
//
//           // ── STEP 3: Compute attention score in registers ──
//           float score = dot(q, subgroupBroadcast(k_decoded, lane));
//           score *= scale;
//
//           // ── STEP 4: Online softmax in registers ──
//           float tile_max = subgroupMax(score);
//           float correction = exp(running_max - max(running_max, tile_max));
//           running_sum = running_sum * correction + subgroupAdd(exp(score - tile_max));
//           running_max = max(running_max, tile_max);
//
//           // ── STEP 5: Decompress V and accumulate in registers ──
//           float v_element = decompress_int4(latent, w_up_v);
//           accum = accum * correction + exp(score - tile_max) * v_broadcast;
//       }
//
//       // ── STEP 6: Final output = accum / running_sum ──
//       vec4 output = accum / running_sum;
//       store_output(q_pos, head, output);  // ONLY VRAM write
//   }

/// Push constants for fused decompression + attention
struct FusedAttentionParams {
    uint32_t batchSize;
    uint32_t seqLen;          // Query sequence length
    uint32_t numHeads;
    uint32_t headDim;
    uint32_t latentDim;       // MLA compressed dimension
    float scale;              // 1/sqrt(head_dim)
    uint32_t cachedTokens;    // Number of tokens in KV cache
    uint32_t tileSizeK;       // Tokens per tile (== wave size for register-only)
    uint32_t hasMask;
    uint32_t waveSize;        // 32 for RDNA 2, 64 for GCN
    uint32_t quantMode;       // 0=FP32, 1=FP8_K+INT4_V, 2=FP8_K+INT2_V
    uint32_t passType;        // 0=init, 1=tile, 2=finalize
};

/// Execute fused subgroup-decompress + flash attention.
///
/// This is the highest-performance path: reads compressed latents from
/// the paged pool, decompresses in Wave32 registers, computes attention
/// entirely in registers using subgroup operations, and writes only the
/// final output to VRAM.
///
/// Requirements:
///   - GPU must support VK_KHR_shader_subgroup with subgroupQuadBroadcast
///   - Paged latent pool must be allocated with MEM_DEVICE_LOCAL
///   - W_up projection matrix must be pre-loaded into GPU buffer
///
/// @param Q          Query tensor (batchSize, numHeads, seqLen, headDim)
/// @param pool       Paged latent pool containing compressed KV cache
/// @param wUp        W_up projection matrix (on GPU)
/// @param mask       Optional attention mask (or nullptr)
/// @param output     Output tensor (batchSize, numHeads, seqLen, headDim)
void fusedSubgroupAttention(
    CommandBatch& batch, BufferPool& bufPool, PipelineCache& cache,
    const float* Q, const PagedLatentPool& pool,
    const GrillyBuffer& wUp, const float* mask,
    float* output,
    uint32_t batchSize, uint32_t seqLen,
    uint32_t numHeads, uint32_t headDim, uint32_t latentDim,
    float scale = 0.0f,
    uint32_t waveSize = 32,
    uint32_t quantMode = 0);

/// CPU reference implementation for fused attention (for verification).
/// Does the same computation but without subgroup ops — sequential.
void fusedAttentionCPU(
    const float* Q, const float* latents, const float* wUp,
    const float* mask, float* output,
    uint32_t batchSize, uint32_t seqLen, uint32_t cachedTokens,
    uint32_t numHeads, uint32_t headDim, uint32_t latentDim,
    float scale = 0.0f);

}  // namespace experimental
}  // namespace grilly
