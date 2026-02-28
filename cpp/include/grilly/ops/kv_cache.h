#pragma once

#include <cstdint>
#include <vector>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace ops {

// ── KV Cache with MLA + H2O + Speculative Eviction + Cross-Layer Sharing ────
//
// This is the most architecturally complex component. The cache stores
// compressed key-value pairs using several cutting-edge techniques:
//
// 1. MLA (Multi-head Latent Attention): Project KV to low-rank latent
//    c_t = W_down * h_t, decode on-the-fly [K_t, V_t] = W_up * c_t.
//    Reduces KV cache memory by compression_ratio (e.g., 4x).
//
// 2. Asymmetric Sub-byte Quantization:
//    - Keys stored in FP8 (E4M3) — sensitive to outliers
//    - Values stored in INT4/INT2 — robust to quantization
//    - Scaling factors kept in Wave32 subgroup registers (AMD RDNA 2)
//      so dequantization requires zero extra memory fetches.
//
// 3. H2O (Heavy Hitter Oracle) Eviction:
//    Track cumulative attention scores per token. When cache is full,
//    evict the token with the lowest cumulative score. Heavy hitters
//    (tokens that consistently receive high attention) are preserved.
//
// 4. Speculative Eviction via Auxiliary Heads:
//    A tiny linear predictor head estimates future attention relevance.
//    Tokens with low predicted future relevance are evicted proactively
//    (before the cache is full). GPU-side linked list of active tokens
//    in a storage buffer; compaction shader defragments periodically.
//
// 5. Cross-Layer KV Sharing:
//    Adjacent layers L and L+1 share the same KV cache. They are
//    differentiated via RoPE rotation (layer L gets standard RoPE,
//    layer L+1 gets a rotated variant). This halves VRAM usage
//    for KV storage.

/// Configuration for KV cache creation.
struct KVCacheConfig {
    uint32_t maxSeqLen;           // Maximum sequence length
    uint32_t numHeads;            // Number of attention heads
    uint32_t headDim;             // Dimension per head
    uint32_t numLayers;           // Total transformer layers
    uint32_t compressionRatio;    // MLA compression ratio (e.g., 4)
    uint32_t maxCacheTokens;      // Max tokens before eviction

    // Quantization
    bool useAsymmetricQuant;      // FP8 keys + INT4 values
    uint32_t valueBits;           // 4 for INT4, 2 for INT2

    // Sharing
    bool crossLayerSharing;       // Adjacent layers share KV

    // Eviction strategy
    bool useH2O;                  // Heavy Hitter Oracle
    bool useSpeculativeEviction;  // Auxiliary head prediction
    float evictionThreshold;      // Score below which to evict (0.0-1.0)
};

/// Per-token metadata for eviction decisions.
struct TokenMeta {
    float cumulativeScore;        // H2O: sum of attention received
    float predictedRelevance;     // Speculative: aux head prediction
    uint32_t tokenIdx;            // Original position in sequence
    uint32_t nextActive;          // Linked list: next active token (GPU-side)
};

// ── Trainable Eviction Head ──────────────────────────────────────────────
//
// Instead of a hand-tuned heuristic, we train a tiny auxiliary MLP to
// predict future attention relevance for each cached token. The MLP:
//
//   input: per-token feature vector (latent or hidden state)
//   hidden: ReLU(W1 @ input + b1)  — small hidden layer (32-64 units)
//   output: sigmoid(W2 @ hidden + b2)  — scalar relevance score in [0,1]
//
// Training signal: the actual attention scores from the latest forward
// pass serve as ground truth. We compute binary cross-entropy loss
// between predicted relevance and a binarized attention signal (tokens
// receiving above-median attention = 1, below = 0).
//
// The head is trained online via SGD after each forward pass. Since it's
// tiny (~2K-8K parameters), training is negligible overhead compared to
// the attention computation itself.
//
// Why this is better than the heuristic:
//   - Adapts to input distribution (the heuristic uses fixed weights)
//   - Captures nonlinear relationships between token features and
//     future relevance
//   - Improves over time as it sees more data

/// Trainable auxiliary head for speculative eviction.
struct EvictionHead {
    // Layer 1: (inputDim, hiddenDim)
    GrillyBuffer w1;
    GrillyBuffer b1;
    // Layer 2: (hiddenDim, 1)
    GrillyBuffer w2;
    GrillyBuffer b2;

    // Gradient buffers (for SGD training)
    GrillyBuffer gradW1;
    GrillyBuffer gradB1;
    GrillyBuffer gradW2;
    GrillyBuffer gradB2;

    uint32_t inputDim;      // Feature dimension per token
    uint32_t hiddenDim;     // Hidden layer size (default: 32)
    float learningRate;     // SGD step size (default: 1e-3)
    uint32_t totalUpdates;  // Number of training steps performed
    bool initialized;       // Has the head been created?
};

/// Opaque handle to a KV cache instance.
/// The actual GPU buffers are managed internally.
struct KVCache {
    // MLA projection matrices (on GPU)
    GrillyBuffer wDown;           // (head_dim, latent_dim) per head
    GrillyBuffer wUp;             // (latent_dim, 2*head_dim) per head

    // Compressed cache storage
    GrillyBuffer latents;         // MLA: compressed c_t per token
    GrillyBuffer keysQuant;       // FP8 quantized keys (if asymmetric)
    GrillyBuffer valuesQuant;     // INT4/INT2 quantized values (if asymmetric)
    GrillyBuffer scaleFactors;    // Per-group quantization scales

    // Eviction metadata
    GrillyBuffer tokenMeta;       // Per-token scores and linked list
    GrillyBuffer activeList;      // Linked list head pointers

    // Trainable eviction head
    EvictionHead evictionHead;

    // State
    uint32_t currentLen;          // Current number of cached tokens
    KVCacheConfig config;
};

/// Create a KV cache with the given configuration.
/// Allocates all GPU buffers via the buffer pool.
KVCache createKVCache(BufferPool& pool, const KVCacheConfig& config);

/// Destroy a KV cache, releasing all GPU buffers.
void destroyKVCache(BufferPool& pool, KVCache& cache);

/// Append new KV pairs to the cache.
///
/// 1. If MLA enabled: compress via c_t = W_down * [k_t; v_t]
/// 2. If asymmetric quant: quantize K to FP8, V to INT4
/// 3. Store in cache at position currentLen
/// 4. If cache full: run eviction (H2O or speculative)
///
/// @param newKeys   (numNewTokens, numHeads, headDim) float32
/// @param newValues (numNewTokens, numHeads, headDim) float32
/// @param numNewTokens Number of new tokens to append
void kvCacheAppend(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                   KVCache& kvCache,
                   const float* newKeys, const float* newValues,
                   uint32_t numNewTokens);

/// Decode KV from cache for attention computation.
///
/// If MLA: [K, V] = W_up * c_t (decoded on-the-fly in GPU registers)
/// If quantized: dequantize using Wave32 subgroup scaling factors
///
/// @param decodedKeys   Output: (cachedTokens, numHeads, headDim)
/// @param decodedValues Output: (cachedTokens, numHeads, headDim)
void kvCacheDecode(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                   const KVCache& kvCache,
                   float* decodedKeys, float* decodedValues);

/// Run H2O eviction: update cumulative attention scores and evict
/// tokens below the threshold.
///
/// @param attentionScores (seqLen, cachedTokens) — attention weights
///                        from the latest forward pass
/// @param numEvict Target number of tokens to evict (0 = auto)
void kvCacheEvictH2O(CommandBatch& batch, BufferPool& pool,
                     PipelineCache& cache, KVCache& kvCache,
                     const float* attentionScores, uint32_t numEvict = 0);

/// Run speculative eviction: auxiliary head predicts future relevance,
/// evicts tokens with low predicted scores.
///
/// @param hiddenStates Current hidden states for aux head input
/// @param hiddenDim    Dimension of hidden states
void kvCacheEvictSpeculative(CommandBatch& batch, BufferPool& pool,
                             PipelineCache& cache, KVCache& kvCache,
                             const float* hiddenStates, uint32_t hiddenDim);

/// Compact the KV cache: defragment the linked list of active tokens
/// into a contiguous array. Should be called periodically after evictions.
void kvCacheCompact(CommandBatch& batch, BufferPool& pool,
                    PipelineCache& cache, KVCache& kvCache);

/// Initialize the trainable eviction head for a KV cache.
/// Must be called before using speculative eviction with training.
///
/// @param inputDim   Feature dimension per token (typically headDim or latentDim)
/// @param hiddenDim  Hidden layer size (default: 32)
/// @param lr         Learning rate for SGD (default: 1e-3)
void kvCacheInitEvictionHead(BufferPool& pool, KVCache& kvCache,
                              uint32_t inputDim, uint32_t hiddenDim = 32,
                              float lr = 1e-3f);

/// Destroy the eviction head, releasing its GPU buffers.
void kvCacheDestroyEvictionHead(BufferPool& pool, KVCache& kvCache);

/// Train the eviction head on attention scores from the latest forward pass.
///
/// Uses the attention pattern as supervision: tokens receiving above-median
/// attention are labeled as relevant (1), below-median as irrelevant (0).
/// The head learns to predict this signal from token features.
///
/// @param tokenFeatures  (currentLen, inputDim) — per-token feature vectors
/// @param attentionScores (seqLen, currentLen) — latest attention weights
/// @param seqLen          Number of query positions in the attention scores
void kvCacheTrainEvictionHead(CommandBatch& batch, BufferPool& pool,
                               PipelineCache& cache, KVCache& kvCache,
                               const float* tokenFeatures,
                               const float* attentionScores,
                               uint32_t seqLen);

/// Get current cache statistics.
struct KVCacheStats {
    uint32_t currentTokens;       // Currently cached tokens
    uint32_t maxTokens;           // Maximum capacity
    uint32_t totalEvicted;        // Total tokens evicted
    uint32_t totalAppended;       // Total tokens ever appended
    float compressionRatio;       // Effective memory compression
    float avgAttentionScore;      // Average H2O score of active tokens
};
KVCacheStats kvCacheGetStats(const KVCache& kvCache);

}  // namespace ops
}  // namespace grilly
