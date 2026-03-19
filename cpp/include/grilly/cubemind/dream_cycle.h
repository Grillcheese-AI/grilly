#pragma once

#include <cstdint>
#include <vector>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/cubemind/cache.h"
#include "grilly/cubemind/types.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace cubemind {

// ── Oja's Rule: Self-Normalizing VSA Plasticity ─────────────────────
//
// Applies Oja's learning rule to VSA memory vectors (pseudo-neurons):
//
//   Δm = η · y · (x - y · m)     where y = m · x (VSA similarity)
//
// Properties:
//   1. Self-normalizing: converges to ||m|| = 1.0
//   2. PCA extraction: amplifies consistent signal, decays noise
//   3. Block-code compatible: preserves per-block normalization
//
// Reference: Oja, E. (1982). J. Math. Biology, 15(3), 267-273.

struct OjaParams {
    uint32_t numVectors;  // Number of memory-input pairs to update
    uint32_t dim;         // VSA dimension per vector
    float eta;            // Learning rate (default 0.01)
};

/// CPU Oja update: apply Oja's rule to a single memory vector.
/// m is updated in place.
void ojaUpdate(float* m, const float* x, uint32_t dim, float eta = 0.01f);

/// CPU batch Oja: update n memory vectors against n input vectors.
/// memories and inputs are contiguous: [n * dim] floats each.
void ojaUpdateBatch(float* memories, const float* inputs,
                     uint32_t n, uint32_t dim, float eta = 0.01f);

/// GPU Oja: dispatch oja-learning.spv shader for batch updates.
/// memories buffer is updated in place on the GPU.
void ojaUpdateGPU(CommandBatch& batch, BufferPool& pool,
                   PipelineCache& pipeCache,
                   float* memories, const float* inputs,
                   uint32_t n, uint32_t dim, float eta = 0.01f);

// ── Sleep Cycle: GPU-Accelerated Memory Consolidation ───────────────
//
// Biologically inspired "dream" consolidation loop:
//
//   1. REPLAY:  Sample high-utility episodic memories from VSACache
//   2. ALIGN:   Find closest semantic prototypes (codebook entries)
//   3. PLASTIC: Apply Oja's rule via GPU shader (oja-learning.spv)
//   4. UPDATE:  Write sharpened prototypes back to codebook
//   5. FORGET:  Decay and prune low-utility episodic traces
//
// The net effect: random superposition noise decays toward 0.0,
// consistent semantic signals converge toward 1.0 (PCA extraction).

struct SleepConfig {
    uint32_t batchSize;       // Memories per dream cycle (default: 256)
    float learningRate;        // Oja eta for consolidation (default: 0.005)
    float utilityDecayRate;    // Decay factor for episodic utilities (default: 0.1)
    float evictionThreshold;   // Evict memories below this utility (default: 0.01)
};

struct SleepStats {
    uint32_t memoriesReplayed;
    uint32_t prototypesUpdated;
    uint32_t memoriesPruned;
    float avgSimilarityBefore;
    float avgSimilarityAfter;
    double wallClockMs;
};

class SleepCycle {
public:
    /// Construct a sleep cycle tied to a cache and codebook.
    /// @param pool       GPU buffer pool for VRAM allocation
    /// @param cache      Hippocampal VSA cache (episodic memories)
    /// @param codebook   Semantic codebook (permanent prototypes)
    /// @param nCodebook  Number of codebook entries
    /// @param dim        VSA dimension
    SleepCycle(BufferPool& pool, VSACache& cache,
               float* codebook, uint32_t nCodebook, uint32_t dim);

    /// Run one dream consolidation cycle.
    /// Samples memories, aligns to prototypes, applies Oja, prunes.
    SleepStats dream(CommandBatch& batch, PipelineCache& pipeCache,
                      const SleepConfig& config = {256, 0.005f, 0.1f, 0.01f});

private:
    BufferPool& pool_;
    VSACache& cache_;
    float* codebook_;
    uint32_t nCodebook_;
    uint32_t dim_;

    /// Find nearest codebook entry for a continuous query vector.
    uint32_t findNearest(const float* query) const;
};

}  // namespace cubemind
}  // namespace grilly
