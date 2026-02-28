#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly {

/// A single recorded dispatch operation within an OpGraph.
struct OpNode {
    std::string shaderName;
    uint32_t numBuffers;
    std::vector<VkDescriptorBufferInfo> bufferInfos;
    std::vector<uint8_t> pushData;
    uint32_t pushSize;
    uint32_t groupX, groupY, groupZ;
};

// ── Operator Fusion ─────────────────────────────────────────────────────
//
// Unlike Triton's JIT kernel fusion, grilly uses a rule-based fusion pass
// that rewrites the op graph before execution. The approach:
//
// 1. Pattern matching: scan consecutive op pairs for known fusible patterns
//    (e.g., fnn-linear followed by activation-relu → fnn-linear-relu)
//
// 2. Buffer elimination: when two ops are fused, the intermediate buffer
//    (output of op A = input of op B) is no longer needed. The fused shader
//    reads A's inputs and writes B's output directly.
//
// 3. Barrier elimination: independent ops (no shared buffer handles) can
//    execute without barriers between them, enabling GPU-level parallelism.
//
// Why not JIT fusion like Triton?
//   Triton generates PTX/AMDGPU IR at runtime, fusing arbitrary ops into
//   custom kernels. This requires an LLVM backend and GPU ISA knowledge.
//   We instead pre-compile fused GLSL shaders (fnn-linear-relu.spv,
//   fnn-linear-gelu.spv, etc.) and select them at graph optimization time.
//   This is simpler, deterministic, and has zero JIT overhead — but limited
//   to patterns we've anticipated and compiled shaders for.

/// Rule for fusing two consecutive operations into one.
struct FusionRule {
    std::string firstShader;     // e.g., "fnn-linear"
    std::string secondShader;    // e.g., "activation-relu"
    std::string fusedShader;     // e.g., "fnn-linear-relu"
    uint32_t fusedNumBuffers;    // Buffer count for fused shader
    uint32_t fusedPushSize;      // Push constant size for fused shader
};

/// Statistics from the optimization pass.
struct FusionStats {
    uint32_t opsFused;           // Number of op pairs merged
    uint32_t barriersEliminated; // Barriers removed between independent ops
    uint32_t originalOps;        // Op count before optimization
    uint32_t optimizedOps;       // Op count after optimization
};

/// Multi-operation graph for batched GPU execution with operator fusion.
///
/// The Python backend submits and waits on a fence after every single dispatch
/// (core.py:743). OpGraph records N dispatches into one CommandBatch with
/// COMPUTE->COMPUTE barriers between them, then submits once.
///
/// The optimize() pass fuses compatible op pairs and eliminates unnecessary
/// barriers, further reducing dispatch count and GPU idle time.
///
/// Usage:
///   OpGraph graph;
///   graph.addOp("fnn-linear",      4, bufInfos, pushData, 16, gx, gy, 1);
///   graph.addOp("activation-relu", 2, bufInfos, pushData, 4,  gx, 1, 1);
///   auto stats = graph.optimize(cache);  // fuses into "fnn-linear-relu"
///   graph.execute(batch, cache);         // single dispatch instead of two
///
/// This eliminates N-1 fence waits per forward pass (for N sequential ops)
/// and reduces dispatch count via operator fusion.
class OpGraph {
public:
    OpGraph() = default;

    /// Record a dispatch into the graph (no GPU work yet).
    void addOp(const std::string& shaderName, uint32_t numBuffers,
               const std::vector<VkDescriptorBufferInfo>& bufferInfos,
               const void* pushData, uint32_t pushSize,
               uint32_t groupX, uint32_t groupY = 1, uint32_t groupZ = 1);

    /// Run the fusion optimization pass on the recorded ops.
    ///
    /// This scans for fusible op pairs (e.g., linear + relu) and replaces
    /// them with a single fused op. Only fuses when the fused shader is
    /// available in the PipelineCache (i.e., the .spv file was loaded).
    ///
    /// Also marks independent ops (no buffer overlap) to skip barriers.
    ///
    /// Returns statistics about what was optimized.
    FusionStats optimize(PipelineCache& cache);

    /// Execute all recorded ops in a single CommandBatch submission.
    /// Inserts COMPUTE->COMPUTE barriers between each dispatch so results
    /// from op N are visible to op N+1 (skipped for independent ops if
    /// optimize() was called).
    void execute(CommandBatch& batch, PipelineCache& cache);

    /// Number of ops recorded.
    size_t size() const { return ops_.size(); }

    /// Clear all recorded ops for reuse.
    void clear() {
        ops_.clear();
        needsBarrier_.clear();
    }

    /// Add a custom fusion rule (in addition to built-in rules).
    void addFusionRule(FusionRule rule);

private:
    /// Check if two ops share any buffer handles (data dependency).
    static bool hasBufferOverlap(const OpNode& a, const OpNode& b);

    /// Get built-in fusion rules.
    static const std::vector<FusionRule>& builtinRules();

    std::vector<OpNode> ops_;
    std::vector<bool> needsBarrier_;  // Per-op: needs barrier before it?
    std::vector<FusionRule> customRules_;
};

}  // namespace grilly
