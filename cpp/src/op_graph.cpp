#include "grilly/op_graph.h"

#include <cstring>
#include <stdexcept>

namespace grilly {

void OpGraph::addOp(const std::string& shaderName, uint32_t numBuffers,
                    const std::vector<VkDescriptorBufferInfo>& bufferInfos,
                    const void* pushData, uint32_t pushSize,
                    uint32_t groupX, uint32_t groupY, uint32_t groupZ) {
    OpNode node;
    node.shaderName = shaderName;
    node.numBuffers = numBuffers;
    node.bufferInfos = bufferInfos;
    node.pushSize = pushSize;
    node.groupX = groupX;
    node.groupY = groupY;
    node.groupZ = groupZ;

    if (pushData && pushSize > 0) {
        node.pushData.resize(pushSize);
        std::memcpy(node.pushData.data(), pushData, pushSize);
    }

    ops_.push_back(std::move(node));
}

// ── Built-in Fusion Rules ───────────────────────────────────────────────
//
// These map pairs of consecutive shaders to pre-compiled fused variants.
// The fused shaders must exist as .spv files. If a fused shader isn't
// loaded in the PipelineCache, that fusion rule is silently skipped.
//
// Current fused shader inventory:
//   fnn-linear + activation-relu  → fnn-linear-relu
//   fnn-linear + activation-gelu  → fnn-linear-gelu
//   fnn-linear + activation-silu  → fnn-linear-silu
//   gemm_mnk   + bias-add         → gemm-bias        (conv2d GEMM path)
//
// Each fused shader combines the buffer bindings of both ops:
//   fnn-linear-relu: 4 buffers (input, weights, bias, output) + 1 push
//   The relu is applied in-place on the output write, so the intermediate
//   buffer (linear output / relu input) never materializes in VRAM.

const std::vector<FusionRule>& OpGraph::builtinRules() {
    static const std::vector<FusionRule> rules = {
        // Linear + Activation fusions
        // The fused shader applies the activation during the output store,
        // eliminating one full buffer read+write cycle.
        {"fnn-linear", "activation-relu", "fnn-linear-relu", 4,
         sizeof(uint32_t) * 4},
        {"fnn-linear", "activation-gelu", "fnn-linear-gelu", 4,
         sizeof(uint32_t) * 4},
        {"fnn-linear", "activation-silu", "fnn-linear-silu", 4,
         sizeof(uint32_t) * 4},
        {"fnn-linear", "activation-tanh", "fnn-linear-tanh", 4,
         sizeof(uint32_t) * 4},

        // GEMM + bias addition (for conv2d GEMM path)
        // Eliminates the CPU download for bias — bias is applied in the
        // same shader that computes the matmul output.
        {"gemm_mnk", "bias-add", "gemm-bias", 4, sizeof(uint32_t) * 4},

        // LayerNorm internal fusion: mean+variance passes
        // When LayerNorm is dispatched as 3 separate passes, the first two
        // (mean and variance) can be fused into a single Welford pass.
        {"fnn-layernorm-mean", "fnn-layernorm-var", "fnn-layernorm-meanvar",
         4, sizeof(uint32_t) * 4},
    };
    return rules;
}

void OpGraph::addFusionRule(FusionRule rule) {
    customRules_.push_back(std::move(rule));
}

// ── Buffer Overlap Detection ────────────────────────────────────────────
//
// Two ops have a data dependency if they share any VkBuffer handle.
// An op's output buffer appears in its bufferInfos; if the next op reads
// from the same handle, we must insert a barrier. If they share no buffers,
// they can execute in parallel on the GPU without a barrier.
//
// This is a conservative analysis — we check buffer handles, not byte
// ranges within buffers. A more precise analysis could use offset+range
// overlap, but buffer-level granularity is sufficient for our use case
// since each op typically gets its own set of buffers from the pool.

bool OpGraph::hasBufferOverlap(const OpNode& a, const OpNode& b) {
    for (const auto& bufA : a.bufferInfos) {
        for (const auto& bufB : b.bufferInfos) {
            if (bufA.buffer == bufB.buffer) {
                return true;
            }
        }
    }
    return false;
}

// ── Fusion Optimization Pass ────────────────────────────────────────────
//
// The optimization runs in two phases:
//
// Phase 1: Op Fusion
//   Scan consecutive pairs. For each pair (ops_[i], ops_[i+1]), check all
//   fusion rules. If a match is found AND the fused shader is loaded in
//   the cache, merge the two ops into one.
//
//   The merged op uses:
//     - The fused shader name
//     - Buffer bindings from the first op (the fused shader is designed
//       to take the same inputs and produce the final output directly)
//     - Push constants from the first op (fused shaders share the same
//       push layout — dimension parameters that both ops need)
//     - Workgroup dimensions from the first op
//
// Phase 2: Barrier Elimination
//   For each consecutive pair in the (potentially fused) op list, check
//   if they share any buffer handles. If not, mark that no barrier is
//   needed between them. This allows the GPU to execute them in parallel.

FusionStats OpGraph::optimize(PipelineCache& cache) {
    FusionStats stats{};
    stats.originalOps = static_cast<uint32_t>(ops_.size());

    if (ops_.size() < 2) {
        stats.optimizedOps = stats.originalOps;
        needsBarrier_.assign(ops_.size(), true);
        return stats;
    }

    // Combine built-in and custom rules
    std::vector<FusionRule> allRules = builtinRules();
    allRules.insert(allRules.end(), customRules_.begin(), customRules_.end());

    // Phase 1: Op Fusion — scan for fusible pairs
    std::vector<OpNode> fused;
    fused.reserve(ops_.size());

    size_t i = 0;
    while (i < ops_.size()) {
        bool merged = false;

        if (i + 1 < ops_.size()) {
            const auto& first = ops_[i];
            const auto& second = ops_[i + 1];

            for (const auto& rule : allRules) {
                if (first.shaderName == rule.firstShader &&
                    second.shaderName == rule.secondShader) {
                    // Check if the fused shader is available
                    if (cache.hasShader(rule.fusedShader)) {
                        // Merge: create fused op with first op's buffers
                        // and push constants (fused shader is designed to
                        // accept the same input layout)
                        OpNode fusedOp;
                        fusedOp.shaderName = rule.fusedShader;
                        fusedOp.numBuffers = rule.fusedNumBuffers;
                        fusedOp.bufferInfos = first.bufferInfos;
                        fusedOp.pushData = first.pushData;
                        fusedOp.pushSize = rule.fusedPushSize;
                        fusedOp.groupX = first.groupX;
                        fusedOp.groupY = first.groupY;
                        fusedOp.groupZ = first.groupZ;

                        fused.push_back(std::move(fusedOp));
                        stats.opsFused++;
                        i += 2;  // skip both ops
                        merged = true;
                        break;
                    }
                }
            }
        }

        if (!merged) {
            fused.push_back(std::move(ops_[i]));
            i++;
        }
    }

    ops_ = std::move(fused);

    // Phase 2: Barrier Elimination — detect independent ops
    needsBarrier_.resize(ops_.size(), true);
    needsBarrier_[0] = false;  // no barrier before the first op

    for (size_t j = 1; j < ops_.size(); ++j) {
        if (!hasBufferOverlap(ops_[j - 1], ops_[j])) {
            needsBarrier_[j] = false;
            stats.barriersEliminated++;
        }
    }

    stats.optimizedOps = static_cast<uint32_t>(ops_.size());
    return stats;
}

// ── Execute: single CommandBatch submission for N ops ──────────────────────
//
// The key optimization: instead of N separate begin/submit/fence-wait cycles
// (which is what the Python backend does), we record all dispatches into one
// command buffer with pipeline barriers between them. The GPU processes the
// chain without CPU intervention.
//
// If optimize() was called, barriers are only inserted between ops that have
// data dependencies (shared buffers). Independent ops can execute in parallel
// on the GPU's multiple compute units.

void OpGraph::execute(CommandBatch& batch, PipelineCache& cache) {
    if (ops_.empty()) return;

    // If optimize() wasn't called, default to barriers everywhere
    if (needsBarrier_.empty()) {
        needsBarrier_.assign(ops_.size(), true);
        needsBarrier_[0] = false;
    }

    batch.begin();

    for (size_t i = 0; i < ops_.size(); ++i) {
        // Insert barrier if needed (data dependency with previous op)
        if (i > 0 && needsBarrier_[i]) {
            batch.barrier();
        }

        const auto& op = ops_[i];

        // Get or create pipeline for this shader
        PipelineEntry pipe = cache.getOrCreate(
            op.shaderName, op.numBuffers, op.pushSize);

        // Allocate descriptor set (LRU cached)
        VkDescriptorSet descSet = cache.allocDescriptorSet(
            op.shaderName, op.bufferInfos);

        // Record the dispatch
        const void* push = op.pushData.empty() ? nullptr : op.pushData.data();
        batch.dispatch(pipe.pipeline, pipe.layout, descSet,
                       op.groupX, op.groupY, op.groupZ,
                       push, op.pushSize);
    }

    batch.submit();
}

}  // namespace grilly
