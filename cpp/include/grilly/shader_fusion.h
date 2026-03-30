#pragma once
/// JIT Shader Fusion Engine — runtime GLSL generation + shaderc compilation.
///
/// Detects fusible op sequences from the traced graph, generates a single
/// GLSL compute shader that chains the operations in thread-local registers
/// (no VRAM round-trips between ops), compiles to SPIR-V via shaderc,
/// and caches the resulting VkPipeline by graph hash.
///
/// This is the grilly equivalent of PyTorch's TorchInductor/Triton codegen.
///
/// Supported fusion patterns:
///   Elementwise chains: linear → gelu, linear → relu, linear → silu,
///     linear → gelu → dropout, add → layernorm, etc.
///   Any sequence of pointwise ops on the same tensor can be fused.
///
/// Architecture:
///   1. Python jit.py traces OpRecords → serializes to FusionGraph
///   2. C++ generate_fused_shader() emits GLSL string from FusionGraph
///   3. C++ compile_glsl_to_spirv() compiles via shaderc (10-20ms first time)
///   4. PipelineCache stores the compiled VkPipeline by graph hash
///   5. Subsequent calls with same graph → instant cache hit

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "grilly/pipeline_cache.h"

namespace grilly {

/// A single op in the fusion graph (passed from Python).
struct FusionOp {
    std::string opType;      // "linear", "gelu", "relu", "dropout", "add", etc.
    uint32_t numInputs;
    uint32_t numOutputs;
    std::unordered_map<std::string, float> params;  // e.g., dropout_p=0.1
};

/// A complete fusion graph to compile into a single shader.
struct FusionGraph {
    std::vector<FusionOp> ops;
    uint32_t totalElements;   // Total elements for dispatch sizing
    uint32_t numInputBuffers; // How many input SSBOs
    uint32_t numOutputBuffers;
};

/// Result of shader fusion compilation.
struct FusionResult {
    std::string shaderName;   // Cache key (hash-based)
    std::string glslSource;   // Generated GLSL (for debugging)
    bool cacheHit;            // True if pipeline was already compiled
    float compileTimeMs;      // Time spent in shaderc (0 on cache hit)
};

/// The JIT Shader Fusion Engine.
class ShaderFusionEngine {
public:
    explicit ShaderFusionEngine(const std::string& snippetDir);

    /// Generate and compile a fused shader for the given op graph.
    /// Returns the shader name (key into PipelineCache).
    /// On cache hit, returns immediately (0ms).
    /// On cache miss, generates GLSL + compiles via shaderc (~10-20ms).
    FusionResult fuse(const FusionGraph& graph, PipelineCache& cache);

    /// Check if a fusion graph has already been compiled.
    bool isCached(const FusionGraph& graph) const;

    /// Get compilation statistics.
    struct Stats {
        uint32_t cacheHits = 0;
        uint32_t cacheMisses = 0;
        float totalCompileMs = 0.0f;
        uint32_t shadersCompiled = 0;
    };
    Stats stats() const { return stats_; }

private:
    /// Generate GLSL source for a fusion graph.
    std::string generateGLSL(const FusionGraph& graph) const;

    /// Compile GLSL string to SPIR-V via shaderc.
    std::vector<uint32_t> compileToSPIRV(const std::string& glsl,
                                          const std::string& name) const;

    /// Compute a deterministic hash for a fusion graph.
    std::string computeHash(const FusionGraph& graph) const;

    /// Load a snippet file from disk.
    std::string loadSnippet(const std::string& name) const;

    std::string snippetDir_;
    std::unordered_map<std::string, std::string> snippetCache_;
    std::unordered_map<std::string, bool> compiledGraphs_;
    Stats stats_;
};

}  // namespace grilly
