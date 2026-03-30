/// shader_fusion.cpp — JIT Shader Fusion Engine.
///
/// Generates fused GLSL from traced op graphs, compiles to SPIR-V via
/// shaderc at runtime, caches compiled pipelines by graph hash.

#include "grilly/shader_fusion.h"

#include <blake3.h>

#ifdef GRILLY_HAS_SHADERC
#include <shaderc/shaderc.hpp>
#endif

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace grilly {

ShaderFusionEngine::ShaderFusionEngine(const std::string& snippetDir)
    : snippetDir_(snippetDir) {
    // Pre-load all snippet files
    namespace fs = std::filesystem;
    if (fs::exists(snippetDir)) {
        for (const auto& entry : fs::directory_iterator(snippetDir)) {
            if (entry.path().extension() == ".glsl") {
                std::string name = entry.path().stem().string();
                std::ifstream f(entry.path());
                std::string content((std::istreambuf_iterator<char>(f)),
                                    std::istreambuf_iterator<char>());
                snippetCache_[name] = content;
            }
        }
    }
}

std::string ShaderFusionEngine::loadSnippet(const std::string& name) const {
    auto it = snippetCache_.find(name);
    if (it != snippetCache_.end()) return it->second;
    return "";  // Snippet not found — op will be inlined
}

std::string ShaderFusionEngine::computeHash(const FusionGraph& graph) const {
    // BLAKE3 hash of op types + params → deterministic cache key
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);

    for (const auto& op : graph.ops) {
        blake3_hasher_update(&hasher, op.opType.data(), op.opType.size());
        for (const auto& [k, v] : op.params) {
            blake3_hasher_update(&hasher, k.data(), k.size());
            blake3_hasher_update(&hasher, &v, sizeof(v));
        }
    }
    uint32_t n = graph.totalElements;
    blake3_hasher_update(&hasher, &n, sizeof(n));

    uint8_t hash[16];
    blake3_hasher_finalize(&hasher, hash, 16);

    // Hex string
    char hex[33];
    for (int i = 0; i < 16; i++)
        snprintf(hex + i * 2, 3, "%02x", hash[i]);
    return std::string("fused_") + hex;
}

// ── GLSL Code Generation ─────────────────────────────────────────────

std::string ShaderFusionEngine::generateGLSL(const FusionGraph& graph) const {
    std::stringstream ss;

    // Header
    ss << "#version 450\n";
    ss << "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n";

    // Determine buffer bindings
    uint32_t binding = 0;
    ss << "layout(std430, set = 0, binding = " << binding++
       << ") readonly buffer InBuf { float Input[]; };\n";

    // Check if we need a second input (e.g., for add/residual)
    bool needsSecondInput = false;
    for (const auto& op : graph.ops) {
        if (op.opType == "add" || op.opType == "residual") {
            needsSecondInput = true;
            break;
        }
    }
    if (needsSecondInput) {
        ss << "layout(std430, set = 0, binding = " << binding++
           << ") readonly buffer In2Buf { float Input2[]; };\n";
    }

    ss << "layout(std430, set = 0, binding = " << binding++
       << ") writeonly buffer OutBuf { float Output[]; };\n\n";

    // Push constants
    ss << "layout(push_constant) uniform Params {\n";
    ss << "    uint total_elements;\n";
    // Add any op-specific params
    bool needsSeed = false;
    float dropoutP = 0.0f;
    for (const auto& op : graph.ops) {
        if (op.opType == "dropout") {
            needsSeed = true;
            auto it = op.params.find("p");
            if (it != op.params.end()) dropoutP = it->second;
        }
    }
    if (needsSeed) {
        ss << "    uint seed;\n";
        ss << "    float dropout_p;\n";
    }
    ss << "} p;\n\n";

    // Inject required snippets
    std::unordered_set<std::string> injected;
    for (const auto& op : graph.ops) {
        if (injected.count(op.opType)) continue;
        std::string snippet = loadSnippet(op.opType);
        if (!snippet.empty()) {
            ss << "// --- Snippet: " << op.opType << " ---\n";
            ss << snippet << "\n";
            injected.insert(op.opType);
        }
    }

    // Generate fused main()
    ss << "void main() {\n";
    ss << "    uint idx = gl_GlobalInvocationID.x;\n";
    ss << "    if (idx >= p.total_elements) return;\n\n";
    ss << "    float val = Input[idx];\n\n";

    // Chain operations — each reads from `val` and writes back to `val`
    for (const auto& op : graph.ops) {
        if (op.opType == "gelu") {
            ss << "    val = op_gelu(val);\n";
        } else if (op.opType == "relu") {
            ss << "    val = op_relu(val);\n";
        } else if (op.opType == "silu") {
            ss << "    val = op_silu(val);\n";
        } else if (op.opType == "add" || op.opType == "residual") {
            ss << "    val = val + Input2[idx];\n";
        } else if (op.opType == "dropout") {
            ss << "    val = op_dropout(val, idx, p.seed, p.dropout_p);\n";
        }
        // linear/matmul ops are NOT fused into elementwise chains
        // They're handled separately as GEMM dispatches
    }

    ss << "\n    Output[idx] = val;\n";
    ss << "}\n";

    return ss.str();
}

// ── SPIR-V Compilation via shaderc ───────────────────────────────────

std::vector<uint32_t> ShaderFusionEngine::compileToSPIRV(
    const std::string& glsl, const std::string& name) const {

#ifdef GRILLY_HAS_SHADERC
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    options.SetOptimizationLevel(shaderc_optimization_level_performance);
    options.SetTargetEnvironment(shaderc_target_env_vulkan,
                                 shaderc_env_version_vulkan_1_2);

    shaderc::SpvCompilationResult result = compiler.CompileGlslToSpv(
        glsl.c_str(), glsl.size(),
        shaderc_compute_shader,
        name.c_str(),
        options);

    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        throw std::runtime_error(
            "Shader fusion compile failed for " + name + ": " +
            result.GetErrorMessage());
    }

    return std::vector<uint32_t>(result.cbegin(), result.cend());
#else
    throw std::runtime_error(
        "JIT shader fusion requires shaderc (not found at build time)");
#endif
}

// ── Public API ───────────────────────────────────────────────────────

bool ShaderFusionEngine::isCached(const FusionGraph& graph) const {
    return compiledGraphs_.count(computeHash(graph)) > 0;
}

FusionResult ShaderFusionEngine::fuse(const FusionGraph& graph,
                                       PipelineCache& cache) {
    FusionResult result;
    result.shaderName = computeHash(graph);

    // Cache hit?
    if (compiledGraphs_.count(result.shaderName)) {
        result.cacheHit = true;
        result.compileTimeMs = 0.0f;
        stats_.cacheHits++;
        return result;
    }

    // Cache miss — generate + compile
    auto t0 = std::chrono::high_resolution_clock::now();

    result.glslSource = generateGLSL(graph);
    std::vector<uint32_t> spirv = compileToSPIRV(result.glslSource,
                                                   result.shaderName);

    // Load into PipelineCache (convert uint32_t SPIR-V to uint8_t bytes)
    std::vector<uint8_t> spirvBytes(
        reinterpret_cast<const uint8_t*>(spirv.data()),
        reinterpret_cast<const uint8_t*>(spirv.data() + spirv.size()));
    cache.loadSPIRV(result.shaderName, spirvBytes);

    auto t1 = std::chrono::high_resolution_clock::now();
    result.compileTimeMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
    result.cacheHit = false;

    compiledGraphs_[result.shaderName] = true;
    stats_.cacheMisses++;
    stats_.shadersCompiled++;
    stats_.totalCompileMs += result.compileTimeMs;

    return result;
}

}  // namespace grilly
