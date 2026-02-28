#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "grilly/compute_backend.h"
#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/device.h"
#include "grilly/op_graph.h"
#include "grilly/ops/activations.h"
#include "grilly/ops/attention.h"
#include "grilly/ops/conv.h"
#include "grilly/ops/kv_cache.h"
#include "grilly/ops/layernorm.h"
#include "grilly/ops/rmsnorm.h"
#include "grilly/ops/linear.h"
#include "grilly/ops/snn.h"
#include "grilly/ops/attention_ops.h"
#include "grilly/ops/pooling.h"
#include "grilly/ops/batchnorm.h"
#include "grilly/ops/loss.h"
#include "grilly/ops/optimizer.h"
#include "grilly/ops/embedding.h"
#include "grilly/ops/learning.h"
#include "grilly/ops/swizzle.h"
#include "grilly/pipeline_cache.h"
#include "grilly/experimental/paged_latent_pool.h"
#include "grilly/experimental/fused_attention.h"
#include "grilly/cubemind/types.h"
#include "grilly/cubemind/vsa.h"
#include "grilly/cubemind/cube.h"
#include "grilly/cubemind/cache.h"
#include "grilly/cubemind/text_encoder.h"
#include "grilly/cubemind/semantic_assigner.h"
#include "grilly/cubemind/resonator.h"
#include "grilly/training/pipeline.h"
#include "grilly/cognitive/world_model.h"
#include "grilly/temporal/temporal_encoder.h"
#include "grilly/temporal/counterfactual.h"
#include "grilly/temporal/vulkan_temporal.h"
#include "grilly/temporal/hippocampus.h"
#include "grilly/autograd/autograd.h"
#include "grilly/system_profile.h"

// ── NN framework ──
#include "grilly/nn/tensor.h"
#include "grilly/nn/parameter.h"
#include "grilly/nn/module.h"
#include "grilly/nn/surrogate.h"
#include "grilly/nn/snn.h"
#include "grilly/nn/containers.h"
#include "grilly/nn/optimizer.h"
#include "grilly/nn/dataloader.h"

namespace py = pybind11;

/// Context holds all Vulkan state so Python only sees one object.
/// Internally it owns GrillyDevice -> BufferPool -> PipelineCache -> CommandBatch.
struct GrillyCoreContext {
    grilly::GrillyDevice device;
    grilly::BufferPool pool;
    grilly::PipelineCache cache;
    grilly::CommandBatch batch;
    bool shadersLoaded = false;

    GrillyCoreContext()
        : device(), pool(device), cache(device), batch(device) {}

    /// Load all .spv shaders from a directory into the pipeline cache.
    void loadShaders(const std::string& shaderDir) {
        namespace fs = std::filesystem;
        fs::path dir(shaderDir);

        if (!fs::exists(dir))
            throw std::runtime_error("Shader directory not found: " +
                                     shaderDir);

        int count = 0;
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.path().extension() == ".spv") {
                std::string name = entry.path().stem().string();
                cache.loadSPIRVFile(name, entry.path().string());
                count++;
            }
        }
        shadersLoaded = true;

        // Try to find sibling grilly repo shaders as fallback
        if (count == 0) {
            fs::path grillyShaders =
                fs::path(shaderDir).parent_path().parent_path() / "grilly" /
                "shaders" / "spv";
            if (fs::exists(grillyShaders)) {
                for (const auto& entry :
                     fs::directory_iterator(grillyShaders)) {
                    if (entry.path().extension() == ".spv") {
                        std::string name = entry.path().stem().string();
                        cache.loadSPIRVFile(name, entry.path().string());
                        count++;
                    }
                }
            }
        }
    }
};

/// Persistent GPU cache for Hamming search benchmarking.
/// Bitpacks and uploads cache once; only the query is uploaded per search call.
struct HammingSearchBench {
    GrillyCoreContext* ctx = nullptr;
    std::vector<uint32_t> cachePacked;
    grilly::GrillyBuffer gpuCache{};
    grilly::GrillyBuffer gpuQuery{};   // Persistent query buffer (1.3 KB)
    grilly::GrillyBuffer gpuDist{};    // Persistent distance buffer (N × 4 B)
    VkDescriptorSet descSet = VK_NULL_HANDLE;  // Reused every call
    grilly::PipelineEntry pipe{};              // Cached pipeline
    uint32_t numEntries = 0;
    uint32_t wordsPerVec = 0;
    uint32_t dim = 0;
    bool gpuReady = false;  // True if shader + descriptors are set up

    // Top-1 argmin shader: atomicMin on packed uint64 — reads back 8 bytes
    // Uses SoA-transposed cache for coalesced GPU memory access.
    std::vector<uint32_t> cachePackedSoA;  // Transposed: [word][entry]
    grilly::GrillyBuffer gpuCacheSoA{};    // SoA cache in VRAM
    grilly::GrillyBuffer gpuResult{};      // 8-byte result: (distance<<32)|index
    VkDescriptorSet descSetTop1 = VK_NULL_HANDLE;
    grilly::PipelineEntry pipeTop1{};
    bool top1Ready = false;

    // Subgroup size for dispatch calculations
    uint32_t subgroupSize = 64;    // Default; queried from device
    uint32_t entriesPerWG = 4;     // 256 / subgroupSize

    // GPU timestamp query pool for measuring actual shader execution time
    VkQueryPool tsPool = VK_NULL_HANDLE;
    float timestampPeriod = 0.0f;  // nanoseconds per tick
    double lastGpuMs = 0.0;

    ~HammingSearchBench() {
        if (!ctx) return;
        if (tsPool != VK_NULL_HANDLE)
            vkDestroyQueryPool(ctx->device.device(), tsPool, nullptr);
        auto alloc = ctx->pool.allocator();
        if (gpuCache.handle != VK_NULL_HANDLE)
            vmaDestroyBuffer(alloc, gpuCache.handle, gpuCache.allocation);
        if (gpuQuery.handle != VK_NULL_HANDLE)
            vmaDestroyBuffer(alloc, gpuQuery.handle, gpuQuery.allocation);
        if (gpuDist.handle != VK_NULL_HANDLE)
            vmaDestroyBuffer(alloc, gpuDist.handle, gpuDist.allocation);
        if (gpuResult.handle != VK_NULL_HANDLE)
            vmaDestroyBuffer(alloc, gpuResult.handle, gpuResult.allocation);
        if (gpuCacheSoA.handle != VK_NULL_HANDLE)
            vmaDestroyBuffer(alloc, gpuCacheSoA.handle, gpuCacheSoA.allocation);
    }
};

// ── Helper: extract flat dimensions from numpy array ────────────────────

static std::pair<uint32_t, uint32_t> extractBatchAndLastDim(
    const py::buffer_info& buf) {
    uint32_t lastDim = static_cast<uint32_t>(buf.shape[buf.ndim - 1]);
    uint32_t batch = 1;
    for (int i = 0; i < buf.ndim - 1; ++i)
        batch *= static_cast<uint32_t>(buf.shape[i]);
    if (buf.ndim == 1) batch = 1;
    return {batch, lastDim};
}

PYBIND11_MODULE(grilly_core, m) {
    m.doc() = "grilly C++ Vulkan backend — eliminates Python->C boundary "
              "crossings for GPU dispatch";

    // ── Device info ──
    py::class_<GrillyCoreContext>(m, "Device")
        .def(py::init<>(), "Initialize Vulkan device, buffer pool, and "
                           "pipeline cache")
        .def("load_shaders", &GrillyCoreContext::loadShaders,
             py::arg("shader_dir"),
             "Load all .spv shaders from a directory")
        .def_property_readonly("device_name",
                               [](const GrillyCoreContext& ctx) {
                                   return ctx.device.deviceName();
                               })
        .def_property_readonly("has_cooperative_matrix",
                               [](const GrillyCoreContext& ctx) {
                                   return ctx.device.hasCooperativeMatrix();
                               })
        .def_property_readonly("has_float16",
                               [](const GrillyCoreContext& ctx) {
                                   return ctx.device.hasFloat16();
                               })
        .def("pool_stats",
             [](const GrillyCoreContext& ctx) {
                 auto s = ctx.pool.stats();
                 py::dict d;
                 d["hits"] = s.hits;
                 d["misses"] = s.misses;
                 d["allocations"] = s.allocations;
                 d["total_acquired"] = s.totalAcquired;
                 d["total_released"] = s.totalReleased;
                 return d;
             })
        .def("cache_stats", [](const GrillyCoreContext& ctx) {
            auto s = ctx.cache.cacheStats();
            py::dict d;
            d["hits"] = s.hits;
            d["misses"] = s.misses;
            d["evictions"] = s.evictions;
            d["cached_sets"] = s.cachedSets;
            return d;
        });

    // ═══════════════════════════════════════════════════════════════════════
    // OP GRAPH (batched execution with operator fusion)
    // ═══════════════════════════════════════════════════════════════════════

    py::class_<grilly::OpGraph>(m, "OpGraph")
        .def(py::init<>())
        .def("size", &grilly::OpGraph::size,
             "Number of ops recorded in the graph")
        .def("clear", &grilly::OpGraph::clear,
             "Clear all recorded ops for reuse")
        .def("optimize",
             [](grilly::OpGraph& graph, GrillyCoreContext& ctx) -> py::dict {
                 auto stats = graph.optimize(ctx.cache);
                 py::dict d;
                 d["ops_fused"] = stats.opsFused;
                 d["barriers_eliminated"] = stats.barriersEliminated;
                 d["original_ops"] = stats.originalOps;
                 d["optimized_ops"] = stats.optimizedOps;
                 return d;
             },
             py::arg("device"),
             "Run fusion optimization pass. Returns fusion statistics.")
        .def("execute",
             [](grilly::OpGraph& graph, GrillyCoreContext& ctx) {
                 graph.execute(ctx.batch, ctx.cache);
             },
             py::arg("device"),
             "Execute all recorded ops in a single GPU submission");

    // ═══════════════════════════════════════════════════════════════════════
    // LINEAR
    // ═══════════════════════════════════════════════════════════════════════

    m.def(
        "linear",
        [](GrillyCoreContext& ctx, py::array_t<float> x,
           py::array_t<float> weights,
           std::optional<py::array_t<float>> bias) -> py::array_t<float> {
            auto xBuf = x.request();
            auto wBuf = weights.request();

            if (xBuf.ndim < 1 || xBuf.ndim > 3)
                throw std::runtime_error(
                    "x must be 1D, 2D, or 3D (batch, seq, input_dim)");
            if (wBuf.ndim != 2)
                throw std::runtime_error(
                    "weights must be 2D (output_dim, input_dim)");

            auto [batchSeq, inputDim] = extractBatchAndLastDim(xBuf);
            uint32_t outputDim = static_cast<uint32_t>(wBuf.shape[0]);

            if (static_cast<uint32_t>(wBuf.shape[1]) != inputDim)
                throw std::runtime_error(
                    "Weight input_dim mismatch: " +
                    std::to_string(wBuf.shape[1]) + " vs " +
                    std::to_string(inputDim));

            const float* biasPtr = nullptr;
            uint32_t hasBias = 0;
            if (bias.has_value()) {
                auto bBuf = bias->request();
                if (bBuf.ndim != 1 ||
                    static_cast<uint32_t>(bBuf.shape[0]) != outputDim)
                    throw std::runtime_error("bias must be 1D with size output_dim");
                biasPtr = static_cast<const float*>(bBuf.ptr);
                hasBias = 1;
            }

            grilly::ops::LinearParams p{batchSeq, inputDim, outputDim, hasBias};

            std::vector<py::ssize_t> outShape;
            for (int i = 0; i < xBuf.ndim - 1; ++i)
                outShape.push_back(xBuf.shape[i]);
            if (xBuf.ndim == 1) outShape.push_back(1);
            outShape.push_back(outputDim);

            py::array_t<float> result(outShape);
            auto rBuf = result.request();

            grilly::ops::linear(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(xBuf.ptr),
                static_cast<const float*>(wBuf.ptr), biasPtr,
                static_cast<float*>(rBuf.ptr), p);

            if (xBuf.ndim == 1)
                result = result.reshape({static_cast<py::ssize_t>(outputDim)});

            return result;
        },
        py::arg("device"), py::arg("x"), py::arg("weights"),
        py::arg("bias") = py::none(),
        "GPU linear projection: output = x @ W^T + bias");

    m.def(
        "linear_cpu",
        [](py::array_t<float> x, py::array_t<float> weights,
           std::optional<py::array_t<float>> bias) -> py::array_t<float> {
            auto xBuf = x.request();
            auto wBuf = weights.request();

            auto [batchSeq, inputDim] = extractBatchAndLastDim(xBuf);
            uint32_t outputDim = static_cast<uint32_t>(wBuf.shape[0]);

            const float* biasPtr = nullptr;
            uint32_t hasBias = 0;
            if (bias.has_value()) {
                biasPtr = static_cast<const float*>(bias->request().ptr);
                hasBias = 1;
            }

            grilly::ops::LinearParams p{batchSeq, inputDim, outputDim, hasBias};
            std::vector<float> out = grilly::ops::linearCPU(
                static_cast<const float*>(xBuf.ptr),
                static_cast<const float*>(wBuf.ptr), biasPtr, p);

            std::vector<py::ssize_t> outShape;
            for (int i = 0; i < xBuf.ndim - 1; ++i)
                outShape.push_back(xBuf.shape[i]);
            if (xBuf.ndim == 1) outShape.push_back(1);
            outShape.push_back(outputDim);

            py::array_t<float> result(outShape);
            std::memcpy(result.request().ptr, out.data(),
                        out.size() * sizeof(float));

            if (xBuf.ndim == 1)
                result = result.reshape({static_cast<py::ssize_t>(outputDim)});

            return result;
        },
        py::arg("x"), py::arg("weights"), py::arg("bias") = py::none(),
        "CPU linear projection using Eigen (for verification)");

    // ═══════════════════════════════════════════════════════════════════════
    // ACTIVATIONS
    // ═══════════════════════════════════════════════════════════════════════

    auto defActivation = [&m](const char* name, auto fn) {
        m.def(
            name,
            [fn](GrillyCoreContext& ctx,
                 py::array_t<float> input) -> py::array_t<float> {
                auto inBuf = input.request();
                uint32_t total = 1;
                for (int i = 0; i < inBuf.ndim; ++i)
                    total *= static_cast<uint32_t>(inBuf.shape[i]);

                py::array_t<float> result(input.request().shape);
                auto rBuf = result.request();

                fn(ctx.batch, ctx.pool, ctx.cache,
                   static_cast<const float*>(inBuf.ptr),
                   static_cast<float*>(rBuf.ptr), total);

                return result;
            },
            py::arg("device"), py::arg("input"));
    };

    defActivation("relu", grilly::ops::relu);
    defActivation("gelu", grilly::ops::gelu);
    defActivation("silu", grilly::ops::silu);
    defActivation("tanh_act", grilly::ops::tanh_act);

    // ── Activation backward ops ──────────────────────────────────────────
    // 3 buffers: grad_output, input (original forward input), grad_input.
    // Same push constant (uint total_elements) and dispatch pattern.

    auto defActivationBackward = [&m](const char* name, auto fn) {
        m.def(
            name,
            [fn](GrillyCoreContext& ctx,
                 py::array_t<float> grad_output,
                 py::array_t<float> input) -> py::array_t<float> {
                auto gBuf = grad_output.request();
                auto iBuf = input.request();
                uint32_t total = 1;
                for (int i = 0; i < gBuf.ndim; ++i)
                    total *= static_cast<uint32_t>(gBuf.shape[i]);

                py::array_t<float> result(gBuf.shape);
                auto rBuf = result.request();

                fn(ctx.batch, ctx.pool, ctx.cache,
                   static_cast<const float*>(gBuf.ptr),
                   static_cast<const float*>(iBuf.ptr),
                   static_cast<float*>(rBuf.ptr), total);

                return result;
            },
            py::arg("device"), py::arg("grad_output"), py::arg("input"));
    };

    defActivationBackward("relu_backward", grilly::ops::reluBackward);
    defActivationBackward("gelu_backward", grilly::ops::geluBackward);
    defActivationBackward("silu_backward", grilly::ops::siluBackward);

    // ═══════════════════════════════════════════════════════════════════════
    // SNN STANDALONE OPS
    // ═══════════════════════════════════════════════════════════════════════

    m.def(
        "lif_step",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input, py::array_t<float> v_mem,
           py::array_t<float> t_refrac,
           float dt, float tau_mem, float v_rest, float v_reset,
           float v_thresh, float r_mem,
           float t_refrac_period) -> py::dict {
            auto inBuf = input.request();
            uint32_t n = 1;
            for (int i = 0; i < inBuf.ndim; ++i)
                n *= static_cast<uint32_t>(inBuf.shape[i]);

            // Copy state arrays since they're updated in-place
            py::array_t<float> vMemOut(v_mem.request().shape);
            py::array_t<float> refracOut(t_refrac.request().shape);
            py::array_t<float> spikes(inBuf.shape);

            std::memcpy(vMemOut.request().ptr, v_mem.request().ptr,
                        n * sizeof(float));
            std::memcpy(refracOut.request().ptr, t_refrac.request().ptr,
                        n * sizeof(float));

            grilly::ops::LIFParams p{n, dt, tau_mem, v_rest, v_reset,
                                     v_thresh, r_mem, t_refrac_period};

            grilly::ops::lifStep(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(inBuf.ptr),
                static_cast<float*>(vMemOut.request().ptr),
                static_cast<float*>(refracOut.request().ptr),
                static_cast<float*>(spikes.request().ptr), p);

            py::dict result;
            result["spikes"] = spikes;
            result["v_mem"] = vMemOut;
            result["t_refrac"] = refracOut;
            return result;
        },
        py::arg("device"), py::arg("input"), py::arg("v_mem"),
        py::arg("t_refrac"),
        py::arg("dt") = 1.0f, py::arg("tau_mem") = 20.0f,
        py::arg("v_rest") = 0.0f, py::arg("v_reset") = 0.0f,
        py::arg("v_thresh") = 1.0f, py::arg("r_mem") = 1.0f,
        py::arg("t_refrac_period") = 0.0f,
        "GPU LIF neuron step");

    m.def(
        "snn_node_forward",
        [](GrillyCoreContext& ctx,
           py::array_t<float> x_in, py::array_t<float> v_mem,
           py::array_t<float> tau_param,
           uint32_t neuron_type, float tau, float v_threshold,
           float v_reset, uint32_t reset_mode,
           uint32_t decay_input) -> py::dict {
            auto xBuf = x_in.request();
            uint32_t n = 1;
            for (int i = 0; i < xBuf.ndim; ++i)
                n *= static_cast<uint32_t>(xBuf.shape[i]);

            py::array_t<float> vMemOut(v_mem.request().shape);
            py::array_t<float> spikes(xBuf.shape);
            py::array_t<float> hOut(xBuf.shape);

            std::memcpy(vMemOut.request().ptr, v_mem.request().ptr,
                        n * sizeof(float));

            grilly::ops::SNNNodeForwardParams p{
                n, neuron_type, tau, v_threshold, v_reset,
                reset_mode, decay_input};

            grilly::ops::snnNodeForward(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(xBuf.ptr),
                static_cast<float*>(vMemOut.request().ptr),
                static_cast<float*>(spikes.request().ptr),
                static_cast<float*>(hOut.request().ptr),
                static_cast<const float*>(tau_param.request().ptr), p);

            py::dict result;
            result["spikes"] = spikes;
            result["v_mem"] = vMemOut;
            result["h_out"] = hOut;
            return result;
        },
        py::arg("device"), py::arg("x_in"), py::arg("v_mem"),
        py::arg("tau_param"),
        py::arg("neuron_type") = 1,
        py::arg("tau") = 2.0f, py::arg("v_threshold") = 1.0f,
        py::arg("v_reset") = 0.0f, py::arg("reset_mode") = 0,
        py::arg("decay_input") = 0,
        "GPU SNN node forward (IF/LIF/PLIF)");

    m.def(
        "snn_node_backward",
        [](GrillyCoreContext& ctx,
           py::array_t<float> grad_spike, py::array_t<float> h_cache,
           float alpha, uint32_t surrogate_type,
           float v_threshold) -> py::array_t<float> {
            auto gBuf = grad_spike.request();
            uint32_t n = 1;
            for (int i = 0; i < gBuf.ndim; ++i)
                n *= static_cast<uint32_t>(gBuf.shape[i]);

            py::array_t<float> gradX(gBuf.shape);

            grilly::ops::SNNNodeBackwardParams p{
                n, alpha, surrogate_type, v_threshold};

            grilly::ops::snnNodeBackward(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(gBuf.ptr),
                static_cast<const float*>(h_cache.request().ptr),
                static_cast<float*>(gradX.request().ptr), p);

            return gradX;
        },
        py::arg("device"), py::arg("grad_spike"), py::arg("h_cache"),
        py::arg("alpha") = 2.0f, py::arg("surrogate_type") = 0,
        py::arg("v_threshold") = 1.0f,
        "GPU SNN node backward (surrogate gradient)");

    m.def(
        "hebbian_learning",
        [](GrillyCoreContext& ctx,
           py::array_t<float> pre, py::array_t<float> post,
           py::array_t<float> weights,
           uint32_t batch_size, uint32_t time_steps,
           float learning_rate,
           float weight_decay) -> py::array_t<float> {
            auto preBuf = pre.request();
            auto postBuf = post.request();
            auto wBuf = weights.request();

            uint32_t pre_dim = static_cast<uint32_t>(
                preBuf.shape[preBuf.ndim - 1]);
            uint32_t post_dim = static_cast<uint32_t>(
                postBuf.shape[postBuf.ndim - 1]);

            py::array_t<float> wOut(wBuf.shape);
            std::memcpy(wOut.request().ptr, wBuf.ptr,
                        size_t(pre_dim) * post_dim * sizeof(float));

            grilly::ops::HebbianParams p{batch_size, time_steps,
                                         pre_dim, post_dim,
                                         learning_rate, weight_decay};

            grilly::ops::hebbianLearning(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(preBuf.ptr),
                static_cast<const float*>(postBuf.ptr),
                static_cast<float*>(wOut.request().ptr), p);

            return wOut;
        },
        py::arg("device"), py::arg("pre"), py::arg("post"),
        py::arg("weights"),
        py::arg("batch_size") = 1, py::arg("time_steps") = 1,
        py::arg("learning_rate") = 0.01f,
        py::arg("weight_decay") = 0.0f,
        "GPU Hebbian learning rule");

    m.def(
        "stdp_learning",
        [](GrillyCoreContext& ctx,
           py::array_t<float> pre, py::array_t<float> post,
           py::array_t<float> weights,
           py::array_t<float> pre_trace, py::array_t<float> post_trace,
           uint32_t batch_size, uint32_t time_steps,
           float lr_pot, float lr_dep,
           float trace_decay) -> py::dict {
            auto preBuf = pre.request();
            auto postBuf = post.request();
            auto wBuf = weights.request();

            uint32_t pre_dim = static_cast<uint32_t>(
                preBuf.shape[preBuf.ndim - 1]);
            uint32_t post_dim = static_cast<uint32_t>(
                postBuf.shape[postBuf.ndim - 1]);

            py::array_t<float> wOut(wBuf.shape);
            py::array_t<float> preTraceOut(pre_trace.request().shape);
            py::array_t<float> postTraceOut(post_trace.request().shape);

            std::memcpy(wOut.request().ptr, wBuf.ptr,
                        size_t(pre_dim) * post_dim * sizeof(float));
            std::memcpy(preTraceOut.request().ptr, pre_trace.request().ptr,
                        size_t(batch_size) * pre_dim * sizeof(float));
            std::memcpy(postTraceOut.request().ptr, post_trace.request().ptr,
                        size_t(batch_size) * post_dim * sizeof(float));

            grilly::ops::STDPParams p{batch_size, time_steps,
                                      pre_dim, post_dim,
                                      lr_pot, lr_dep, trace_decay, 0};

            grilly::ops::stdpLearning(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(preBuf.ptr),
                static_cast<const float*>(postBuf.ptr),
                static_cast<float*>(wOut.request().ptr),
                static_cast<float*>(preTraceOut.request().ptr),
                static_cast<float*>(postTraceOut.request().ptr), p);

            py::dict result;
            result["weights"] = wOut;
            result["pre_trace"] = preTraceOut;
            result["post_trace"] = postTraceOut;
            return result;
        },
        py::arg("device"), py::arg("pre"), py::arg("post"),
        py::arg("weights"), py::arg("pre_trace"), py::arg("post_trace"),
        py::arg("batch_size") = 1, py::arg("time_steps") = 1,
        py::arg("lr_pot") = 0.01f, py::arg("lr_dep") = 0.01f,
        py::arg("trace_decay") = 0.95f,
        "GPU STDP learning (2-pass: traces then weights)");

    m.def(
        "synapse_filter",
        [](GrillyCoreContext& ctx,
           py::array_t<float> x_in,
           py::array_t<float> y_state,
           float decay) -> py::array_t<float> {
            auto xBuf = x_in.request();
            uint32_t n = 1;
            for (int i = 0; i < xBuf.ndim; ++i)
                n *= static_cast<uint32_t>(xBuf.shape[i]);

            py::array_t<float> yOut(y_state.request().shape);
            std::memcpy(yOut.request().ptr, y_state.request().ptr,
                        n * sizeof(float));

            grilly::ops::SynapseFilterParams p{n, decay};

            grilly::ops::synapseFilter(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(xBuf.ptr),
                static_cast<float*>(yOut.request().ptr), p);

            return yOut;
        },
        py::arg("device"), py::arg("x_in"), py::arg("y_state"),
        py::arg("decay") = 0.95f,
        "GPU exponential synapse filter");

    m.def(
        "gif_neuron_step",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input, py::array_t<float> v_mem,
           py::array_t<float> i_adapt,
           py::array_t<float> g_input, py::array_t<float> g_forget,
           py::array_t<float> t_refrac, py::array_t<float> t_last_spike,
           float dt, float current_time,
           float tau_mem, float v_rest, float v_reset, float v_thresh,
           float r_mem, float tau_adapt, float delta_adapt, float b_adapt,
           float tau_gate, float gate_strength,
           float t_refrac_period) -> py::dict {
            auto inBuf = input.request();
            uint32_t n = 1;
            for (int i = 0; i < inBuf.ndim; ++i)
                n *= static_cast<uint32_t>(inBuf.shape[i]);

            py::array_t<float> vMemOut(v_mem.request().shape);
            py::array_t<float> iAdaptOut(i_adapt.request().shape);
            py::array_t<float> gInputOut(g_input.request().shape);
            py::array_t<float> gForgetOut(g_forget.request().shape);
            py::array_t<float> refracOut(t_refrac.request().shape);
            py::array_t<float> spikes(inBuf.shape);
            py::array_t<float> tLastOut(t_last_spike.request().shape);

            std::memcpy(vMemOut.request().ptr, v_mem.request().ptr,
                        n * sizeof(float));
            std::memcpy(iAdaptOut.request().ptr, i_adapt.request().ptr,
                        n * sizeof(float));
            std::memcpy(gInputOut.request().ptr, g_input.request().ptr,
                        n * sizeof(float));
            std::memcpy(gForgetOut.request().ptr, g_forget.request().ptr,
                        n * sizeof(float));
            std::memcpy(refracOut.request().ptr, t_refrac.request().ptr,
                        n * sizeof(float));
            std::memcpy(tLastOut.request().ptr, t_last_spike.request().ptr,
                        n * sizeof(float));

            grilly::ops::GIFParams p{
                n, dt, current_time, tau_mem, v_rest, v_reset, v_thresh,
                r_mem, tau_adapt, delta_adapt, b_adapt, tau_gate,
                gate_strength, t_refrac_period};

            grilly::ops::gifNeuronStep(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(inBuf.ptr),
                static_cast<float*>(vMemOut.request().ptr),
                static_cast<float*>(iAdaptOut.request().ptr),
                static_cast<float*>(gInputOut.request().ptr),
                static_cast<float*>(gForgetOut.request().ptr),
                static_cast<float*>(refracOut.request().ptr),
                static_cast<float*>(spikes.request().ptr),
                static_cast<float*>(tLastOut.request().ptr), p);

            py::dict result;
            result["spikes"] = spikes;
            result["v_mem"] = vMemOut;
            result["i_adapt"] = iAdaptOut;
            result["g_input"] = gInputOut;
            result["g_forget"] = gForgetOut;
            result["t_refrac"] = refracOut;
            result["t_last_spike"] = tLastOut;
            return result;
        },
        py::arg("device"), py::arg("input"), py::arg("v_mem"),
        py::arg("i_adapt"), py::arg("g_input"), py::arg("g_forget"),
        py::arg("t_refrac"), py::arg("t_last_spike"),
        py::arg("dt") = 1.0f, py::arg("current_time") = 0.0f,
        py::arg("tau_mem") = 20.0f, py::arg("v_rest") = 0.0f,
        py::arg("v_reset") = 0.0f, py::arg("v_thresh") = 1.0f,
        py::arg("r_mem") = 1.0f, py::arg("tau_adapt") = 100.0f,
        py::arg("delta_adapt") = 0.1f, py::arg("b_adapt") = 0.0f,
        py::arg("tau_gate") = 50.0f, py::arg("gate_strength") = 1.0f,
        py::arg("t_refrac_period") = 0.0f,
        "GPU GIF (Generalized Integrate-and-Fire) neuron step");

    // ═══════════════════════════════════════════════════════════════════════
    // LAYERNORM
    // ═══════════════════════════════════════════════════════════════════════

    m.def(
        "layernorm",
        [](GrillyCoreContext& ctx, py::array_t<float> input,
           py::array_t<float> gamma, py::array_t<float> beta,
           float eps) -> py::array_t<float> {
            auto inBuf = input.request();
            auto gBuf = gamma.request();

            if (inBuf.ndim < 2)
                throw std::runtime_error("input must be at least 2D");

            uint32_t features = static_cast<uint32_t>(
                inBuf.shape[inBuf.ndim - 1]);
            uint32_t totalBatch = 1;
            for (int i = 0; i < inBuf.ndim - 1; ++i)
                totalBatch *= static_cast<uint32_t>(inBuf.shape[i]);

            // Treat as (batchSize=1, seqLen=totalBatch, features)
            py::array_t<float> result(inBuf.shape);
            auto rBuf = result.request();

            grilly::ops::layernorm(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(inBuf.ptr),
                static_cast<float*>(rBuf.ptr),
                static_cast<const float*>(gBuf.ptr),
                static_cast<const float*>(beta.request().ptr),
                1, totalBatch, features, eps);

            return result;
        },
        py::arg("device"), py::arg("input"), py::arg("gamma"),
        py::arg("beta"), py::arg("eps") = 1e-5f,
        "GPU LayerNorm: gamma * (x - mean) / sqrt(var + eps) + beta");

    // ── LayerNorm backward ───────────────────────────────────────────────
    // Returns dict with grad_input, grad_gamma, grad_beta.
    // Requires mean and variance from the forward pass.

    m.def(
        "layernorm_backward",
        [](GrillyCoreContext& ctx,
           py::array_t<float> grad_output, py::array_t<float> input,
           py::array_t<float> gamma, py::array_t<float> mean,
           py::array_t<float> var, float eps) -> py::dict {
            auto goBuf = grad_output.request();
            auto inBuf = input.request();
            auto gBuf  = gamma.request();

            if (inBuf.ndim < 2)
                throw std::runtime_error("input must be at least 2D");

            uint32_t features = static_cast<uint32_t>(
                inBuf.shape[inBuf.ndim - 1]);
            uint32_t totalBatch = 1;
            for (int i = 0; i < inBuf.ndim - 1; ++i)
                totalBatch *= static_cast<uint32_t>(inBuf.shape[i]);

            py::array_t<float> gradInput(inBuf.shape);
            py::array_t<float> gradGamma({static_cast<py::ssize_t>(features)});
            py::array_t<float> gradBeta({static_cast<py::ssize_t>(features)});

            grilly::ops::layernormBackward(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(goBuf.ptr),
                static_cast<const float*>(inBuf.ptr),
                static_cast<const float*>(gBuf.ptr),
                static_cast<const float*>(mean.request().ptr),
                static_cast<const float*>(var.request().ptr),
                static_cast<float*>(gradInput.request().ptr),
                static_cast<float*>(gradGamma.request().ptr),
                static_cast<float*>(gradBeta.request().ptr),
                1, totalBatch, features, eps);

            py::dict result;
            result["grad_input"] = gradInput;
            result["grad_gamma"] = gradGamma;
            result["grad_beta"] = gradBeta;
            return result;
        },
        py::arg("device"), py::arg("grad_output"), py::arg("input"),
        py::arg("gamma"), py::arg("mean"), py::arg("var"),
        py::arg("eps") = 1e-5f,
        "GPU LayerNorm backward: computes grad_input, grad_gamma, grad_beta");

    // ── RMSNorm ──────────────────────────────────────────────────────────
    m.def(
        "rmsnorm",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input,
           py::array_t<float> weight,
           float eps) -> py::array_t<float> {
            auto inBuf = input.request();

            uint32_t features, batchSize, seqLen;
            if (inBuf.ndim == 1) {
                batchSize = 1; seqLen = 1;
                features = static_cast<uint32_t>(inBuf.shape[0]);
            } else if (inBuf.ndim == 2) {
                batchSize = static_cast<uint32_t>(inBuf.shape[0]);
                seqLen = 1;
                features = static_cast<uint32_t>(inBuf.shape[1]);
            } else if (inBuf.ndim == 3) {
                batchSize = static_cast<uint32_t>(inBuf.shape[0]);
                seqLen = static_cast<uint32_t>(inBuf.shape[1]);
                features = static_cast<uint32_t>(inBuf.shape[2]);
            } else {
                throw std::runtime_error("RMSNorm input must be 1D, 2D, or 3D");
            }

            py::array_t<float> result(inBuf.shape);
            auto rBuf = result.request();

            grilly::ops::rmsnorm(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(inBuf.ptr),
                static_cast<float*>(rBuf.ptr),
                static_cast<const float*>(weight.request().ptr),
                batchSize, seqLen, features, eps);

            return result;
        },
        py::arg("device"), py::arg("input"),
        py::arg("weight"), py::arg("eps") = 1e-5f,
        "GPU RMSNorm: weight * x * rsqrt(mean(x^2) + eps)");

    // ═══════════════════════════════════════════════════════════════════════
    // FLASH ATTENTION 2
    // ═══════════════════════════════════════════════════════════════════════

    m.def(
        "flash_attention2",
        [](GrillyCoreContext& ctx,
           py::array_t<float> Q, py::array_t<float> K, py::array_t<float> V,
           std::optional<py::array_t<float>> mask,
           float scale, uint32_t tileSizeQ,
           uint32_t tileSizeK) -> py::array_t<float> {
            auto qBuf = Q.request();

            if (qBuf.ndim != 4)
                throw std::runtime_error(
                    "Q must be 4D (batch, heads, seq_len, head_dim)");

            uint32_t batchSize = static_cast<uint32_t>(qBuf.shape[0]);
            uint32_t numHeads  = static_cast<uint32_t>(qBuf.shape[1]);
            uint32_t seqLen    = static_cast<uint32_t>(qBuf.shape[2]);
            uint32_t headDim   = static_cast<uint32_t>(qBuf.shape[3]);

            const float* maskPtr = nullptr;
            if (mask.has_value()) {
                maskPtr = static_cast<const float*>(mask->request().ptr);
            }

            py::array_t<float> result({
                static_cast<py::ssize_t>(batchSize),
                static_cast<py::ssize_t>(numHeads),
                static_cast<py::ssize_t>(seqLen),
                static_cast<py::ssize_t>(headDim)
            });
            auto rBuf = result.request();

            grilly::ops::flashAttention2(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(qBuf.ptr),
                static_cast<const float*>(K.request().ptr),
                static_cast<const float*>(V.request().ptr),
                maskPtr,
                static_cast<float*>(rBuf.ptr),
                batchSize, seqLen, numHeads, headDim,
                scale, tileSizeQ, tileSizeK);

            return result;
        },
        py::arg("device"), py::arg("Q"), py::arg("K"), py::arg("V"),
        py::arg("mask") = py::none(), py::arg("scale") = 0.0f,
        py::arg("tile_size_q") = 64, py::arg("tile_size_k") = 64,
        "GPU Flash Attention 2 with online softmax tiling");

    // ═══════════════════════════════════════════════════════════════════════
    // CONV2D / CONV1D
    // ═══════════════════════════════════════════════════════════════════════

    m.def(
        "conv2d",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input, py::array_t<float> weight,
           std::optional<py::array_t<float>> bias,
           std::vector<uint32_t> stride, std::vector<uint32_t> padding,
           std::vector<uint32_t> dilation,
           uint32_t groups) -> py::array_t<float> {
            auto inBuf = input.request();
            auto wBuf = weight.request();

            if (inBuf.ndim != 4)
                throw std::runtime_error(
                    "input must be 4D (batch, channels, height, width)");
            if (wBuf.ndim != 4)
                throw std::runtime_error(
                    "weight must be 4D (out_ch, in_ch/groups, kH, kW)");

            uint32_t batchSize  = static_cast<uint32_t>(inBuf.shape[0]);
            uint32_t inChannels = static_cast<uint32_t>(inBuf.shape[1]);
            uint32_t inH        = static_cast<uint32_t>(inBuf.shape[2]);
            uint32_t inW        = static_cast<uint32_t>(inBuf.shape[3]);
            uint32_t outChannels = static_cast<uint32_t>(wBuf.shape[0]);
            uint32_t kH          = static_cast<uint32_t>(wBuf.shape[2]);
            uint32_t kW          = static_cast<uint32_t>(wBuf.shape[3]);

            uint32_t sH = stride.size() >= 1 ? stride[0] : 1;
            uint32_t sW = stride.size() >= 2 ? stride[1] : sH;
            uint32_t pH = padding.size() >= 1 ? padding[0] : 0;
            uint32_t pW = padding.size() >= 2 ? padding[1] : pH;
            uint32_t dH = dilation.size() >= 1 ? dilation[0] : 1;
            uint32_t dW = dilation.size() >= 2 ? dilation[1] : dH;

            uint32_t outH = grilly::ops::convOutputSize(inH, kH, sH, pH, dH);
            uint32_t outW = grilly::ops::convOutputSize(inW, kW, sW, pW, dW);

            const float* biasPtr = nullptr;
            if (bias.has_value())
                biasPtr = static_cast<const float*>(bias->request().ptr);

            py::array_t<float> result({
                static_cast<py::ssize_t>(batchSize),
                static_cast<py::ssize_t>(outChannels),
                static_cast<py::ssize_t>(outH),
                static_cast<py::ssize_t>(outW)
            });
            auto rBuf = result.request();

            grilly::ops::conv2d(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(inBuf.ptr),
                static_cast<const float*>(wBuf.ptr),
                biasPtr,
                static_cast<float*>(rBuf.ptr),
                batchSize, inChannels, inH, inW,
                outChannels, kH, kW,
                sH, sW, pH, pW, dH, dW, groups);

            return result;
        },
        py::arg("device"), py::arg("input"), py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = std::vector<uint32_t>{1, 1},
        py::arg("padding") = std::vector<uint32_t>{0, 0},
        py::arg("dilation") = std::vector<uint32_t>{1, 1},
        py::arg("groups") = 1,
        "GPU Conv2d forward (direct or GEMM path)");

    m.def(
        "conv1d",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input, py::array_t<float> weight,
           std::optional<py::array_t<float>> bias,
           uint32_t stride, uint32_t padding,
           uint32_t dilation, uint32_t groups) -> py::array_t<float> {
            auto inBuf = input.request();
            auto wBuf = weight.request();

            if (inBuf.ndim != 3)
                throw std::runtime_error(
                    "input must be 3D (batch, channels, length)");

            uint32_t batchSize  = static_cast<uint32_t>(inBuf.shape[0]);
            uint32_t inChannels = static_cast<uint32_t>(inBuf.shape[1]);
            uint32_t length     = static_cast<uint32_t>(inBuf.shape[2]);
            uint32_t outChannels = static_cast<uint32_t>(wBuf.shape[0]);
            uint32_t kSize       = static_cast<uint32_t>(wBuf.shape[2]);

            uint32_t outLen = grilly::ops::convOutputSize(
                length, kSize, stride, padding, dilation);

            const float* biasPtr = nullptr;
            if (bias.has_value())
                biasPtr = static_cast<const float*>(bias->request().ptr);

            py::array_t<float> result({
                static_cast<py::ssize_t>(batchSize),
                static_cast<py::ssize_t>(outChannels),
                static_cast<py::ssize_t>(outLen)
            });
            auto rBuf = result.request();

            grilly::ops::conv1d(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(inBuf.ptr),
                static_cast<const float*>(wBuf.ptr),
                biasPtr,
                static_cast<float*>(rBuf.ptr),
                batchSize, inChannels, length,
                outChannels, kSize,
                stride, padding, dilation, groups);

            return result;
        },
        py::arg("device"), py::arg("input"), py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1, py::arg("padding") = 0,
        py::arg("dilation") = 1, py::arg("groups") = 1,
        "GPU Conv1d forward (wrapper around Conv2d)");

    // ═══════════════════════════════════════════════════════════════════════
    // KV CACHE
    // ═══════════════════════════════════════════════════════════════════════

    py::class_<grilly::ops::KVCache>(m, "KVCache")
        .def_readonly("current_len", &grilly::ops::KVCache::currentLen)
        .def("stats", [](const grilly::ops::KVCache& kv) {
            auto s = grilly::ops::kvCacheGetStats(kv);
            py::dict d;
            d["current_tokens"] = s.currentTokens;
            d["max_tokens"] = s.maxTokens;
            d["total_evicted"] = s.totalEvicted;
            d["total_appended"] = s.totalAppended;
            d["compression_ratio"] = s.compressionRatio;
            d["avg_attention_score"] = s.avgAttentionScore;
            return d;
        });

    m.def(
        "create_kv_cache",
        [](GrillyCoreContext& ctx,
           uint32_t maxSeqLen, uint32_t numHeads, uint32_t headDim,
           uint32_t numLayers, uint32_t compressionRatio,
           uint32_t maxCacheTokens, bool useAsymmetricQuant,
           uint32_t valueBits, bool crossLayerSharing,
           bool useH2O, bool useSpeculativeEviction,
           float evictionThreshold) -> grilly::ops::KVCache {
            grilly::ops::KVCacheConfig cfg;
            cfg.maxSeqLen = maxSeqLen;
            cfg.numHeads = numHeads;
            cfg.headDim = headDim;
            cfg.numLayers = numLayers;
            cfg.compressionRatio = compressionRatio;
            cfg.maxCacheTokens = maxCacheTokens;
            cfg.useAsymmetricQuant = useAsymmetricQuant;
            cfg.valueBits = valueBits;
            cfg.crossLayerSharing = crossLayerSharing;
            cfg.useH2O = useH2O;
            cfg.useSpeculativeEviction = useSpeculativeEviction;
            cfg.evictionThreshold = evictionThreshold;

            return grilly::ops::createKVCache(ctx.pool, cfg);
        },
        py::arg("device"),
        py::arg("max_seq_len") = 2048,
        py::arg("num_heads") = 8,
        py::arg("head_dim") = 64,
        py::arg("num_layers") = 12,
        py::arg("compression_ratio") = 4,
        py::arg("max_cache_tokens") = 2048,
        py::arg("use_asymmetric_quant") = false,
        py::arg("value_bits") = 4,
        py::arg("cross_layer_sharing") = false,
        py::arg("use_h2o") = true,
        py::arg("use_speculative_eviction") = false,
        py::arg("eviction_threshold") = 0.1f,
        "Create a KV cache with MLA compression and H2O eviction");

    m.def(
        "kv_cache_append",
        [](GrillyCoreContext& ctx, grilly::ops::KVCache& kvCache,
           py::array_t<float> newKeys,
           py::array_t<float> newValues) {
            auto kBuf = newKeys.request();
            uint32_t numNew = static_cast<uint32_t>(kBuf.shape[0]);

            grilly::ops::kvCacheAppend(
                ctx.batch, ctx.pool, ctx.cache, kvCache,
                static_cast<const float*>(kBuf.ptr),
                static_cast<const float*>(newValues.request().ptr),
                numNew);
        },
        py::arg("device"), py::arg("kv_cache"),
        py::arg("new_keys"), py::arg("new_values"),
        "Append new KV pairs to cache (with MLA compression)");

    m.def(
        "kv_cache_decode",
        [](GrillyCoreContext& ctx,
           const grilly::ops::KVCache& kvCache) -> py::dict {
            const auto& cfg = kvCache.config;
            uint32_t tokens = kvCache.currentLen;

            size_t kvSize = size_t(tokens) * cfg.numHeads * cfg.headDim;
            py::array_t<float> keys({
                static_cast<py::ssize_t>(tokens),
                static_cast<py::ssize_t>(cfg.numHeads),
                static_cast<py::ssize_t>(cfg.headDim)
            });
            py::array_t<float> values({
                static_cast<py::ssize_t>(tokens),
                static_cast<py::ssize_t>(cfg.numHeads),
                static_cast<py::ssize_t>(cfg.headDim)
            });

            grilly::ops::kvCacheDecode(
                ctx.batch, ctx.pool, ctx.cache, kvCache,
                static_cast<float*>(keys.request().ptr),
                static_cast<float*>(values.request().ptr));

            py::dict result;
            result["keys"] = keys;
            result["values"] = values;
            return result;
        },
        py::arg("device"), py::arg("kv_cache"),
        "Decode KV from compressed cache");

    m.def(
        "kv_cache_evict_h2o",
        [](GrillyCoreContext& ctx, grilly::ops::KVCache& kvCache,
           std::optional<py::array_t<float>> attentionScores,
           uint32_t numEvict) {
            const float* scoresPtr = nullptr;
            if (attentionScores.has_value())
                scoresPtr = static_cast<const float*>(
                    attentionScores->request().ptr);

            grilly::ops::kvCacheEvictH2O(
                ctx.batch, ctx.pool, ctx.cache, kvCache,
                scoresPtr, numEvict);
        },
        py::arg("device"), py::arg("kv_cache"),
        py::arg("attention_scores") = py::none(),
        py::arg("num_evict") = 0,
        "Run H2O eviction on KV cache");

    m.def(
        "kv_cache_compact",
        [](GrillyCoreContext& ctx, grilly::ops::KVCache& kvCache) {
            grilly::ops::kvCacheCompact(
                ctx.batch, ctx.pool, ctx.cache, kvCache);
        },
        py::arg("device"), py::arg("kv_cache"),
        "Compact KV cache after eviction");

    m.def(
        "destroy_kv_cache",
        [](GrillyCoreContext& ctx, grilly::ops::KVCache& kvCache) {
            grilly::ops::destroyKVCache(ctx.pool, kvCache);
        },
        py::arg("device"), py::arg("kv_cache"),
        "Destroy KV cache and release GPU buffers");

    m.def(
        "kv_cache_init_eviction_head",
        [](GrillyCoreContext& ctx, grilly::ops::KVCache& kvCache,
           uint32_t inputDim, uint32_t hiddenDim, float lr) {
            grilly::ops::kvCacheInitEvictionHead(ctx.pool, kvCache,
                                                  inputDim, hiddenDim, lr);
        },
        py::arg("device"), py::arg("kv_cache"),
        py::arg("input_dim"), py::arg("hidden_dim") = 32,
        py::arg("lr") = 1e-3f,
        "Initialize trainable eviction head for speculative eviction");

    m.def(
        "kv_cache_train_eviction_head",
        [](GrillyCoreContext& ctx, grilly::ops::KVCache& kvCache,
           py::array_t<float> tokenFeatures,
           py::array_t<float> attentionScores,
           uint32_t seqLen) {
            grilly::ops::kvCacheTrainEvictionHead(
                ctx.batch, ctx.pool, ctx.cache, kvCache,
                static_cast<const float*>(tokenFeatures.request().ptr),
                static_cast<const float*>(attentionScores.request().ptr),
                seqLen);
        },
        py::arg("device"), py::arg("kv_cache"),
        py::arg("token_features"), py::arg("attention_scores"),
        py::arg("seq_len"),
        "Train the eviction head on attention patterns from latest forward pass");

    m.def(
        "kv_cache_evict_speculative",
        [](GrillyCoreContext& ctx, grilly::ops::KVCache& kvCache,
           std::optional<py::array_t<float>> hiddenStates,
           uint32_t hiddenDim) {
            const float* hsPtr = nullptr;
            if (hiddenStates.has_value())
                hsPtr = static_cast<const float*>(
                    hiddenStates->request().ptr);

            grilly::ops::kvCacheEvictSpeculative(
                ctx.batch, ctx.pool, ctx.cache, kvCache,
                hsPtr, hiddenDim);
        },
        py::arg("device"), py::arg("kv_cache"),
        py::arg("hidden_states") = py::none(),
        py::arg("hidden_dim") = 64,
        "Run speculative eviction using trained auxiliary head");

    // ═══════════════════════════════════════════════════════════════════════
    // SUBGROUP-SWIZZLED BLOCK CACHING
    // ═══════════════════════════════════════════════════════════════════════

    m.def(
        "swizzle_kv",
        [](GrillyCoreContext& ctx, py::array_t<float> input,
           uint32_t waveSize, bool reverse) -> py::array_t<float> {
            auto inBuf = input.request();

            if (inBuf.ndim != 4)
                throw std::runtime_error(
                    "input must be 4D (batch, heads, seq_len, head_dim)");

            uint32_t batchSize = static_cast<uint32_t>(inBuf.shape[0]);
            uint32_t numHeads  = static_cast<uint32_t>(inBuf.shape[1]);
            uint32_t seqLen    = static_cast<uint32_t>(inBuf.shape[2]);
            uint32_t headDim   = static_cast<uint32_t>(inBuf.shape[3]);

            // Swizzled output may be padded to wave boundaries
            uint32_t seqPadded = ((seqLen + waveSize - 1) / waveSize) * waveSize;
            uint32_t dimPadded = ((headDim + waveSize - 1) / waveSize) * waveSize;

            py::array_t<float> result;
            if (reverse) {
                // Unswizzle: output is standard shape
                result = py::array_t<float>({
                    static_cast<py::ssize_t>(batchSize),
                    static_cast<py::ssize_t>(numHeads),
                    static_cast<py::ssize_t>(seqLen),
                    static_cast<py::ssize_t>(headDim)
                });
            } else {
                // Swizzle: output is padded flat buffer
                size_t outSize = grilly::ops::swizzledBufferSize(
                    batchSize, numHeads, seqLen, headDim, waveSize);
                result = py::array_t<float>(outSize / sizeof(float));
            }
            auto rBuf = result.request();

            grilly::ops::swizzle(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(inBuf.ptr),
                static_cast<float*>(rBuf.ptr),
                batchSize, numHeads, seqLen, headDim,
                waveSize, reverse);

            return result;
        },
        py::arg("device"), py::arg("input"),
        py::arg("wave_size") = 32, py::arg("reverse") = false,
        "Swizzle/unswizzle KV tensor for AMD Wave32 memory channel alignment.\n"
        "Transforms [B,H,S,D] -> [B,S/W,D/W,W,W] for optimal bandwidth.");

    // ═══════════════════════════════════════════════════════════════════════
    // EXPERIMENTAL: FUSED SUBGROUP-DECOMPRESS + FLASH ATTENTION
    // ═══════════════════════════════════════════════════════════════════════

    m.def(
        "fused_attention_cpu",
        [](py::array_t<float> Q, py::array_t<float> latents,
           py::array_t<float> wUp,
           std::optional<py::array_t<float>> mask,
           uint32_t cachedTokens, uint32_t numHeads,
           uint32_t headDim, uint32_t latentDim,
           float scale) -> py::array_t<float> {
            auto qBuf = Q.request();

            if (qBuf.ndim < 2)
                throw std::runtime_error("Q must be at least 2D");

            // Infer batchSize and seqLen from Q shape
            // Q is (batchSize, numHeads, seqLen, headDim) or flat
            uint32_t batchSize = 1;
            uint32_t seqLen = 1;
            if (qBuf.ndim == 4) {
                batchSize = static_cast<uint32_t>(qBuf.shape[0]);
                seqLen = static_cast<uint32_t>(qBuf.shape[2]);
            } else if (qBuf.ndim == 3) {
                batchSize = static_cast<uint32_t>(qBuf.shape[0]);
                seqLen = static_cast<uint32_t>(qBuf.shape[1]);
            } else {
                // 2D: (seqLen, headDim)
                seqLen = static_cast<uint32_t>(qBuf.shape[0]);
            }

            const float* maskPtr = nullptr;
            if (mask.has_value())
                maskPtr = static_cast<const float*>(mask->request().ptr);

            size_t outSize = size_t(batchSize) * numHeads * seqLen * headDim;
            py::array_t<float> result(outSize);
            auto rBuf = result.request();

            grilly::experimental::fusedAttentionCPU(
                static_cast<const float*>(qBuf.ptr),
                static_cast<const float*>(latents.request().ptr),
                static_cast<const float*>(wUp.request().ptr),
                maskPtr,
                static_cast<float*>(rBuf.ptr),
                batchSize, seqLen, cachedTokens,
                numHeads, headDim, latentDim, scale);

            // Reshape to (batchSize, numHeads, seqLen, headDim)
            result = result.reshape({
                static_cast<py::ssize_t>(batchSize),
                static_cast<py::ssize_t>(numHeads),
                static_cast<py::ssize_t>(seqLen),
                static_cast<py::ssize_t>(headDim)
            });

            return result;
        },
        py::arg("Q"), py::arg("latents"), py::arg("w_up"),
        py::arg("mask") = py::none(),
        py::arg("cached_tokens") = 0,
        py::arg("num_heads") = 8, py::arg("head_dim") = 64,
        py::arg("latent_dim") = 16, py::arg("scale") = 0.0f,
        "CPU reference for fused MLA decompression + attention.\n"
        "Decompresses latents via W_up matmul, then computes standard attention.");

    // ═══════════════════════════════════════════════════════════════════════
    // CUBEMIND: VSA ENCODING
    // ═══════════════════════════════════════════════════════════════════════

    m.def(
        "blake3_role",
        [](const std::string& key, uint32_t dim,
           const std::string& domain) -> py::array_t<int8_t> {
            auto result = grilly::cubemind::blake3Role(key, dim, domain);
            py::array_t<int8_t> arr(dim);
            std::memcpy(arr.request().ptr, result.data(),
                        dim * sizeof(int8_t));
            return arr;
        },
        py::arg("key"), py::arg("dim"),
        py::arg("domain") = "grilly.cubemind",
        "Generate a deterministic bipolar {-1,+1} role vector via BLAKE3");

    m.def(
        "vsa_bind",
        [](py::array_t<int8_t> a, py::array_t<int8_t> b) -> py::array_t<int8_t> {
            auto aBuf = a.request();
            auto bBuf = b.request();
            uint32_t dim = static_cast<uint32_t>(aBuf.shape[0]);
            if (static_cast<uint32_t>(bBuf.shape[0]) != dim)
                throw std::runtime_error("VSA bind: dimension mismatch");

            auto result = grilly::cubemind::vsaBind(
                static_cast<const int8_t*>(aBuf.ptr),
                static_cast<const int8_t*>(bBuf.ptr), dim);

            py::array_t<int8_t> arr(dim);
            std::memcpy(arr.request().ptr, result.data(),
                        dim * sizeof(int8_t));
            return arr;
        },
        py::arg("a"), py::arg("b"),
        "Bipolar binding: element-wise multiply (self-inverse)");

    m.def(
        "vsa_bundle",
        [](std::vector<py::array_t<int8_t>> vectors) -> py::array_t<int8_t> {
            if (vectors.empty())
                throw std::runtime_error("VSA bundle: empty vector list");

            uint32_t dim = static_cast<uint32_t>(vectors[0].request().shape[0]);
            std::vector<const int8_t*> ptrs;
            ptrs.reserve(vectors.size());
            for (auto& v : vectors)
                ptrs.push_back(static_cast<const int8_t*>(v.request().ptr));

            auto result = grilly::cubemind::vsaBundle(ptrs, dim);

            py::array_t<int8_t> arr(dim);
            std::memcpy(arr.request().ptr, result.data(),
                        dim * sizeof(int8_t));
            return arr;
        },
        py::arg("vectors"),
        "Bipolar bundling: majority vote superposition");

    m.def(
        "vsa_bitpack",
        [](py::array_t<int8_t> bipolar) -> py::array_t<uint32_t> {
            auto buf = bipolar.request();
            uint32_t dim = static_cast<uint32_t>(buf.shape[0]);
            auto packed = grilly::cubemind::vsaBitpack(
                static_cast<const int8_t*>(buf.ptr), dim);

            py::array_t<uint32_t> arr(packed.numWords());
            std::memcpy(arr.request().ptr, packed.data.data(),
                        packed.numWords() * sizeof(uint32_t));
            return arr;
        },
        py::arg("bipolar"),
        "Bitpack bipolar {-1,+1} int8 to packed uint32 bits");

    m.def(
        "vsa_encode",
        [](std::vector<std::string> roles,
           std::vector<py::array_t<int8_t>> fillers,
           uint32_t dim) -> py::array_t<uint32_t> {
            if (roles.size() != fillers.size())
                throw std::runtime_error("VSA encode: roles/fillers length mismatch");

            std::vector<const int8_t*> fillerPtrs;
            fillerPtrs.reserve(fillers.size());
            for (auto& f : fillers)
                fillerPtrs.push_back(static_cast<const int8_t*>(f.request().ptr));

            auto packed = grilly::cubemind::vsaEncode(roles, fillerPtrs, dim);

            py::array_t<uint32_t> arr(packed.numWords());
            std::memcpy(arr.request().ptr, packed.data.data(),
                        packed.numWords() * sizeof(uint32_t));
            return arr;
        },
        py::arg("roles"), py::arg("fillers"), py::arg("dim"),
        "Full VSA encode pipeline: BLAKE3 roles + bind + bundle + bitpack");

    // ═══════════════════════════════════════════════════════════════════════
    // CUBEMIND: HAMMING SEARCH
    // ═══════════════════════════════════════════════════════════════════════

    m.def(
        "hamming_search",
        [](GrillyCoreContext& ctx,
           py::array_t<int8_t> query,
           py::array_t<int8_t> cache_data) -> py::array_t<uint32_t> {
            // Accept bipolar int8 arrays, bitpack internally
            auto qBuf = query.request();
            auto cBuf = cache_data.request();

            uint32_t dim = static_cast<uint32_t>(qBuf.shape[0]);
            uint32_t numEntries;
            if (cBuf.ndim == 2) {
                numEntries = static_cast<uint32_t>(cBuf.shape[0]);
                if (static_cast<uint32_t>(cBuf.shape[1]) != dim)
                    throw std::runtime_error("Cache dim mismatch with query");
            } else {
                numEntries = static_cast<uint32_t>(cBuf.size) / dim;
            }

            // Bitpack query
            auto queryPacked = grilly::cubemind::vsaBitpack(
                static_cast<const int8_t*>(qBuf.ptr), dim);

            // Bitpack cache
            uint32_t wordsPerVec = queryPacked.numWords();
            std::vector<uint32_t> cachePacked(size_t(numEntries) * wordsPerVec);
            for (uint32_t i = 0; i < numEntries; ++i) {
                auto packed = grilly::cubemind::vsaBitpack(
                    static_cast<const int8_t*>(cBuf.ptr) + i * dim, dim);
                std::memcpy(cachePacked.data() + size_t(i) * wordsPerVec,
                            packed.data.data(),
                            wordsPerVec * sizeof(uint32_t));
            }

            // GPU Hamming search
            py::array_t<uint32_t> result(numEntries);
            auto rBuf = result.request();

            grilly::cubemind::hammingSearch(
                ctx.batch, ctx.pool, ctx.cache,
                queryPacked.data.data(), cachePacked.data(),
                static_cast<uint32_t*>(rBuf.ptr),
                numEntries, wordsPerVec);

            return result;
        },
        py::arg("device"), py::arg("query"), py::arg("cache"),
        "GPU Hamming search: distances from query to all cache entries.\n"
        "Accepts bipolar int8 arrays, bitpacks internally.");

    m.def(
        "hamming_search_cpu",
        [](py::array_t<int8_t> query,
           py::array_t<int8_t> cache_data) -> py::array_t<uint32_t> {
            auto qBuf = query.request();
            auto cBuf = cache_data.request();

            uint32_t dim = static_cast<uint32_t>(qBuf.shape[0]);
            uint32_t numEntries;
            if (cBuf.ndim == 2) {
                numEntries = static_cast<uint32_t>(cBuf.shape[0]);
            } else {
                numEntries = static_cast<uint32_t>(cBuf.size) / dim;
            }

            auto queryPacked = grilly::cubemind::vsaBitpack(
                static_cast<const int8_t*>(qBuf.ptr), dim);

            uint32_t wordsPerVec = queryPacked.numWords();
            std::vector<uint32_t> cachePacked(size_t(numEntries) * wordsPerVec);
            for (uint32_t i = 0; i < numEntries; ++i) {
                auto packed = grilly::cubemind::vsaBitpack(
                    static_cast<const int8_t*>(cBuf.ptr) + i * dim, dim);
                std::memcpy(cachePacked.data() + size_t(i) * wordsPerVec,
                            packed.data.data(),
                            wordsPerVec * sizeof(uint32_t));
            }

            auto distances = grilly::cubemind::hammingSearchCPU(
                queryPacked.data.data(), cachePacked.data(),
                numEntries, wordsPerVec);

            py::array_t<uint32_t> result(numEntries);
            std::memcpy(result.request().ptr, distances.data(),
                        numEntries * sizeof(uint32_t));
            return result;
        },
        py::arg("query"), py::arg("cache"),
        "CPU reference for Hamming search (for verification)");

    // ═══════════════════════════════════════════════════════════════════════
    // CUBEMIND: PERSISTENT GPU HAMMING SEARCH (for benchmarking)
    // ═══════════════════════════════════════════════════════════════════════
    //
    // HammingSearchBench: bitpack cache once, upload once, search many times.
    // This avoids the per-call PCIe transfer of the entire cache that was
    // making the naive hamming_search() 1000x slower than it should be.

    py::class_<HammingSearchBench>(m, "HammingSearchBench")
        .def(py::init([](GrillyCoreContext& ctx,
                         py::array_t<int8_t> cache_data,
                         uint32_t dim) {
                 auto bench = std::make_unique<HammingSearchBench>();
                 bench->ctx = &ctx;
                 bench->dim = dim;

                 auto cBuf = cache_data.request();
                 bench->numEntries = (cBuf.ndim == 2)
                     ? static_cast<uint32_t>(cBuf.shape[0])
                     : static_cast<uint32_t>(cBuf.size) / dim;

                 // Bitpack cache once on CPU
                 auto probe = grilly::cubemind::vsaBitpack(
                     static_cast<const int8_t*>(cBuf.ptr), dim);
                 bench->wordsPerVec = probe.numWords();

                 bench->cachePacked.resize(
                     size_t(bench->numEntries) * bench->wordsPerVec);
                 std::memcpy(bench->cachePacked.data(),
                             probe.data.data(),
                             bench->wordsPerVec * sizeof(uint32_t));

                 for (uint32_t i = 1; i < bench->numEntries; ++i) {
                     auto packed = grilly::cubemind::vsaBitpack(
                         static_cast<const int8_t*>(cBuf.ptr) + i * dim, dim);
                     std::memcpy(
                         bench->cachePacked.data() + size_t(i) * bench->wordsPerVec,
                         packed.data.data(),
                         bench->wordsPerVec * sizeof(uint32_t));
                 }

                 // ── Pre-allocate ALL GPU resources once ──────────────
                 size_t cacheBytes = size_t(bench->numEntries) *
                                     bench->wordsPerVec * sizeof(uint32_t);
                 size_t queryBytes = size_t(bench->wordsPerVec) * sizeof(uint32_t);
                 size_t distBytes  = size_t(bench->numEntries) * sizeof(uint32_t);

                 // 1. Cache buffer in VRAM — uploaded once via staging, never touched again.
                 //    GPU reads at 288 GB/s (VRAM) instead of 14 GB/s (PCIe).
                 //    At 490K entries: 627 MB → 2.2ms from VRAM vs 45ms from system RAM.
                 bench->gpuCache = ctx.pool.acquireDeviceLocal(cacheBytes);
                 ctx.pool.uploadStaged(bench->gpuCache,
                                       bench->cachePacked.data(),
                                       cacheBytes);

                 // ── VRAM placement diagnostic ──────────────────────
                 {
                     VkMemoryPropertyFlags memFlags;
                     vmaGetMemoryTypeProperties(
                         ctx.pool.allocator(),
                         bench->gpuCache.info.memoryType,
                         &memFlags);
                     bool devLocal = (memFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
                     bool hostVis  = (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
                     std::cout << "[VRAM] Cache: memType="
                               << bench->gpuCache.info.memoryType
                               << " DEVICE_LOCAL=" << devLocal
                               << " HOST_VISIBLE=" << hostVis
                               << " size=" << (cacheBytes / (1024*1024))
                               << " MB" << std::endl;
                     if (!devLocal) {
                         std::cerr << "[WARN] Cache NOT in VRAM!" << std::endl;
                     }
                 }

                 // 2. Query buffer — write-only from CPU (WC is fine)
                 bench->gpuQuery = ctx.pool.acquire(queryBytes);
                 // 3. Distance buffer — READ from CPU, must be cached RAM!
                 //    WC memory reads at ~39 MB/s. Cached reads at ~10 GB/s.
                 bench->gpuDist  = ctx.pool.acquireReadback(distBytes);

                 // Diagnostic: compare memory types
                 {
                     VkMemoryPropertyFlags qFlags, dFlags;
                     vmaGetMemoryTypeProperties(ctx.pool.allocator(),
                         bench->gpuQuery.info.memoryType, &qFlags);
                     vmaGetMemoryTypeProperties(ctx.pool.allocator(),
                         bench->gpuDist.info.memoryType, &dFlags);
                     std::cout << "[MEM] Query: type=" << bench->gpuQuery.info.memoryType
                               << " HOST_CACHED=" << ((qFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0)
                               << std::endl;
                     std::cout << "[MEM] Dist:  type=" << bench->gpuDist.info.memoryType
                               << " HOST_CACHED=" << ((dFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0)
                               << std::endl;
                 }

                 // 3. Pipeline + descriptor set — allocated once, reused every call
                 if (ctx.cache.hasShader("hamming-search")) {
                     bench->pipe = ctx.cache.getOrCreate(
                         "hamming-search", 3,
                         sizeof(grilly::cubemind::HammingSearchParams));

                     std::vector<VkDescriptorBufferInfo> bufInfos = {
                         {bench->gpuQuery.handle, 0, queryBytes},
                         {bench->gpuCache.handle, 0, cacheBytes},
                         {bench->gpuDist.handle,  0, distBytes},
                     };
                     bench->descSet = ctx.cache.allocDescriptorSet(
                         "hamming-search", bufInfos);
                     bench->gpuReady = true;
                 }

                 // ── Top-1 argmin shader: atomicMin on packed uint64 ──
                 // Uses SoA-transposed cache for coalesced GPU memory access.
                 // AoS: [entry][word] — strided reads, ~12% BW utilization
                 // SoA: [word][entry] — coalesced reads, ~80% BW utilization
                 if (ctx.cache.hasShader("hamming-top1")) {
                     bench->pipeTop1 = ctx.cache.getOrCreate(
                         "hamming-top1", 3,  // 3 bindings: query, cache, result
                         sizeof(grilly::cubemind::HammingSearchParams));

                     // Transpose cache: AoS → SoA
                     size_t N = bench->numEntries;
                     size_t W = bench->wordsPerVec;
                     bench->cachePackedSoA.resize(N * W);
                     for (size_t w = 0; w < W; w++) {
                         for (size_t e = 0; e < N; e++) {
                             bench->cachePackedSoA[w * N + e] =
                                 bench->cachePacked[e * W + w];
                         }
                     }

                     // Upload SoA cache to VRAM
                     bench->gpuCacheSoA = ctx.pool.acquireDeviceLocal(cacheBytes);
                     ctx.pool.uploadStaged(bench->gpuCacheSoA,
                                           bench->cachePackedSoA.data(),
                                           cacheBytes);

                     // 8-byte result buffer — host-visible for readback
                     bench->gpuResult = ctx.pool.acquire(sizeof(uint64_t));

                     std::vector<VkDescriptorBufferInfo> top1BufInfos = {
                         {bench->gpuQuery.handle, 0, queryBytes},
                         {bench->gpuCache.handle, 0, cacheBytes},  // AoS (subgroup-strided)
                         {bench->gpuResult.handle, 0, sizeof(uint64_t)},
                     };
                     bench->descSetTop1 = ctx.cache.allocDescriptorSet(
                         "hamming-top1", top1BufInfos);
                     bench->top1Ready = true;
                     std::cout << "[OK] hamming-top1 shader loaded (SoA + atomicMin)"
                               << std::endl;
                 }

                 // Query subgroup size for dispatch calculations
                 {
                     VkPhysicalDeviceSubgroupProperties sgProps{};
                     sgProps.sType =
                         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
                     VkPhysicalDeviceProperties2 props2{};
                     props2.sType =
                         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
                     props2.pNext = &sgProps;
                     vkGetPhysicalDeviceProperties2(
                         ctx.device.physicalDevice(), &props2);
                     bench->subgroupSize = sgProps.subgroupSize;
                     bench->entriesPerWG = 256 / bench->subgroupSize;
                     std::cout << "[OK] Subgroup size: "
                               << bench->subgroupSize
                               << " (entries/WG=" << bench->entriesPerWG
                               << ")" << std::endl;
                 }

                 // Create GPU timestamp query pool (2 timestamps: before + after dispatch)
                 {
                     VkPhysicalDeviceProperties props;
                     vkGetPhysicalDeviceProperties(ctx.device.physicalDevice(), &props);
                     bench->timestampPeriod = props.limits.timestampPeriod;

                     VkQueryPoolCreateInfo qpci{};
                     qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
                     qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
                     qpci.queryCount = 2;
                     vkCreateQueryPool(ctx.device.device(), &qpci, nullptr,
                                       &bench->tsPool);
                     std::cout << "[OK] GPU timestamps: period="
                               << bench->timestampPeriod << "ns" << std::endl;
                 }

                 return bench;
             }),
             py::arg("device"), py::arg("cache"), py::arg("dim") = 10240,
             py::keep_alive<1, 2>())  // Keep GrillyCoreContext alive while bench exists
        .def("search",
             [](HammingSearchBench& bench,
                py::array_t<int8_t> query) -> py::array_t<uint32_t> {
                 auto qBuf = query.request();
                 auto queryPacked = grilly::cubemind::vsaBitpack(
                     static_cast<const int8_t*>(qBuf.ptr), bench.dim);

                 py::array_t<uint32_t> result(bench.numEntries);
                 auto rBuf = result.request();

                 if (bench.gpuReady) {
                     // ── SINGLE-SUBMISSION HOT PATH ─────────────────────
                     // Query + distances in host-visible (memcpy, no staging).
                     // Cache in DEVICE_LOCAL VRAM. One GPU round-trip.
                     const size_t queryBytes = size_t(bench.wordsPerVec) *
                                               sizeof(uint32_t);
                     const size_t distBytes  = size_t(bench.numEntries) *
                                               sizeof(uint32_t);

                     // 1. Upload query via memcpy (1.3 KB, no GPU submit)
                     bench.ctx->pool.upload(
                         bench.gpuQuery,
                         reinterpret_cast<const float*>(queryPacked.data.data()),
                         queryBytes);

                     // 2. Dispatch compute shader
                     // Scalar: one thread per cache entry, workgroup = 256
                     grilly::cubemind::HammingSearchParams push{
                         bench.numEntries, bench.wordsPerVec, 1, 0};
                     uint32_t gx = (bench.numEntries + 255) / 256;

                     bench.ctx->batch.begin();
                     auto cmd = bench.ctx->batch.cmdBuffer();
                     // GPU timestamps: measure actual shader execution
                     vkCmdResetQueryPool(cmd, bench.tsPool, 0, 2);
                     vkCmdWriteTimestamp(cmd,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         bench.tsPool, 0);
                     bench.ctx->batch.dispatch(
                         bench.pipe.pipeline, bench.pipe.layout,
                         bench.descSet, gx, 1, 1,
                         &push, sizeof(push));
                     vkCmdWriteTimestamp(cmd,
                         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         bench.tsPool, 1);
                     bench.ctx->batch.submit();

                     // Read GPU timestamps
                     uint64_t ts[2] = {0, 0};
                     VkResult tsResult = vkGetQueryPoolResults(
                         bench.ctx->device.device(), bench.tsPool,
                         0, 2, sizeof(ts), ts, sizeof(uint64_t),
                         VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
                     if (bench.lastGpuMs == 0.0) {
                         // First call: print raw values for debugging
                         double diffNs = double(ts[1] - ts[0]) * double(bench.timestampPeriod);
                         std::cout << "[TS] t0=" << ts[0]
                                   << " t1=" << ts[1]
                                   << " diff_ticks=" << (ts[1] - ts[0])
                                   << " diff_ms=" << (diffNs / 1e6)
                                   << " vkResult=" << tsResult
                                   << std::endl;
                     }
                     bench.lastGpuMs = double(ts[1] - ts[0]) *
                                       double(bench.timestampPeriod) / 1e6;

                     // 3. Download distances via memcpy (no staging)
                     bench.ctx->pool.download(
                         bench.gpuDist,
                         reinterpret_cast<float*>(
                             static_cast<uint32_t*>(rBuf.ptr)),
                         distBytes);
                 } else {
                     // CPU fallback
                     auto distances = grilly::cubemind::hammingSearchCPU(
                         queryPacked.data.data(),
                         bench.cachePacked.data(),
                         bench.numEntries, bench.wordsPerVec);
                     std::memcpy(rBuf.ptr, distances.data(),
                                 bench.numEntries * sizeof(uint32_t));
                 }

                 return result;
             },
             py::arg("query"),
             "Zero-alloc GPU Hamming search (pre-built descriptor set)")
        .def("search_top1",
             [](HammingSearchBench& bench,
                py::array_t<int8_t> query) -> py::dict {
                 if (!bench.top1Ready)
                     throw std::runtime_error(
                         "hamming-top1 shader not loaded");

                 auto qBuf = query.request();
                 auto queryPacked = grilly::cubemind::vsaBitpack(
                     static_cast<const int8_t*>(qBuf.ptr), bench.dim);

                 const size_t queryBytes = size_t(bench.wordsPerVec) *
                                           sizeof(uint32_t);

                 // 1. Upload query via memcpy (1.3 KB)
                 bench.ctx->pool.upload(
                     bench.gpuQuery,
                     reinterpret_cast<const float*>(queryPacked.data.data()),
                     queryBytes);

                 // 2. Reset result to max uint64 so atomicMin can find minimum.
                 // 0xFFFFFFFFFFFFFFFF = max distance (0xFFFFFFFF) + max index.
                 uint64_t sentinel = 0xFFFFFFFFFFFFFFFFULL;
                 std::memcpy(bench.gpuResult.mappedPtr, &sentinel,
                             sizeof(uint64_t));

                 // 3. Dispatch top-1 argmin shader
                 // entries_per_wg = 256/subgroupSize (4 for Wave64, 8 for Wave32)
                 grilly::cubemind::HammingSearchParams push{
                     bench.numEntries, bench.wordsPerVec,
                     bench.entriesPerWG, 0};
                 uint32_t gx = (bench.numEntries + bench.entriesPerWG - 1)
                             / bench.entriesPerWG;

                 bench.ctx->batch.begin();
                 auto cmd = bench.ctx->batch.cmdBuffer();
                 vkCmdResetQueryPool(cmd, bench.tsPool, 0, 2);
                 vkCmdWriteTimestamp(cmd,
                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                     bench.tsPool, 0);
                 bench.ctx->batch.dispatch(
                     bench.pipeTop1.pipeline, bench.pipeTop1.layout,
                     bench.descSetTop1, gx, 1, 1,
                     &push, sizeof(push));
                 vkCmdWriteTimestamp(cmd,
                     VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                     bench.tsPool, 1);
                 bench.ctx->batch.submit();

                 // 4. Read GPU timestamps
                 uint64_t ts[2] = {0, 0};
                 vkGetQueryPoolResults(
                     bench.ctx->device.device(), bench.tsPool,
                     0, 2, sizeof(ts), ts, sizeof(uint64_t),
                     VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
                 bench.lastGpuMs = double(ts[1] - ts[0]) *
                                   double(bench.timestampPeriod) / 1e6;

                 // 5. Read back 8 bytes: packed (distance << 32) | index
                 uint64_t packed = 0;
                 std::memcpy(&packed, bench.gpuResult.mappedPtr,
                             sizeof(uint64_t));

                 uint32_t bestDist  = static_cast<uint32_t>(packed >> 32);
                 uint32_t bestIndex = static_cast<uint32_t>(packed & 0xFFFFFFFF);

                 py::dict result;
                 result["index"] = bestIndex;
                 result["distance"] = bestDist;
                 return result;
             },
             py::arg("query"),
             "GPU top-1 Hamming search via atomicMin on packed uint64.\n"
             "Returns dict with 'index' and 'distance'. Reads back 8 bytes\n"
             "instead of N*4, eliminating the PCIe readback bottleneck.")
        .def_property_readonly("gpu_time_ms",
             [](const HammingSearchBench& b) { return b.lastGpuMs; },
             "GPU-side execution time of last search (ms)");

    // ═══════════════════════════════════════════════════════════════════════
    // CUBEMIND: RUBIK'S CUBE STATE GENERATOR
    // ═══════════════════════════════════════════════════════════════════════

    m.def(
        "cube_solved",
        [](uint32_t size) -> py::array_t<uint8_t> {
            auto cs = (size == 2) ? grilly::cubemind::CubeSize::Cube2x2
                                  : grilly::cubemind::CubeSize::Cube3x3;
            auto state = grilly::cubemind::cubeSolved(cs);
            py::array_t<uint8_t> arr(state.facelets.size());
            std::memcpy(arr.request().ptr, state.facelets.data(),
                        state.facelets.size());
            return arr;
        },
        py::arg("size") = 3,
        "Create solved Rubik's cube state (2 or 3)");

    m.def(
        "cube_apply_move",
        [](py::array_t<uint8_t> state, uint32_t size,
           uint8_t move) -> py::array_t<uint8_t> {
            auto buf = state.request();
            grilly::cubemind::CubeState cs;
            cs.size = (size == 2) ? grilly::cubemind::CubeSize::Cube2x2
                                  : grilly::cubemind::CubeSize::Cube3x3;
            cs.facelets.assign(static_cast<uint8_t*>(buf.ptr),
                               static_cast<uint8_t*>(buf.ptr) + buf.size);

            auto result = grilly::cubemind::cubeApplyMove(
                cs, static_cast<grilly::cubemind::CubeMove>(move));

            py::array_t<uint8_t> arr(result.facelets.size());
            std::memcpy(arr.request().ptr, result.facelets.data(),
                        result.facelets.size());
            return arr;
        },
        py::arg("state"), py::arg("size") = 3, py::arg("move") = 0,
        "Apply a move to a cube state");

    m.def(
        "cube_random_walk",
        [](uint32_t size, uint32_t numMoves,
           uint32_t seed) -> py::array_t<uint8_t> {
            auto cs = (size == 2) ? grilly::cubemind::CubeSize::Cube2x2
                                  : grilly::cubemind::CubeSize::Cube3x3;
            auto state = grilly::cubemind::cubeRandomWalk(cs, numMoves, seed);
            py::array_t<uint8_t> arr(state.facelets.size());
            std::memcpy(arr.request().ptr, state.facelets.data(),
                        state.facelets.size());
            return arr;
        },
        py::arg("size") = 3, py::arg("num_moves") = 20,
        py::arg("seed") = 0,
        "Random walk from solved state");

    m.def(
        "cube_estimate_distance",
        [](py::array_t<uint8_t> state, uint32_t size) -> uint32_t {
            auto buf = state.request();
            grilly::cubemind::CubeState cs;
            cs.size = (size == 2) ? grilly::cubemind::CubeSize::Cube2x2
                                  : grilly::cubemind::CubeSize::Cube3x3;
            cs.facelets.assign(static_cast<uint8_t*>(buf.ptr),
                               static_cast<uint8_t*>(buf.ptr) + buf.size);
            return grilly::cubemind::cubeEstimateDistance(cs);
        },
        py::arg("state"), py::arg("size") = 3,
        "Estimate distance from solved via facelet mismatch heuristic");

    m.def(
        "cube_to_vsa",
        [](py::array_t<uint8_t> state, uint32_t size,
           uint32_t dim) -> py::array_t<int8_t> {
            auto buf = state.request();
            grilly::cubemind::CubeState cs;
            cs.size = (size == 2) ? grilly::cubemind::CubeSize::Cube2x2
                                  : grilly::cubemind::CubeSize::Cube3x3;
            cs.facelets.assign(static_cast<uint8_t*>(buf.ptr),
                               static_cast<uint8_t*>(buf.ptr) + buf.size);

            auto result = grilly::cubemind::cubeToVSA(cs, dim);

            py::array_t<int8_t> arr(dim);
            std::memcpy(arr.request().ptr, result.data(),
                        dim * sizeof(int8_t));
            return arr;
        },
        py::arg("state"), py::arg("size") = 3, py::arg("dim") = 10240,
        "Encode cube state as bipolar VSA hypervector");

    // ═══════════════════════════════════════════════════════════════════════
    // CUBEMIND: VSA CACHE
    // ═══════════════════════════════════════════════════════════════════════

    py::class_<grilly::cubemind::VSACache>(m, "VSACache")
        .def(py::init([](GrillyCoreContext& ctx,
                         uint32_t initialCapacity, uint32_t maxCapacity,
                         uint32_t dim, float surpriseThreshold,
                         float utilityDecay) {
                 grilly::cubemind::CacheConfig cfg;
                 cfg.initialCapacity = initialCapacity;
                 cfg.maxCapacity = maxCapacity;
                 cfg.dim = dim;
                 cfg.surpriseThreshold = surpriseThreshold;
                 cfg.utilityDecay = utilityDecay;
                 return std::make_unique<grilly::cubemind::VSACache>(
                     ctx.pool, cfg);
             }),
             py::arg("device"),
             py::arg("initial_capacity") = 1024,
             py::arg("max_capacity") = 500000,
             py::arg("dim") = 10240,
             py::arg("surprise_threshold") = 0.3f,
             py::arg("utility_decay") = 0.99f,
             py::keep_alive<1, 2>())  // Keep GrillyCoreContext alive while cache exists
        .def("lookup",
             [](grilly::cubemind::VSACache& cache,
                GrillyCoreContext& ctx,
                py::array_t<int8_t> query,
                uint32_t topK) -> py::dict {
                 auto buf = query.request();
                 uint32_t dim = static_cast<uint32_t>(buf.shape[0]);
                 auto packed = grilly::cubemind::vsaBitpack(
                     static_cast<const int8_t*>(buf.ptr), dim);

                 auto result = cache.lookup(ctx.batch, ctx.cache, packed, topK);

                 py::dict d;
                 d["indices"] = py::array_t<uint32_t>(
                     result.indices.size(), result.indices.data());
                 d["distances"] = py::array_t<uint32_t>(
                     result.distances.size(), result.distances.data());
                 d["surprise"] = result.querySurprise;
                 return d;
             },
             py::arg("device"), py::arg("query"), py::arg("top_k") = 5)
        .def("lookup_packed",
             [](grilly::cubemind::VSACache& cache,
                GrillyCoreContext& ctx,
                py::array_t<uint32_t> query_packed,
                uint32_t topK) -> py::dict {
                 auto buf = query_packed.request();
                 uint32_t words = static_cast<uint32_t>(buf.shape[0]);

                 grilly::cubemind::BitpackedVec packed;
                 packed.dim = words * 32;
                 packed.data.assign(
                     static_cast<uint32_t*>(buf.ptr),
                     static_cast<uint32_t*>(buf.ptr) + words);

                 auto result = cache.lookup(ctx.batch, ctx.cache, packed, topK);

                 py::dict d;
                 d["indices"] = py::array_t<uint32_t>(
                     result.indices.size(), result.indices.data());
                 d["distances"] = py::array_t<uint32_t>(
                     result.distances.size(), result.distances.data());
                 d["surprise"] = result.querySurprise;
                 return d;
             },
             py::arg("device"), py::arg("query_packed"), py::arg("top_k") = 5,
             "Lookup using pre-bitpacked uint32 query (avoids unpack/repack)")
        .def("insert_packed",
             [](grilly::cubemind::VSACache& cache,
                py::array_t<uint32_t> key_packed,
                float surprise, float stress) -> bool {
                 auto buf = key_packed.request();
                 uint32_t words = static_cast<uint32_t>(buf.shape[0]);

                 grilly::cubemind::BitpackedVec packed;
                 packed.dim = words * 32;
                 packed.data.assign(
                     static_cast<uint32_t*>(buf.ptr),
                     static_cast<uint32_t*>(buf.ptr) + words);

                 grilly::cubemind::EmotionState emo{surprise, stress};
                 return cache.insert(packed, emo);
             },
             py::arg("key_packed"),
             py::arg("surprise") = 1.0f, py::arg("stress") = 0.0f,
             "Insert using pre-bitpacked uint32 key (avoids bitpack overhead)")
        .def("insert",
             [](grilly::cubemind::VSACache& cache,
                py::array_t<int8_t> key,
                float surprise, float stress) -> bool {
                 auto buf = key.request();
                 uint32_t dim = static_cast<uint32_t>(buf.shape[0]);
                 auto packed = grilly::cubemind::vsaBitpack(
                     static_cast<const int8_t*>(buf.ptr), dim);
                 grilly::cubemind::EmotionState emo{surprise, stress};
                 return cache.insert(packed, emo);
             },
             py::arg("key"), py::arg("surprise") = 1.0f,
             py::arg("stress") = 0.0f)
        .def("evict", &grilly::cubemind::VSACache::evict, py::arg("count"))
        .def("size", &grilly::cubemind::VSACache::size)
        .def("stats",
             [](const grilly::cubemind::VSACache& cache) -> py::dict {
                 auto s = cache.stats();
                 py::dict d;
                 d["size"] = s.size;
                 d["capacity"] = s.capacity;
                 d["total_inserts"] = s.totalInserts;
                 d["total_evictions"] = s.totalEvictions;
                 d["total_lookups"] = s.totalLookups;
                 d["avg_surprise"] = s.avgSurprise;
                 d["avg_utility"] = s.avgUtility;
                 d["last_lookup_ms"] = s.lastLookupMs;
                 return d;
             });

    // ── CubeMind: TextEncoder (FastText LSH → Bipolar VSA) ────────────

    py::class_<grilly::cubemind::TextEncoder>(m, "TextEncoder")
        .def(py::init<uint32_t, uint32_t>(),
             py::arg("dim") = 10240, py::arg("ft_dim") = 300)
        .def("encode_sentence",
             [](grilly::cubemind::TextEncoder& enc,
                const std::vector<std::string>& tokens,
                const std::vector<std::string>& roles,
                const std::vector<uint32_t>& positions) -> py::dict {
                 auto packed = enc.encode_sentence(tokens, roles, positions);
                 py::dict d;
                 d["data"] = py::array_t<uint32_t>(
                     packed.data.size(), packed.data.data());
                 d["dim"] = packed.dim;
                 d["num_words"] = packed.numWords();
                 return d;
             },
             py::arg("tokens"), py::arg("dependency_roles"),
             py::arg("positions"))
        .def("load_fillers", &grilly::cubemind::TextEncoder::load_fillers,
             py::arg("path"))
        .def("project_to_bipolar",
             [](const grilly::cubemind::TextEncoder& enc,
                py::array_t<float> ft_vec) -> py::array_t<int8_t> {
                 auto buf = ft_vec.request();
                 auto result = enc.project_to_bipolar(
                     static_cast<const float*>(buf.ptr));
                 return py::array_t<int8_t>(result.size(), result.data());
             },
             py::arg("ft_vec"))
        .def("add_filler",
             [](grilly::cubemind::TextEncoder& enc,
                const std::string& token,
                py::array_t<int8_t> bipolar) {
                 auto buf = bipolar.request();
                 std::vector<int8_t> vec(
                     static_cast<int8_t*>(buf.ptr),
                     static_cast<int8_t*>(buf.ptr) + buf.size);
                 enc.add_filler(token, vec);
             },
             py::arg("token"), py::arg("bipolar"))
        .def("has_filler", &grilly::cubemind::TextEncoder::has_filler,
             py::arg("token"))
        .def("vocab_size", &grilly::cubemind::TextEncoder::vocab_size)
        .def_property_readonly("dim", &grilly::cubemind::TextEncoder::dim)
        .def_property_readonly("ft_dim", &grilly::cubemind::TextEncoder::ft_dim);

    // ── CubeMind: SemanticAssigner (Memoized LSH Projection Cache) ──────

    py::class_<grilly::cubemind::SemanticAssigner>(m, "SemanticAssigner")
        .def(py::init<uint32_t, uint32_t>(),
             py::arg("dim") = 10240, py::arg("ft_dim") = 300)
        .def("get_semantic_filler",
             [](grilly::cubemind::SemanticAssigner& sa,
                const std::string& token) -> py::dict {
                 auto packed = sa.get_semantic_filler(token);
                 py::dict d;
                 d["data"] = py::array_t<uint32_t>(
                     packed.data.size(), packed.data.data());
                 d["dim"] = packed.dim;
                 d["num_words"] = packed.numWords();
                 return d;
             },
             py::arg("token"))
        .def("add_float_vector",
             [](grilly::cubemind::SemanticAssigner& sa,
                const std::string& token,
                py::array_t<float> vec) {
                 auto buf = vec.request();
                 sa.add_float_vector(token, static_cast<const float*>(buf.ptr));
             },
             py::arg("token"), py::arg("vec"))
        .def("add_float_vectors_batch",
             [](grilly::cubemind::SemanticAssigner& sa,
                const std::vector<std::string>& tokens,
                py::array_t<float> vectors) {
                 auto buf = vectors.request();
                 sa.add_float_vectors_batch(
                     tokens, static_cast<const float*>(buf.ptr));
             },
             py::arg("tokens"), py::arg("vectors"))
        .def("prewarm", &grilly::cubemind::SemanticAssigner::prewarm)
        .def("add_bipolar_filler",
             [](grilly::cubemind::SemanticAssigner& sa,
                const std::string& token,
                py::array_t<int8_t> bipolar) {
                 auto buf = bipolar.request();
                 std::vector<int8_t> vec(
                     static_cast<int8_t*>(buf.ptr),
                     static_cast<int8_t*>(buf.ptr) + buf.size);
                 sa.add_bipolar_filler(token, vec);
             },
             py::arg("token"), py::arg("bipolar"))
        .def("load_fillers",
             &grilly::cubemind::SemanticAssigner::load_fillers,
             py::arg("path"))
        .def("project_to_bipolar",
             [](const grilly::cubemind::SemanticAssigner& sa,
                py::array_t<float> vec) -> py::array_t<int8_t> {
                 auto buf = vec.request();
                 auto result = sa.project_to_bipolar(
                     static_cast<const float*>(buf.ptr));
                 return py::array_t<int8_t>(result.size(), result.data());
             },
             py::arg("vec"))
        .def_property_readonly("cache_size",
             &grilly::cubemind::SemanticAssigner::cache_size)
        .def_property_readonly("float_vocab_size",
             &grilly::cubemind::SemanticAssigner::float_vocab_size)
        .def_property_readonly("cache_hits",
             &grilly::cubemind::SemanticAssigner::cache_hits)
        .def_property_readonly("cache_misses",
             &grilly::cubemind::SemanticAssigner::cache_misses)
        .def_property_readonly("hit_rate",
             &grilly::cubemind::SemanticAssigner::hit_rate)
        .def("reset_stats",
             &grilly::cubemind::SemanticAssigner::reset_stats)
        .def_property_readonly("dim",
             &grilly::cubemind::SemanticAssigner::dim)
        .def_property_readonly("ft_dim",
             &grilly::cubemind::SemanticAssigner::ft_dim);

    // ── CubeMind: ResonatorNetwork (GPU Hamming Similarity Generation) ──

    py::class_<grilly::cubemind::ResonatorNetwork>(m, "ResonatorNetwork")
        .def(py::init(
                 [](GrillyCoreContext& ctx, uint32_t dim) {
                     return new grilly::cubemind::ResonatorNetwork(
                         ctx.pool, ctx.batch, ctx.cache, dim);
                 }),
             py::arg("device"), py::arg("dim") = 10240,
             py::keep_alive<1, 2>())
        .def("load_codebook",
             [](grilly::cubemind::ResonatorNetwork& res,
                const std::vector<std::string>& words,
                py::array_t<uint32_t> vectors) {
                 auto buf = vectors.request();
                 res.load_codebook(words,
                     static_cast<const uint32_t*>(buf.ptr));
             },
             py::arg("words"), py::arg("vectors"))
        .def("load_codebook_bipolar",
             [](grilly::cubemind::ResonatorNetwork& res,
                const std::vector<std::string>& words,
                py::array_t<int8_t> vectors) {
                 auto buf = vectors.request();
                 res.load_codebook_bipolar(words,
                     static_cast<const int8_t*>(buf.ptr));
             },
             py::arg("words"), py::arg("vectors"))
        .def("resonate",
             [](grilly::cubemind::ResonatorNetwork& res,
                py::array_t<uint32_t> query_packed,
                bool return_all) -> py::dict {
                 auto buf = query_packed.request();
                 grilly::cubemind::BitpackedVec q;
                 q.dim = res.codebook_size() > 0 ? 10240 : 0;
                 q.data.assign(
                     static_cast<uint32_t*>(buf.ptr),
                     static_cast<uint32_t*>(buf.ptr) + buf.size);
                 q.dim = q.data.size() * 32;

                 auto result = res.resonate(q, return_all);

                 py::dict d;
                 d["best_index"] = result.best_index;
                 d["best_similarity"] = result.best_similarity;
                 d["best_word"] = res.get_word(result.best_index);
                 if (return_all) {
                     d["similarities"] = py::array_t<float>(
                         result.all_similarities.size(),
                         result.all_similarities.data());
                 }
                 return d;
             },
             py::arg("query_packed"), py::arg("return_all") = false)
        .def("generate_sentence",
             [](grilly::cubemind::ResonatorNetwork& res,
                py::array_t<uint32_t> bundle_packed,
                std::vector<std::string> dependency_roles,
                std::vector<uint32_t> positions,
                bool explain_away) -> py::list {
                 auto buf = bundle_packed.request();
                 grilly::cubemind::BitpackedVec bundle;
                 bundle.data.assign(
                     static_cast<uint32_t*>(buf.ptr),
                     static_cast<uint32_t*>(buf.ptr) + buf.size);
                 bundle.dim = bundle.data.size() * 32;

                 auto result = res.generate_sentence(
                     bundle, dependency_roles, positions, explain_away);

                 py::list out;
                 for (auto& [word, sim] : result) {
                     py::dict entry;
                     entry["word"] = word;
                     entry["similarity"] = sim;
                     out.append(entry);
                 }
                 return out;
             },
             py::arg("bundle_packed"),
             py::arg("dependency_roles"),
             py::arg("positions"),
             py::arg("explain_away") = true)
        .def_property_readonly("codebook_size",
             &grilly::cubemind::ResonatorNetwork::codebook_size)
        .def("get_word",
             &grilly::cubemind::ResonatorNetwork::get_word,
             py::arg("index"))
        .def_property_readonly("total_resonations",
             &grilly::cubemind::ResonatorNetwork::total_resonations)
        .def_property_readonly("last_resonate_ms",
             &grilly::cubemind::ResonatorNetwork::last_resonate_ms);

    // ── Training Pipeline: Producer-Consumer Data Loading ───────────────

    py::class_<grilly::training::ParsedDocument>(m, "ParsedDocument")
        .def(py::init<>())
        .def_readwrite("tokens",
                        &grilly::training::ParsedDocument::tokens)
        .def_readwrite("dependency_roles",
                        &grilly::training::ParsedDocument::dependency_roles)
        .def_readwrite("positions",
                        &grilly::training::ParsedDocument::positions)
        .def_readwrite("llm_token_ids",
                        &grilly::training::ParsedDocument::llm_token_ids);

    py::class_<grilly::TrainingPayload>(m, "TrainingPayload")
        .def(py::init<>())
        .def_property_readonly("vsa_data",
             [](const grilly::TrainingPayload& p) -> py::array_t<uint32_t> {
                 return py::array_t<uint32_t>(
                     p.vsa_state.data.size(), p.vsa_state.data.data());
             })
        .def_property_readonly("vsa_dim",
             [](const grilly::TrainingPayload& p) { return p.vsa_state.dim; })
        .def_readwrite("llm_input_tokens",
                        &grilly::TrainingPayload::llm_input_tokens)
        .def_readwrite("sequence_id",
                        &grilly::TrainingPayload::sequence_id)
        .def_property_readonly("surprise",
             [](const grilly::TrainingPayload& p) {
                 return p.emotion.surprise;
             })
        .def_property_readonly("stress",
             [](const grilly::TrainingPayload& p) {
                 return p.emotion.stress;
             })
        .def_readonly("svc_subject",
             &grilly::TrainingPayload::svc_subject)
        .def_readonly("svc_verb",
             &grilly::TrainingPayload::svc_verb)
        .def_readonly("svc_complement",
             &grilly::TrainingPayload::svc_complement)
        .def_property_readonly("has_svc",
             [](const grilly::TrainingPayload& p) {
                 return !p.svc_subject.empty() && !p.svc_verb.empty()
                        && !p.svc_complement.empty();
             });

    py::class_<grilly::training::TrainingPipeline>(m, "TrainingPipeline")
        .def(py::init<uint32_t, uint32_t, size_t>(),
             py::arg("dim") = 10240,
             py::arg("ft_dim") = 300,
             py::arg("queue_depth") = 1024)
        .def("start",
             [](grilly::training::TrainingPipeline& pipe,
                const std::vector<grilly::training::ParsedDocument>& docs) {
                 pipe.start(docs);
             },
             py::arg("documents"),
             py::call_guard<py::gil_scoped_release>())
        .def("start_with_files",
             [](grilly::training::TrainingPipeline& pipe,
                const std::vector<std::string>& paths) {
                 pipe.start_with_files(paths);
             },
             py::arg("paths"),
             py::call_guard<py::gil_scoped_release>())
        .def("pop",
             [](grilly::training::TrainingPipeline& pipe) -> py::object {
                 grilly::TrainingPayload payload;
                 bool ok;
                 {
                     py::gil_scoped_release release;
                     ok = pipe.pop(payload);
                 }
                 if (!ok) return py::none();
                 return py::cast(std::move(payload));
             })
        .def("try_pop",
             [](grilly::training::TrainingPipeline& pipe) -> py::object {
                 grilly::TrainingPayload payload;
                 if (!pipe.try_pop(payload)) return py::none();
                 return py::cast(std::move(payload));
             })
        .def("stop", &grilly::training::TrainingPipeline::stop,
             py::call_guard<py::gil_scoped_release>())
        .def("join", &grilly::training::TrainingPipeline::join,
             py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("queue_size",
             &grilly::training::TrainingPipeline::queue_size)
        .def("encoder",
             &grilly::training::TrainingPipeline::encoder,
             py::return_value_policy::reference_internal)
        .def("assigner",
             &grilly::training::TrainingPipeline::assigner,
             py::return_value_policy::reference_internal)
        .def("stats",
             [](const grilly::training::TrainingPipeline& pipe) -> py::dict {
                 auto s = pipe.stats();
                 py::dict d;
                 d["documents_encoded"] = s.documents_encoded;
                 d["payloads_consumed"] = s.payloads_consumed;
                 d["queue_current_size"] = s.queue_current_size;
                 d["encoding_docs_per_sec"] = s.encoding_docs_per_sec;
                 d["elapsed_seconds"] = s.elapsed_seconds;
                 d["producer_busy_pct"] = s.producer_busy_pct;
                 return d;
             });

    // ── Cognitive: WorldModel (Dual VSACache Fact Engine) ────────────────

    py::class_<grilly::cognitive::CoherenceResult>(m, "CoherenceResult")
        .def_readonly("support", &grilly::cognitive::CoherenceResult::support)
        .def_readonly("violation", &grilly::cognitive::CoherenceResult::violation)
        .def_readonly("score", &grilly::cognitive::CoherenceResult::score)
        .def_readonly("coherent", &grilly::cognitive::CoherenceResult::coherent)
        .def_readonly("nearest_fact_idx",
                      &grilly::cognitive::CoherenceResult::nearestFactIdx)
        .def_readonly("nearest_constraint_idx",
                      &grilly::cognitive::CoherenceResult::nearestConstraintIdx);

    py::class_<grilly::cognitive::WorldModel>(m, "WorldModel")
        .def(py::init([](GrillyCoreContext& ctx,
                         uint32_t dim,
                         uint32_t fact_capacity,
                         uint32_t constraint_capacity,
                         float coherence_threshold,
                         float surprise_threshold) {
                 grilly::cognitive::WorldModelConfig cfg;
                 cfg.dim = dim;
                 cfg.factCapacity = fact_capacity;
                 cfg.constraintCapacity = constraint_capacity;
                 cfg.coherenceThreshold = coherence_threshold;
                 cfg.surpriseThreshold = surprise_threshold;
                 return std::make_unique<grilly::cognitive::WorldModel>(
                     ctx.pool, cfg);
             }),
             py::arg("device"),
             py::arg("dim") = 10240,
             py::arg("fact_capacity") = 500000,
             py::arg("constraint_capacity") = 500000,
             py::arg("coherence_threshold") = 0.3f,
             py::arg("surprise_threshold") = 0.3f,
             py::keep_alive<1, 2>())
        .def("add_fact",
             &grilly::cognitive::WorldModel::add_fact,
             py::arg("subject"), py::arg("relation"), py::arg("object"),
             "Add a (S, R, O) fact and auto-generate negation constraint")
        .def("check_coherence",
             [](grilly::cognitive::WorldModel& wm,
                GrillyCoreContext& ctx,
                const std::string& subject,
                const std::string& relation,
                const std::string& object) {
                 return wm.check_coherence(ctx.batch, ctx.cache,
                                           subject, relation, object);
             },
             py::arg("device"),
             py::arg("subject"), py::arg("relation"), py::arg("object"),
             "Check coherence of a (S, R, O) triple via GPU Hamming search")
        .def("check_coherence_vec",
             [](grilly::cognitive::WorldModel& wm,
                GrillyCoreContext& ctx,
                py::array_t<uint32_t> statement) {
                 auto buf = statement.request();
                 uint32_t dim = wm.dim();
                 uint32_t words = (dim + 31) / 32;

                 grilly::cubemind::BitpackedVec vec;
                 vec.dim = dim;
                 vec.data.assign(
                     static_cast<uint32_t*>(buf.ptr),
                     static_cast<uint32_t*>(buf.ptr) + words);

                 return wm.check_coherence(ctx.batch, ctx.cache, vec);
             },
             py::arg("device"), py::arg("statement"),
             "Check coherence of a pre-encoded bitpacked vector")
        .def("check_coherence_cpu",
             [](grilly::cognitive::WorldModel& wm,
                py::array_t<uint32_t> statement) {
                 auto buf = statement.request();
                 uint32_t dim = wm.dim();
                 uint32_t words = (dim + 31) / 32;

                 grilly::cubemind::BitpackedVec vec;
                 vec.dim = dim;
                 vec.data.assign(
                     static_cast<uint32_t*>(buf.ptr),
                     static_cast<uint32_t*>(buf.ptr) + words);

                 return wm.check_coherence_cpu(vec);
             },
             py::arg("statement"),
             "Check coherence via CPU (no GPU needed, for testing)")
        .def("encode_triple",
             [](const grilly::cognitive::WorldModel& wm,
                const std::string& subject,
                const std::string& relation,
                const std::string& object) -> py::array_t<uint32_t> {
                 auto vec = wm.encode_triple(subject, relation, object);
                 py::array_t<uint32_t> arr(vec.data.size());
                 std::memcpy(arr.mutable_data(), vec.data.data(),
                             vec.data.size() * sizeof(uint32_t));
                 return arr;
             },
             py::arg("subject"), py::arg("relation"), py::arg("object"),
             "Encode a (S, R, O) triple into a bitpacked vector")
        .def("add_fact_vec",
             [](grilly::cognitive::WorldModel& wm,
                py::array_t<uint32_t> fact_vec) {
                 auto buf = fact_vec.request();
                 uint32_t dim = wm.dim();
                 uint32_t words = (dim + 31) / 32;

                 grilly::cubemind::BitpackedVec vec;
                 vec.dim = dim;
                 vec.data.assign(
                     static_cast<uint32_t*>(buf.ptr),
                     static_cast<uint32_t*>(buf.ptr) + words);

                 wm.add_fact_vec(vec);
             },
             py::arg("fact_vec"),
             "Add a pre-encoded bitpacked fact vector (no negation)")
        .def_property_readonly("fact_count",
             &grilly::cognitive::WorldModel::fact_count)
        .def_property_readonly("constraint_count",
             &grilly::cognitive::WorldModel::constraint_count)
        .def_property_readonly("dim",
             &grilly::cognitive::WorldModel::dim);

    // ── Hippocampal Consolidator (Offline Dream Cycle) ──────────────────

    py::class_<grilly::DreamReport>(m, "DreamReport")
        .def_readonly("episodes_replayed",
                      &grilly::DreamReport::episodes_replayed)
        .def_readonly("synthetic_dreams",
                      &grilly::DreamReport::synthetic_dreams)
        .def_readonly("new_rules_extracted",
                      &grilly::DreamReport::new_rules_extracted);

    py::class_<grilly::HippocampalConsolidator>(m, "HippocampalConsolidator")
        .def(py::init<uint32_t>(),
             py::arg("max_capacity") = 10000,
             "Create a hippocampal consolidator with FIFO episode buffer")
        .def("record_episode",
             [](grilly::HippocampalConsolidator& hc,
                py::array_t<uint32_t> state_t,
                py::array_t<uint32_t> state_t1) {
                 auto buf_t = state_t.request();
                 auto buf_t1 = state_t1.request();

                 grilly::cubemind::BitpackedVec vec_t;
                 vec_t.dim = static_cast<uint32_t>(buf_t.size) * 32;
                 vec_t.data.assign(
                     static_cast<uint32_t*>(buf_t.ptr),
                     static_cast<uint32_t*>(buf_t.ptr) + buf_t.size);

                 grilly::cubemind::BitpackedVec vec_t1;
                 vec_t1.dim = static_cast<uint32_t>(buf_t1.size) * 32;
                 vec_t1.data.assign(
                     static_cast<uint32_t*>(buf_t1.ptr),
                     static_cast<uint32_t*>(buf_t1.ptr) + buf_t1.size);

                 hc.record_episode(vec_t, vec_t1);
             },
             py::arg("state_t"), py::arg("state_t1"),
             "Record a (state_t, state_t+1) transition episode")
        .def("dream",
             [](grilly::HippocampalConsolidator& hc,
                grilly::cognitive::WorldModel& wm,
                uint32_t cycles) {
                 return hc.dream(wm, cycles);
             },
             py::arg("world_model"), py::arg("cycles") = 128,
             "Run offline dream consolidation, returns DreamReport")
        .def_property_readonly("buffer_size",
             &grilly::HippocampalConsolidator::buffer_size);

    // ── SystemProfile: Hardware Configuration Loader ────────────────────

    py::class_<grilly::SystemProfile>(m, "SystemProfile")
        .def_readonly("device_name", &grilly::SystemProfile::deviceName)
        .def_readonly("subgroup_size", &grilly::SystemProfile::subgroupSize)
        .def_readonly("arena_size_bytes", &grilly::SystemProfile::arenaSizeBytes)
        .def_readonly("vsa_dim", &grilly::SystemProfile::vsaDim)
        .def_readonly("max_cache_capacity", &grilly::SystemProfile::maxCacheCapacity)
        .def_readonly("max_constraint_capacity", &grilly::SystemProfile::maxConstraintCapacity)
        .def_readonly("surprise_threshold", &grilly::SystemProfile::surpriseThreshold)
        .def_readonly("coherence_threshold", &grilly::SystemProfile::coherenceThreshold)
        .def_readonly("thinking_steps", &grilly::SystemProfile::thinkingSteps)
        .def_readonly("batch_size", &grilly::SystemProfile::batchSize)
        .def_readonly("workgroup_size", &grilly::SystemProfile::workgroupSize)
        .def_property_readonly("entries_per_wg",
             &grilly::SystemProfile::entriesPerWG)
        .def_static("load", &grilly::SystemProfile::load,
             py::arg("path"), py::arg("profile_name"),
             "Load a hardware profile from profiles.json");

    // ── Autograd: TapeArena + Wengert List Backward Engine ──────────────

    py::enum_<grilly::autograd::OpType>(m, "OpType")
        .value("Add", grilly::autograd::OpType::Add)
        .value("Sub", grilly::autograd::OpType::Sub)
        .value("Mul", grilly::autograd::OpType::Mul)
        .value("Div", grilly::autograd::OpType::Div)
        .value("Neg", grilly::autograd::OpType::Neg)
        .value("Pow", grilly::autograd::OpType::Pow)
        .value("MatMul", grilly::autograd::OpType::MatMul)
        .value("Linear", grilly::autograd::OpType::Linear)
        .value("ReLU", grilly::autograd::OpType::ReLU)
        .value("GELU", grilly::autograd::OpType::GELU)
        .value("SiLU", grilly::autograd::OpType::SiLU)
        .value("Tanh", grilly::autograd::OpType::Tanh)
        .value("Sigmoid", grilly::autograd::OpType::Sigmoid)
        .value("Softmax", grilly::autograd::OpType::Softmax)
        .value("LayerNorm", grilly::autograd::OpType::LayerNorm)
        .value("RMSNorm", grilly::autograd::OpType::RMSNorm)
        .value("FlashAttention2", grilly::autograd::OpType::FlashAttention2)
        .value("Conv2d", grilly::autograd::OpType::Conv2d)
        .value("Conv1d", grilly::autograd::OpType::Conv1d)
        .value("Sum", grilly::autograd::OpType::Sum)
        .value("Mean", grilly::autograd::OpType::Mean)
        .value("Max", grilly::autograd::OpType::Max)
        .value("Min", grilly::autograd::OpType::Min)
        .value("Reshape", grilly::autograd::OpType::Reshape)
        .value("Transpose", grilly::autograd::OpType::Transpose)
        .value("Slice", grilly::autograd::OpType::Slice)
        .value("CrossEntropy", grilly::autograd::OpType::CrossEntropy)
        .value("MSELoss", grilly::autograd::OpType::MSELoss)
        .value("CubeMindSurprise", grilly::autograd::OpType::CubeMindSurprise)
        .value("TemporalSurprise", grilly::autograd::OpType::TemporalSurprise)
        .export_values();

    py::class_<grilly::autograd::TensorRef>(m, "TensorRef")
        .def(py::init<>())
        .def_readwrite("buffer_id", &grilly::autograd::TensorRef::buffer_id)
        .def_readwrite("ndim", &grilly::autograd::TensorRef::ndim)
        .def_readwrite("dtype", &grilly::autograd::TensorRef::dtype)
        .def_readwrite("requires_grad", &grilly::autograd::TensorRef::requires_grad)
        .def("numel", &grilly::autograd::TensorRef::numel)
        .def("size_bytes", &grilly::autograd::TensorRef::size_bytes)
        .def("valid", &grilly::autograd::TensorRef::valid)
        .def_static("none", &grilly::autograd::TensorRef::none)
        .def("set_shape",
             [](grilly::autograd::TensorRef& ref,
                const std::vector<uint32_t>& shape) {
                 ref.ndim = static_cast<uint32_t>(
                     std::min(shape.size(), size_t(8)));
                 for (uint32_t i = 0; i < ref.ndim; ++i) {
                     ref.shape[i] = shape[i];
                 }
             },
             py::arg("shape"))
        .def("get_shape",
             [](const grilly::autograd::TensorRef& ref) -> std::vector<uint32_t> {
                 return std::vector<uint32_t>(ref.shape, ref.shape + ref.ndim);
             });

    py::class_<grilly::autograd::TapeContext>(m, "TapeContext")
        .def(py::init(
                 [](GrillyCoreContext& ctx, size_t capacity) {
                     return new grilly::autograd::TapeContext(
                         ctx.pool, ctx.batch, ctx.cache, capacity);
                 }),
             py::arg("device"),
             py::arg("arena_capacity") = grilly::autograd::TapeArena::kDefaultCapacity,
             py::keep_alive<1, 2>())
        .def("begin", &grilly::autograd::TapeContext::begin)
        .def("record_op",
             [](grilly::autograd::TapeContext& tape,
                grilly::autograd::OpType op,
                const std::vector<grilly::autograd::TensorRef>& inputs,
                const std::vector<grilly::autograd::TensorRef>& outputs)
                 -> grilly::autograd::Node* {
                 return tape.record_op(
                     op,
                     inputs.data(),
                     static_cast<uint32_t>(inputs.size()),
                     outputs.data(),
                     static_cast<uint32_t>(outputs.size()));
             },
             py::arg("op"), py::arg("inputs"), py::arg("outputs"),
             py::return_value_policy::reference)
        .def("save_for_backward",
             [](grilly::autograd::TapeContext& tape,
                grilly::autograd::Node* node,
                const std::vector<uint32_t>& buffer_ids) {
                 tape.save_for_backward(
                     node, buffer_ids.data(),
                     static_cast<uint32_t>(buffer_ids.size()));
             },
             py::arg("node"), py::arg("buffer_ids"))
        .def("backward",
             [](grilly::autograd::TapeContext& tape,
                grilly::autograd::Node* loss_node,
                uint32_t grad_output_buffer) {
                 tape.backward(loss_node, grad_output_buffer);
             },
             py::arg("loss_node"), py::arg("grad_output_buffer"))
        .def("get_grad_buffer",
             &grilly::autograd::TapeContext::get_grad_buffer,
             py::arg("input_buffer_id"))
        .def("record_temporal_surprise",
             [](grilly::autograd::TapeContext& tape,
                const grilly::autograd::TensorRef& input,
                const grilly::autograd::TensorRef& output,
                float avg_coherence,
                float avg_contradiction,
                uint32_t num_branches,
                uint32_t dt,
                float alpha) -> grilly::autograd::Node* {
                 // Record the TemporalSurprise node on the tape
                 grilly::autograd::TensorRef ins[1] = {input};
                 grilly::autograd::TensorRef outs[1] = {output};

                 // Pre-compute the temporal multiplier:
                 //   1.0 = futures are fully coherent (pass gradient through)
                 //   0.0 = half-contradictory (attenuate gradient)
                 //  -1.0 = fully contradictory (reverse gradient direction)
                 float multiplier = 1.0f - 2.0f * avg_contradiction;

                 grilly::autograd::TemporalSurpriseParams tparams;
                 tparams.avg_coherence = avg_coherence;
                 tparams.avg_contradiction = avg_contradiction;
                 tparams.temporal_multiplier = multiplier;
                 tparams.alpha = alpha;
                 tparams.num_branches = num_branches;
                 tparams.dt = dt;

                 auto* node = tape.record_op(
                     grilly::autograd::OpType::TemporalSurprise,
                     ins, 1, outs, 1,
                     &tparams, sizeof(tparams));
                 return node;
             },
             py::arg("input"),
             py::arg("output"),
             py::arg("avg_coherence"),
             py::arg("avg_contradiction"),
             py::arg("num_branches") = 128,
             py::arg("dt") = 1,
             py::arg("alpha") = 1.0f,
             py::return_value_policy::reference,
             "Record a TemporalSurprise node on the tape.\n"
             "Stores counterfactual results for gradient modulation during backward.")
        .def("end", &grilly::autograd::TapeContext::end)
        .def("is_recording", &grilly::autograd::TapeContext::is_recording)
        .def("arena_bytes_used", &grilly::autograd::TapeContext::arena_bytes_used)
        .def("arena_utilization", &grilly::autograd::TapeContext::arena_utilization)
        .def("last_backward_stats",
             [](const grilly::autograd::TapeContext& tape) -> py::dict {
                 auto s = tape.last_backward_stats();
                 py::dict d;
                 d["nodes_visited"] = s.nodes_visited;
                 d["nodes_with_grad"] = s.nodes_with_grad;
                 d["shaders_dispatched"] = s.shaders_dispatched;
                 d["cpu_fallbacks"] = s.cpu_fallbacks;
                 return d;
             });

    // Expose Node for inspection (read-only from Python)
    py::class_<grilly::autograd::Node>(m, "AutogradNode")
        .def_readonly("op", &grilly::autograd::Node::op)
        .def_readonly("seq", &grilly::autograd::Node::seq)
        .def_readonly("num_inputs", &grilly::autograd::Node::num_inputs)
        .def_readonly("num_outputs", &grilly::autograd::Node::num_outputs)
        .def_readonly("num_saved", &grilly::autograd::Node::num_saved)
        .def_readonly("grad_output_buffer",
                      &grilly::autograd::Node::grad_output_buffer)
        .def("get_grad_input_buffer",
             [](const grilly::autograd::Node& node, uint32_t idx) -> uint32_t {
                 if (idx >= grilly::autograd::kMaxNodeIO) return 0;
                 return node.grad_input_buffers[idx];
             },
             py::arg("index"))
        .def("get_input",
             [](const grilly::autograd::Node& node, uint32_t idx)
                 -> grilly::autograd::TensorRef {
                 if (idx >= grilly::autograd::kMaxNodeIO)
                     return grilly::autograd::TensorRef::none();
                 return node.inputs[idx];
             },
             py::arg("index"))
        .def("get_output",
             [](const grilly::autograd::Node& node, uint32_t idx)
                 -> grilly::autograd::TensorRef {
                 if (idx >= grilly::autograd::kMaxNodeIO)
                     return grilly::autograd::TensorRef::none();
                 return node.outputs[idx];
             },
             py::arg("index"));

    // ── Temporal: TemporalEncoder + CounterfactualReasoner ────────────────

    py::class_<grilly::temporal::CounterfactualResult>(m, "CounterfactualResult")
        .def_readonly("coherence", &grilly::temporal::CounterfactualResult::coherence)
        .def_readonly("surprise", &grilly::temporal::CounterfactualResult::surprise)
        .def_readonly("best_idx", &grilly::temporal::CounterfactualResult::best_idx);

    // TemporalEncoder: static methods exposed as module-level functions
    m.def("temporal_bind",
         [](py::array_t<uint32_t> state, uint32_t t,
            uint32_t dim) -> py::array_t<uint32_t> {
             auto buf = state.request();
             grilly::cubemind::BitpackedVec vec;
             vec.dim = dim;
             uint32_t words = (dim + 31) / 32;
             vec.data.assign(
                 static_cast<uint32_t*>(buf.ptr),
                 static_cast<uint32_t*>(buf.ptr) + words);

             auto result = grilly::temporal::TemporalEncoder::bind_time(vec, t);

             py::array_t<uint32_t> arr(result.data.size());
             std::memcpy(arr.mutable_data(), result.data.data(),
                         result.data.size() * sizeof(uint32_t));
             return arr;
         },
         py::arg("state"), py::arg("t"), py::arg("dim") = 10240,
         "Bind a bitpacked VSA state to time step t (circular right shift)");

    m.def("temporal_unbind",
         [](py::array_t<uint32_t> state, uint32_t t,
            uint32_t dim) -> py::array_t<uint32_t> {
             auto buf = state.request();
             grilly::cubemind::BitpackedVec vec;
             vec.dim = dim;
             uint32_t words = (dim + 31) / 32;
             vec.data.assign(
                 static_cast<uint32_t*>(buf.ptr),
                 static_cast<uint32_t*>(buf.ptr) + words);

             auto result = grilly::temporal::TemporalEncoder::unbind_time(vec, t);

             py::array_t<uint32_t> arr(result.data.size());
             std::memcpy(arr.mutable_data(), result.data.data(),
                         result.data.size() * sizeof(uint32_t));
             return arr;
         },
         py::arg("state"), py::arg("t"), py::arg("dim") = 10240,
         "Unbind time step t from a previously bound state (circular left shift)");

    m.def("xor_bind",
         [](py::array_t<uint32_t> a, py::array_t<uint32_t> b,
            uint32_t dim) -> py::array_t<uint32_t> {
             auto buf_a = a.request();
             auto buf_b = b.request();
             uint32_t words = (dim + 31) / 32;

             grilly::cubemind::BitpackedVec va, vb;
             va.dim = dim; vb.dim = dim;
             va.data.assign(
                 static_cast<uint32_t*>(buf_a.ptr),
                 static_cast<uint32_t*>(buf_a.ptr) + words);
             vb.data.assign(
                 static_cast<uint32_t*>(buf_b.ptr),
                 static_cast<uint32_t*>(buf_b.ptr) + words);

             auto result = grilly::temporal::TemporalEncoder::xor_bind(va, vb);

             py::array_t<uint32_t> arr(result.data.size());
             std::memcpy(arr.mutable_data(), result.data.data(),
                         result.data.size() * sizeof(uint32_t));
             return arr;
         },
         py::arg("a"), py::arg("b"), py::arg("dim") = 10240,
         "XOR-bind two bitpacked VSA vectors (self-inverse binding)");

    py::class_<grilly::temporal::CounterfactualReasoner>(m, "CounterfactualReasoner")
        .def(py::init<>())
        .def("evaluate",
             [](grilly::temporal::CounterfactualReasoner& cf,
                GrillyCoreContext& ctx,
                py::array_t<uint32_t> timeline,
                py::array_t<uint32_t> actual_fact,
                py::array_t<uint32_t> what_if_fact,
                grilly::cubemind::VSACache& world_cache,
                uint32_t dt, uint32_t dim) {
                 auto buf_t = timeline.request();
                 auto buf_a = actual_fact.request();
                 auto buf_w = what_if_fact.request();
                 uint32_t words = (dim + 31) / 32;

                 grilly::cubemind::BitpackedVec vt, va, vw;
                 vt.dim = dim; va.dim = dim; vw.dim = dim;
                 vt.data.assign(
                     static_cast<uint32_t*>(buf_t.ptr),
                     static_cast<uint32_t*>(buf_t.ptr) + words);
                 va.data.assign(
                     static_cast<uint32_t*>(buf_a.ptr),
                     static_cast<uint32_t*>(buf_a.ptr) + words);
                 vw.data.assign(
                     static_cast<uint32_t*>(buf_w.ptr),
                     static_cast<uint32_t*>(buf_w.ptr) + words);

                 return cf.evaluate(vt, va, vw, world_cache,
                                    ctx.batch, ctx.cache, dt);
             },
             py::arg("device"),
             py::arg("timeline"),
             py::arg("actual_fact"),
             py::arg("what_if_fact"),
             py::arg("world_cache"),
             py::arg("dt") = 1,
             py::arg("dim") = 10240,
             "Evaluate a counterfactual: erase actual_fact, insert what_if_fact, "
             "project dt steps forward, and check coherence against world_cache")
        .def("evaluate_cpu",
             [](grilly::temporal::CounterfactualReasoner& cf,
                py::array_t<uint32_t> timeline,
                py::array_t<uint32_t> actual_fact,
                py::array_t<uint32_t> what_if_fact,
                grilly::cubemind::VSACache& world_cache,
                uint32_t dt, uint32_t dim) {
                 auto buf_t = timeline.request();
                 auto buf_a = actual_fact.request();
                 auto buf_w = what_if_fact.request();
                 uint32_t words = (dim + 31) / 32;

                 grilly::cubemind::BitpackedVec vt, va, vw;
                 vt.dim = dim; va.dim = dim; vw.dim = dim;
                 vt.data.assign(
                     static_cast<uint32_t*>(buf_t.ptr),
                     static_cast<uint32_t*>(buf_t.ptr) + words);
                 va.data.assign(
                     static_cast<uint32_t*>(buf_a.ptr),
                     static_cast<uint32_t*>(buf_a.ptr) + words);
                 vw.data.assign(
                     static_cast<uint32_t*>(buf_w.ptr),
                     static_cast<uint32_t*>(buf_w.ptr) + words);

                 return cf.evaluate_cpu(vt, va, vw, world_cache, dt);
             },
             py::arg("timeline"),
             py::arg("actual_fact"),
             py::arg("what_if_fact"),
             py::arg("world_cache"),
             py::arg("dt") = 1,
             py::arg("dim") = 10240,
             "CPU-only counterfactual evaluation (no GPU needed)");

    // ── Temporal: VulkanTemporalDispatcher (GPU Batch Operations) ─────────

    m.def("batch_temporal_shift",
         [](GrillyCoreContext& ctx,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> input,
            uint32_t shift_amount,
            uint32_t mode,
            uint32_t dim) -> py::array_t<uint32_t> {
             auto buf = input.request();
             if (buf.ndim != 2) {
                 throw std::runtime_error(
                     "input must be 2D array of shape (batch_size, words_per_vec)");
             }

             uint32_t batch_size = static_cast<uint32_t>(buf.shape[0]);
             uint32_t words_per_vec = (dim + 31) / 32;

             auto result = grilly::temporal::VulkanTemporalDispatcher::dispatch(
                 ctx.batch, ctx.pool, ctx.cache,
                 static_cast<const uint32_t*>(buf.ptr),
                 batch_size, words_per_vec, shift_amount, mode);

             py::array_t<uint32_t> arr({batch_size, words_per_vec});
             std::memcpy(arr.mutable_data(), result.data.data(),
                         result.data.size() * sizeof(uint32_t));
             return arr;
         },
         py::arg("device"),
         py::arg("input"),
         py::arg("shift_amount") = 1,
         py::arg("mode") = 0,
         py::arg("dim") = 10240,
         "GPU batch circular shift: shift N bitpacked vectors simultaneously.\n"
         "mode 0 = right shift (bind_time), mode 1 = left shift (unbind_time)");

    m.def("batch_counterfactuals",
         [](GrillyCoreContext& ctx,
            py::array_t<uint32_t> base_timeline,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> actual_facts,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> what_if_facts,
            uint32_t dt,
            uint32_t dim) -> py::array_t<uint32_t> {
             auto buf_base = base_timeline.request();
             auto buf_actual = actual_facts.request();
             auto buf_whatif = what_if_facts.request();

             if (buf_actual.ndim != 2 || buf_whatif.ndim != 2) {
                 throw std::runtime_error(
                     "actual_facts and what_if_facts must be 2D (N, words_per_vec)");
             }

             uint32_t n = static_cast<uint32_t>(buf_actual.shape[0]);
             uint32_t words_per_vec = (dim + 31) / 32;

             auto result = grilly::temporal::VulkanTemporalDispatcher
                 ::batch_counterfactuals(
                     ctx.batch, ctx.pool, ctx.cache,
                     static_cast<const uint32_t*>(buf_base.ptr),
                     static_cast<const uint32_t*>(buf_actual.ptr),
                     static_cast<const uint32_t*>(buf_whatif.ptr),
                     n, words_per_vec, dt);

             py::array_t<uint32_t> arr({n, words_per_vec});
             std::memcpy(arr.mutable_data(), result.data.data(),
                         result.data.size() * sizeof(uint32_t));
             return arr;
         },
         py::arg("device"),
         py::arg("base_timeline"),
         py::arg("actual_facts"),
         py::arg("what_if_facts"),
         py::arg("dt") = 1,
         py::arg("dim") = 10240,
         "GPU batch counterfactual evaluation:\n"
         "  1. XOR-erase actual_facts from base_timeline (CPU, instant)\n"
         "  2. XOR-insert what_if_facts (CPU, instant)\n"
         "  3. Circular shift all N timelines forward by dt steps (GPU)\n"
         "Returns (N, words_per_vec) array of shifted alternate futures.");

    // ═══════════════════════════════════════════════════════════════════════
    // COMPUTE BACKEND (abstract, extensible — Vulkan / OpenGL / OpenCL)
    // ═══════════════════════════════════════════════════════════════════════

    py::enum_<grilly::BufferDesc::Usage>(m, "BufferUsage")
        .value("HostVisible", grilly::BufferDesc::HostVisible)
        .value("DeviceLocal", grilly::BufferDesc::DeviceLocal)
        .value("Readback", grilly::BufferDesc::Readback)
        .value("PreferDevice", grilly::BufferDesc::PreferDevice);

    py::class_<grilly::ComputeBackend>(m, "ComputeBackend")
        .def_property_readonly("name", &grilly::ComputeBackend::name,
             "Backend name: 'vulkan', 'opengl', 'opencl'")
        .def_property_readonly("device_name", &grilly::ComputeBackend::deviceName,
             "GPU device name reported by the driver")
        .def("create_buffer",
             [](grilly::ComputeBackend& b, size_t size,
                grilly::BufferDesc::Usage usage) -> uint64_t {
                 grilly::BufferDesc desc;
                 desc.size = size;
                 desc.usage = usage;
                 return b.createBuffer(desc);
             },
             py::arg("size"),
             py::arg("usage") = grilly::BufferDesc::HostVisible,
             "Allocate a GPU buffer. Returns an opaque handle.")
        .def("destroy_buffer", &grilly::ComputeBackend::destroyBuffer,
             py::arg("handle"), "Free a GPU buffer")
        .def("upload",
             [](grilly::ComputeBackend& b, uint64_t handle,
                py::array_t<float> data) {
                 auto buf = data.request();
                 b.upload(handle, buf.ptr, buf.size * sizeof(float));
             },
             py::arg("handle"), py::arg("data"))
        .def("download",
             [](grilly::ComputeBackend& b, uint64_t handle,
                size_t count) -> py::array_t<float> {
                 py::array_t<float> out(count);
                 b.download(handle, out.mutable_data(), count * sizeof(float));
                 return out;
             },
             py::arg("handle"), py::arg("count"),
             "Download float data from GPU buffer")
        .def("load_shader", &grilly::ComputeBackend::loadShader,
             py::arg("name"), py::arg("path"))
        .def("load_shader_dir", &grilly::ComputeBackend::loadShaderDir,
             py::arg("dir"), "Load all shaders from a directory")
        .def("has_shader", &grilly::ComputeBackend::hasShader,
             py::arg("name"))
        .def("begin_batch", &grilly::ComputeBackend::beginBatch)
        .def("dispatch",
             [](grilly::ComputeBackend& b, const std::string& shader,
                const std::vector<uint64_t>& buffers,
                uint32_t gx, uint32_t gy, uint32_t gz,
                std::optional<py::array_t<uint8_t>> push) {
                 if (push) {
                     auto buf = push->request();
                     b.dispatch(shader, buffers, gx, gy, gz,
                                buf.ptr, static_cast<uint32_t>(buf.size));
                 } else {
                     b.dispatch(shader, buffers, gx, gy, gz);
                 }
             },
             py::arg("shader"), py::arg("buffers"),
             py::arg("gx"), py::arg("gy") = 1, py::arg("gz") = 1,
             py::arg("push") = py::none(),
             "Dispatch a compute shader")
        .def("barrier", &grilly::ComputeBackend::barrier)
        .def("end_batch", &grilly::ComputeBackend::endBatch,
             "Submit current batch and wait for completion")
        .def_property_readonly("has_cooperative_matrix",
             &grilly::ComputeBackend::hasCooperativeMatrix)
        .def_property_readonly("has_float16",
             &grilly::ComputeBackend::hasFloat16);

    m.def("create_backend", &grilly::createBackend,
          py::arg("type") = "vulkan",
          py::return_value_policy::move,
          "Create a GPU compute backend.\n"
          "Supported: 'vulkan'. Coming soon: 'opengl', 'opencl'.");

    // ═══════════════════════════════════════════════════════════════════════
    // NN FRAMEWORK — Tensor, Parameter, Module, SNN, Optimizer, DataLoader
    // ═══════════════════════════════════════════════════════════════════════

    using namespace grilly::nn;

    // ── DType enum ──
    py::enum_<DType>(m, "DType")
        .value("Float32", DType::Float32)
        .value("Float16", DType::Float16)
        .value("Int32", DType::Int32)
        .value("Int64", DType::Int64);

    // ── Tensor ──
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::vector<int64_t>, DType, grilly::ComputeBackend*>(),
             py::arg("shape"),
             py::arg("dtype") = DType::Float32,
             py::arg("backend") = nullptr)
        .def_static("from_numpy", &Tensor::from_numpy,
             py::arg("arr"), py::arg("backend") = nullptr,
             "Create tensor from numpy array")
        .def_static("zeros", &Tensor::zeros,
             py::arg("shape"), py::arg("backend") = nullptr,
             "Create zero-filled tensor")
        .def_static("empty", &Tensor::empty,
             py::arg("shape"), py::arg("backend") = nullptr,
             "Create uninitialized tensor")
        .def("numpy", &Tensor::numpy,
             "Download to CPU and return as numpy array")
        .def("gpu_handle", &Tensor::gpu_handle,
             "Upload to GPU and return buffer handle")
        .def("mark_gpu_modified", &Tensor::mark_gpu_modified)
        .def("mark_cpu_modified", &Tensor::mark_cpu_modified)
        .def("reshape", &Tensor::reshape, py::arg("shape"))
        .def("view", &Tensor::view, py::arg("shape"))
        .def("ensure_gpu", &Tensor::ensure_gpu)
        .def("release_gpu", &Tensor::release_gpu)
        .def_property_readonly("shape",
             [](const Tensor& t) { return t.shape(); })
        .def_property_readonly("ndim", &Tensor::ndim)
        .def_property_readonly("numel", &Tensor::numel)
        .def_property_readonly("nbytes", &Tensor::nbytes)
        .def_property_readonly("dtype", &Tensor::dtype)
        .def_property_readonly("on_gpu", &Tensor::on_gpu)
        .def_property_readonly("on_cpu", &Tensor::on_cpu)
        .def_property_readonly("valid", &Tensor::valid)
        .def_property("requires_grad", &Tensor::requires_grad,
                      &Tensor::set_requires_grad)
        .def("__repr__", [](const Tensor& t) {
            std::string s = "Tensor(shape=[";
            for (size_t i = 0; i < t.shape().size(); i++) {
                if (i > 0) s += ", ";
                s += std::to_string(t.shape()[i]);
            }
            s += "], dtype=float32";
            if (t.on_gpu()) s += ", gpu";
            if (t.requires_grad()) s += ", requires_grad=True";
            s += ")";
            return s;
        });

    // ── Parameter ──
    py::class_<Parameter, Tensor>(m, "Parameter")
        .def(py::init<Tensor, bool>(),
             py::arg("data"),
             py::arg("requires_grad") = true)
        .def(py::init<std::vector<int64_t>, grilly::ComputeBackend*, bool>(),
             py::arg("shape"),
             py::arg("backend") = nullptr,
             py::arg("requires_grad") = true)
        .def("grad", static_cast<Tensor& (Parameter::*)()>(&Parameter::grad_ref),
             py::return_value_policy::reference)
        .def("set_grad",
             static_cast<void (Parameter::*)(const Tensor&)>(&Parameter::set_grad),
             py::arg("grad"))
        .def("has_grad", &Parameter::has_grad)
        .def("zero_grad", &Parameter::zero_grad);

    // ── Module (with trampoline for Python subclassing) ──
    py::class_<Module, PyModule, std::shared_ptr<Module>>(m, "Module")
        .def(py::init<>())
        .def("forward", &Module::forward, py::arg("input"))
        .def("__call__", &Module::operator(), py::arg("input"))
        .def("register_parameter", &Module::register_parameter,
             py::arg("name"), py::arg("param"))
        .def("register_module", &Module::register_module,
             py::arg("name"), py::arg("module"))
        .def("parameters", &Module::parameters,
             py::return_value_policy::reference)
        .def("named_parameters", &Module::named_parameters,
             py::return_value_policy::reference)
        .def("train", &Module::train, py::arg("mode") = true)
        .def("eval", &Module::eval)
        .def_property_readonly("training", &Module::is_training)
        .def("state_dict", &Module::state_dict)
        .def("load_state_dict", &Module::load_state_dict,
             py::arg("state"))
        .def("gpu_mode", &Module::gpu_mode,
             py::arg("enable") = true, py::arg("device_local") = true)
        .def("to", &Module::to, py::arg("device"));

    // ── SurrogateFunction ──
    py::enum_<SurrogateType>(m, "SurrogateType")
        .value("ATan", SurrogateType::ATan)
        .value("Sigmoid", SurrogateType::Sigmoid)
        .value("FastSigmoid", SurrogateType::FastSigmoid);

    py::class_<SurrogateFunction>(m, "SurrogateFunction")
        .def(py::init<SurrogateType, float>(),
             py::arg("type") = SurrogateType::ATan,
             py::arg("alpha") = 2.0f)
        .def("forward", &SurrogateFunction::forward, py::arg("x"))
        .def("gradient", &SurrogateFunction::gradient, py::arg("x"))
        .def_readwrite("alpha", &SurrogateFunction::alpha)
        .def_readwrite("type", &SurrogateFunction::type);

    // ── MemoryModule ──
    py::class_<MemoryModule, Module, std::shared_ptr<MemoryModule>>(
            m, "MemoryModule")
        .def("register_memory", &MemoryModule::register_memory,
             py::arg("name"), py::arg("default_value"))
        .def("reset", &MemoryModule::reset)
        .def("detach", &MemoryModule::detach);

    // ── BaseNode ──
    py::class_<BaseNode, MemoryModule, std::shared_ptr<BaseNode>>(
            m, "BaseNode")
        .def("single_step_forward", &BaseNode::single_step_forward,
             py::arg("x"))
        .def("multi_step_forward", &BaseNode::multi_step_forward,
             py::arg("x_seq"))
        .def("backward", &BaseNode::backward, py::arg("grad_output"))
        .def_property_readonly("v_threshold", &BaseNode::v_threshold)
        .def_property_readonly("v_reset", &BaseNode::v_reset)
        .def_property("step_mode", &BaseNode::step_mode,
                      &BaseNode::set_step_mode);

    // ── IFNode ──
    py::class_<IFNode, BaseNode, std::shared_ptr<IFNode>>(m, "IFNode")
        .def(py::init<float, float, SurrogateFunction, bool, std::string,
                      bool>(),
             py::arg("v_threshold") = 1.0f,
             py::arg("v_reset") = 0.0f,
             py::arg("surrogate") = ATan(),
             py::arg("detach_reset") = false,
             py::arg("step_mode") = "s",
             py::arg("store_v_seq") = false);

    // ── LIFNode ──
    py::class_<LIFNode, BaseNode, std::shared_ptr<LIFNode>>(m, "LIFNode")
        .def(py::init<float, bool, float, float, SurrogateFunction,
                      bool, std::string, bool>(),
             py::arg("tau") = 2.0f,
             py::arg("decay_input") = false,
             py::arg("v_threshold") = 1.0f,
             py::arg("v_reset") = 0.0f,
             py::arg("surrogate") = ATan(),
             py::arg("detach_reset") = false,
             py::arg("step_mode") = "s",
             py::arg("store_v_seq") = false)
        .def_property_readonly("tau", &LIFNode::tau);

    // ── ParametricLIFNode ──
    py::class_<ParametricLIFNode, BaseNode,
               std::shared_ptr<ParametricLIFNode>>(m, "ParametricLIFNode")
        .def(py::init<float, bool, float, float, SurrogateFunction,
                      bool, std::string, bool>(),
             py::arg("init_tau") = 2.0f,
             py::arg("decay_input") = false,
             py::arg("v_threshold") = 1.0f,
             py::arg("v_reset") = 0.0f,
             py::arg("surrogate") = ATan(),
             py::arg("detach_reset") = false,
             py::arg("step_mode") = "s",
             py::arg("store_v_seq") = false);

    // ── Containers ──
    py::class_<MultiStepContainer, Module,
               std::shared_ptr<MultiStepContainer>>(m, "MultiStepContainer")
        .def(py::init<std::shared_ptr<Module>>(), py::arg("module"),
             py::keep_alive<1, 2>())
        .def("backward", &MultiStepContainer::backward,
             py::arg("grad_output"));

    py::class_<SeqToANNContainer, Module,
               std::shared_ptr<SeqToANNContainer>>(m, "SeqToANNContainer")
        .def(py::init<std::vector<std::shared_ptr<Module>>>(),
             py::arg("modules"),
             py::keep_alive<1, 2>())
        .def("backward", &SeqToANNContainer::backward,
             py::arg("grad_output"));

    py::class_<Flatten, Module, std::shared_ptr<Flatten>>(m, "Flatten")
        .def(py::init<int, int>(),
             py::arg("start_dim") = 1, py::arg("end_dim") = -1)
        .def("backward", &Flatten::backward, py::arg("grad_output"));

    // ── Optimizers ──
    py::class_<Optimizer>(m, "Optimizer")
        .def("step", &Optimizer::step)
        .def("zero_grad", &Optimizer::zero_grad)
        .def("state_dict", &Optimizer::state_dict)
        .def("load_state_dict", &Optimizer::load_state_dict,
             py::arg("state"))
        .def_property_readonly("param_count", &Optimizer::param_count);

    py::class_<Adam, Optimizer>(m, "Adam")
        .def(py::init([](py::list params, float lr,
                         std::pair<float, float> betas, float eps,
                         float weight_decay, bool amsgrad) {
                 std::vector<Parameter*> ptrs;
                 for (auto& item : params) {
                     ptrs.push_back(item.cast<Parameter*>());
                 }
                 auto opt = new Adam(std::move(ptrs), lr, betas, eps,
                                     weight_decay, amsgrad);
                 opt->py_params_ = params;
                 return opt;
             }),
             py::arg("params"),
             py::arg("lr") = 1e-3f,
             py::arg("betas") = std::make_pair(0.9f, 0.999f),
             py::arg("eps") = 1e-8f,
             py::arg("weight_decay") = 0.0f,
             py::arg("amsgrad") = false)
        .def_property("lr", &Adam::lr, &Adam::set_lr);

    py::class_<AdamW, Optimizer>(m, "AdamW")
        .def(py::init([](py::list params, float lr,
                         std::pair<float, float> betas, float eps,
                         float weight_decay, bool amsgrad) {
                 std::vector<Parameter*> ptrs;
                 for (auto& item : params) {
                     ptrs.push_back(item.cast<Parameter*>());
                 }
                 auto opt = new AdamW(std::move(ptrs), lr, betas, eps,
                                      weight_decay, amsgrad);
                 opt->py_params_ = params;
                 return opt;
             }),
             py::arg("params"),
             py::arg("lr") = 1e-3f,
             py::arg("betas") = std::make_pair(0.9f, 0.999f),
             py::arg("eps") = 1e-8f,
             py::arg("weight_decay") = 0.01f,
             py::arg("amsgrad") = false)
        .def_property("lr", &AdamW::lr, &AdamW::set_lr);

    py::class_<SGD, Optimizer>(m, "SGD")
        .def(py::init([](py::list params, float lr, float momentum,
                         float weight_decay, bool nesterov) {
                 std::vector<Parameter*> ptrs;
                 for (auto& item : params) {
                     ptrs.push_back(item.cast<Parameter*>());
                 }
                 auto opt = new SGD(std::move(ptrs), lr, momentum,
                                    weight_decay, nesterov);
                 opt->py_params_ = params;
                 return opt;
             }),
             py::arg("params"),
             py::arg("lr") = 0.01f,
             py::arg("momentum") = 0.0f,
             py::arg("weight_decay") = 0.0f,
             py::arg("nesterov") = false)
        .def_property("lr", &SGD::lr, &SGD::set_lr);

    // ── DataLoader ──
    py::class_<Batch>(m, "Batch")
        .def_readwrite("data", &Batch::data)
        .def_readwrite("target", &Batch::target);

    py::class_<DataLoader>(m, "DataLoader")
        .def(py::init<py::object, int, bool, int, bool>(),
             py::arg("dataset"),
             py::arg("batch_size") = 1,
             py::arg("shuffle") = false,
             py::arg("num_workers") = 0,
             py::arg("drop_last") = false)
        .def_property_readonly("num_batches", &DataLoader::num_batches)
        .def_property_readonly("dataset_size", &DataLoader::dataset_size)
        .def("__iter__", &DataLoader::iter, py::return_value_policy::reference)
        .def("iter", &DataLoader::iter, py::return_value_policy::reference)
        .def("__next__", &DataLoader::next);

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 3-6: ADDITIONAL OP BINDINGS
    // ═══════════════════════════════════════════════════════════════════════

    // ── Tanh backward ───────────────────────────────────────────────────
    defActivationBackward("tanh_backward", grilly::ops::tanhBackward);

    // ── Softmax ─────────────────────────────────────────────────────────
    m.def(
        "softmax",
        [](GrillyCoreContext& ctx, py::array_t<float> input,
           int dim) -> py::array_t<float> {
            auto inBuf = input.request();
            if (inBuf.ndim < 1)
                throw std::runtime_error("input must be at least 1D");

            uint32_t features = static_cast<uint32_t>(
                inBuf.shape[inBuf.ndim - 1]);
            uint32_t totalBatch = 1;
            for (int i = 0; i < inBuf.ndim - 1; ++i)
                totalBatch *= static_cast<uint32_t>(inBuf.shape[i]);
            if (inBuf.ndim == 1) totalBatch = 1;

            py::array_t<float> result(inBuf.shape);
            grilly::ops::softmax(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(inBuf.ptr),
                static_cast<float*>(result.request().ptr),
                1, totalBatch, features);
            return result;
        },
        py::arg("device"), py::arg("input"), py::arg("dim") = -1,
        "GPU Softmax (3-pass: max, sum_exp, normalize)");

    // ── Softmax backward ────────────────────────────────────────────────
    m.def(
        "softmax_backward",
        [](GrillyCoreContext& ctx,
           py::array_t<float> grad_output,
           py::array_t<float> softmax_output) -> py::array_t<float> {
            auto gBuf = grad_output.request();
            uint32_t numClasses = static_cast<uint32_t>(
                gBuf.shape[gBuf.ndim - 1]);
            uint32_t batchSeq = 1;
            for (int i = 0; i < gBuf.ndim - 1; ++i)
                batchSeq *= static_cast<uint32_t>(gBuf.shape[i]);
            if (gBuf.ndim == 1) batchSeq = 1;

            py::array_t<float> result(gBuf.shape);
            grilly::ops::softmaxBackward(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(gBuf.ptr),
                static_cast<const float*>(softmax_output.request().ptr),
                static_cast<float*>(result.request().ptr),
                1, batchSeq, numClasses);
            return result;
        },
        py::arg("device"), py::arg("grad_output"), py::arg("softmax_output"),
        "GPU Softmax backward");

    // ── Linear backward ─────────────────────────────────────────────────
    m.def(
        "linear_backward",
        [](GrillyCoreContext& ctx,
           py::array_t<float> grad_output, py::array_t<float> input,
           py::array_t<float> weights) -> py::dict {
            auto gBuf = grad_output.request();
            auto iBuf = input.request();
            auto wBuf = weights.request();

            auto [batchSeq, outputDim] = extractBatchAndLastDim(gBuf);
            uint32_t inputDim = static_cast<uint32_t>(
                iBuf.shape[iBuf.ndim - 1]);

            grilly::ops::LinearParams p{batchSeq, inputDim, outputDim, 1};

            py::array_t<float> gradInput(iBuf.shape);
            py::array_t<float> gradWeight(wBuf.shape);
            py::array_t<float> gradBias({static_cast<py::ssize_t>(outputDim)});

            grilly::ops::linearBackward(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(gBuf.ptr),
                static_cast<const float*>(iBuf.ptr),
                static_cast<const float*>(wBuf.ptr),
                static_cast<float*>(gradInput.request().ptr),
                static_cast<float*>(gradWeight.request().ptr),
                static_cast<float*>(gradBias.request().ptr), p);

            py::dict result;
            result["grad_input"] = gradInput;
            result["grad_weight"] = gradWeight;
            result["grad_bias"] = gradBias;
            return result;
        },
        py::arg("device"), py::arg("grad_output"), py::arg("input"),
        py::arg("weights"),
        "GPU linear backward: grad_input, grad_weight, grad_bias");

    // ── Dropout ─────────────────────────────────────────────────────────
    m.def(
        "dropout",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input, py::array_t<float> random_mask,
           float p, bool training) -> py::array_t<float> {
            auto inBuf = input.request();
            uint32_t total = 1;
            for (int i = 0; i < inBuf.ndim; ++i)
                total *= static_cast<uint32_t>(inBuf.shape[i]);

            py::array_t<float> result(inBuf.shape);
            grilly::ops::dropout(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(inBuf.ptr),
                static_cast<const float*>(random_mask.request().ptr),
                static_cast<float*>(result.request().ptr),
                total, p, training);
            return result;
        },
        py::arg("device"), py::arg("input"), py::arg("random_mask"),
        py::arg("p") = 0.5f, py::arg("training") = true,
        "GPU dropout with inverted scaling");

    // ── Conv2d backward ─────────────────────────────────────────────────
    m.def(
        "conv2d_backward_input",
        [](GrillyCoreContext& ctx,
           py::array_t<float> grad_output, py::array_t<float> weight,
           std::vector<uint32_t> input_shape,
           std::vector<uint32_t> stride, std::vector<uint32_t> padding,
           std::vector<uint32_t> dilation,
           uint32_t groups) -> py::array_t<float> {
            auto gBuf = grad_output.request();
            auto wBuf = weight.request();

            uint32_t batchSize  = input_shape[0];
            uint32_t inChannels = input_shape[1];
            uint32_t inH        = input_shape[2];
            uint32_t inW        = input_shape[3];
            uint32_t outChannels = static_cast<uint32_t>(gBuf.shape[1]);
            uint32_t outH        = static_cast<uint32_t>(gBuf.shape[2]);
            uint32_t outW        = static_cast<uint32_t>(gBuf.shape[3]);
            uint32_t kH          = static_cast<uint32_t>(wBuf.shape[2]);
            uint32_t kW          = static_cast<uint32_t>(wBuf.shape[3]);

            uint32_t sH = stride.size() >= 1 ? stride[0] : 1;
            uint32_t sW = stride.size() >= 2 ? stride[1] : sH;
            uint32_t pH = padding.size() >= 1 ? padding[0] : 0;
            uint32_t pW = padding.size() >= 2 ? padding[1] : pH;
            uint32_t dH = dilation.size() >= 1 ? dilation[0] : 1;
            uint32_t dW = dilation.size() >= 2 ? dilation[1] : dH;

            grilly::ops::Conv2dBackwardInputParams p{
                batchSize, inChannels, inH, inW, outChannels, outH, outW,
                kH, kW, sH, sW, pH, pW, dH, dW, groups};

            py::array_t<float> result({
                static_cast<py::ssize_t>(batchSize),
                static_cast<py::ssize_t>(inChannels),
                static_cast<py::ssize_t>(inH),
                static_cast<py::ssize_t>(inW)});

            grilly::ops::conv2dBackwardInput(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(gBuf.ptr),
                static_cast<const float*>(wBuf.ptr),
                static_cast<float*>(result.request().ptr), p);

            return result;
        },
        py::arg("device"), py::arg("grad_output"), py::arg("weight"),
        py::arg("input_shape"),
        py::arg("stride") = std::vector<uint32_t>{1, 1},
        py::arg("padding") = std::vector<uint32_t>{0, 0},
        py::arg("dilation") = std::vector<uint32_t>{1, 1},
        py::arg("groups") = 1,
        "GPU Conv2d backward w.r.t. input");

    m.def(
        "conv2d_backward_weight",
        [](GrillyCoreContext& ctx,
           py::array_t<float> grad_output, py::array_t<float> input,
           std::vector<uint32_t> weight_shape,
           std::vector<uint32_t> stride, std::vector<uint32_t> padding,
           std::vector<uint32_t> dilation,
           uint32_t groups, bool has_bias) -> py::dict {
            auto gBuf = grad_output.request();
            auto iBuf = input.request();

            uint32_t batchSize  = static_cast<uint32_t>(iBuf.shape[0]);
            uint32_t inChannels = static_cast<uint32_t>(iBuf.shape[1]);
            uint32_t inH        = static_cast<uint32_t>(iBuf.shape[2]);
            uint32_t inW        = static_cast<uint32_t>(iBuf.shape[3]);
            uint32_t outChannels = weight_shape[0];
            uint32_t outH        = static_cast<uint32_t>(gBuf.shape[2]);
            uint32_t outW        = static_cast<uint32_t>(gBuf.shape[3]);
            uint32_t kH          = weight_shape[2];
            uint32_t kW          = weight_shape[3];

            uint32_t sH = stride.size() >= 1 ? stride[0] : 1;
            uint32_t sW = stride.size() >= 2 ? stride[1] : sH;
            uint32_t pH = padding.size() >= 1 ? padding[0] : 0;
            uint32_t pW = padding.size() >= 2 ? padding[1] : pH;
            uint32_t dH = dilation.size() >= 1 ? dilation[0] : 1;
            uint32_t dW = dilation.size() >= 2 ? dilation[1] : dH;

            grilly::ops::Conv2dBackwardWeightParams p{
                batchSize, inChannels, inH, inW, outChannels, outH, outW,
                kH, kW, sH, sW, pH, pW, dH, dW, groups,
                has_bias ? 1u : 0u};

            py::array_t<float> gradWeight({
                static_cast<py::ssize_t>(weight_shape[0]),
                static_cast<py::ssize_t>(weight_shape[1]),
                static_cast<py::ssize_t>(weight_shape[2]),
                static_cast<py::ssize_t>(weight_shape[3])});

            py::array_t<float> gradBias(
                has_bias ? std::vector<py::ssize_t>{
                    static_cast<py::ssize_t>(outChannels)}
                : std::vector<py::ssize_t>{1});

            grilly::ops::conv2dBackwardWeight(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(gBuf.ptr),
                static_cast<const float*>(iBuf.ptr),
                static_cast<float*>(gradWeight.request().ptr),
                has_bias ? static_cast<float*>(gradBias.request().ptr)
                         : nullptr,
                p);

            py::dict result;
            result["grad_weight"] = gradWeight;
            if (has_bias) result["grad_bias"] = gradBias;
            return result;
        },
        py::arg("device"), py::arg("grad_output"), py::arg("input"),
        py::arg("weight_shape"),
        py::arg("stride") = std::vector<uint32_t>{1, 1},
        py::arg("padding") = std::vector<uint32_t>{0, 0},
        py::arg("dilation") = std::vector<uint32_t>{1, 1},
        py::arg("groups") = 1, py::arg("has_bias") = false,
        "GPU Conv2d backward w.r.t. weight and bias");

    // ── Attention decomposed ops ────────────────────────────────────────
    m.def(
        "attention_scores",
        [](GrillyCoreContext& ctx,
           py::array_t<float> Q, py::array_t<float> K,
           float scale) -> py::array_t<float> {
            auto qBuf = Q.request();
            if (qBuf.ndim != 4)
                throw std::runtime_error("Q must be 4D (B, H, S, D)");

            uint32_t B = static_cast<uint32_t>(qBuf.shape[0]);
            uint32_t H = static_cast<uint32_t>(qBuf.shape[1]);
            uint32_t S = static_cast<uint32_t>(qBuf.shape[2]);
            uint32_t D = static_cast<uint32_t>(qBuf.shape[3]);

            if (scale == 0.0f) scale = 1.0f / std::sqrt(float(D));

            py::array_t<float> result({
                static_cast<py::ssize_t>(B), static_cast<py::ssize_t>(H),
                static_cast<py::ssize_t>(S), static_cast<py::ssize_t>(S)});

            grilly::ops::AttentionScoresParams p{B, S, H, D, scale, 0};
            grilly::ops::attentionScores(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(qBuf.ptr),
                static_cast<const float*>(K.request().ptr),
                static_cast<float*>(result.request().ptr), p);
            return result;
        },
        py::arg("device"), py::arg("Q"), py::arg("K"),
        py::arg("scale") = 0.0f,
        "GPU attention scores: Q @ K^T / sqrt(d_h)");

    m.def(
        "attention_mask",
        [](GrillyCoreContext& ctx,
           py::array_t<float> scores,
           std::optional<py::array_t<float>> mask,
           bool causal, float mask_value) -> py::array_t<float> {
            auto sBuf = scores.request();
            if (sBuf.ndim != 4)
                throw std::runtime_error("scores must be 4D (B, H, S, S)");

            uint32_t B = static_cast<uint32_t>(sBuf.shape[0]);
            uint32_t H = static_cast<uint32_t>(sBuf.shape[1]);
            uint32_t S = static_cast<uint32_t>(sBuf.shape[2]);

            py::array_t<float> result(sBuf.shape);
            std::memcpy(result.mutable_data(), sBuf.ptr,
                        sBuf.size * sizeof(float));

            const float* maskPtr = nullptr;
            if (mask.has_value())
                maskPtr = static_cast<const float*>(mask->request().ptr);

            grilly::ops::AttentionMaskParams p{
                B, H, S, causal ? 1u : 0u, mask_value};
            grilly::ops::attentionMask(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<float*>(result.mutable_data()), maskPtr, p);
            return result;
        },
        py::arg("device"), py::arg("scores"),
        py::arg("mask") = py::none(),
        py::arg("causal") = true, py::arg("mask_value") = -1e9f,
        "GPU attention mask (causal or custom)");

    m.def(
        "attention_output",
        [](GrillyCoreContext& ctx,
           py::array_t<float> weights, py::array_t<float> V) -> py::array_t<float> {
            auto wBuf = weights.request();
            auto vBuf = V.request();
            if (wBuf.ndim != 4 || vBuf.ndim != 4)
                throw std::runtime_error("weights and V must be 4D");

            uint32_t B = static_cast<uint32_t>(vBuf.shape[0]);
            uint32_t H = static_cast<uint32_t>(vBuf.shape[1]);
            uint32_t S = static_cast<uint32_t>(vBuf.shape[2]);
            uint32_t D = static_cast<uint32_t>(vBuf.shape[3]);

            py::array_t<float> result({
                static_cast<py::ssize_t>(B), static_cast<py::ssize_t>(H),
                static_cast<py::ssize_t>(S), static_cast<py::ssize_t>(D)});

            grilly::ops::AttentionOutputParams p{B, S, H, D};
            grilly::ops::attentionOutput(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(wBuf.ptr),
                static_cast<const float*>(vBuf.ptr),
                static_cast<float*>(result.request().ptr), p);
            return result;
        },
        py::arg("device"), py::arg("weights"), py::arg("V"),
        "GPU attention output: softmax(scores) @ V");

    m.def(
        "attention_concat_heads",
        [](GrillyCoreContext& ctx,
           py::array_t<float> mh_output) -> py::array_t<float> {
            auto inBuf = mh_output.request();
            if (inBuf.ndim != 4)
                throw std::runtime_error("input must be 4D (B, H, S, D)");

            uint32_t B = static_cast<uint32_t>(inBuf.shape[0]);
            uint32_t H = static_cast<uint32_t>(inBuf.shape[1]);
            uint32_t S = static_cast<uint32_t>(inBuf.shape[2]);
            uint32_t D = static_cast<uint32_t>(inBuf.shape[3]);

            py::array_t<float> result({
                static_cast<py::ssize_t>(B),
                static_cast<py::ssize_t>(S),
                static_cast<py::ssize_t>(H * D)});

            grilly::ops::ConcatHeadsParams p{B, S, H, D};
            grilly::ops::attentionConcatHeads(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(inBuf.ptr),
                static_cast<float*>(result.request().ptr), p);
            return result;
        },
        py::arg("device"), py::arg("mh_output"),
        "GPU concat multi-head attention output: (B,H,S,D) -> (B,S,H*D)");

    m.def(
        "rope",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input,
           std::optional<py::array_t<float>> cos_table,
           std::optional<py::array_t<float>> sin_table,
           float base, float scaling) -> py::array_t<float> {
            auto inBuf = input.request();
            if (inBuf.ndim != 4)
                throw std::runtime_error("input must be 4D (B, H, S, D)");

            uint32_t B = static_cast<uint32_t>(inBuf.shape[0]);
            uint32_t H = static_cast<uint32_t>(inBuf.shape[1]);
            uint32_t S = static_cast<uint32_t>(inBuf.shape[2]);
            uint32_t D = static_cast<uint32_t>(inBuf.shape[3]);

            bool precomputed = cos_table.has_value() && sin_table.has_value();
            const float* cosPtr = precomputed ?
                static_cast<const float*>(cos_table->request().ptr) : nullptr;
            const float* sinPtr = precomputed ?
                static_cast<const float*>(sin_table->request().ptr) : nullptr;

            py::array_t<float> result(inBuf.shape);
            grilly::ops::RoPEParams p{
                B, S, H, D, base, precomputed ? 1u : 0u, scaling};
            grilly::ops::applyRoPE(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(inBuf.ptr),
                static_cast<float*>(result.request().ptr),
                cosPtr, sinPtr, p);
            return result;
        },
        py::arg("device"), py::arg("input"),
        py::arg("cos_table") = py::none(), py::arg("sin_table") = py::none(),
        py::arg("base") = 10000.0f, py::arg("scaling") = 1.0f,
        "GPU Rotary Position Embeddings");

    // ── Pooling ops ─────────────────────────────────────────────────────
    m.def(
        "maxpool2d",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input,
           std::vector<uint32_t> kernel_size,
           std::vector<uint32_t> stride,
           std::vector<uint32_t> padding,
           std::vector<uint32_t> dilation) -> py::dict {
            auto inBuf = input.request();
            if (inBuf.ndim != 4)
                throw std::runtime_error("input must be 4D (B, C, H, W)");

            uint32_t B  = static_cast<uint32_t>(inBuf.shape[0]);
            uint32_t C  = static_cast<uint32_t>(inBuf.shape[1]);
            uint32_t iH = static_cast<uint32_t>(inBuf.shape[2]);
            uint32_t iW = static_cast<uint32_t>(inBuf.shape[3]);
            uint32_t kH = kernel_size[0];
            uint32_t kW = kernel_size.size() > 1 ? kernel_size[1] : kH;
            uint32_t sH = stride[0]; uint32_t sW = stride.size() > 1 ? stride[1] : sH;
            uint32_t pH = padding[0]; uint32_t pW = padding.size() > 1 ? padding[1] : pH;
            uint32_t dH = dilation[0]; uint32_t dW = dilation.size() > 1 ? dilation[1] : dH;

            uint32_t oH = grilly::ops::convOutputSize(iH, kH, sH, pH, dH);
            uint32_t oW = grilly::ops::convOutputSize(iW, kW, sW, pW, dW);

            py::array_t<float> result({
                static_cast<py::ssize_t>(B), static_cast<py::ssize_t>(C),
                static_cast<py::ssize_t>(oH), static_cast<py::ssize_t>(oW)});
            py::array_t<uint32_t> indices({
                static_cast<py::ssize_t>(B), static_cast<py::ssize_t>(C),
                static_cast<py::ssize_t>(oH), static_cast<py::ssize_t>(oW)});

            grilly::ops::MaxPool2dParams p{B, C, iH, iW, oH, oW,
                                           kH, kW, sH, sW, pH, pW, dH, dW};
            grilly::ops::maxpool2dForward(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(inBuf.ptr),
                static_cast<float*>(result.request().ptr),
                static_cast<uint32_t*>(indices.request().ptr), p);

            py::dict out;
            out["output"] = result;
            out["indices"] = indices;
            return out;
        },
        py::arg("device"), py::arg("input"),
        py::arg("kernel_size"), py::arg("stride") = std::vector<uint32_t>{2, 2},
        py::arg("padding") = std::vector<uint32_t>{0, 0},
        py::arg("dilation") = std::vector<uint32_t>{1, 1},
        "GPU MaxPool2d forward");

    m.def(
        "avgpool2d",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input,
           std::vector<uint32_t> kernel_size,
           std::vector<uint32_t> stride,
           std::vector<uint32_t> padding,
           bool count_include_pad) -> py::array_t<float> {
            auto inBuf = input.request();
            if (inBuf.ndim != 4)
                throw std::runtime_error("input must be 4D");

            uint32_t B  = static_cast<uint32_t>(inBuf.shape[0]);
            uint32_t C  = static_cast<uint32_t>(inBuf.shape[1]);
            uint32_t iH = static_cast<uint32_t>(inBuf.shape[2]);
            uint32_t iW = static_cast<uint32_t>(inBuf.shape[3]);
            uint32_t kH = kernel_size[0];
            uint32_t kW = kernel_size.size() > 1 ? kernel_size[1] : kH;
            uint32_t sH = stride[0]; uint32_t sW = stride.size() > 1 ? stride[1] : sH;
            uint32_t pH = padding[0]; uint32_t pW = padding.size() > 1 ? padding[1] : pH;

            uint32_t oH = grilly::ops::convOutputSize(iH, kH, sH, pH, 1);
            uint32_t oW = grilly::ops::convOutputSize(iW, kW, sW, pW, 1);

            py::array_t<float> result({
                static_cast<py::ssize_t>(B), static_cast<py::ssize_t>(C),
                static_cast<py::ssize_t>(oH), static_cast<py::ssize_t>(oW)});

            grilly::ops::AvgPool2dParams p{
                B, C, iH, iW, oH, oW, kH, kW, sH, sW, pH, pW,
                count_include_pad ? 1u : 0u};
            grilly::ops::avgpool2dForward(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(inBuf.ptr),
                static_cast<float*>(result.request().ptr), p);
            return result;
        },
        py::arg("device"), py::arg("input"),
        py::arg("kernel_size"), py::arg("stride") = std::vector<uint32_t>{2, 2},
        py::arg("padding") = std::vector<uint32_t>{0, 0},
        py::arg("count_include_pad") = true,
        "GPU AvgPool2d forward");

    // ── BatchNorm2d ─────────────────────────────────────────────────────
    m.def(
        "batchnorm2d_forward",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input,
           py::array_t<float> gamma, py::array_t<float> beta,
           py::array_t<float> running_mean, py::array_t<float> running_var,
           float eps, float momentum, bool training) -> py::dict {
            auto inBuf = input.request();
            if (inBuf.ndim != 4)
                throw std::runtime_error("input must be 4D (B, C, H, W)");

            uint32_t B = static_cast<uint32_t>(inBuf.shape[0]);
            uint32_t C = static_cast<uint32_t>(inBuf.shape[1]);
            uint32_t H = static_cast<uint32_t>(inBuf.shape[2]);
            uint32_t W = static_cast<uint32_t>(inBuf.shape[3]);

            py::array_t<float> output(inBuf.shape);
            py::array_t<float> rmOut(running_mean.request().shape);
            py::array_t<float> rvOut(running_var.request().shape);
            py::array_t<float> bMean({static_cast<py::ssize_t>(C)});
            py::array_t<float> bVar({static_cast<py::ssize_t>(C)});

            std::memcpy(rmOut.mutable_data(), running_mean.data(),
                        C * sizeof(float));
            std::memcpy(rvOut.mutable_data(), running_var.data(),
                        C * sizeof(float));

            grilly::ops::BatchNorm2dForwardParams p{
                B, C, H, W, eps, momentum, training ? 1u : 0u, 1u};
            grilly::ops::batchnorm2dForward(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(inBuf.ptr),
                static_cast<float*>(output.request().ptr),
                static_cast<const float*>(gamma.request().ptr),
                static_cast<const float*>(beta.request().ptr),
                static_cast<float*>(rmOut.mutable_data()),
                static_cast<float*>(rvOut.mutable_data()),
                static_cast<float*>(bMean.mutable_data()),
                static_cast<float*>(bVar.mutable_data()), p);

            py::dict result;
            result["output"] = output;
            result["running_mean"] = rmOut;
            result["running_var"] = rvOut;
            result["batch_mean"] = bMean;
            result["batch_var"] = bVar;
            return result;
        },
        py::arg("device"), py::arg("input"),
        py::arg("gamma"), py::arg("beta"),
        py::arg("running_mean"), py::arg("running_var"),
        py::arg("eps") = 1e-5f, py::arg("momentum") = 0.1f,
        py::arg("training") = true,
        "GPU BatchNorm2d forward");

    // ── Cross-entropy loss ──────────────────────────────────────────────
    m.def(
        "cross_entropy_loss",
        [](GrillyCoreContext& ctx,
           py::array_t<float> logits, py::array_t<uint32_t> targets,
           float label_smoothing) -> py::array_t<float> {
            auto lBuf = logits.request();

            uint32_t batchSize, seqLen, vocabSize;
            if (lBuf.ndim == 2) {
                batchSize = static_cast<uint32_t>(lBuf.shape[0]);
                seqLen = 1;
                vocabSize = static_cast<uint32_t>(lBuf.shape[1]);
            } else if (lBuf.ndim == 3) {
                batchSize = static_cast<uint32_t>(lBuf.shape[0]);
                seqLen = static_cast<uint32_t>(lBuf.shape[1]);
                vocabSize = static_cast<uint32_t>(lBuf.shape[2]);
            } else {
                throw std::runtime_error("logits must be 2D or 3D");
            }

            uint32_t totalPos = batchSize * seqLen;
            py::array_t<float> losses(totalPos);

            grilly::ops::CrossEntropyParams p{
                batchSize, seqLen, vocabSize, 0, label_smoothing};
            grilly::ops::crossEntropyLoss(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(lBuf.ptr),
                static_cast<const uint32_t*>(targets.request().ptr),
                static_cast<float*>(losses.request().ptr), p);
            return losses;
        },
        py::arg("device"), py::arg("logits"), py::arg("targets"),
        py::arg("label_smoothing") = 0.0f,
        "GPU cross-entropy loss forward");

    m.def(
        "cross_entropy_backward",
        [](GrillyCoreContext& ctx,
           py::array_t<float> logits,
           py::array_t<uint32_t> targets) -> py::array_t<float> {
            auto lBuf = logits.request();
            uint32_t batchSize = static_cast<uint32_t>(lBuf.shape[0]);
            uint32_t numClasses = static_cast<uint32_t>(lBuf.shape[1]);

            py::array_t<float> gradLogits(lBuf.shape);

            grilly::ops::CrossEntropyBackwardParams p{batchSize, numClasses};
            grilly::ops::crossEntropyBackward(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(lBuf.ptr),
                static_cast<const uint32_t*>(targets.request().ptr),
                static_cast<float*>(gradLogits.request().ptr), p);
            return gradLogits;
        },
        py::arg("device"), py::arg("logits"), py::arg("targets"),
        "GPU cross-entropy backward");

    // ── Optimizer GPU ops ───────────────────────────────────────────────
    m.def(
        "adam_update",
        [](GrillyCoreContext& ctx,
           py::array_t<float> weights, py::array_t<float> grad,
           py::array_t<float> m_state, py::array_t<float> v_state,
           float lr, float beta1, float beta2, float eps,
           float beta1_t, float beta2_t, bool clear_grad) -> py::dict {
            auto wBuf = weights.request();
            uint32_t total = 1;
            for (int i = 0; i < wBuf.ndim; ++i)
                total *= static_cast<uint32_t>(wBuf.shape[i]);

            py::array_t<float> wOut(wBuf.shape);
            py::array_t<float> gOut(wBuf.shape);
            py::array_t<float> mOut(wBuf.shape);
            py::array_t<float> vOut(wBuf.shape);

            std::memcpy(wOut.mutable_data(), wBuf.ptr, total * sizeof(float));
            std::memcpy(gOut.mutable_data(), grad.data(), total * sizeof(float));
            std::memcpy(mOut.mutable_data(), m_state.data(), total * sizeof(float));
            std::memcpy(vOut.mutable_data(), v_state.data(), total * sizeof(float));

            grilly::ops::AdamParams p{total, lr, beta1, beta2, eps,
                                      beta1_t, beta2_t, clear_grad ? 1u : 0u};
            grilly::ops::adamUpdate(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<float*>(wOut.mutable_data()),
                static_cast<float*>(gOut.mutable_data()),
                static_cast<float*>(mOut.mutable_data()),
                static_cast<float*>(vOut.mutable_data()), p);

            py::dict result;
            result["weights"] = wOut;
            result["grad"] = gOut;
            result["m"] = mOut;
            result["v"] = vOut;
            return result;
        },
        py::arg("device"), py::arg("weights"), py::arg("grad"),
        py::arg("m"), py::arg("v"),
        py::arg("lr") = 1e-3f, py::arg("beta1") = 0.9f,
        py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f,
        py::arg("beta1_t") = 0.9f, py::arg("beta2_t") = 0.999f,
        py::arg("clear_grad") = false,
        "GPU Adam optimizer step");

    m.def(
        "adamw_update",
        [](GrillyCoreContext& ctx,
           py::array_t<float> weights, py::array_t<float> grad,
           py::array_t<float> m_state, py::array_t<float> v_state,
           float lr, float beta1, float beta2, float eps,
           float weight_decay,
           float beta1_t, float beta2_t, bool clear_grad) -> py::dict {
            auto wBuf = weights.request();
            uint32_t total = 1;
            for (int i = 0; i < wBuf.ndim; ++i)
                total *= static_cast<uint32_t>(wBuf.shape[i]);

            py::array_t<float> wOut(wBuf.shape);
            py::array_t<float> gOut(wBuf.shape);
            py::array_t<float> mOut(wBuf.shape);
            py::array_t<float> vOut(wBuf.shape);

            std::memcpy(wOut.mutable_data(), wBuf.ptr, total * sizeof(float));
            std::memcpy(gOut.mutable_data(), grad.data(), total * sizeof(float));
            std::memcpy(mOut.mutable_data(), m_state.data(), total * sizeof(float));
            std::memcpy(vOut.mutable_data(), v_state.data(), total * sizeof(float));

            grilly::ops::AdamWParams p{total, lr, beta1, beta2, eps,
                                       weight_decay, beta1_t, beta2_t,
                                       clear_grad ? 1u : 0u};
            grilly::ops::adamwUpdate(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<float*>(wOut.mutable_data()),
                static_cast<float*>(gOut.mutable_data()),
                static_cast<float*>(mOut.mutable_data()),
                static_cast<float*>(vOut.mutable_data()), p);

            py::dict result;
            result["weights"] = wOut;
            result["grad"] = gOut;
            result["m"] = mOut;
            result["v"] = vOut;
            return result;
        },
        py::arg("device"), py::arg("weights"), py::arg("grad"),
        py::arg("m"), py::arg("v"),
        py::arg("lr") = 1e-3f, py::arg("beta1") = 0.9f,
        py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f,
        py::arg("weight_decay") = 0.01f,
        py::arg("beta1_t") = 0.9f, py::arg("beta2_t") = 0.999f,
        py::arg("clear_grad") = false,
        "GPU AdamW optimizer step");

    // ── Embedding lookup ────────────────────────────────────────────────
    m.def(
        "embedding_lookup",
        [](GrillyCoreContext& ctx,
           py::array_t<uint32_t> token_ids,
           py::array_t<float> embeddings) -> py::array_t<float> {
            auto idBuf = token_ids.request();
            auto eBuf = embeddings.request();

            uint32_t batchSize = 1, seqLen;
            if (idBuf.ndim == 1) {
                seqLen = static_cast<uint32_t>(idBuf.shape[0]);
            } else {
                batchSize = static_cast<uint32_t>(idBuf.shape[0]);
                seqLen = static_cast<uint32_t>(idBuf.shape[1]);
            }
            uint32_t vocabSize = static_cast<uint32_t>(eBuf.shape[0]);
            uint32_t embDim = static_cast<uint32_t>(eBuf.shape[1]);

            py::array_t<float> result({
                static_cast<py::ssize_t>(batchSize),
                static_cast<py::ssize_t>(seqLen),
                static_cast<py::ssize_t>(embDim)});

            grilly::ops::EmbeddingParams p{batchSize, seqLen, vocabSize, embDim};
            grilly::ops::embeddingLookup(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const uint32_t*>(idBuf.ptr),
                static_cast<const float*>(eBuf.ptr),
                static_cast<float*>(result.request().ptr), p);

            if (idBuf.ndim == 1)
                result = result.reshape({
                    static_cast<py::ssize_t>(seqLen),
                    static_cast<py::ssize_t>(embDim)});
            return result;
        },
        py::arg("device"), py::arg("token_ids"), py::arg("embeddings"),
        "GPU embedding table lookup");

    // ── Mean pooling ────────────────────────────────────────────────────
    m.def(
        "mean_pool",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input) -> py::array_t<float> {
            auto inBuf = input.request();
            if (inBuf.ndim != 3)
                throw std::runtime_error("input must be 3D (B, S, D)");

            uint32_t B = static_cast<uint32_t>(inBuf.shape[0]);
            uint32_t S = static_cast<uint32_t>(inBuf.shape[1]);
            uint32_t D = static_cast<uint32_t>(inBuf.shape[2]);

            py::array_t<float> result({
                static_cast<py::ssize_t>(B), static_cast<py::ssize_t>(D)});

            grilly::ops::MeanPoolParams p{B, S, D};
            grilly::ops::meanPool(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(inBuf.ptr),
                static_cast<float*>(result.request().ptr), p);
            return result;
        },
        py::arg("device"), py::arg("input"),
        "GPU mean pooling over sequence dim: (B,S,D) -> (B,D)");
}
