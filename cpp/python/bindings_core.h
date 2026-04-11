#pragma once
/// Shared header for split pybind11 binding files.
///
/// Provides common includes, the GrillyCoreContext struct, helper functions,
/// and forward declarations for all register_*_ops() functions that each
/// split file implements.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "grilly/compute_backend.h"
#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/device.h"
#include "grilly/pipeline_cache.h"

// ── NN framework ──
#include "grilly/nn/tensor.h"
#include "grilly/nn/parameter.h"
#include "grilly/nn/module.h"

namespace py = pybind11;

// ═══════════════════════════════════════════════════════════════════════════
// GrillyCoreContext — holds all Vulkan state so Python sees one object.
// Internally owns GrillyDevice -> BufferPool -> PipelineCache -> CommandBatch.
// ═══════════════════════════════════════════════════════════════════════════

struct GrillyCoreContext {
    grilly::GrillyDevice device;
    grilly::BufferPool pool;
    grilly::PipelineCache cache;
    grilly::CommandBatch batch;
    std::mutex ctx_mutex;  // Protects Vulkan command recording (not thread-safe)
    bool shadersLoaded = false;

    GrillyCoreContext()
        : device(), pool(device), cache(device), batch(device) {}

    /// Wait for all submitted GPU work to complete.
    /// Call between inference steps to prevent race conditions on
    /// descriptor sets and buffer overwrites.
    void waitIdle() {
        vkQueueWaitIdle(device.computeQueue());
    }

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

// ═══════════════════════════════════════════════════════════════════════════
// Helper utilities shared across binding files
// ═══════════════════════════════════════════════════════════════════════════

/// Require NumPy C-contiguous float32 (kernels assume dense row-major layout).
inline void require_c_contiguous_float(const py::buffer_info& buf) {
    if (buf.itemsize != sizeof(float))
        throw std::runtime_error("expected float32 array");
    if (buf.ndim == 0)
        return;
    py::ssize_t expected_stride = static_cast<py::ssize_t>(sizeof(float));
    for (int i = static_cast<int>(buf.ndim) - 1; i >= 0; --i) {
        if (buf.strides[i] != expected_stride)
            throw std::runtime_error("array must be C-contiguous float32");
        expected_stride *= buf.shape[i];
    }
}

/// Require NumPy C-contiguous int8 (e.g. VSA / Hamming vectors).
inline void require_c_contiguous_int8(const py::buffer_info& buf) {
    if (buf.itemsize != sizeof(int8_t))
        throw std::runtime_error("expected int8 array");
    if (buf.ndim == 0)
        return;
    py::ssize_t expected_stride = static_cast<py::ssize_t>(sizeof(int8_t));
    for (int i = static_cast<int>(buf.ndim) - 1; i >= 0; --i) {
        if (buf.strides[i] != expected_stride)
            throw std::runtime_error("array must be C-contiguous int8");
        expected_stride *= buf.shape[i];
    }
}

/// Require NumPy C-contiguous uint32 (e.g. CE targets).
inline void require_c_contiguous_int32(const py::buffer_info& buf) {
    if (buf.itemsize != sizeof(int32_t))
        throw std::runtime_error("expected int32 array");
    if (buf.ndim == 0)
        return;
    py::ssize_t expected_stride = static_cast<py::ssize_t>(sizeof(int32_t));
    for (int i = static_cast<int>(buf.ndim) - 1; i >= 0; --i) {
        if (buf.strides[i] != expected_stride)
            throw std::runtime_error("array must be C-contiguous int32");
        expected_stride *= buf.shape[i];
    }
}

inline void require_c_contiguous_uint32(const py::buffer_info& buf) {
    if (buf.itemsize != sizeof(uint32_t))
        throw std::runtime_error("expected uint32 array");
    if (buf.ndim == 0)
        return;
    py::ssize_t expected_stride = static_cast<py::ssize_t>(sizeof(uint32_t));
    for (int i = static_cast<int>(buf.ndim) - 1; i >= 0; --i) {
        if (buf.strides[i] != expected_stride)
            throw std::runtime_error("array must be C-contiguous uint32");
        expected_stride *= buf.shape[i];
    }
}

/// Extract flat batch*seq and last-dim from a numpy buffer_info.
inline std::pair<uint32_t, uint32_t> extractBatchAndLastDim(
    const py::buffer_info& buf) {
    uint32_t lastDim = static_cast<uint32_t>(buf.shape[buf.ndim - 1]);
    uint32_t batch = 1;
    for (int i = 0; i < buf.ndim - 1; ++i)
        batch *= static_cast<uint32_t>(buf.shape[i]);
    if (buf.ndim == 1) batch = 1;
    return {batch, lastDim};
}

/// Convert a numpy array to a grilly::nn::Tensor (CPU copy).
inline grilly::nn::Tensor to_tensor(py::array_t<float> arr,
                                    grilly::ComputeBackend* backend = nullptr) {
    return grilly::nn::Tensor::from_numpy(arr, backend);
}

// ═══════════════════════════════════════════════════════════════════════════
// Forward declarations for split binding registration functions.
// Each bindings_*.cpp implements one of these.
// ═══════════════════════════════════════════════════════════════════════════

void register_linear_ops(py::module_& m);
void register_activations_ops(py::module_& m);
void register_conv_ops(py::module_& m);
void register_attention_ops(py::module_& m);
void register_normalization_ops(py::module_& m);
void register_optim_ops(py::module_& m);
void register_loss_ops(py::module_& m);
void register_snn_ops(py::module_& m);
void register_pooling_ops(py::module_& m);
void register_misc_ops(py::module_& m);
void register_perceiver_ops(py::module_& m);
void register_moqe_train_ops(py::module_& m);
void register_moe_ops(py::module_& m);
void register_fusion_ops(py::module_& m);
void register_vsa_lm_ops(py::module_& m);
void register_grl_ops(py::module_& m);
void register_prefix_scan_ops(py::module_& m);
void register_distillation_ops(py::module_& m);
