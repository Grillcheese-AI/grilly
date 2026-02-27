#pragma once
/// GPU-backed tensor with lazy CPU↔GPU synchronization.
///
/// Replaces the numpy ↔ GPU round-trips in the Python backend with a
/// shape-aware wrapper around ComputeBackend buffer handles. Data stays
/// on whichever device last modified it; transfers happen lazily when
/// the other side is accessed.
///
/// Dual-validity model:
///   cpu_valid_  = true  → cpu_data_ matches the authoritative copy
///   gpu_valid_  = true  → buffer_handle_ matches the authoritative copy
///   Both true           → CPU and GPU are in sync
///   Both false          → impossible (invariant violation)

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "grilly/compute_backend.h"

namespace py = pybind11;

namespace grilly {
namespace nn {

enum class DType : uint8_t {
    Float32 = 0,
    Float16 = 1,
    Int32 = 2,
    Int64 = 3,
};

/// Returns the byte size of a single element of the given dtype.
inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::Float32: return 4;
        case DType::Float16: return 2;
        case DType::Int32:   return 4;
        case DType::Int64:   return 8;
    }
    return 4;
}

class Tensor {
public:
    Tensor() = default;

    /// Construct from shape, optionally with a backend for GPU ops.
    Tensor(std::vector<int64_t> shape, DType dtype = DType::Float32,
           ComputeBackend* backend = nullptr);

    /// Construct from existing CPU data + shape.
    Tensor(std::vector<float> data, std::vector<int64_t> shape,
           ComputeBackend* backend = nullptr);

    // -- Factory methods --

    /// Create from a numpy array (zero-copy when contiguous).
    static Tensor from_numpy(py::array_t<float> arr,
                             ComputeBackend* backend = nullptr);

    /// Create a zero-filled tensor.
    static Tensor zeros(std::vector<int64_t> shape,
                        ComputeBackend* backend = nullptr);

    /// Create an uninitialized tensor (CPU data allocated but not zeroed).
    static Tensor empty(std::vector<int64_t> shape,
                        ComputeBackend* backend = nullptr);

    /// Create from an existing GPU buffer handle (takes ownership of sync state).
    static Tensor from_gpu(uint64_t handle, std::vector<int64_t> shape,
                           ComputeBackend* backend);

    // -- Data access (lazy sync) --

    /// Download to CPU if needed, return as numpy array.
    py::array_t<float> numpy() const;

    /// Get raw CPU data pointer. Ensures CPU is valid first.
    const float* data() const;
    float* mutable_data();

    /// Upload to GPU if needed, return buffer handle.
    uint64_t gpu_handle();

    /// Get GPU handle without triggering upload (0 if not on GPU).
    uint64_t gpu_handle_if_valid() const;

    /// Mark that GPU data was modified externally (invalidates CPU copy).
    void mark_gpu_modified();

    /// Mark that CPU data was modified externally (invalidates GPU copy).
    void mark_cpu_modified();

    // -- Shape operations --

    const std::vector<int64_t>& shape() const { return shape_; }
    int64_t shape(int dim) const;
    size_t ndim() const { return shape_.size(); }
    DType dtype() const { return dtype_; }

    /// Total number of elements.
    int64_t numel() const;

    /// Total size in bytes.
    size_t nbytes() const;

    /// Reshape (no data copy, shares underlying storage).
    Tensor reshape(std::vector<int64_t> new_shape) const;

    /// View (alias for reshape, no data copy).
    Tensor view(std::vector<int64_t> new_shape) const;

    // -- GPU lifecycle --

    /// Ensure data is on GPU (upload if needed).
    void ensure_gpu();

    /// Release GPU buffer, keeping CPU data.
    void release_gpu();

    /// Check if tensor has a valid GPU buffer.
    bool on_gpu() const { return gpu_valid_; }

    /// Check if tensor has valid CPU data.
    bool on_cpu() const { return cpu_valid_; }

    /// Whether this tensor has any data at all.
    bool valid() const { return cpu_valid_ || gpu_valid_; }

    // -- Autograd --

    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool rg) { requires_grad_ = rg; }

    std::shared_ptr<Tensor> grad() const { return grad_; }
    void set_grad(std::shared_ptr<Tensor> g) { grad_ = std::move(g); }

    // -- Backend --

    ComputeBackend* backend() const { return backend_; }
    void set_backend(ComputeBackend* b) { backend_ = b; }

private:
    void ensure_cpu() const;

    ComputeBackend* backend_ = nullptr;  // Non-owning
    uint64_t buffer_handle_ = 0;         // GPU buffer (0 = CPU-only)
    std::vector<int64_t> shape_;
    DType dtype_ = DType::Float32;

    // Lazy sync state
    mutable std::vector<float> cpu_data_;
    mutable bool gpu_valid_ = false;
    mutable bool cpu_valid_ = true;

    // Autograd
    bool requires_grad_ = false;
    std::shared_ptr<Tensor> grad_;
};

}  // namespace nn
}  // namespace grilly
