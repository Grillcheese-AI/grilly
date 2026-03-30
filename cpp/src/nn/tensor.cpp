#include "grilly/nn/tensor.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <numeric>
#include <stdexcept>

namespace grilly {
namespace nn {

// ── Global default backend ───────────────────────────────────────────
// FIX #4: Thread-safe initialization via std::call_once.

static ComputeBackend* g_default_backend = nullptr;
static std::once_flag g_backend_init_flag;

ComputeBackend* default_backend() {
    std::call_once(g_backend_init_flag, [] {
        try {
            static auto backend = createBackend("vulkan");
            g_default_backend = backend.get();

            const char* shader_dirs[] = {
                "shaders/spv",
                "../shaders/spv",
                "shaders",
            };
            for (const char* dir : shader_dirs) {
                try {
                    g_default_backend->loadShaderDir(dir);
                    break;
                } catch (...) {}
            }
        } catch (...) {
            // Vulkan not available — fall back to CPU-only
        }
    });
    return g_default_backend;
}

void set_default_backend(ComputeBackend* backend) {
    g_default_backend = backend;
}

// ── Helper: ensure storage exists ────────────────────────────────────

void Tensor::ensure_storage() {
    if (!storage_) {
        storage_ = std::make_shared<TensorStorage>();
    }
}

// ── Constructors ──────────────────────────────────────────────────────

Tensor::Tensor(std::vector<int64_t> shape, DType dtype,
               ComputeBackend* backend)
    : storage_(std::make_shared<TensorStorage>()),
      shape_(std::move(shape)) {
    storage_->dtype = dtype;
    storage_->backend = backend;
    storage_->cpu_valid = true;
    storage_->gpu_valid = false;
    size_t n = static_cast<size_t>(numel());
    storage_->cpu_data.resize(n, 0.0f);
}

Tensor::Tensor(std::vector<float> data, std::vector<int64_t> shape,
               ComputeBackend* backend)
    : storage_(std::make_shared<TensorStorage>()),
      shape_(std::move(shape)) {
    storage_->dtype = DType::Float32;
    storage_->backend = backend;
    storage_->cpu_data = std::move(data);
    storage_->cpu_valid = true;
    storage_->gpu_valid = false;
    if (static_cast<int64_t>(storage_->cpu_data.size()) != numel()) {
        throw std::invalid_argument(
            "Tensor: data size (" + std::to_string(storage_->cpu_data.size()) +
            ") does not match shape (" + std::to_string(numel()) + ")");
    }
}

// ── Factory methods ───────────────────────────────────────────────────

Tensor Tensor::from_numpy(py::array_t<float> arr, ComputeBackend* backend) {
    if (!backend) backend = default_backend();

    auto buf = arr.request();
    std::vector<int64_t> shape(buf.ndim);
    for (int i = 0; i < buf.ndim; i++) {
        shape[i] = static_cast<int64_t>(buf.shape[i]);
    }

    size_t count = static_cast<size_t>(buf.size);
    std::vector<float> data(count);
    std::memcpy(data.data(), buf.ptr, count * sizeof(float));

    return Tensor(std::move(data), std::move(shape), backend);
}

Tensor Tensor::zeros(std::vector<int64_t> shape, ComputeBackend* backend) {
    if (!backend) backend = default_backend();
    return Tensor(std::move(shape), DType::Float32, backend);
}

Tensor Tensor::empty(std::vector<int64_t> shape, ComputeBackend* backend) {
    if (!backend) backend = default_backend();
    Tensor t;
    t.storage_ = std::make_shared<TensorStorage>();
    t.shape_ = std::move(shape);
    t.storage_->dtype = DType::Float32;
    t.storage_->backend = backend;
    t.storage_->cpu_data.resize(static_cast<size_t>(t.numel()));
    t.storage_->cpu_valid = true;
    t.storage_->gpu_valid = false;
    return t;
}

Tensor Tensor::from_gpu(uint64_t handle, std::vector<int64_t> shape,
                        ComputeBackend* backend) {
    Tensor t;
    t.storage_ = std::make_shared<TensorStorage>();
    t.storage_->buffer_handle = handle;
    t.storage_->backend = backend;
    t.storage_->gpu_valid = true;
    t.storage_->cpu_valid = false;
    t.shape_ = std::move(shape);
    return t;
}

// ── Data access ───────────────────────────────────────────────────────

py::array_t<float> Tensor::numpy() const {
    ensure_cpu();

    std::vector<py::ssize_t> py_shape(shape_.begin(), shape_.end());
    py::array_t<float> arr(py_shape);
    auto buf = arr.request();
    std::memcpy(buf.ptr, storage_->cpu_data.data(),
                storage_->cpu_data.size() * sizeof(float));
    return arr;
}

const float* Tensor::data() const {
    ensure_cpu();
    return storage_->cpu_data.data();
}

float* Tensor::mutable_data() {
    ensure_cpu();
    storage_->gpu_valid = false;  // CPU will be authoritative after mutation
    return storage_->cpu_data.data();
}

uint64_t Tensor::gpu_handle() {
    ensure_gpu();
    return storage_->buffer_handle;
}

uint64_t Tensor::gpu_handle_if_valid() const {
    return (storage_ && storage_->gpu_valid) ? storage_->buffer_handle : 0;
}

void Tensor::mark_gpu_modified() {
    ensure_storage();
    storage_->gpu_valid = true;
    storage_->cpu_valid = false;
}

void Tensor::mark_cpu_modified() {
    ensure_storage();
    storage_->cpu_valid = true;
    storage_->gpu_valid = false;
}

// ── Shape operations ──────────────────────────────────────────────────

int64_t Tensor::shape(int dim) const {
    int ndims = static_cast<int>(shape_.size());
    if (dim < 0) dim += ndims;
    if (dim < 0 || dim >= ndims) {
        throw std::out_of_range("Tensor::shape: dim " + std::to_string(dim) +
                                " out of range for " + std::to_string(ndims) +
                                "-D tensor");
    }
    return shape_[dim];
}

int64_t Tensor::numel() const {
    if (shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(), int64_t(1),
                           std::multiplies<int64_t>());
}

size_t Tensor::nbytes() const {
    DType dt = storage_ ? storage_->dtype : DType::Float32;
    return static_cast<size_t>(numel()) * dtype_size(dt);
}

Tensor Tensor::reshape(std::vector<int64_t> new_shape) const {
    // Resolve -1 dimension
    int64_t total = numel();
    int neg_idx = -1;
    int64_t known_product = 1;
    for (size_t i = 0; i < new_shape.size(); i++) {
        if (new_shape[i] == -1) {
            if (neg_idx >= 0) {
                throw std::invalid_argument("Tensor::reshape: only one -1 allowed");
            }
            neg_idx = static_cast<int>(i);
        } else {
            known_product *= new_shape[i];
        }
    }
    if (neg_idx >= 0) {
        if (known_product == 0) {
            throw std::invalid_argument(
                "Tensor::reshape: cannot infer dim with zero-size dims");
        }
        new_shape[neg_idx] = total / known_product;
    }

    // Validate total elements match
    int64_t new_total = std::accumulate(new_shape.begin(), new_shape.end(),
                                        int64_t(1), std::multiplies<int64_t>());
    if (new_total != total) {
        throw std::invalid_argument(
            "Tensor::reshape: cannot reshape " + std::to_string(total) +
            " elements into shape with " + std::to_string(new_total) + " elements");
    }

    // FIX #1: Share TensorStorage via shared_ptr — O(1), no memcpy,
    // no dangling VkBuffer (ref-counted destructor handles cleanup).
    Tensor t;
    t.storage_ = storage_;  // shared_ptr copy — O(1)
    t.shape_ = std::move(new_shape);
    t.requires_grad_ = requires_grad_;
    t.grad_ = grad_;
    return t;
}

Tensor Tensor::view(std::vector<int64_t> new_shape) const {
    return reshape(std::move(new_shape));
}

// ── GPU lifecycle ─────────────────────────────────────────────────────

void Tensor::ensure_gpu() {
    ensure_storage();
    if (storage_->gpu_valid) return;

    if (!storage_->backend) {
        throw std::runtime_error(
            "Tensor::ensure_gpu: no ComputeBackend attached");
    }

    size_t bytes = nbytes();

    // Allocate GPU buffer if needed
    if (storage_->buffer_handle == 0) {
        BufferDesc desc;
        desc.size = bytes;
        desc.usage = BufferDesc::DeviceLocal;
        storage_->buffer_handle = storage_->backend->createBuffer(desc);
    }

    // Upload CPU data
    if (storage_->cpu_valid && !storage_->cpu_data.empty()) {
        storage_->backend->upload(storage_->buffer_handle,
                                  storage_->cpu_data.data(), bytes);
    }

    storage_->gpu_valid = true;
}

void Tensor::ensure_cpu() const {
    if (!storage_) {
        throw std::runtime_error(
            "Tensor::ensure_cpu: no storage allocated");
    }
    if (storage_->cpu_valid) return;

    if (!storage_->gpu_valid) {
        throw std::runtime_error(
            "Tensor::ensure_cpu: neither CPU nor GPU data is valid");
    }

    if (!storage_->backend) {
        throw std::runtime_error(
            "Tensor::ensure_cpu: no ComputeBackend for GPU download");
    }

    size_t count = static_cast<size_t>(numel());
    storage_->cpu_data.resize(count);
    storage_->backend->download(storage_->buffer_handle,
                                storage_->cpu_data.data(),
                                count * sizeof(float));
    storage_->cpu_valid = true;
}

void Tensor::release_gpu() {
    if (!storage_) return;

    if (storage_->buffer_handle != 0 && storage_->backend) {
        // Ensure CPU has a copy before releasing
        if (storage_->gpu_valid && !storage_->cpu_valid) {
            ensure_cpu();
        }
        storage_->backend->destroyBuffer(storage_->buffer_handle);
        storage_->buffer_handle = 0;
    }
    storage_->gpu_valid = false;
}

}  // namespace nn
}  // namespace grilly
