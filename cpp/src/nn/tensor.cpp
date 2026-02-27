#include "grilly/nn/tensor.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace grilly {
namespace nn {

// ── Constructors ──────────────────────────────────────────────────────

Tensor::Tensor(std::vector<int64_t> shape, DType dtype,
               ComputeBackend* backend)
    : backend_(backend),
      shape_(std::move(shape)),
      dtype_(dtype),
      cpu_valid_(true),
      gpu_valid_(false) {
    size_t n = static_cast<size_t>(numel());
    cpu_data_.resize(n, 0.0f);
}

Tensor::Tensor(std::vector<float> data, std::vector<int64_t> shape,
               ComputeBackend* backend)
    : backend_(backend),
      shape_(std::move(shape)),
      dtype_(DType::Float32),
      cpu_data_(std::move(data)),
      cpu_valid_(true),
      gpu_valid_(false) {
    if (static_cast<int64_t>(cpu_data_.size()) != numel()) {
        throw std::invalid_argument(
            "Tensor: data size (" + std::to_string(cpu_data_.size()) +
            ") does not match shape (" + std::to_string(numel()) + ")");
    }
}

// ── Factory methods ───────────────────────────────────────────────────

Tensor Tensor::from_numpy(py::array_t<float> arr, ComputeBackend* backend) {
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
    return Tensor(std::move(shape), DType::Float32, backend);
}

Tensor Tensor::empty(std::vector<int64_t> shape, ComputeBackend* backend) {
    Tensor t;
    t.shape_ = std::move(shape);
    t.dtype_ = DType::Float32;
    t.backend_ = backend;
    t.cpu_data_.resize(static_cast<size_t>(t.numel()));
    t.cpu_valid_ = true;
    t.gpu_valid_ = false;
    return t;
}

Tensor Tensor::from_gpu(uint64_t handle, std::vector<int64_t> shape,
                        ComputeBackend* backend) {
    Tensor t;
    t.buffer_handle_ = handle;
    t.shape_ = std::move(shape);
    t.dtype_ = DType::Float32;
    t.backend_ = backend;
    t.gpu_valid_ = true;
    t.cpu_valid_ = false;
    return t;
}

// ── Data access ───────────────────────────────────────────────────────

py::array_t<float> Tensor::numpy() const {
    ensure_cpu();

    std::vector<py::ssize_t> py_shape(shape_.begin(), shape_.end());
    py::array_t<float> arr(py_shape);
    auto buf = arr.request();
    std::memcpy(buf.ptr, cpu_data_.data(), cpu_data_.size() * sizeof(float));
    return arr;
}

const float* Tensor::data() const {
    ensure_cpu();
    return cpu_data_.data();
}

float* Tensor::mutable_data() {
    ensure_cpu();
    gpu_valid_ = false;  // CPU will be authoritative after mutation
    return cpu_data_.data();
}

uint64_t Tensor::gpu_handle() {
    ensure_gpu();
    return buffer_handle_;
}

uint64_t Tensor::gpu_handle_if_valid() const {
    return gpu_valid_ ? buffer_handle_ : 0;
}

void Tensor::mark_gpu_modified() {
    gpu_valid_ = true;
    cpu_valid_ = false;
}

void Tensor::mark_cpu_modified() {
    cpu_valid_ = true;
    gpu_valid_ = false;
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
    return static_cast<size_t>(numel()) * dtype_size(dtype_);
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
            throw std::invalid_argument("Tensor::reshape: cannot infer dim with zero-size dims");
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

    // Share data — this is a view, not a copy
    Tensor t;
    t.backend_ = backend_;
    t.buffer_handle_ = buffer_handle_;
    t.shape_ = std::move(new_shape);
    t.dtype_ = dtype_;
    t.cpu_data_ = cpu_data_;  // Shared copy (could use shared_ptr for zero-copy)
    t.gpu_valid_ = gpu_valid_;
    t.cpu_valid_ = cpu_valid_;
    t.requires_grad_ = requires_grad_;
    t.grad_ = grad_;
    return t;
}

Tensor Tensor::view(std::vector<int64_t> new_shape) const {
    return reshape(std::move(new_shape));
}

// ── GPU lifecycle ─────────────────────────────────────────────────────

void Tensor::ensure_gpu() {
    if (gpu_valid_) return;

    if (!backend_) {
        throw std::runtime_error(
            "Tensor::ensure_gpu: no ComputeBackend attached");
    }

    size_t bytes = nbytes();

    // Allocate GPU buffer if needed
    if (buffer_handle_ == 0) {
        BufferDesc desc;
        desc.size = bytes;
        desc.usage = BufferDesc::DeviceLocal;
        buffer_handle_ = backend_->createBuffer(desc);
    }

    // Upload CPU data
    if (cpu_valid_ && !cpu_data_.empty()) {
        backend_->upload(buffer_handle_, cpu_data_.data(), bytes);
    }

    gpu_valid_ = true;
}

void Tensor::ensure_cpu() const {
    if (cpu_valid_) return;

    if (!gpu_valid_) {
        throw std::runtime_error(
            "Tensor::ensure_cpu: neither CPU nor GPU data is valid");
    }

    if (!backend_) {
        throw std::runtime_error(
            "Tensor::ensure_cpu: no ComputeBackend for GPU download");
    }

    size_t count = static_cast<size_t>(numel());
    cpu_data_.resize(count);
    backend_->download(buffer_handle_, cpu_data_.data(), count * sizeof(float));
    cpu_valid_ = true;
}

void Tensor::release_gpu() {
    if (buffer_handle_ != 0 && backend_) {
        // Ensure CPU has a copy before releasing
        if (gpu_valid_ && !cpu_valid_) {
            ensure_cpu();
        }
        backend_->destroyBuffer(buffer_handle_);
        buffer_handle_ = 0;
    }
    gpu_valid_ = false;
}

}  // namespace nn
}  // namespace grilly
