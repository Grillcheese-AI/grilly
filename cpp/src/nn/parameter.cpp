#include "grilly/nn/parameter.h"

#include <cstring>

namespace grilly {
namespace nn {

Parameter::Parameter(Tensor data, bool requires_grad) : Tensor(std::move(data)) {
    set_requires_grad(requires_grad);
}

Parameter::Parameter(std::vector<int64_t> shape, ComputeBackend* backend,
                     bool requires_grad)
    : Tensor(std::move(shape), DType::Float32, backend) {
    set_requires_grad(requires_grad);
}

Tensor& Parameter::grad_ref() {
    if (!Tensor::grad()) {
        // Lazily create gradient tensor matching this parameter's shape
        auto g = std::make_shared<Tensor>(
            Tensor::zeros(this->shape(), this->backend()));
        Tensor::set_grad(std::move(g));
    }
    return *Tensor::grad();
}

const Tensor& Parameter::grad_ref() const {
    if (!Tensor::grad()) {
        throw std::runtime_error("Parameter has no gradient");
    }
    return *Tensor::grad();
}

bool Parameter::has_grad() const {
    return Tensor::grad() != nullptr && Tensor::grad()->valid();
}

void Parameter::set_grad(const Tensor& new_grad) {
    auto g = std::make_shared<Tensor>(new_grad);
    Tensor::set_grad(std::move(g));
}

void Parameter::zero_grad() {
    if (Tensor::grad()) {
        size_t count = static_cast<size_t>(Tensor::grad()->numel());
        float* ptr = Tensor::grad()->mutable_data();
        std::memset(ptr, 0, count * sizeof(float));
    }
}

}  // namespace nn
}  // namespace grilly
