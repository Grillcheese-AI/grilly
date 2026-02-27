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
    if (!grad()) {
        // Lazily create gradient tensor matching this parameter's shape
        auto g = std::make_shared<Tensor>(
            Tensor::zeros(this->shape(), this->backend()));
        set_grad(g);
    }
    return *grad();
}

const Tensor& Parameter::grad_ref() const {
    if (!grad()) {
        throw std::runtime_error("Parameter has no gradient");
    }
    return *grad();
}

bool Parameter::has_grad() const {
    return grad() != nullptr && grad()->valid();
}

void Parameter::zero_grad() {
    if (grad()) {
        size_t count = static_cast<size_t>(grad()->numel());
        float* ptr = grad()->mutable_data();
        std::memset(ptr, 0, count * sizeof(float));
    }
}

}  // namespace nn
}  // namespace grilly
