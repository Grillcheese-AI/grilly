#pragma once
/// Trainable parameter — extends Tensor with gradient tracking.
///
/// Replaces the Python `Parameter(np.ndarray)` subclass. Every Parameter
/// has requires_grad=true by default, owns a gradient tensor, and supports
/// zero_grad() for optimizer integration.

#include "grilly/nn/tensor.h"

namespace grilly {
namespace nn {

class Parameter : public Tensor {
public:
    Parameter() = default;

    /// Construct from an existing tensor.
    explicit Parameter(Tensor data, bool requires_grad = true);

    /// Construct from shape (zero-initialized).
    explicit Parameter(std::vector<int64_t> shape,
                       ComputeBackend* backend = nullptr,
                       bool requires_grad = true);

    /// Access the gradient tensor.
    Tensor& grad_ref();
    const Tensor& grad_ref() const;
    bool has_grad() const;

    /// Set the gradient tensor explicitly.
    void set_grad(const Tensor& grad);

    /// Zero out the gradient tensor.
    void zero_grad();
};

}  // namespace nn
}  // namespace grilly
