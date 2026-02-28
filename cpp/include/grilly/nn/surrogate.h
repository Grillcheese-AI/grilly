#pragma once
/// Surrogate gradient functions for spiking neural networks.
///
/// SNNs use the Heaviside step function for firing (non-differentiable).
/// During backpropagation, we substitute a smooth surrogate gradient:
///   - ATan:        α / (2 * (1 + (π * α * x / 2)²))
///   - Sigmoid:     α * σ(αx) * (1 - σ(αx))
///   - FastSigmoid: α / (2 * (1 + α|x|)²)
///
/// The alpha parameter controls sharpness — higher alpha = closer to
/// true Heaviside but harder to train (vanishing gradients far from 0).

#include <cstdint>

#include "grilly/nn/tensor.h"

namespace grilly {
namespace nn {

enum class SurrogateType : uint8_t {
    ATan = 0,
    Sigmoid = 1,
    FastSigmoid = 2,
};

class SurrogateFunction {
public:
    SurrogateType type;
    float alpha;

    SurrogateFunction(SurrogateType type = SurrogateType::ATan,
                      float alpha = 2.0f);

    /// Forward: Heaviside step function (x >= 0 → 1, else 0).
    Tensor forward(const Tensor& x) const;

    /// Backward: smooth surrogate gradient for BPTT.
    Tensor gradient(const Tensor& x) const;
};

/// Convenience factory functions.
inline SurrogateFunction ATan(float alpha = 2.0f) {
    return SurrogateFunction(SurrogateType::ATan, alpha);
}
inline SurrogateFunction Sigmoid(float alpha = 4.0f) {
    return SurrogateFunction(SurrogateType::Sigmoid, alpha);
}
inline SurrogateFunction FastSigmoid(float alpha = 2.0f) {
    return SurrogateFunction(SurrogateType::FastSigmoid, alpha);
}

}  // namespace nn
}  // namespace grilly
