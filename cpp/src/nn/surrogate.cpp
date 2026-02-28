#include "grilly/nn/surrogate.h"

#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace grilly {
namespace nn {

SurrogateFunction::SurrogateFunction(SurrogateType type, float alpha)
    : type(type), alpha(alpha) {}

Tensor SurrogateFunction::forward(const Tensor& x) const {
    // Heaviside step function: x >= 0 → 1.0, else 0.0
    int64_t n = x.numel();
    const float* src = x.data();
    std::vector<float> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; i++) {
        out[i] = (src[i] >= 0.0f) ? 1.0f : 0.0f;
    }
    return Tensor(std::move(out), x.shape(), x.backend());
}

Tensor SurrogateFunction::gradient(const Tensor& x) const {
    int64_t n = x.numel();
    const float* src = x.data();
    std::vector<float> out(static_cast<size_t>(n));

    switch (type) {
        case SurrogateType::ATan: {
            // α / (2 * (1 + (π * α * x / 2)²))
            float half_pi_alpha = static_cast<float>(M_PI) * alpha * 0.5f;
            for (int64_t i = 0; i < n; i++) {
                float u = half_pi_alpha * src[i];
                out[i] = alpha / (2.0f * (1.0f + u * u));
            }
            break;
        }
        case SurrogateType::Sigmoid: {
            // α * σ(αx) * (1 - σ(αx))
            for (int64_t i = 0; i < n; i++) {
                float s = 1.0f / (1.0f + std::exp(-alpha * src[i]));
                out[i] = alpha * s * (1.0f - s);
            }
            break;
        }
        case SurrogateType::FastSigmoid: {
            // α / (2 * (1 + α|x|)²)
            for (int64_t i = 0; i < n; i++) {
                float denom = 1.0f + alpha * std::fabs(src[i]);
                out[i] = alpha / (2.0f * denom * denom);
            }
            break;
        }
    }

    return Tensor(std::move(out), x.shape(), x.backend());
}

}  // namespace nn
}  // namespace grilly
