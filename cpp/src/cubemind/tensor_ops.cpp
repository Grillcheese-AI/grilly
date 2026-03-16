// Tensor operations for CubeMind — CPU reference via Eigen SIMD.
//
// These replace numpy for the zero-numpy cubemind codebase.
// Each function has a corresponding Vulkan shader for GPU dispatch.

#include "grilly/cubemind/tensor_ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace grilly {
namespace cubemind {

// ── Allocation ───────────────────────────────────────────────────────

std::vector<float> zeros(int64_t n) {
    return std::vector<float>(n, 0.0f);
}

std::vector<float> ones(int64_t n) {
    return std::vector<float>(n, 1.0f);
}

std::vector<float> full(int64_t n, float val) {
    return std::vector<float>(n, val);
}

std::vector<float> arange(int64_t n) {
    std::vector<float> out(n);
    for (int64_t i = 0; i < n; ++i) out[i] = static_cast<float>(i);
    return out;
}

// ── Element-wise Math ────────────────────────────────────────────────

std::vector<float> exp(const float* x, int64_t n) {
    std::vector<float> out(n);
    Eigen::Map<const Eigen::VectorXf> in(x, n);
    Eigen::Map<Eigen::VectorXf> o(out.data(), n);
    o = in.array().exp();
    return out;
}

std::vector<float> log_safe(const float* x, int64_t n) {
    std::vector<float> out(n);
    Eigen::Map<const Eigen::VectorXf> in(x, n);
    Eigen::Map<Eigen::VectorXf> o(out.data(), n);
    o = in.array().max(1e-30f).log();
    return out;
}

std::vector<float> sqrt_safe(const float* x, int64_t n) {
    std::vector<float> out(n);
    Eigen::Map<const Eigen::VectorXf> in(x, n);
    Eigen::Map<Eigen::VectorXf> o(out.data(), n);
    o = in.array().max(0.0f).sqrt();
    return out;
}

std::vector<float> abs(const float* x, int64_t n) {
    std::vector<float> out(n);
    Eigen::Map<const Eigen::VectorXf> in(x, n);
    Eigen::Map<Eigen::VectorXf> o(out.data(), n);
    o = in.array().abs();
    return out;
}

std::vector<float> sign(const float* x, int64_t n) {
    std::vector<float> out(n);
    Eigen::Map<const Eigen::VectorXf> in(x, n);
    Eigen::Map<Eigen::VectorXf> o(out.data(), n);
    o = in.array().sign();
    return out;
}

std::vector<float> clip(const float* x, int64_t n, float lo, float hi) {
    std::vector<float> out(n);
    Eigen::Map<const Eigen::VectorXf> in(x, n);
    Eigen::Map<Eigen::VectorXf> o(out.data(), n);
    o = in.array().max(lo).min(hi);
    return out;
}

std::vector<float> pow(const float* x, int64_t n, float p) {
    std::vector<float> out(n);
    Eigen::Map<const Eigen::VectorXf> in(x, n);
    Eigen::Map<Eigen::VectorXf> o(out.data(), n);
    o = in.array().pow(p);
    return out;
}

std::vector<float> where(const bool* cond, const float* a, const float* b, int64_t n) {
    std::vector<float> out(n);
    for (int64_t i = 0; i < n; ++i) out[i] = cond[i] ? a[i] : b[i];
    return out;
}

std::vector<float> maximum(const float* a, const float* b, int64_t n) {
    std::vector<float> out(n);
    Eigen::Map<const Eigen::VectorXf> va(a, n);
    Eigen::Map<const Eigen::VectorXf> vb(b, n);
    Eigen::Map<Eigen::VectorXf> o(out.data(), n);
    o = va.array().max(vb.array());
    return out;
}

// ── Reductions ───────────────────────────────────────────────────────

float sum(const float* x, int64_t n) {
    return Eigen::Map<const Eigen::VectorXf>(x, n).sum();
}

float mean(const float* x, int64_t n) {
    return Eigen::Map<const Eigen::VectorXf>(x, n).mean();
}

float var(const float* x, int64_t n) {
    Eigen::Map<const Eigen::VectorXf> v(x, n);
    float m = v.mean();
    return (v.array() - m).square().mean();
}

float max(const float* x, int64_t n) {
    return Eigen::Map<const Eigen::VectorXf>(x, n).maxCoeff();
}

float min(const float* x, int64_t n) {
    return Eigen::Map<const Eigen::VectorXf>(x, n).minCoeff();
}

int64_t argmax(const float* x, int64_t n) {
    Eigen::Index idx;
    Eigen::Map<const Eigen::VectorXf>(x, n).maxCoeff(&idx);
    return static_cast<int64_t>(idx);
}

int64_t argmin(const float* x, int64_t n) {
    Eigen::Index idx;
    Eigen::Map<const Eigen::VectorXf>(x, n).minCoeff(&idx);
    return static_cast<int64_t>(idx);
}

std::vector<int64_t> argsort(const float* x, int64_t n) {
    std::vector<int64_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [x](int64_t a, int64_t b) { return x[a] < x[b]; });
    return indices;
}

// ── Linear Algebra ───────────────────────────────────────────────────

float dot(const float* a, const float* b, int64_t n) {
    return Eigen::Map<const Eigen::VectorXf>(a, n).dot(
           Eigen::Map<const Eigen::VectorXf>(b, n));
}

float norm_l2(const float* x, int64_t n) {
    return Eigen::Map<const Eigen::VectorXf>(x, n).norm();
}

float norm_l1(const float* x, int64_t n) {
    return Eigen::Map<const Eigen::VectorXf>(x, n).lpNorm<1>();
}

std::vector<float> matmul(const float* a, const float* b,
                           int64_t m, int64_t k, int64_t n) {
    std::vector<float> out(m * n);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        ma(a, m, k);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        mb(b, k, n);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        mc(out.data(), m, n);
    mc.noalias() = ma * mb;
    return out;
}

// ── FFT ──────────────────────────────────────────────────────────────

void fft_1d(const float* x, float* out_re, float* out_im, int64_t n) {
    const double PI2 = 6.283185307179586;
    for (int64_t k = 0; k < n; ++k) {
        double sr = 0.0, si = 0.0;
        for (int64_t j = 0; j < n; ++j) {
            double angle = PI2 * k * j / n;
            sr += x[j] * std::cos(angle);
            si -= x[j] * std::sin(angle);
        }
        out_re[k] = static_cast<float>(sr);
        out_im[k] = static_cast<float>(si);
    }
}

void ifft_1d(const float* re, const float* im, float* out, int64_t n) {
    const double PI2 = 6.283185307179586;
    for (int64_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (int64_t k = 0; k < n; ++k) {
            double angle = PI2 * k * j / n;
            s += re[k] * std::cos(angle) - im[k] * std::sin(angle);
        }
        out[j] = static_cast<float>(s / n);
    }
}

// ── RNG ──────────────────────────────────────────────────────────────

std::vector<float> rand_uniform(int64_t n, uint32_t seed) {
    std::vector<float> out(n);
    uint64_t state = seed;
    for (int64_t i = 0; i < n; ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        out[i] = static_cast<float>((state >> 33) & 0x7FFFFFFF) / 2147483647.0f;
    }
    return out;
}

std::vector<float> rand_normal(int64_t n, uint32_t seed) {
    std::vector<float> out(n);
    uint64_t state = seed;
    for (int64_t i = 0; i < n; i += 2) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        float u1 = std::max(1e-10f,
            static_cast<float>((state >> 33) & 0x7FFFFFFF) / 2147483647.0f);
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        float u2 = static_cast<float>((state >> 33) & 0x7FFFFFFF) / 2147483647.0f;
        float r = std::sqrt(-2.0f * std::log(u1));
        float theta = 6.283185307f * u2;
        out[i] = r * std::cos(theta);
        if (i + 1 < n) out[i + 1] = r * std::sin(theta);
    }
    return out;
}

std::vector<int64_t> rand_int(int64_t n, int64_t low, int64_t high, uint32_t seed) {
    std::vector<int64_t> out(n);
    uint64_t state = seed;
    uint64_t range = static_cast<uint64_t>(high - low);
    for (int64_t i = 0; i < n; ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        out[i] = low + static_cast<int64_t>((state >> 33) % range);
    }
    return out;
}

}  // namespace cubemind
}  // namespace grilly
