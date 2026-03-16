#pragma once

/// Tensor operations for CubeMind — replaces numpy for zero-numpy codebase.
///
/// All functions operate on contiguous float32 arrays. Uses Eigen internally
/// for SIMD-optimized element-wise math and reductions.
///
/// These are the CPU reference implementations. The Vulkan compute shader
/// equivalents (activation-*.glsl, fft-*.glsl, etc.) are dispatched via
/// ComputeBackend when available.

#include <Eigen/Dense>

#include <cstdint>
#include <vector>

namespace grilly {
namespace cubemind {

// ── Allocation ───────────────────────────────────────────────────────

std::vector<float> zeros(int64_t n);
std::vector<float> ones(int64_t n);
std::vector<float> full(int64_t n, float val);
std::vector<float> arange(int64_t n);

// ── Element-wise Math (Eigen SIMD) ───────────────────────────────────

std::vector<float> exp(const float* x, int64_t n);
std::vector<float> log_safe(const float* x, int64_t n);
std::vector<float> sqrt_safe(const float* x, int64_t n);
std::vector<float> abs(const float* x, int64_t n);
std::vector<float> sign(const float* x, int64_t n);
std::vector<float> clip(const float* x, int64_t n, float lo, float hi);
std::vector<float> pow(const float* x, int64_t n, float p);
std::vector<float> where(const bool* cond, const float* a, const float* b, int64_t n);
std::vector<float> maximum(const float* a, const float* b, int64_t n);

// ── Reductions (Eigen SIMD) ──────────────────────────────────────────

float sum(const float* x, int64_t n);
float mean(const float* x, int64_t n);
float var(const float* x, int64_t n);
float max(const float* x, int64_t n);
float min(const float* x, int64_t n);
int64_t argmax(const float* x, int64_t n);
int64_t argmin(const float* x, int64_t n);
std::vector<int64_t> argsort(const float* x, int64_t n);

// ── Linear Algebra (Eigen BLAS) ──────────────────────────────────────

float dot(const float* a, const float* b, int64_t n);
float norm_l2(const float* x, int64_t n);
float norm_l1(const float* x, int64_t n);

/// Row-major matmul: C[m,n] = A[m,k] @ B[k,n]
std::vector<float> matmul(const float* a, const float* b,
                           int64_t m, int64_t k, int64_t n);

// ── FFT (direct DFT for block-code bind) ─────────────────────────────

/// 1D DFT. out_re and out_im must be pre-allocated to length n.
void fft_1d(const float* x, float* out_re, float* out_im, int64_t n);

/// Inverse 1D DFT. Returns real part only. out must be pre-allocated.
void ifft_1d(const float* re, const float* im, float* out, int64_t n);

// ── RNG (deterministic PCG-like LCG) ─────────────────────────────────

std::vector<float> rand_uniform(int64_t n, uint32_t seed = 42);
std::vector<float> rand_normal(int64_t n, uint32_t seed = 42);
std::vector<int64_t> rand_int(int64_t n, int64_t low, int64_t high, uint32_t seed = 42);

}  // namespace cubemind
}  // namespace grilly
