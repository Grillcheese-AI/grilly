/// siglip_encoder.cpp — Full SigLIP2 encoder in pure C++ (AVX2 auto-vectorized).
///
/// Runs all 12 transformer layers in a single C++ call.
/// No Python loops, no numpy, no per-op PCIe round-trips.
/// Uses GPU for the expensive matmuls (flash_attention, linear) and
/// C++ for everything else (layernorm, residual, GELU, reshape).
///
/// Expected speedup: 12.8s → ~2-4s (eliminating Python/numpy overhead).

#include "grilly/ops/fused.h"
#include <cmath>
#include <cstring>
#include <vector>

namespace grilly {
namespace ops {

// ── Inline CPU ops (auto-vectorized by MSVC/GCC) ────────────────────────

static void layerNorm(const float* x, const float* w, const float* b,
                      float* out, uint32_t seqLen, uint32_t dim) {
    for (uint32_t s = 0; s < seqLen; ++s) {
        const float* row = x + s * dim;
        float* orow = out + s * dim;
        float mean = 0.0f;
        for (uint32_t i = 0; i < dim; ++i) mean += row[i];
        mean /= dim;
        float var = 0.0f;
        for (uint32_t i = 0; i < dim; ++i) {
            float d = row[i] - mean;
            var += d * d;
        }
        float inv_std = 1.0f / sqrtf(var / dim + 1e-6f);
        for (uint32_t i = 0; i < dim; ++i)
            orow[i] = (row[i] - mean) * inv_std * w[i] + b[i];
    }
}

static void linearCPU(const float* x, const float* w, const float* b,
                      float* out, uint32_t M, uint32_t K, uint32_t N) {
    // out[M,N] = x[M,K] @ w[N,K]^T + b[N]
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float acc = b ? b[n] : 0.0f;
            const float* xrow = x + m * K;
            const float* wrow = w + n * K;
            for (uint32_t k = 0; k < K; ++k)
                acc += xrow[k] * wrow[k];
            out[m * N + n] = acc;
        }
    }
}

static inline float geluScalar(float x) {
    if (x > 10.0f) return x;
    if (x < -10.0f) return 0.0f;
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static void geluInPlace(float* x, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i)
        x[i] = geluScalar(x[i]);
}

static void addResidual(float* x, const float* residual, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i)
        x[i] += residual[i];
}

static void softmaxRows(float* scores, uint32_t rows, uint32_t cols) {
    for (uint32_t r = 0; r < rows; ++r) {
        float* row = scores + r * cols;
        float maxVal = row[0];
        for (uint32_t c = 1; c < cols; ++c)
            if (row[c] > maxVal) maxVal = row[c];
        float sum = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) {
            row[c] = expf(row[c] - maxVal);
            sum += row[c];
        }
        float inv_sum = 1.0f / (sum + 1e-8f);
        for (uint32_t c = 0; c < cols; ++c)
            row[c] *= inv_sum;
    }
}

// ── Full SigLIP2 Transformer Layer (pure C++) ───────────────────────────

/// Run one transformer layer: LN→QKV→Attention→OutProj→Residual→LN→MLP→Residual
static void transformerLayer(
    float* x, float* temp, float* qkv, float* attn_out, float* scores,
    const float* ln1_w, const float* ln1_b,
    const float* qkv_w, const float* qkv_b,
    const float* out_w, const float* out_b,
    const float* ln2_w, const float* ln2_b,
    const float* mlp_w1, const float* mlp_b1,
    const float* mlp_w2, const float* mlp_b2,
    uint32_t seqLen, uint32_t hidden, uint32_t numHeads, uint32_t headDim, uint32_t mlpDim,
    CommandBatch& batch, BufferPool& pool, PipelineCache& cache) {

    // 1. LayerNorm + QKV projection (use GPU fused op)
    fusedLayernormLinear(batch, pool, cache,
                         x, ln1_w, ln1_b, qkv_w, qkv_b,
                         qkv, seqLen, hidden, hidden * 3);

    // 2. Multi-head self-attention (CPU — avoid GPU upload/download per head)
    float* Q = qkv;
    float* K = qkv + seqLen * hidden;
    float* V = qkv + 2 * seqLen * hidden;
    float scale = 1.0f / sqrtf(float(headDim));

    // Per-head attention
    std::memset(attn_out, 0, seqLen * hidden * sizeof(float));
    for (uint32_t h = 0; h < numHeads; ++h) {
        // Extract head slices (strided access)
        // Q[s][h*headDim : (h+1)*headDim] for each s
        // scores[seq, seq] = Q_h @ K_h^T / sqrt(d)
        for (uint32_t sq = 0; sq < seqLen; ++sq) {
            for (uint32_t sk = 0; sk < seqLen; ++sk) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < headDim; ++d)
                    dot += Q[sq * hidden + h * headDim + d] * K[sk * hidden + h * headDim + d];
                scores[sq * seqLen + sk] = dot * scale;
            }
        }

        // Softmax per row
        softmaxRows(scores, seqLen, seqLen);

        // attn_out_h = scores @ V_h
        for (uint32_t sq = 0; sq < seqLen; ++sq) {
            for (uint32_t d = 0; d < headDim; ++d) {
                float acc = 0.0f;
                for (uint32_t sk = 0; sk < seqLen; ++sk)
                    acc += scores[sq * seqLen + sk] * V[sk * hidden + h * headDim + d];
                attn_out[sq * hidden + h * headDim + d] = acc;
            }
        }
    }

    // 3. Output projection (GPU fused would be ideal, CPU for now)
    linearCPU(attn_out, out_w, out_b, temp, seqLen, hidden, hidden);

    // 4. Residual
    addResidual(x, temp, seqLen * hidden);

    // 5. LN + MLP (GPU fused)
    // LayerNorm
    layerNorm(x, ln2_w, ln2_b, temp, seqLen, hidden);

    // Fused MLP: fc1 → GELU → fc2
    fusedMlpGelu(batch, pool, cache,
                 temp, mlp_w1, mlp_b1, mlp_w2, mlp_b2,
                 attn_out, seqLen, hidden, mlpDim, hidden);

    // 6. Residual
    addResidual(x, attn_out, seqLen * hidden);
}

// ── Pybind11-callable entry point ───────────────────────────────────────

void siglipEncodeFull(
    CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
    const float* patches, float* embedding,
    const float* const* layerData, uint32_t numLayers,
    const float* postLnW, const float* postLnB,
    uint32_t seqLen, uint32_t hiddenSize, uint32_t numHeads, uint32_t mlpDim) {

    const uint32_t headDim = hiddenSize / numHeads;

    // Allocate work buffers
    std::vector<float> x(seqLen * hiddenSize);
    std::vector<float> temp(seqLen * hiddenSize);
    std::vector<float> qkv(seqLen * hiddenSize * 3);
    std::vector<float> attn_out(seqLen * hiddenSize);
    std::vector<float> scores(seqLen * seqLen);  // 1024×1024 = 4MB

    std::memcpy(x.data(), patches, seqLen * hiddenSize * sizeof(float));

    for (uint32_t layer = 0; layer < numLayers; ++layer) {
        transformerLayer(
            x.data(), temp.data(), qkv.data(), attn_out.data(), scores.data(),
            layerData[layer * 12 + 4], layerData[layer * 12 + 5],  // ln1
            layerData[layer * 12 + 0], layerData[layer * 12 + 1],  // qkv
            layerData[layer * 12 + 2], layerData[layer * 12 + 3],  // out_proj
            layerData[layer * 12 + 6], layerData[layer * 12 + 7],  // ln2
            layerData[layer * 12 + 8], layerData[layer * 12 + 9],  // mlp fc1
            layerData[layer * 12 + 10], layerData[layer * 12 + 11], // mlp fc2
            seqLen, hiddenSize, numHeads, headDim, mlpDim,
            batch, pool, cache);
    }

    // Post-layernorm
    layerNorm(x.data(), postLnW, postLnB, temp.data(), seqLen, hiddenSize);

    // Mean pool → embedding
    for (uint32_t i = 0; i < hiddenSize; ++i) {
        float sum = 0.0f;
        for (uint32_t s = 0; s < seqLen; ++s)
            sum += temp[s * hiddenSize + i];
        embedding[i] = sum / seqLen;
    }

    // L2 normalize
    float norm = 0.0f;
    for (uint32_t i = 0; i < hiddenSize; ++i)
        norm += embedding[i] * embedding[i];
    norm = sqrtf(norm + 1e-8f);
    for (uint32_t i = 0; i < hiddenSize; ++i)
        embedding[i] /= norm;
}

}  // namespace ops
}  // namespace grilly
