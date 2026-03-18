/// bindings_attention.cpp — Attention and RoPE op bindings.
///
/// Migrated from monolithic bindings.cpp to use Tensor-based I/O.

#include "bindings_core.h"
#include "grilly/ops/attention.h"
#include "grilly/ops/attention_ops.h"

void register_attention_ops(py::module_& m) {
    using namespace grilly::nn;

    // ── Flash Attention 2 ────────────────────────────────────────────────
    m.def(
        "flash_attention2",
        [](GrillyCoreContext& ctx,
           py::array_t<float> Q, py::array_t<float> K, py::array_t<float> V,
           std::optional<py::array_t<float>> mask,
           float scale, uint32_t tileSizeQ,
           uint32_t tileSizeK) -> Tensor {
            auto qBuf = Q.request();

            if (qBuf.ndim != 4)
                throw std::runtime_error(
                    "Q must be 4D (batch, heads, seq_len, head_dim)");

            uint32_t batchSize = static_cast<uint32_t>(qBuf.shape[0]);
            uint32_t numHeads  = static_cast<uint32_t>(qBuf.shape[1]);
            uint32_t seqLen    = static_cast<uint32_t>(qBuf.shape[2]);
            uint32_t headDim   = static_cast<uint32_t>(qBuf.shape[3]);

            const float* maskPtr = nullptr;
            if (mask.has_value()) {
                maskPtr = static_cast<const float*>(mask->request().ptr);
            }

            py::array_t<float> result({
                static_cast<py::ssize_t>(batchSize),
                static_cast<py::ssize_t>(numHeads),
                static_cast<py::ssize_t>(seqLen),
                static_cast<py::ssize_t>(headDim)
            });
            auto rBuf = result.request();

            grilly::ops::flashAttention2(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(qBuf.ptr),
                static_cast<const float*>(K.request().ptr),
                static_cast<const float*>(V.request().ptr),
                maskPtr,
                static_cast<float*>(rBuf.ptr),
                batchSize, seqLen, numHeads, headDim,
                scale, tileSizeQ, tileSizeK);

            return Tensor::from_numpy(result);
        },
        py::arg("device"), py::arg("Q"), py::arg("K"), py::arg("V"),
        py::arg("mask") = py::none(), py::arg("scale") = 0.0f,
        py::arg("tile_size_q") = 64, py::arg("tile_size_k") = 64,
        "GPU Flash Attention 2 with online softmax tiling");

    // ── Attention scores: Q @ K^T / sqrt(d_h) ───────────────────────────
    m.def(
        "attention_scores",
        [](GrillyCoreContext& ctx,
           py::array_t<float> Q, py::array_t<float> K,
           float scale) -> Tensor {
            auto qBuf = Q.request();
            if (qBuf.ndim != 4)
                throw std::runtime_error("Q must be 4D (B, H, S, D)");

            uint32_t B = static_cast<uint32_t>(qBuf.shape[0]);
            uint32_t H = static_cast<uint32_t>(qBuf.shape[1]);
            uint32_t S = static_cast<uint32_t>(qBuf.shape[2]);
            uint32_t D = static_cast<uint32_t>(qBuf.shape[3]);

            if (scale == 0.0f) scale = 1.0f / std::sqrt(float(D));

            py::array_t<float> result({
                static_cast<py::ssize_t>(B), static_cast<py::ssize_t>(H),
                static_cast<py::ssize_t>(S), static_cast<py::ssize_t>(S)});

            grilly::ops::AttentionScoresParams p{B, S, H, D, scale, 0};
            grilly::ops::attentionScores(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(qBuf.ptr),
                static_cast<const float*>(K.request().ptr),
                static_cast<float*>(result.request().ptr), p);
            return Tensor::from_numpy(result);
        },
        py::arg("device"), py::arg("Q"), py::arg("K"),
        py::arg("scale") = 0.0f,
        "GPU attention scores: Q @ K^T / sqrt(d_h)");

    // ── Attention mask (causal or custom) ────────────────────────────────
    m.def(
        "attention_mask",
        [](GrillyCoreContext& ctx,
           py::array_t<float> scores,
           std::optional<py::array_t<float>> mask,
           bool causal, float mask_value) -> Tensor {
            auto sBuf = scores.request();
            if (sBuf.ndim != 4)
                throw std::runtime_error("scores must be 4D (B, H, S, S)");

            uint32_t B = static_cast<uint32_t>(sBuf.shape[0]);
            uint32_t H = static_cast<uint32_t>(sBuf.shape[1]);
            uint32_t S = static_cast<uint32_t>(sBuf.shape[2]);

            py::array_t<float> result(sBuf.shape);
            std::memcpy(result.mutable_data(), sBuf.ptr,
                        sBuf.size * sizeof(float));

            const float* maskPtr = nullptr;
            if (mask.has_value())
                maskPtr = static_cast<const float*>(mask->request().ptr);

            grilly::ops::AttentionMaskParams p{
                B, H, S, causal ? 1u : 0u, mask_value};
            grilly::ops::attentionMask(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<float*>(result.mutable_data()), maskPtr, p);
            return Tensor::from_numpy(result);
        },
        py::arg("device"), py::arg("scores"),
        py::arg("mask") = py::none(),
        py::arg("causal") = true, py::arg("mask_value") = -1e9f,
        "GPU attention mask (causal or custom)");

    // ── Attention output: softmax(scores) @ V ────────────────────────────
    m.def(
        "attention_output",
        [](GrillyCoreContext& ctx,
           py::array_t<float> weights,
           py::array_t<float> V) -> Tensor {
            auto wBuf = weights.request();
            auto vBuf = V.request();
            if (wBuf.ndim != 4 || vBuf.ndim != 4)
                throw std::runtime_error("weights and V must be 4D");

            uint32_t B = static_cast<uint32_t>(vBuf.shape[0]);
            uint32_t H = static_cast<uint32_t>(vBuf.shape[1]);
            uint32_t S = static_cast<uint32_t>(vBuf.shape[2]);
            uint32_t D = static_cast<uint32_t>(vBuf.shape[3]);

            py::array_t<float> result({
                static_cast<py::ssize_t>(B), static_cast<py::ssize_t>(H),
                static_cast<py::ssize_t>(S), static_cast<py::ssize_t>(D)});

            grilly::ops::AttentionOutputParams p{B, S, H, D};
            grilly::ops::attentionOutput(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(wBuf.ptr),
                static_cast<const float*>(vBuf.ptr),
                static_cast<float*>(result.request().ptr), p);
            return Tensor::from_numpy(result);
        },
        py::arg("device"), py::arg("weights"), py::arg("V"),
        "GPU attention output: softmax(scores) @ V");

    // ── Concat multi-head: (B,H,S,D) -> (B,S,H*D) ──────────────────────
    m.def(
        "attention_concat_heads",
        [](GrillyCoreContext& ctx,
           py::array_t<float> mh_output) -> Tensor {
            auto inBuf = mh_output.request();
            if (inBuf.ndim != 4)
                throw std::runtime_error("input must be 4D (B, H, S, D)");

            uint32_t B = static_cast<uint32_t>(inBuf.shape[0]);
            uint32_t H = static_cast<uint32_t>(inBuf.shape[1]);
            uint32_t S = static_cast<uint32_t>(inBuf.shape[2]);
            uint32_t D = static_cast<uint32_t>(inBuf.shape[3]);

            py::array_t<float> result({
                static_cast<py::ssize_t>(B),
                static_cast<py::ssize_t>(S),
                static_cast<py::ssize_t>(H * D)});

            grilly::ops::ConcatHeadsParams p{B, S, H, D};
            grilly::ops::attentionConcatHeads(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(inBuf.ptr),
                static_cast<float*>(result.request().ptr), p);
            return Tensor::from_numpy(result);
        },
        py::arg("device"), py::arg("mh_output"),
        "GPU concat multi-head attention output: (B,H,S,D) -> (B,S,H*D)");

    // ── RoPE (Rotary Position Embeddings) ────────────────────────────────
    m.def(
        "rope",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input,
           std::optional<py::array_t<float>> cos_table,
           std::optional<py::array_t<float>> sin_table,
           float base, float scaling) -> Tensor {
            auto inBuf = input.request();
            if (inBuf.ndim != 4)
                throw std::runtime_error("input must be 4D (B, H, S, D)");

            uint32_t B = static_cast<uint32_t>(inBuf.shape[0]);
            uint32_t H = static_cast<uint32_t>(inBuf.shape[1]);
            uint32_t S = static_cast<uint32_t>(inBuf.shape[2]);
            uint32_t D = static_cast<uint32_t>(inBuf.shape[3]);

            bool precomputed = cos_table.has_value() &&
                               sin_table.has_value();
            const float* cosPtr = precomputed ?
                static_cast<const float*>(cos_table->request().ptr)
                : nullptr;
            const float* sinPtr = precomputed ?
                static_cast<const float*>(sin_table->request().ptr)
                : nullptr;

            py::array_t<float> result(inBuf.shape);
            grilly::ops::RoPEParams p{
                B, S, H, D, base, precomputed ? 1u : 0u, scaling};
            grilly::ops::applyRoPE(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(inBuf.ptr),
                static_cast<float*>(result.request().ptr),
                cosPtr, sinPtr, p);
            return Tensor::from_numpy(result);
        },
        py::arg("device"), py::arg("input"),
        py::arg("cos_table") = py::none(),
        py::arg("sin_table") = py::none(),
        py::arg("base") = 10000.0f, py::arg("scaling") = 1.0f,
        "GPU Rotary Position Embeddings");
}
