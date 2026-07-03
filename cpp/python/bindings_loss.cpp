/// bindings_loss.cpp — Loss function op bindings.
///
/// Migrated from monolithic bindings.cpp to use Tensor-based I/O.

#include "bindings_core.h"
#include "grilly/ops/loss.h"

void register_loss_ops(py::module_& m) {
    using namespace grilly::nn;

    // ── Cross-entropy loss forward ───────────────────────────────────────
    m.def(
        "cross_entropy_loss",
        [](GrillyCoreContext& ctx,
           py::array_t<float> logits, py::array_t<uint32_t> targets,
           float label_smoothing) -> Tensor {
            auto lBuf = logits.request();
            require_c_contiguous_float(lBuf);

            uint32_t batchSize, seqLen, vocabSize;
            if (lBuf.ndim == 2) {
                batchSize = static_cast<uint32_t>(lBuf.shape[0]);
                seqLen = 1;
                vocabSize = static_cast<uint32_t>(lBuf.shape[1]);
            } else if (lBuf.ndim == 3) {
                batchSize = static_cast<uint32_t>(lBuf.shape[0]);
                seqLen = static_cast<uint32_t>(lBuf.shape[1]);
                vocabSize = static_cast<uint32_t>(lBuf.shape[2]);
            } else {
                throw std::runtime_error("logits must be 2D or 3D");
            }

            uint32_t totalPos = batchSize * seqLen;
            py::array_t<float> losses(totalPos);
            auto tBuf = targets.request();
            require_c_contiguous_uint32(tBuf);
            auto lossBuf = losses.request();

            grilly::ops::CrossEntropyParams p{
                batchSize, seqLen, vocabSize, 0, label_smoothing};
            {
                py::gil_scoped_release release;
                grilly::ops::crossEntropyLoss(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(lBuf.ptr),
                    static_cast<const uint32_t*>(tBuf.ptr),
                    static_cast<float*>(lossBuf.ptr), p);
            }
            return Tensor::from_numpy(losses);
        },
        py::arg("device"), py::arg("logits"), py::arg("targets"),
        py::arg("label_smoothing") = 0.0f,
        "GPU cross-entropy loss forward");

    // ── Cross-entropy loss backward ──────────────────────────────────────
    m.def(
        "cross_entropy_backward",
        [](GrillyCoreContext& ctx,
           py::array_t<float> logits,
           py::array_t<uint32_t> targets) -> Tensor {
            auto lBuf = logits.request();
            require_c_contiguous_float(lBuf);
            uint32_t batchSize = static_cast<uint32_t>(lBuf.shape[0]);
            uint32_t numClasses = static_cast<uint32_t>(lBuf.shape[1]);

            py::array_t<float> gradLogits(lBuf.shape);
            auto tBuf = targets.request();
            require_c_contiguous_uint32(tBuf);
            auto gBuf = gradLogits.request();

            grilly::ops::CrossEntropyBackwardParams p{
                batchSize, numClasses};
            {
                py::gil_scoped_release release;
                grilly::ops::crossEntropyBackward(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(lBuf.ptr),
                    static_cast<const uint32_t*>(tBuf.ptr),
                    static_cast<float*>(gBuf.ptr), p);
            }
            return Tensor::from_numpy(gradLogits);
        },
        py::arg("device"), py::arg("logits"), py::arg("targets"),
        "GPU cross-entropy backward");

    // â”€â”€ Cross-entropy FUSED loss + gradient â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    m.def(
        "cross_entropy_fused",
        [](GrillyCoreContext& ctx,
           py::array_t<float> logits,
           py::array_t<uint32_t> targets) -> py::tuple {
            auto lBuf = logits.request();
            require_c_contiguous_float(lBuf);
            if (lBuf.ndim != 2)
                throw std::runtime_error("cross_entropy_fused: logits must be 2D (batch, num_classes)");
            uint32_t batchSize = static_cast<uint32_t>(lBuf.shape[0]);
            uint32_t numClasses = static_cast<uint32_t>(lBuf.shape[1]);

            py::array_t<float> losses(batchSize);
            py::array_t<float> gradLogits(lBuf.shape);
            auto tBuf = targets.request();
            require_c_contiguous_uint32(tBuf);
            auto lossBuf = losses.request();
            auto gBuf = gradLogits.request();

            grilly::ops::CrossEntropyFusedParams p{batchSize, numClasses};
            {
                py::gil_scoped_release release;
                grilly::ops::crossEntropyFused(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(lBuf.ptr),
                    static_cast<const uint32_t*>(tBuf.ptr),
                    static_cast<float*>(lossBuf.ptr),
                    static_cast<float*>(gBuf.ptr), p);
            }
            return py::make_tuple(Tensor::from_numpy(losses),
                                  Tensor::from_numpy(gradLogits));
        },
        py::arg("device"), py::arg("logits"), py::arg("targets"),
        "GPU fused cross-entropy loss + gradient (one dispatch)");

    // ── Sampled-BCE FUSED loss + dH + dW (softmax-free vocab head) ───────
    m.def(
        "sampled_bce_fused",
        [](GrillyCoreContext& ctx,
           py::array_t<float> hidden,      // (N, d)
           py::array_t<float> table,       // (V, d)
           py::array_t<uint32_t> ids)      // (N, 1+K), col 0 = target
            -> py::tuple {
            auto hBuf = hidden.request();
            auto wBuf = table.request();
            require_c_contiguous_float(hBuf);
            require_c_contiguous_float(wBuf);
            if (hBuf.ndim != 2)
                throw std::runtime_error("sampled_bce_fused: hidden must be 2D (n_tokens, dim)");
            if (wBuf.ndim != 2)
                throw std::runtime_error("sampled_bce_fused: table must be 2D (vocab, dim)");
            if (hBuf.shape[1] != wBuf.shape[1])
                throw std::runtime_error("sampled_bce_fused: hidden/table dim mismatch");

            auto iBuf = ids.request();
            require_c_contiguous_uint32(iBuf);
            if (iBuf.ndim != 2 || iBuf.shape[0] != hBuf.shape[0])
                throw std::runtime_error("sampled_bce_fused: ids must be 2D (n_tokens, n_cand)");

            uint32_t nTokens = static_cast<uint32_t>(hBuf.shape[0]);
            uint32_t dim     = static_cast<uint32_t>(hBuf.shape[1]);
            uint32_t vocab   = static_cast<uint32_t>(wBuf.shape[0]);
            uint32_t nCand   = static_cast<uint32_t>(iBuf.shape[1]);

            py::array_t<float> losses(nTokens);
            py::array_t<float> gradH(hBuf.shape);
            py::array_t<float> gradW(wBuf.shape);
            auto lBuf  = losses.request();
            auto gHBuf = gradH.request();
            auto gWBuf = gradW.request();

            grilly::ops::SampledBceParams p{
                nTokens, nCand, dim, 0, 1.0f / float(nTokens)};
            {
                py::gil_scoped_release release;
                grilly::ops::sampledBceFused(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(hBuf.ptr),
                    static_cast<const float*>(wBuf.ptr),
                    static_cast<const uint32_t*>(iBuf.ptr),
                    static_cast<float*>(lBuf.ptr),
                    static_cast<float*>(gHBuf.ptr),
                    static_cast<float*>(gWBuf.ptr),
                    vocab, p);
            }
            return py::make_tuple(Tensor::from_numpy(losses),
                                  Tensor::from_numpy(gradH),
                                  Tensor::from_numpy(gradW));
        },
        py::arg("device"), py::arg("hidden"), py::arg("table"), py::arg("ids"),
        "GPU fused sampled-BCE head: per-token loss + grad_hidden + grad_table "
        "in one submit (softmax-free; ids col 0 = target, rest = negatives)");

    // ── NCE FUSED (corrected sampled-BCE) loss + dH + dW + db ────────────
    m.def(
        "nce_fused",
        [](GrillyCoreContext& ctx,
           py::array_t<float> hidden,      // (N, d)
           py::array_t<float> table,       // (V, d)
           py::array_t<uint32_t> ids,      // (N, 1+K), col 0 = target
           py::array_t<float> logkq,       // (V,) log(k*q_id)
           py::array_t<float> bias,        // (V,) learned per-token bias
           bool use_correction) -> py::tuple {
            auto hBuf = hidden.request();
            auto wBuf = table.request();
            require_c_contiguous_float(hBuf);
            require_c_contiguous_float(wBuf);
            if (hBuf.ndim != 2 || wBuf.ndim != 2)
                throw std::runtime_error("nce_fused: hidden/table must be 2D");
            if (hBuf.shape[1] != wBuf.shape[1])
                throw std::runtime_error("nce_fused: hidden/table dim mismatch");

            auto iBuf = ids.request();
            require_c_contiguous_uint32(iBuf);
            if (iBuf.ndim != 2 || iBuf.shape[0] != hBuf.shape[0])
                throw std::runtime_error("nce_fused: ids must be (n_tokens, n_cand)");

            auto qBuf = logkq.request();
            auto bBuf = bias.request();
            require_c_contiguous_float(qBuf);
            require_c_contiguous_float(bBuf);
            if (qBuf.size != wBuf.shape[0] || bBuf.size != wBuf.shape[0])
                throw std::runtime_error("nce_fused: logkq/bias must be (vocab,)");

            uint32_t nTokens = static_cast<uint32_t>(hBuf.shape[0]);
            uint32_t dim     = static_cast<uint32_t>(hBuf.shape[1]);
            uint32_t vocab   = static_cast<uint32_t>(wBuf.shape[0]);
            uint32_t nCand   = static_cast<uint32_t>(iBuf.shape[1]);

            py::array_t<float> losses(nTokens);
            py::array_t<float> gradH(hBuf.shape);
            py::array_t<float> gradW(wBuf.shape);
            py::array_t<float> gradB(static_cast<py::ssize_t>(vocab));
            auto lBuf  = losses.request();
            auto gHBuf = gradH.request();
            auto gWBuf = gradW.request();
            auto gBBuf = gradB.request();

            grilly::ops::NceParams p{
                nTokens, nCand, dim, 0, 1.0f / float(nTokens),
                use_correction ? 1u : 0u};
            {
                py::gil_scoped_release release;
                grilly::ops::nceFused(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(hBuf.ptr),
                    static_cast<const float*>(wBuf.ptr),
                    static_cast<const uint32_t*>(iBuf.ptr),
                    static_cast<const float*>(qBuf.ptr),
                    static_cast<const float*>(bBuf.ptr),
                    static_cast<float*>(lBuf.ptr),
                    static_cast<float*>(gHBuf.ptr),
                    static_cast<float*>(gWBuf.ptr),
                    static_cast<float*>(gBBuf.ptr),
                    vocab, p);
            }
            return py::make_tuple(Tensor::from_numpy(losses),
                                  Tensor::from_numpy(gradH),
                                  Tensor::from_numpy(gradW),
                                  Tensor::from_numpy(gradB));
        },
        py::arg("device"), py::arg("hidden"), py::arg("table"), py::arg("ids"),
        py::arg("logkq"), py::arg("bias"), py::arg("use_correction") = true,
        "GPU fused NCE head: loss + grad_hidden + grad_table + grad_bias in one "
        "submit (noise-corrected softmax-free; use_correction=false = SGNS)");

    // ── MSE Loss ─────────────────────────────────────────────────────────
    m.def(
        "mse_loss",
        [](GrillyCoreContext& ctx, py::array_t<float> preds, py::array_t<float> targets) -> Tensor {
            auto pBuf = preds.request();
            auto tBuf = targets.request();
            require_c_contiguous_float(pBuf);
            require_c_contiguous_float(tBuf);
            
            if (pBuf.size != tBuf.size)
                throw std::runtime_error("mse_loss: preds and targets must have same size");
                
            uint32_t n = static_cast<uint32_t>(pBuf.size);
            py::array_t<float> losses(n);
            auto lBuf = losses.request();
            
            grilly::ops::MSELossParams p{n};
            {
                py::gil_scoped_release release;
                grilly::ops::mseLoss(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(pBuf.ptr),
                    static_cast<const float*>(tBuf.ptr),
                    static_cast<float*>(lBuf.ptr), p);
            }
            return Tensor::from_numpy(losses);
        },
        py::arg("device"), py::arg("preds"), py::arg("targets"),
        "GPU MSE loss forward");

    // ── Cosine Similarity Loss ───────────────────────────────────────────
    m.def(
        "cosine_similarity_loss",
        [](GrillyCoreContext& ctx, py::array_t<float> preds, py::array_t<float> targets) -> Tensor {
            auto pBuf = preds.request();
            auto tBuf = targets.request();
            require_c_contiguous_float(pBuf);
            require_c_contiguous_float(tBuf);
            
            if (pBuf.ndim != 2 || tBuf.ndim != 2 || pBuf.shape[0] != tBuf.shape[0] || pBuf.shape[1] != tBuf.shape[1])
                throw std::runtime_error("cosine_similarity_loss: inputs must be 2D and equal shape");
                
            uint32_t batchSize = static_cast<uint32_t>(pBuf.shape[0]);
            uint32_t dim = static_cast<uint32_t>(pBuf.shape[1]);
            
            py::array_t<float> losses(batchSize);
            auto lBuf = losses.request();
            
            grilly::ops::CosineLossParams p{batchSize, dim};
            {
                py::gil_scoped_release release;
                grilly::ops::cosineSimilarityLoss(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(pBuf.ptr),
                    static_cast<const float*>(tBuf.ptr),
                    static_cast<float*>(lBuf.ptr), p);
            }
            return Tensor::from_numpy(losses);
        },
        py::arg("device"), py::arg("preds"), py::arg("targets"),
        "GPU Cosine Similarity loss forward");
}
