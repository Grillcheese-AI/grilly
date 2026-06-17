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
