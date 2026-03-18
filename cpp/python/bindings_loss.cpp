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

            grilly::ops::CrossEntropyParams p{
                batchSize, seqLen, vocabSize, 0, label_smoothing};
            grilly::ops::crossEntropyLoss(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(lBuf.ptr),
                static_cast<const uint32_t*>(targets.request().ptr),
                static_cast<float*>(losses.request().ptr), p);
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
            uint32_t batchSize = static_cast<uint32_t>(lBuf.shape[0]);
            uint32_t numClasses = static_cast<uint32_t>(lBuf.shape[1]);

            py::array_t<float> gradLogits(lBuf.shape);

            grilly::ops::CrossEntropyBackwardParams p{
                batchSize, numClasses};
            grilly::ops::crossEntropyBackward(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(lBuf.ptr),
                static_cast<const uint32_t*>(targets.request().ptr),
                static_cast<float*>(gradLogits.request().ptr), p);
            return Tensor::from_numpy(gradLogits);
        },
        py::arg("device"), py::arg("logits"), py::arg("targets"),
        "GPU cross-entropy backward");
}
