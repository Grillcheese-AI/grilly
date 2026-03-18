/// bindings_pooling.cpp — Pooling op bindings.
///
/// Migrated from monolithic bindings.cpp to use Tensor-based I/O.

#include "bindings_core.h"
#include "grilly/ops/pooling.h"
#include "grilly/ops/conv.h"  // for convOutputSize

void register_pooling_ops(py::module_& m) {
    using namespace grilly::nn;

    // ── MaxPool2d forward ────────────────────────────────────────────────
    m.def(
        "maxpool2d",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input,
           std::vector<uint32_t> kernel_size,
           std::vector<uint32_t> stride,
           std::vector<uint32_t> padding,
           std::vector<uint32_t> dilation) -> py::dict {
            auto inBuf = input.request();
            if (inBuf.ndim != 4)
                throw std::runtime_error(
                    "input must be 4D (B, C, H, W)");

            uint32_t B  = static_cast<uint32_t>(inBuf.shape[0]);
            uint32_t C  = static_cast<uint32_t>(inBuf.shape[1]);
            uint32_t iH = static_cast<uint32_t>(inBuf.shape[2]);
            uint32_t iW = static_cast<uint32_t>(inBuf.shape[3]);
            uint32_t kH = kernel_size[0];
            uint32_t kW = kernel_size.size() > 1
                              ? kernel_size[1] : kH;
            uint32_t sH = stride[0];
            uint32_t sW = stride.size() > 1 ? stride[1] : sH;
            uint32_t pH = padding[0];
            uint32_t pW = padding.size() > 1 ? padding[1] : pH;
            uint32_t dH = dilation[0];
            uint32_t dW = dilation.size() > 1 ? dilation[1] : dH;

            uint32_t oH = grilly::ops::convOutputSize(
                iH, kH, sH, pH, dH);
            uint32_t oW = grilly::ops::convOutputSize(
                iW, kW, sW, pW, dW);

            py::array_t<float> result({
                static_cast<py::ssize_t>(B),
                static_cast<py::ssize_t>(C),
                static_cast<py::ssize_t>(oH),
                static_cast<py::ssize_t>(oW)});
            py::array_t<uint32_t> indices({
                static_cast<py::ssize_t>(B),
                static_cast<py::ssize_t>(C),
                static_cast<py::ssize_t>(oH),
                static_cast<py::ssize_t>(oW)});

            grilly::ops::MaxPool2dParams p{B, C, iH, iW, oH, oW,
                                           kH, kW, sH, sW,
                                           pH, pW, dH, dW};
            grilly::ops::maxpool2dForward(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(inBuf.ptr),
                static_cast<float*>(result.request().ptr),
                static_cast<uint32_t*>(indices.request().ptr), p);

            py::dict out;
            out["output"] = Tensor::from_numpy(result);
            // indices stay as numpy uint32 — not a float Tensor
            out["indices"] = indices;
            return out;
        },
        py::arg("device"), py::arg("input"),
        py::arg("kernel_size"),
        py::arg("stride") = std::vector<uint32_t>{2, 2},
        py::arg("padding") = std::vector<uint32_t>{0, 0},
        py::arg("dilation") = std::vector<uint32_t>{1, 1},
        "GPU MaxPool2d forward");

    // ── AvgPool2d forward ────────────────────────────────────────────────
    m.def(
        "avgpool2d",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input,
           std::vector<uint32_t> kernel_size,
           std::vector<uint32_t> stride,
           std::vector<uint32_t> padding,
           bool count_include_pad) -> Tensor {
            auto inBuf = input.request();
            if (inBuf.ndim != 4)
                throw std::runtime_error("input must be 4D");

            uint32_t B  = static_cast<uint32_t>(inBuf.shape[0]);
            uint32_t C  = static_cast<uint32_t>(inBuf.shape[1]);
            uint32_t iH = static_cast<uint32_t>(inBuf.shape[2]);
            uint32_t iW = static_cast<uint32_t>(inBuf.shape[3]);
            uint32_t kH = kernel_size[0];
            uint32_t kW = kernel_size.size() > 1
                              ? kernel_size[1] : kH;
            uint32_t sH = stride[0];
            uint32_t sW = stride.size() > 1 ? stride[1] : sH;
            uint32_t pH = padding[0];
            uint32_t pW = padding.size() > 1 ? padding[1] : pH;

            uint32_t oH = grilly::ops::convOutputSize(
                iH, kH, sH, pH, 1);
            uint32_t oW = grilly::ops::convOutputSize(
                iW, kW, sW, pW, 1);

            py::array_t<float> result({
                static_cast<py::ssize_t>(B),
                static_cast<py::ssize_t>(C),
                static_cast<py::ssize_t>(oH),
                static_cast<py::ssize_t>(oW)});

            grilly::ops::AvgPool2dParams p{
                B, C, iH, iW, oH, oW, kH, kW, sH, sW, pH, pW,
                count_include_pad ? 1u : 0u};
            grilly::ops::avgpool2dForward(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(inBuf.ptr),
                static_cast<float*>(result.request().ptr), p);
            return Tensor::from_numpy(result);
        },
        py::arg("device"), py::arg("input"),
        py::arg("kernel_size"),
        py::arg("stride") = std::vector<uint32_t>{2, 2},
        py::arg("padding") = std::vector<uint32_t>{0, 0},
        py::arg("count_include_pad") = true,
        "GPU AvgPool2d forward");

    // ── Mean pooling (B,S,D) -> (B,D) ───────────────────────────────────
    m.def(
        "mean_pool",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input) -> Tensor {
            auto inBuf = input.request();
            if (inBuf.ndim != 3)
                throw std::runtime_error(
                    "input must be 3D (B, S, D)");

            uint32_t B = static_cast<uint32_t>(inBuf.shape[0]);
            uint32_t S = static_cast<uint32_t>(inBuf.shape[1]);
            uint32_t D = static_cast<uint32_t>(inBuf.shape[2]);

            py::array_t<float> result({
                static_cast<py::ssize_t>(B),
                static_cast<py::ssize_t>(D)});

            grilly::ops::MeanPoolParams p{B, S, D};
            grilly::ops::meanPool(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(inBuf.ptr),
                static_cast<float*>(result.request().ptr), p);
            return Tensor::from_numpy(result);
        },
        py::arg("device"), py::arg("input"),
        "GPU mean pooling over sequence dim: (B,S,D) -> (B,D)");
}
