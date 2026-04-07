/// bindings_conv.cpp — Convolution op bindings (forward + backward).
///
/// Migrated from monolithic bindings.cpp to use Tensor-based I/O.

#include "bindings_core.h"
#include "grilly/ops/conv.h"

void register_conv_ops(py::module_& m) {
    using namespace grilly::nn;

    // ── Conv2d forward ───────────────────────────────────────────────────
    m.def(
        "conv2d",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input, py::array_t<float> weight,
           std::optional<py::array_t<float>> bias,
           std::vector<uint32_t> stride, std::vector<uint32_t> padding,
           std::vector<uint32_t> dilation,
           uint32_t groups) -> Tensor {
            auto inBuf = input.request();
            auto wBuf = weight.request();
            require_c_contiguous_float(inBuf);
            require_c_contiguous_float(wBuf);

            if (inBuf.ndim != 4)
                throw std::runtime_error(
                    "input must be 4D (batch, channels, height, width)");
            if (wBuf.ndim != 4)
                throw std::runtime_error(
                    "weight must be 4D (out_ch, in_ch/groups, kH, kW)");

            uint32_t batchSize  = static_cast<uint32_t>(inBuf.shape[0]);
            uint32_t inChannels = static_cast<uint32_t>(inBuf.shape[1]);
            uint32_t inH        = static_cast<uint32_t>(inBuf.shape[2]);
            uint32_t inW        = static_cast<uint32_t>(inBuf.shape[3]);
            uint32_t outChannels = static_cast<uint32_t>(wBuf.shape[0]);
            uint32_t kH          = static_cast<uint32_t>(wBuf.shape[2]);
            uint32_t kW          = static_cast<uint32_t>(wBuf.shape[3]);

            uint32_t sH = stride.size() >= 1 ? stride[0] : 1;
            uint32_t sW = stride.size() >= 2 ? stride[1] : sH;
            uint32_t pH = padding.size() >= 1 ? padding[0] : 0;
            uint32_t pW = padding.size() >= 2 ? padding[1] : pH;
            uint32_t dH = dilation.size() >= 1 ? dilation[0] : 1;
            uint32_t dW = dilation.size() >= 2 ? dilation[1] : dH;

            uint32_t outH = grilly::ops::convOutputSize(inH, kH, sH, pH, dH);
            uint32_t outW = grilly::ops::convOutputSize(inW, kW, sW, pW, dW);

            const float* biasPtr = nullptr;
            if (bias.has_value()) {
                auto bBuf = bias->request();
                require_c_contiguous_float(bBuf);
                biasPtr = static_cast<const float*>(bBuf.ptr);
            }

            py::array_t<float> result({
                static_cast<py::ssize_t>(batchSize),
                static_cast<py::ssize_t>(outChannels),
                static_cast<py::ssize_t>(outH),
                static_cast<py::ssize_t>(outW)
            });
            auto rBuf = result.request();

            {
                py::gil_scoped_release release;
                grilly::ops::conv2d(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(inBuf.ptr),
                    static_cast<const float*>(wBuf.ptr),
                    biasPtr,
                    static_cast<float*>(rBuf.ptr),
                    batchSize, inChannels, inH, inW,
                    outChannels, kH, kW,
                    sH, sW, pH, pW, dH, dW, groups);
            }

            return Tensor::from_numpy(result);
        },
        py::arg("device"), py::arg("input"), py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = std::vector<uint32_t>{1, 1},
        py::arg("padding") = std::vector<uint32_t>{0, 0},
        py::arg("dilation") = std::vector<uint32_t>{1, 1},
        py::arg("groups") = 1,
        "GPU Conv2d forward (direct or GEMM path)");

    // ── Conv1d forward ───────────────────────────────────────────────────
    m.def(
        "conv1d",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input, py::array_t<float> weight,
           std::optional<py::array_t<float>> bias,
           uint32_t stride, uint32_t padding,
           uint32_t dilation, uint32_t groups) -> Tensor {
            auto inBuf = input.request();
            auto wBuf = weight.request();
            require_c_contiguous_float(inBuf);
            require_c_contiguous_float(wBuf);

            if (inBuf.ndim != 3)
                throw std::runtime_error(
                    "input must be 3D (batch, channels, length)");

            uint32_t batchSize  = static_cast<uint32_t>(inBuf.shape[0]);
            uint32_t inChannels = static_cast<uint32_t>(inBuf.shape[1]);
            uint32_t length     = static_cast<uint32_t>(inBuf.shape[2]);
            uint32_t outChannels = static_cast<uint32_t>(wBuf.shape[0]);
            uint32_t kSize       = static_cast<uint32_t>(wBuf.shape[2]);

            uint32_t outLen = grilly::ops::convOutputSize(
                length, kSize, stride, padding, dilation);

            const float* biasPtr = nullptr;
            if (bias.has_value()) {
                auto bBuf = bias->request();
                require_c_contiguous_float(bBuf);
                biasPtr = static_cast<const float*>(bBuf.ptr);
            }

            py::array_t<float> result({
                static_cast<py::ssize_t>(batchSize),
                static_cast<py::ssize_t>(outChannels),
                static_cast<py::ssize_t>(outLen)
            });
            auto rBuf = result.request();

            {
                py::gil_scoped_release release;
                grilly::ops::conv1d(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(inBuf.ptr),
                    static_cast<const float*>(wBuf.ptr),
                    biasPtr,
                    static_cast<float*>(rBuf.ptr),
                    batchSize, inChannels, length,
                    outChannels, kSize,
                    stride, padding, dilation, groups);
            }

            return Tensor::from_numpy(result);
        },
        py::arg("device"), py::arg("input"), py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1, py::arg("padding") = 0,
        py::arg("dilation") = 1, py::arg("groups") = 1,
        "GPU Conv1d forward (wrapper around Conv2d)");

    // ── Conv2d backward w.r.t. input ─────────────────────────────────────
    m.def(
        "conv2d_backward_input",
        [](GrillyCoreContext& ctx,
           py::array_t<float> grad_output, py::array_t<float> weight,
           std::vector<uint32_t> input_shape,
           std::vector<uint32_t> stride, std::vector<uint32_t> padding,
           std::vector<uint32_t> dilation,
           uint32_t groups) -> Tensor {
            auto gBuf = grad_output.request();
            auto wBuf = weight.request();
            require_c_contiguous_float(gBuf);
            require_c_contiguous_float(wBuf);

            uint32_t batchSize  = input_shape[0];
            uint32_t inChannels = input_shape[1];
            uint32_t inH        = input_shape[2];
            uint32_t inW        = input_shape[3];
            uint32_t outChannels = static_cast<uint32_t>(gBuf.shape[1]);
            uint32_t outH        = static_cast<uint32_t>(gBuf.shape[2]);
            uint32_t outW        = static_cast<uint32_t>(gBuf.shape[3]);
            uint32_t kH          = static_cast<uint32_t>(wBuf.shape[2]);
            uint32_t kW          = static_cast<uint32_t>(wBuf.shape[3]);

            uint32_t sH = stride.size() >= 1 ? stride[0] : 1;
            uint32_t sW = stride.size() >= 2 ? stride[1] : sH;
            uint32_t pH = padding.size() >= 1 ? padding[0] : 0;
            uint32_t pW = padding.size() >= 2 ? padding[1] : pH;
            uint32_t dH = dilation.size() >= 1 ? dilation[0] : 1;
            uint32_t dW = dilation.size() >= 2 ? dilation[1] : dH;

            grilly::ops::Conv2dBackwardInputParams p{
                batchSize, inChannels, inH, inW, outChannels, outH, outW,
                kH, kW, sH, sW, pH, pW, dH, dW, groups};

            py::array_t<float> result({
                static_cast<py::ssize_t>(batchSize),
                static_cast<py::ssize_t>(inChannels),
                static_cast<py::ssize_t>(inH),
                static_cast<py::ssize_t>(inW)});
            auto resBuf = result.request();

            {
                py::gil_scoped_release release;
                grilly::ops::conv2dBackwardInput(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(gBuf.ptr),
                    static_cast<const float*>(wBuf.ptr),
                    static_cast<float*>(resBuf.ptr), p);
            }

            return Tensor::from_numpy(result);
        },
        py::arg("device"), py::arg("grad_output"), py::arg("weight"),
        py::arg("input_shape"),
        py::arg("stride") = std::vector<uint32_t>{1, 1},
        py::arg("padding") = std::vector<uint32_t>{0, 0},
        py::arg("dilation") = std::vector<uint32_t>{1, 1},
        py::arg("groups") = 1,
        "GPU Conv2d backward w.r.t. input");

    // ── Conv2d backward w.r.t. weight ────────────────────────────────────
    m.def(
        "conv2d_backward_weight",
        [](GrillyCoreContext& ctx,
           py::array_t<float> grad_output, py::array_t<float> input,
           std::vector<uint32_t> weight_shape,
           std::vector<uint32_t> stride, std::vector<uint32_t> padding,
           std::vector<uint32_t> dilation,
           uint32_t groups, bool has_bias) -> py::dict {
            auto gBuf = grad_output.request();
            auto iBuf = input.request();
            require_c_contiguous_float(gBuf);
            require_c_contiguous_float(iBuf);

            uint32_t batchSize  = static_cast<uint32_t>(iBuf.shape[0]);
            uint32_t inChannels = static_cast<uint32_t>(iBuf.shape[1]);
            uint32_t inH        = static_cast<uint32_t>(iBuf.shape[2]);
            uint32_t inW        = static_cast<uint32_t>(iBuf.shape[3]);
            uint32_t outChannels = weight_shape[0];
            uint32_t outH        = static_cast<uint32_t>(gBuf.shape[2]);
            uint32_t outW        = static_cast<uint32_t>(gBuf.shape[3]);
            uint32_t kH          = weight_shape[2];
            uint32_t kW          = weight_shape[3];

            uint32_t sH = stride.size() >= 1 ? stride[0] : 1;
            uint32_t sW = stride.size() >= 2 ? stride[1] : sH;
            uint32_t pH = padding.size() >= 1 ? padding[0] : 0;
            uint32_t pW = padding.size() >= 2 ? padding[1] : pH;
            uint32_t dH = dilation.size() >= 1 ? dilation[0] : 1;
            uint32_t dW = dilation.size() >= 2 ? dilation[1] : dH;

            grilly::ops::Conv2dBackwardWeightParams p{
                batchSize, inChannels, inH, inW, outChannels, outH, outW,
                kH, kW, sH, sW, pH, pW, dH, dW, groups,
                has_bias ? 1u : 0u};

            py::array_t<float> gradWeight({
                static_cast<py::ssize_t>(weight_shape[0]),
                static_cast<py::ssize_t>(weight_shape[1]),
                static_cast<py::ssize_t>(weight_shape[2]),
                static_cast<py::ssize_t>(weight_shape[3])});

            py::array_t<float> gradBias(
                has_bias ? std::vector<py::ssize_t>{
                    static_cast<py::ssize_t>(outChannels)}
                : std::vector<py::ssize_t>{1});
            auto gwBuf = gradWeight.request();
            auto gbBuf = gradBias.request();

            {
                py::gil_scoped_release release;
                grilly::ops::conv2dBackwardWeight(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(gBuf.ptr),
                    static_cast<const float*>(iBuf.ptr),
                    static_cast<float*>(gwBuf.ptr),
                    has_bias ? static_cast<float*>(gbBuf.ptr) : nullptr,
                    p);
            }

            py::dict result;
            result["grad_weight"] = Tensor::from_numpy(gradWeight);
            if (has_bias)
                result["grad_bias"] = Tensor::from_numpy(gradBias);
            return result;
        },
        py::arg("device"), py::arg("grad_output"), py::arg("input"),
        py::arg("weight_shape"),
        py::arg("stride") = std::vector<uint32_t>{1, 1},
        py::arg("padding") = std::vector<uint32_t>{0, 0},
        py::arg("dilation") = std::vector<uint32_t>{1, 1},
        py::arg("groups") = 1, py::arg("has_bias") = false,
        "GPU Conv2d backward w.r.t. weight and bias");
}
