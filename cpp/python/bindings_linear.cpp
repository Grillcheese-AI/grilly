/// bindings_linear.cpp — Linear layer op bindings (forward + backward).
///
/// Migrated from monolithic bindings.cpp to use Tensor-based I/O.

#include "bindings_core.h"
#include "grilly/ops/linear.h"

void register_linear_ops(py::module_& m) {
    using namespace grilly::nn;

    // ── GPU linear forward ───────────────────────────────────────────────
    m.def(
        "linear",
        [](GrillyCoreContext& ctx, py::array_t<float> x,
           py::array_t<float> weights,
           std::optional<py::array_t<float>> bias) -> Tensor {
            auto xBuf = x.request();
            auto wBuf = weights.request();

            if (xBuf.ndim < 1 || xBuf.ndim > 3)
                throw std::runtime_error(
                    "x must be 1D, 2D, or 3D (batch, seq, input_dim)");
            if (wBuf.ndim != 2)
                throw std::runtime_error(
                    "weights must be 2D (output_dim, input_dim)");

            auto [batchSeq, inputDim] = extractBatchAndLastDim(xBuf);
            uint32_t outputDim = static_cast<uint32_t>(wBuf.shape[0]);

            if (static_cast<uint32_t>(wBuf.shape[1]) != inputDim)
                throw std::runtime_error(
                    "Weight input_dim mismatch: " +
                    std::to_string(wBuf.shape[1]) + " vs " +
                    std::to_string(inputDim));

            const float* biasPtr = nullptr;
            uint32_t hasBias = 0;
            if (bias.has_value()) {
                auto bBuf = bias->request();
                if (bBuf.ndim != 1 ||
                    static_cast<uint32_t>(bBuf.shape[0]) != outputDim)
                    throw std::runtime_error(
                        "bias must be 1D with size output_dim");
                biasPtr = static_cast<const float*>(bBuf.ptr);
                hasBias = 1;
            }

            grilly::ops::LinearParams p{batchSeq, inputDim, outputDim, hasBias};

            std::vector<py::ssize_t> outShape;
            for (int i = 0; i < xBuf.ndim - 1; ++i)
                outShape.push_back(xBuf.shape[i]);
            if (xBuf.ndim == 1) outShape.push_back(1);
            outShape.push_back(outputDim);

            py::array_t<float> result(outShape);
            auto rBuf = result.request();

            // Extract raw pointers before GIL release
            const float* xPtr = static_cast<const float*>(xBuf.ptr);
            const float* wPtr = static_cast<const float*>(wBuf.ptr);
            float* oPtr = static_cast<float*>(rBuf.ptr);

            {
                // Release GIL during GPU GEMM dispatch
                py::gil_scoped_release release;
                grilly::ops::linear(
                    ctx.batch, ctx.pool, ctx.cache,
                    xPtr, wPtr, biasPtr, oPtr, p);
            }

            if (xBuf.ndim == 1)
                result = result.reshape({static_cast<py::ssize_t>(outputDim)});

            return Tensor::from_numpy(result);
        },
        py::arg("device"), py::arg("x"), py::arg("weights"),
        py::arg("bias") = py::none(),
        "GPU linear projection: output = x @ W^T + bias");

    // ── GPU linear forward (Tensor I/O — no numpy copies) ──────────────
    m.def(
        "linear_t",
        [](GrillyCoreContext& ctx, Tensor& x, Tensor& weights,
           std::optional<Tensor> bias) -> Tensor {
            auto& xShape = x.shape();
            auto& wShape = weights.shape();

            if (xShape.size() < 1 || xShape.size() > 3)
                throw std::runtime_error(
                    "x must be 1D, 2D, or 3D (batch, seq, input_dim)");
            if (wShape.size() != 2)
                throw std::runtime_error(
                    "weights must be 2D (output_dim, input_dim)");

            uint32_t inputDim = static_cast<uint32_t>(xShape.back());
            uint32_t outputDim = static_cast<uint32_t>(wShape[0]);
            uint32_t batchSeq = 1;
            for (size_t i = 0; i + 1 < xShape.size(); i++)
                batchSeq *= static_cast<uint32_t>(xShape[i]);
            if (xShape.size() == 1) batchSeq = 1;

            if (static_cast<uint32_t>(wShape[1]) != inputDim)
                throw std::runtime_error(
                    "Weight input_dim mismatch: " +
                    std::to_string(wShape[1]) + " vs " +
                    std::to_string(inputDim));

            // Direct CPU data access (no numpy allocation/copy)
            const float* xPtr = x.data();
            const float* wPtr = weights.data();

            const float* biasPtr = nullptr;
            uint32_t hasBias = 0;
            if (bias.has_value()) {
                biasPtr = bias->data();
                hasBias = 1;
            }

            grilly::ops::LinearParams p{batchSeq, inputDim, outputDim, hasBias};

            // Allocate output tensor (CPU-valid)
            std::vector<int64_t> outShape;
            for (size_t i = 0; i + 1 < xShape.size(); i++)
                outShape.push_back(xShape[i]);
            if (xShape.size() == 1) outShape.push_back(1);
            outShape.push_back(static_cast<int64_t>(outputDim));

            Tensor result(outShape);

            grilly::ops::linear(
                ctx.batch, ctx.pool, ctx.cache,
                xPtr, wPtr, biasPtr,
                result.mutable_data(), p);

            if (xShape.size() == 1)
                result = result.reshape(
                    {static_cast<int64_t>(outputDim)});

            return result;
        },
        py::arg("device"), py::arg("x"), py::arg("weights"),
        py::arg("bias") = py::none(),
        "GPU linear with Tensor I/O (avoids numpy allocation/copy overhead)");

    // ── CPU linear forward (verification) ────────────────────────────────
    m.def(
        "linear_cpu",
        [](py::array_t<float> x, py::array_t<float> weights,
           std::optional<py::array_t<float>> bias) -> Tensor {
            auto xBuf = x.request();
            auto wBuf = weights.request();

            auto [batchSeq, inputDim] = extractBatchAndLastDim(xBuf);
            uint32_t outputDim = static_cast<uint32_t>(wBuf.shape[0]);

            const float* biasPtr = nullptr;
            uint32_t hasBias = 0;
            if (bias.has_value()) {
                biasPtr = static_cast<const float*>(bias->request().ptr);
                hasBias = 1;
            }

            grilly::ops::LinearParams p{batchSeq, inputDim, outputDim, hasBias};
            std::vector<float> out = grilly::ops::linearCPU(
                static_cast<const float*>(xBuf.ptr),
                static_cast<const float*>(wBuf.ptr), biasPtr, p);

            std::vector<py::ssize_t> outShape;
            for (int i = 0; i < xBuf.ndim - 1; ++i)
                outShape.push_back(xBuf.shape[i]);
            if (xBuf.ndim == 1) outShape.push_back(1);
            outShape.push_back(outputDim);

            py::array_t<float> result(outShape);
            std::memcpy(result.request().ptr, out.data(),
                        out.size() * sizeof(float));

            if (xBuf.ndim == 1)
                result = result.reshape({static_cast<py::ssize_t>(outputDim)});

            return Tensor::from_numpy(result);
        },
        py::arg("x"), py::arg("weights"), py::arg("bias") = py::none(),
        "CPU linear projection using Eigen (for verification)");

    // ── GPU linear backward ──────────────────────────────────────────────
    m.def(
        "linear_backward",
        [](GrillyCoreContext& ctx,
           py::array_t<float> grad_output, py::array_t<float> input,
           py::array_t<float> weights) -> py::dict {
            auto gBuf = grad_output.request();
            auto iBuf = input.request();
            auto wBuf = weights.request();

            auto [batchSeq, outputDim] = extractBatchAndLastDim(gBuf);
            uint32_t inputDim = static_cast<uint32_t>(
                iBuf.shape[iBuf.ndim - 1]);

            grilly::ops::LinearParams p{batchSeq, inputDim, outputDim, 1};

            py::array_t<float> gradInput(iBuf.shape);
            py::array_t<float> gradWeight(wBuf.shape);
            py::array_t<float> gradBias(
                {static_cast<py::ssize_t>(outputDim)});

            grilly::ops::linearBackward(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(gBuf.ptr),
                static_cast<const float*>(iBuf.ptr),
                static_cast<const float*>(wBuf.ptr),
                static_cast<float*>(gradInput.request().ptr),
                static_cast<float*>(gradWeight.request().ptr),
                static_cast<float*>(gradBias.request().ptr), p);

            py::dict result;
            result["grad_input"] = Tensor::from_numpy(gradInput);
            result["grad_weight"] = Tensor::from_numpy(gradWeight);
            result["grad_bias"] = Tensor::from_numpy(gradBias);
            return result;
        },
        py::arg("device"), py::arg("grad_output"), py::arg("input"),
        py::arg("weights"),
        "GPU linear backward: grad_input, grad_weight, grad_bias");
}
