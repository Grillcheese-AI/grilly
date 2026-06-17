/// bindings_linear.cpp — Linear layer op bindings (forward + backward).
///
/// Migrated from monolithic bindings.cpp to use Tensor-based I/O.

#include "bindings_core.h"
#include "grilly/ops/linear.h"

void register_linear_ops(py::module_& m) {
    using namespace grilly::nn;

    // ── GPU linear forward (fp32 + fp16) ───────────────────────────────
    //
    // Accepts generic numpy arrays — inspects ``itemsize`` to detect fp32
    // (itemsize=4) vs fp16 (itemsize=2). Input and weights must share the
    // same dtype; bias is always fp32 (stability + broadcast simplicity).
    // Output is always fp32 regardless of input dtype because the
    // cooperative-matrix accumulator runs in fp32 — callers needing fp16
    // output can ``astype(np.float16)`` on the returned array.
    m.def(
        "linear",
        [](GrillyCoreContext& ctx, py::array x, py::array weights,
           std::optional<py::array> bias) -> py::array {
            auto xBuf = x.request();
            auto wBuf = weights.request();

            if (xBuf.ndim < 1 || xBuf.ndim > 3)
                throw std::runtime_error(
                    "x must be 1D, 2D, or 3D (batch, seq, input_dim)");
            if (wBuf.ndim != 2)
                throw std::runtime_error(
                    "weights must be 2D (output_dim, input_dim)");

            if (xBuf.itemsize != 2 && xBuf.itemsize != 4)
                throw std::runtime_error(
                    "grilly linear: x must be fp32 (itemsize=4) or fp16 (itemsize=2)");
            if (xBuf.itemsize != wBuf.itemsize)
                throw std::runtime_error(
                    "grilly linear: x and weights must share the same dtype");

            const uint32_t elemSize = static_cast<uint32_t>(xBuf.itemsize);

            auto [batchSeq, inputDim] = extractBatchAndLastDim(xBuf);
            uint32_t outputDim = static_cast<uint32_t>(wBuf.shape[0]);

            if (static_cast<uint32_t>(wBuf.shape[1]) != inputDim)
                throw std::runtime_error(
                    "Weight input_dim mismatch: " +
                    std::to_string(wBuf.shape[1]) + " vs " +
                    std::to_string(inputDim));

            // Bias is always fp32. If the caller passed fp16 bias, we
            // require them to upcast it on the Python side — simpler and
            // matches the C++ contract.
            const void* biasPtr = nullptr;
            uint32_t hasBias = 0;
            if (bias.has_value()) {
                auto bBuf = bias->request();
                if (bBuf.itemsize != 4)
                    throw std::runtime_error(
                        "grilly linear: bias must be fp32 (cast via .astype(np.float32))");
                if (bBuf.ndim != 1 ||
                    static_cast<uint32_t>(bBuf.shape[0]) != outputDim)
                    throw std::runtime_error(
                        "bias must be 1D with size output_dim");
                biasPtr = bBuf.ptr;
                hasBias = 1;
            }

            grilly::ops::LinearParams p{
                batchSeq, inputDim, outputDim, hasBias, elemSize};

            std::vector<py::ssize_t> outShape;
            for (int i = 0; i < xBuf.ndim - 1; ++i)
                outShape.push_back(xBuf.shape[i]);
            if (xBuf.ndim == 1) outShape.push_back(1);
            outShape.push_back(outputDim);

            // Output is always fp32 — format code 'f'. The coopmat shader
            // accumulates in fp32, and so does fnn-linear.
            py::array result(py::dtype("f"), outShape);
            auto rBuf = result.request();

            const void* xPtr = xBuf.ptr;
            const void* wPtr = wBuf.ptr;
            void* oPtr = rBuf.ptr;

            {
                py::gil_scoped_release release;
                grilly::ops::linear(
                    ctx.batch, ctx.pool, ctx.cache,
                    xPtr, wPtr, biasPtr, oPtr, p);
            }

            if (xBuf.ndim == 1)
                result = result.reshape({static_cast<py::ssize_t>(outputDim)});

            return result;
        },
        py::arg("device"), py::arg("x"), py::arg("weights"),
        py::arg("bias") = py::none(),
        "GPU linear projection (fp32 or fp16 input; output always fp32)");

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

            // ``linear_t`` is the native C++ Tensor path — always fp32.
            grilly::ops::LinearParams p{batchSeq, inputDim, outputDim,
                                         hasBias, 4u};

            // Allocate output tensor (CPU-valid)
            std::vector<int64_t> outShape;
            for (size_t i = 0; i + 1 < xShape.size(); i++)
                outShape.push_back(xShape[i]);
            if (xShape.size() == 1) outShape.push_back(1);
            outShape.push_back(static_cast<int64_t>(outputDim));

            Tensor result(outShape);
            float* oPtr = result.mutable_data();

            {
                py::gil_scoped_release release;
                grilly::ops::linear(
                    ctx.batch, ctx.pool, ctx.cache,
                    xPtr, wPtr, biasPtr, oPtr, p);
            }

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

            // ``linear_cpu`` is the Eigen reference path — always fp32.
            grilly::ops::LinearParams p{batchSeq, inputDim, outputDim,
                                         hasBias, 4u};
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

    // ── GPU linear backward (fp32 only for now; interface is fp16-ready) ──
    //
    // Returns a py::tuple (grad_input, grad_weight, grad_bias). Tuples are
    // drastically cheaper than dicts to construct/destruct on every backward
    // step (no string hashing, no dict allocation).
    m.def(
        "linear_backward",
        [](GrillyCoreContext& ctx,
           py::array grad_output, py::array input,
           py::array weights) -> py::tuple {
            const auto gItem = grad_output.itemsize();
            const auto iItem = input.itemsize();
            const auto wItem = weights.itemsize();

            if (gItem != 2 && gItem != 4)
                throw std::runtime_error(
                    "grilly linear_backward: grad_output must be fp32 or fp16");
            if (gItem != iItem || gItem != wItem)
                throw std::runtime_error(
                    "grilly linear_backward: grad_output, input, weights "
                    "must share the same dtype");

            const int gNdim = grad_output.ndim();
            const int iNdim = input.ndim();
            if (gNdim < 1 || iNdim < 1)
                throw std::runtime_error(
                    "grilly linear_backward: inputs must have rank >= 1");

            const uint32_t elemSize = static_cast<uint32_t>(gItem);
            const char npFormat = (elemSize == 2) ? 'e' : 'f';

            // Flatten (batch, [seq,] out) -> batchSeq × outputDim.
            uint32_t batchSeq = 1;
            for (int i = 0; i < gNdim - 1; ++i)
                batchSeq *= static_cast<uint32_t>(grad_output.shape(i));
            if (gNdim == 1) batchSeq = 1;
            const uint32_t outputDim =
                static_cast<uint32_t>(grad_output.shape(gNdim - 1));
            const uint32_t inputDim =
                static_cast<uint32_t>(input.shape(iNdim - 1));

            grilly::ops::LinearParams p{
                batchSeq, inputDim, outputDim, 1u, elemSize};

            // Build output shape vectors directly from the array API (no
            // buffer_info allocation).
            std::vector<py::ssize_t> inputShape(iNdim);
            for (int i = 0; i < iNdim; ++i)
                inputShape[i] = input.shape(i);
            std::vector<py::ssize_t> weightShape(weights.ndim());
            for (int i = 0; i < weights.ndim(); ++i)
                weightShape[i] = weights.shape(i);
            std::vector<py::ssize_t> biasShape = {
                static_cast<py::ssize_t>(outputDim)};

            py::array gradInput(py::dtype(std::string(1, npFormat)),
                                inputShape);
            py::array gradWeight(py::dtype(std::string(1, npFormat)),
                                 weightShape);
            py::array gradBias(py::dtype(std::string(1, npFormat)),
                               biasShape);

            // Cache raw pointers BEFORE releasing the GIL — mutable_data()
            // / data() are Python-API-adjacent and must be called under GIL.
            const void* gPtr  = grad_output.data();
            const void* iPtr  = input.data();
            const void* wPtr  = weights.data();
            void* gInPtr  = gradInput.mutable_data();
            void* gWPtr   = gradWeight.mutable_data();
            void* gBPtr   = gradBias.mutable_data();

            {
                py::gil_scoped_release release;
                grilly::ops::linearBackward(
                    ctx.batch, ctx.pool, ctx.cache,
                    gPtr, iPtr, wPtr,
                    gInPtr, gWPtr, gBPtr, p);
            }

            return py::make_tuple(gradInput, gradWeight, gradBias);
        },
        py::arg("device"), py::arg("grad_output"), py::arg("input"),
        py::arg("weights"),
        "GPU linear backward — returns (grad_input, grad_weight, grad_bias).");

    // â”€â”€ GPU linear backward via fp16 coopmat (fp32 in/out) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    //
    // Mirrors linear_backward but routes grad_input / grad_weight through the
    // fp16 cooperative-matrix kernel. Inputs and outputs are fp32. Caller MUST
    // pass coopmat-aligned shapes: batchSeq%16, outputDim%16, inputDim%64.
    // For parity testing the fp16 path against the fp32 linear_backward.
    m.def(
        "linear_backward_coopmat",
        [](GrillyCoreContext& ctx,
           py::array_t<float> grad_output, py::array_t<float> input,
           py::array_t<float> weights) -> py::tuple {
            const int gNdim = grad_output.ndim();
            const int iNdim = input.ndim();
            if (gNdim < 1 || iNdim < 1)
                throw std::runtime_error(
                    "linear_backward_coopmat: inputs must have rank >= 1");

            uint32_t batchSeq = 1;
            for (int i = 0; i < gNdim - 1; ++i)
                batchSeq *= static_cast<uint32_t>(grad_output.shape(i));
            if (gNdim == 1) batchSeq = 1;
            const uint32_t outputDim =
                static_cast<uint32_t>(grad_output.shape(gNdim - 1));
            const uint32_t inputDim =
                static_cast<uint32_t>(input.shape(iNdim - 1));

            if (batchSeq % 16u || outputDim % 16u || inputDim % 64u)
                throw std::runtime_error(
                    "linear_backward_coopmat: shape not coopmat-aligned "
                    "(need batchSeq%16, outputDim%16, inputDim%64)");

            grilly::ops::LinearParams p{
                batchSeq, inputDim, outputDim, 1u, 4u};

            std::vector<py::ssize_t> inputShape(iNdim);
            for (int i = 0; i < iNdim; ++i)
                inputShape[i] = input.shape(i);
            std::vector<py::ssize_t> weightShape(weights.ndim());
            for (int i = 0; i < weights.ndim(); ++i)
                weightShape[i] = weights.shape(i);
            std::vector<py::ssize_t> biasShape = {
                static_cast<py::ssize_t>(outputDim)};

            py::array_t<float> gradInput(inputShape);
            py::array_t<float> gradWeight(weightShape);
            py::array_t<float> gradBias(biasShape);

            const void* gPtr = grad_output.data();
            const void* iPtr = input.data();
            const void* wPtr = weights.data();
            void* gInPtr = gradInput.mutable_data();
            void* gWPtr  = gradWeight.mutable_data();
            void* gBPtr  = gradBias.mutable_data();

            {
                py::gil_scoped_release release;
                grilly::ops::linearBackwardCoopmat(
                    ctx.batch, ctx.pool, ctx.cache,
                    gPtr, iPtr, wPtr,
                    gInPtr, gWPtr, gBPtr, p);
            }

            return py::make_tuple(gradInput, gradWeight, gradBias);
        },
        py::arg("device"), py::arg("grad_output"), py::arg("input"),
        py::arg("weights"),
        "fp16-coopmat linear backward (fp32 in/out) for parity testing.");
}
