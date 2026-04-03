/// bindings_normalization.cpp — Normalization op bindings.
///
/// Migrated from monolithic bindings.cpp to use Tensor-based I/O.

#include "bindings_core.h"
#include "grilly/ops/layernorm.h"
#include "grilly/ops/rmsnorm.h"
#include "grilly/ops/batchnorm.h"
#include "grilly/ops/activations.h"

void register_normalization_ops(py::module_& m) {
    using namespace grilly::nn;

    // ── LayerNorm forward ────────────────────────────────────────────────
    m.def(
        "layernorm",
        [](GrillyCoreContext& ctx, py::array_t<float> input,
           py::array_t<float> gamma, py::array_t<float> beta,
           float eps) -> Tensor {
            auto inBuf = input.request();
            auto gBuf = gamma.request();
            require_c_contiguous_float(inBuf);
            require_c_contiguous_float(gBuf);

            if (inBuf.ndim < 2)
                throw std::runtime_error("input must be at least 2D");

            uint32_t features = static_cast<uint32_t>(
                inBuf.shape[inBuf.ndim - 1]);
            uint32_t totalBatch = 1;
            for (int i = 0; i < inBuf.ndim - 1; ++i)
                totalBatch *= static_cast<uint32_t>(inBuf.shape[i]);

            py::array_t<float> result(inBuf.shape);
            auto rBuf = result.request();
            auto bBuf = beta.request();
            require_c_contiguous_float(bBuf);

            {
                py::gil_scoped_release release;
                grilly::ops::layernorm(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(inBuf.ptr),
                    static_cast<float*>(rBuf.ptr),
                    static_cast<const float*>(gBuf.ptr),
                    static_cast<const float*>(bBuf.ptr),
                    1, totalBatch, features, eps);
            }

            return Tensor::from_numpy(result);
        },
        py::arg("device"), py::arg("input"), py::arg("gamma"),
        py::arg("beta"), py::arg("eps") = 1e-5f,
        "GPU LayerNorm: gamma * (x - mean) / sqrt(var + eps) + beta");

    // ── LayerNorm backward ───────────────────────────────────────────────
    m.def(
        "layernorm_backward",
        [](GrillyCoreContext& ctx,
           py::array_t<float> grad_output, py::array_t<float> input,
           py::array_t<float> gamma, py::array_t<float> mean,
           py::array_t<float> var, float eps) -> py::dict {
            auto goBuf = grad_output.request();
            auto inBuf = input.request();
            auto gBuf  = gamma.request();
            require_c_contiguous_float(goBuf);
            require_c_contiguous_float(inBuf);
            require_c_contiguous_float(gBuf);

            if (inBuf.ndim < 2)
                throw std::runtime_error("input must be at least 2D");

            uint32_t features = static_cast<uint32_t>(
                inBuf.shape[inBuf.ndim - 1]);
            uint32_t totalBatch = 1;
            for (int i = 0; i < inBuf.ndim - 1; ++i)
                totalBatch *= static_cast<uint32_t>(inBuf.shape[i]);

            py::array_t<float> gradInput(inBuf.shape);
            py::array_t<float> gradGamma(
                {static_cast<py::ssize_t>(features)});
            py::array_t<float> gradBeta(
                {static_cast<py::ssize_t>(features)});
            auto meanBuf = mean.request();
            auto varBuf = var.request();
            auto giBuf = gradInput.request();
            auto ggBuf = gradGamma.request();
            auto gbBuf = gradBeta.request();
            require_c_contiguous_float(meanBuf);
            require_c_contiguous_float(varBuf);

            {
                py::gil_scoped_release release;
                grilly::ops::layernormBackward(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(goBuf.ptr),
                    static_cast<const float*>(inBuf.ptr),
                    static_cast<const float*>(gBuf.ptr),
                    static_cast<const float*>(meanBuf.ptr),
                    static_cast<const float*>(varBuf.ptr),
                    static_cast<float*>(giBuf.ptr),
                    static_cast<float*>(ggBuf.ptr),
                    static_cast<float*>(gbBuf.ptr),
                    1, totalBatch, features, eps);
            }

            py::dict result;
            result["grad_input"] = Tensor::from_numpy(gradInput);
            result["grad_gamma"] = Tensor::from_numpy(gradGamma);
            result["grad_beta"] = Tensor::from_numpy(gradBeta);
            return result;
        },
        py::arg("device"), py::arg("grad_output"), py::arg("input"),
        py::arg("gamma"), py::arg("mean"), py::arg("var"),
        py::arg("eps") = 1e-5f,
        "GPU LayerNorm backward: computes grad_input, grad_gamma, "
        "grad_beta");

    // ── RMSNorm ──────────────────────────────────────────────────────────
    m.def(
        "rmsnorm",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input,
           py::array_t<float> weight,
           float eps) -> Tensor {
            auto inBuf = input.request();
            require_c_contiguous_float(inBuf);

            uint32_t features, batchSize, seqLen;
            if (inBuf.ndim == 1) {
                batchSize = 1; seqLen = 1;
                features = static_cast<uint32_t>(inBuf.shape[0]);
            } else if (inBuf.ndim == 2) {
                batchSize = static_cast<uint32_t>(inBuf.shape[0]);
                seqLen = 1;
                features = static_cast<uint32_t>(inBuf.shape[1]);
            } else if (inBuf.ndim == 3) {
                batchSize = static_cast<uint32_t>(inBuf.shape[0]);
                seqLen = static_cast<uint32_t>(inBuf.shape[1]);
                features = static_cast<uint32_t>(inBuf.shape[2]);
            } else {
                throw std::runtime_error(
                    "RMSNorm input must be 1D, 2D, or 3D");
            }

            py::array_t<float> result(inBuf.shape);
            auto rBuf = result.request();
            auto wBuf = weight.request();
            require_c_contiguous_float(wBuf);

            {
                py::gil_scoped_release release;
                grilly::ops::rmsnorm(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(inBuf.ptr),
                    static_cast<float*>(rBuf.ptr),
                    static_cast<const float*>(wBuf.ptr),
                    batchSize, seqLen, features, eps);
            }

            return Tensor::from_numpy(result);
        },
        py::arg("device"), py::arg("input"),
        py::arg("weight"), py::arg("eps") = 1e-5f,
        "GPU RMSNorm: weight * x * rsqrt(mean(x^2) + eps)");

    // ── BatchNorm2d forward ──────────────────────────────────────────────
    m.def(
        "batchnorm2d_forward",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input,
           py::array_t<float> gamma, py::array_t<float> beta,
           py::array_t<float> running_mean, py::array_t<float> running_var,
           float eps, float momentum, bool training) -> py::dict {
            auto inBuf = input.request();
            require_c_contiguous_float(inBuf);
            if (inBuf.ndim != 4)
                throw std::runtime_error("input must be 4D (B, C, H, W)");

            uint32_t B = static_cast<uint32_t>(inBuf.shape[0]);
            uint32_t C = static_cast<uint32_t>(inBuf.shape[1]);
            uint32_t H = static_cast<uint32_t>(inBuf.shape[2]);
            uint32_t W = static_cast<uint32_t>(inBuf.shape[3]);

            py::array_t<float> output(inBuf.shape);
            py::array_t<float> rmOut(running_mean.request().shape);
            py::array_t<float> rvOut(running_var.request().shape);
            py::array_t<float> bMean({static_cast<py::ssize_t>(C)});
            py::array_t<float> bVar({static_cast<py::ssize_t>(C)});

            std::memcpy(rmOut.mutable_data(), running_mean.data(),
                        C * sizeof(float));
            std::memcpy(rvOut.mutable_data(), running_var.data(),
                        C * sizeof(float));

            grilly::ops::BatchNorm2dForwardParams p{
                B, C, H, W, eps, momentum, training ? 1u : 0u, 1u};
            auto outBuf = output.request();
            auto gaBuf = gamma.request();
            auto beBuf = beta.request();
            require_c_contiguous_float(gaBuf);
            require_c_contiguous_float(beBuf);

            {
                py::gil_scoped_release release;
                grilly::ops::batchnorm2dForward(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(inBuf.ptr),
                    static_cast<float*>(outBuf.ptr),
                    static_cast<const float*>(gaBuf.ptr),
                    static_cast<const float*>(beBuf.ptr),
                    static_cast<float*>(rmOut.mutable_data()),
                    static_cast<float*>(rvOut.mutable_data()),
                    static_cast<float*>(bMean.mutable_data()),
                    static_cast<float*>(bVar.mutable_data()), p);
            }

            py::dict result;
            result["output"] = Tensor::from_numpy(output);
            result["running_mean"] = Tensor::from_numpy(rmOut);
            result["running_var"] = Tensor::from_numpy(rvOut);
            result["batch_mean"] = Tensor::from_numpy(bMean);
            result["batch_var"] = Tensor::from_numpy(bVar);
            return result;
        },
        py::arg("device"), py::arg("input"),
        py::arg("gamma"), py::arg("beta"),
        py::arg("running_mean"), py::arg("running_var"),
        py::arg("eps") = 1e-5f, py::arg("momentum") = 0.1f,
        py::arg("training") = true,
        "GPU BatchNorm2d forward");

    // ── Softmax ──────────────────────────────────────────────────────────
    m.def(
        "softmax",
        [](GrillyCoreContext& ctx, py::array_t<float> input,
           int dim) -> Tensor {
            auto inBuf = input.request();
            require_c_contiguous_float(inBuf);
            if (inBuf.ndim < 1)
                throw std::runtime_error("input must be at least 1D");

            uint32_t features = static_cast<uint32_t>(
                inBuf.shape[inBuf.ndim - 1]);
            uint32_t totalBatch = 1;
            for (int i = 0; i < inBuf.ndim - 1; ++i)
                totalBatch *= static_cast<uint32_t>(inBuf.shape[i]);
            if (inBuf.ndim == 1) totalBatch = 1;

            py::array_t<float> result(inBuf.shape);
            auto rBuf = result.request();
            {
                py::gil_scoped_release release;
                grilly::ops::softmax(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(inBuf.ptr),
                    static_cast<float*>(rBuf.ptr),
                    1, totalBatch, features);
            }
            return Tensor::from_numpy(result);
        },
        py::arg("device"), py::arg("input"), py::arg("dim") = -1,
        "GPU Softmax (3-pass: max, sum_exp, normalize)");

    // ── Softmax backward ─────────────────────────────────────────────────
    m.def(
        "softmax_backward",
        [](GrillyCoreContext& ctx,
           py::array_t<float> grad_output,
           py::array_t<float> softmax_output) -> Tensor {
            auto gBuf = grad_output.request();
            require_c_contiguous_float(gBuf);
            uint32_t numClasses = static_cast<uint32_t>(
                gBuf.shape[gBuf.ndim - 1]);
            uint32_t batchSeq = 1;
            for (int i = 0; i < gBuf.ndim - 1; ++i)
                batchSeq *= static_cast<uint32_t>(gBuf.shape[i]);
            if (gBuf.ndim == 1) batchSeq = 1;

            py::array_t<float> result(gBuf.shape);
            auto sBuf = softmax_output.request();
            auto rBuf = result.request();
            require_c_contiguous_float(sBuf);

            {
                py::gil_scoped_release release;
                grilly::ops::softmaxBackward(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(gBuf.ptr),
                    static_cast<const float*>(sBuf.ptr),
                    static_cast<float*>(rBuf.ptr),
                    1, batchSeq, numClasses);
            }
            return Tensor::from_numpy(result);
        },
        py::arg("device"), py::arg("grad_output"),
        py::arg("softmax_output"),
        "GPU Softmax backward");
}
