/// bindings_activations.cpp — Activation function bindings (forward + backward).
///
/// Migrated from monolithic bindings.cpp to use Tensor-based I/O.

#include "bindings_core.h"
#include "grilly/ops/activations.h"
#include "grilly/ops/fused.h"

void register_activations_ops(py::module_& m) {
    using namespace grilly::nn;

    // ── Activation forward ops ───────────────────────────────────────────
    // Template: unary element-wise ops with signature (batch, pool, cache,
    // in*, out*, total).

    auto defActivation = [&m](const char* name, auto fn) {
        m.def(
            name,
            [fn](GrillyCoreContext& ctx,
                 py::array_t<float> input) -> Tensor {
                auto inBuf = input.request();
                require_c_contiguous_float(inBuf);
                uint32_t total = static_cast<uint32_t>(inBuf.size);

                py::array_t<float> result(input.request().shape);
                auto rBuf = result.request();

                {
                    py::gil_scoped_release release;
                    fn(ctx.batch, ctx.pool, ctx.cache,
                       static_cast<const float*>(inBuf.ptr),
                       static_cast<float*>(rBuf.ptr), total);
                }

                return Tensor::from_numpy(result);
            },
            py::arg("device"), py::arg("input"));
    };

    defActivation("relu", grilly::ops::relu);
    defActivation("gelu", grilly::ops::gelu);
    defActivation("silu", grilly::ops::silu);
    defActivation("tanh_act", grilly::ops::tanh_act);

    m.def(
        "mf_sigmoid",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input) -> Tensor {
            auto inBuf = input.request();
            require_c_contiguous_float(inBuf);
            uint32_t total = static_cast<uint32_t>(inBuf.size);

            py::array_t<float> result(input.request().shape);
            auto rBuf = result.request();

            {
                py::gil_scoped_release release;
                grilly::ops::mfSigmoid(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(inBuf.ptr),
                    static_cast<float*>(rBuf.ptr), total);
            }

            return Tensor::from_numpy(result);
        },
        py::arg("device"), py::arg("input"),
        "GPU rational sigmoid x/(1+|x|) (shader mf-sigmoid)");

    m.def(
        "mf_softplus",
        [](GrillyCoreContext& ctx, py::array_t<float> input,
           float beta) -> Tensor {
            auto inBuf = input.request();
            require_c_contiguous_float(inBuf);
            uint32_t total = static_cast<uint32_t>(inBuf.size);

            py::array_t<float> result(input.request().shape);
            auto rBuf = result.request();

            {
                py::gil_scoped_release release;
                grilly::ops::mfSoftplus(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(inBuf.ptr),
                    static_cast<float*>(rBuf.ptr), total, beta);
            }

            return Tensor::from_numpy(result);
        },
        py::arg("device"), py::arg("input"), py::arg("beta") = 1.0f,
        "GPU algebraic softplus 0.5*(x+sqrt(x*x+c)), c=4/beta^2 (shader "
        "mf-softplus)");

    // ── Activation backward ops ──────────────────────────────────────────
    // 3 buffers: grad_output, input (original forward input), grad_input.
    // Same push constant (uint total_elements) and dispatch pattern.

    auto defActivationBackward = [&m](const char* name, auto fn) {
        m.def(
            name,
            [fn](GrillyCoreContext& ctx,
                 py::array_t<float> grad_output,
                 py::array_t<float> input) -> Tensor {
                auto gBuf = grad_output.request();
                auto iBuf = input.request();
                require_c_contiguous_float(gBuf);
                require_c_contiguous_float(iBuf);
                uint32_t total = 1;
                for (int i = 0; i < gBuf.ndim; ++i)
                    total *= static_cast<uint32_t>(gBuf.shape[i]);

                py::array_t<float> result(gBuf.shape);
                auto rBuf = result.request();

                {
                    py::gil_scoped_release release;
                    fn(ctx.batch, ctx.pool, ctx.cache,
                       static_cast<const float*>(gBuf.ptr),
                       static_cast<const float*>(iBuf.ptr),
                       static_cast<float*>(rBuf.ptr), total);
                }

                return Tensor::from_numpy(result);
            },
            py::arg("device"), py::arg("grad_output"), py::arg("input"));
    };

    defActivationBackward("relu_backward", grilly::ops::reluBackward);
    defActivationBackward("gelu_backward", grilly::ops::geluBackward);
    defActivationBackward("silu_backward", grilly::ops::siluBackward);
    defActivationBackward("tanh_backward", grilly::ops::tanhBackward);

    // ── Fused transformer ops ─────────────────────────────────────────────

    m.def(
        "fused_mlp_gelu",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input,
           py::array_t<float> w1, py::array_t<float> b1,
           py::array_t<float> w2, py::array_t<float> b2) -> Tensor {
            auto inBuf  = input.request();
            auto w1Buf  = w1.request();
            auto b1Buf  = b1.request();
            auto w2Buf  = w2.request();
            auto b2Buf  = b2.request();

            uint32_t seqLen  = static_cast<uint32_t>(inBuf.shape[0]);
            uint32_t dIn     = static_cast<uint32_t>(inBuf.shape[1]);
            uint32_t dHidden = static_cast<uint32_t>(w1Buf.shape[0]);
            uint32_t dOut    = static_cast<uint32_t>(w2Buf.shape[0]);

            std::vector<py::ssize_t> outShape = {seqLen, dOut};
            py::array_t<float> result(outShape);
            auto rBuf = result.request();

            {
                py::gil_scoped_release release;
                grilly::ops::fusedMlpGelu(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(inBuf.ptr),
                    static_cast<const float*>(w1Buf.ptr),
                    static_cast<const float*>(b1Buf.ptr),
                    static_cast<const float*>(w2Buf.ptr),
                    static_cast<const float*>(b2Buf.ptr),
                    static_cast<float*>(rBuf.ptr),
                    seqLen, dIn, dHidden, dOut);
            }

            return Tensor::from_numpy(result);
        },
        py::arg("device"), py::arg("input"),
        py::arg("w1"), py::arg("b1"), py::arg("w2"), py::arg("b2"),
        "Fused MLP: fc1 -> GELU -> fc2 in one GPU dispatch (hidden in LDS)");

    m.def(
        "fused_layernorm_linear",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input,
           py::array_t<float> ln_weight, py::array_t<float> ln_bias,
           py::array_t<float> proj_weight, py::array_t<float> proj_bias) -> Tensor {
            auto inBuf = input.request();
            auto lnWBuf = ln_weight.request();
            auto lnBBuf = ln_bias.request();
            auto pWBuf = proj_weight.request();
            auto pBBuf = proj_bias.request();

            uint32_t seqLen = static_cast<uint32_t>(inBuf.shape[0]);
            uint32_t dIn    = static_cast<uint32_t>(inBuf.shape[1]);
            uint32_t dOut   = static_cast<uint32_t>(pWBuf.shape[0]);

            std::vector<py::ssize_t> outShape = {seqLen, dOut};
            py::array_t<float> result(outShape);
            auto rBuf = result.request();

            {
                py::gil_scoped_release release;
                grilly::ops::fusedLayernormLinear(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(inBuf.ptr),
                    static_cast<const float*>(lnWBuf.ptr),
                    static_cast<const float*>(lnBBuf.ptr),
                    static_cast<const float*>(pWBuf.ptr),
                    static_cast<const float*>(pBBuf.ptr),
                    static_cast<float*>(rBuf.ptr),
                    seqLen, dIn, dOut);
            }

            return Tensor::from_numpy(result);
        },
        py::arg("device"), py::arg("input"),
        py::arg("ln_weight"), py::arg("ln_bias"),
        py::arg("proj_weight"), py::arg("proj_bias"),
        "Fused LayerNorm + Linear in one GPU dispatch (normalized in LDS)");
}
