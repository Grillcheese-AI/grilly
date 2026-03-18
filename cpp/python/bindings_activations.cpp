/// bindings_activations.cpp — Activation function bindings (forward + backward).
///
/// Migrated from monolithic bindings.cpp to use Tensor-based I/O.

#include "bindings_core.h"
#include "grilly/ops/activations.h"

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
                uint32_t total = 1;
                for (int i = 0; i < inBuf.ndim; ++i)
                    total *= static_cast<uint32_t>(inBuf.shape[i]);

                py::array_t<float> result(input.request().shape);
                auto rBuf = result.request();

                fn(ctx.batch, ctx.pool, ctx.cache,
                   static_cast<const float*>(inBuf.ptr),
                   static_cast<float*>(rBuf.ptr), total);

                return Tensor::from_numpy(result);
            },
            py::arg("device"), py::arg("input"));
    };

    defActivation("relu", grilly::ops::relu);
    defActivation("gelu", grilly::ops::gelu);
    defActivation("silu", grilly::ops::silu);
    defActivation("tanh_act", grilly::ops::tanh_act);

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
                uint32_t total = 1;
                for (int i = 0; i < gBuf.ndim; ++i)
                    total *= static_cast<uint32_t>(gBuf.shape[i]);

                py::array_t<float> result(gBuf.shape);
                auto rBuf = result.request();

                fn(ctx.batch, ctx.pool, ctx.cache,
                   static_cast<const float*>(gBuf.ptr),
                   static_cast<const float*>(iBuf.ptr),
                   static_cast<float*>(rBuf.ptr), total);

                return Tensor::from_numpy(result);
            },
            py::arg("device"), py::arg("grad_output"), py::arg("input"));
    };

    defActivationBackward("relu_backward", grilly::ops::reluBackward);
    defActivationBackward("gelu_backward", grilly::ops::geluBackward);
    defActivationBackward("silu_backward", grilly::ops::siluBackward);
    defActivationBackward("tanh_backward", grilly::ops::tanhBackward);
}
