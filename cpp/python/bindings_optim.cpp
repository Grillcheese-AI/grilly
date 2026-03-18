/// bindings_optim.cpp — Standalone optimizer GPU step op bindings.
///
/// These are the raw GPU dispatch ops (adam_update, adamw_update), not the
/// nn::Optimizer classes (which are in bindings_core.cpp).
///
/// Migrated from monolithic bindings.cpp to use Tensor-based I/O.

#include "bindings_core.h"
#include "grilly/ops/optimizer.h"

void register_optim_ops(py::module_& m) {
    using namespace grilly::nn;

    // ── Adam update ──────────────────────────────────────────────────────
    m.def(
        "adam_update",
        [](GrillyCoreContext& ctx,
           py::array_t<float> weights, py::array_t<float> grad,
           py::array_t<float> m_state, py::array_t<float> v_state,
           float lr, float beta1, float beta2, float eps,
           float beta1_t, float beta2_t, bool clear_grad) -> py::dict {
            auto wBuf = weights.request();
            uint32_t total = 1;
            for (int i = 0; i < wBuf.ndim; ++i)
                total *= static_cast<uint32_t>(wBuf.shape[i]);

            py::array_t<float> wOut(wBuf.shape);
            py::array_t<float> gOut(wBuf.shape);
            py::array_t<float> mOut(wBuf.shape);
            py::array_t<float> vOut(wBuf.shape);

            std::memcpy(wOut.mutable_data(), wBuf.ptr,
                        total * sizeof(float));
            std::memcpy(gOut.mutable_data(), grad.data(),
                        total * sizeof(float));
            std::memcpy(mOut.mutable_data(), m_state.data(),
                        total * sizeof(float));
            std::memcpy(vOut.mutable_data(), v_state.data(),
                        total * sizeof(float));

            grilly::ops::AdamParams p{total, lr, beta1, beta2, eps,
                                      beta1_t, beta2_t,
                                      clear_grad ? 1u : 0u};
            grilly::ops::adamUpdate(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<float*>(wOut.mutable_data()),
                static_cast<float*>(gOut.mutable_data()),
                static_cast<float*>(mOut.mutable_data()),
                static_cast<float*>(vOut.mutable_data()), p);

            py::dict result;
            result["weights"] = Tensor::from_numpy(wOut);
            result["grad"] = Tensor::from_numpy(gOut);
            result["m"] = Tensor::from_numpy(mOut);
            result["v"] = Tensor::from_numpy(vOut);
            return result;
        },
        py::arg("device"), py::arg("weights"), py::arg("grad"),
        py::arg("m"), py::arg("v"),
        py::arg("lr") = 1e-3f, py::arg("beta1") = 0.9f,
        py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f,
        py::arg("beta1_t") = 0.9f, py::arg("beta2_t") = 0.999f,
        py::arg("clear_grad") = false,
        "GPU Adam optimizer step");

    // ── AdamW update ─────────────────────────────────────────────────────
    m.def(
        "adamw_update",
        [](GrillyCoreContext& ctx,
           py::array_t<float> weights, py::array_t<float> grad,
           py::array_t<float> m_state, py::array_t<float> v_state,
           float lr, float beta1, float beta2, float eps,
           float weight_decay,
           float beta1_t, float beta2_t, bool clear_grad) -> py::dict {
            auto wBuf = weights.request();
            uint32_t total = 1;
            for (int i = 0; i < wBuf.ndim; ++i)
                total *= static_cast<uint32_t>(wBuf.shape[i]);

            py::array_t<float> wOut(wBuf.shape);
            py::array_t<float> gOut(wBuf.shape);
            py::array_t<float> mOut(wBuf.shape);
            py::array_t<float> vOut(wBuf.shape);

            std::memcpy(wOut.mutable_data(), wBuf.ptr,
                        total * sizeof(float));
            std::memcpy(gOut.mutable_data(), grad.data(),
                        total * sizeof(float));
            std::memcpy(mOut.mutable_data(), m_state.data(),
                        total * sizeof(float));
            std::memcpy(vOut.mutable_data(), v_state.data(),
                        total * sizeof(float));

            grilly::ops::AdamWParams p{total, lr, beta1, beta2, eps,
                                       weight_decay, beta1_t, beta2_t,
                                       clear_grad ? 1u : 0u};
            grilly::ops::adamwUpdate(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<float*>(wOut.mutable_data()),
                static_cast<float*>(gOut.mutable_data()),
                static_cast<float*>(mOut.mutable_data()),
                static_cast<float*>(vOut.mutable_data()), p);

            py::dict result;
            result["weights"] = Tensor::from_numpy(wOut);
            result["grad"] = Tensor::from_numpy(gOut);
            result["m"] = Tensor::from_numpy(mOut);
            result["v"] = Tensor::from_numpy(vOut);
            return result;
        },
        py::arg("device"), py::arg("weights"), py::arg("grad"),
        py::arg("m"), py::arg("v"),
        py::arg("lr") = 1e-3f, py::arg("beta1") = 0.9f,
        py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f,
        py::arg("weight_decay") = 0.01f,
        py::arg("beta1_t") = 0.9f, py::arg("beta2_t") = 0.999f,
        py::arg("clear_grad") = false,
        "GPU AdamW optimizer step");
}
