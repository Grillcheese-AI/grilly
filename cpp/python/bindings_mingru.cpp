/// bindings_mingru.cpp — MinGRU fused forward/backward bindings.
///
/// Exposes ``mingru_forward`` and ``mingru_backward`` to Python.
/// Fuses G, V, D projections activations and causal scan.

#include "bindings_core.h"
#include "grilly/ops/mingru.h"

void register_mingru_ops(py::module_& m) {
    using namespace grilly::ops;

    m.def(
        "mingru_forward",
        [](GrillyCoreContext& ctx,
           py::array_t<float> g, py::array_t<float> v, py::array_t<float> d) -> py::array_t<float> {
            auto gBuf = g.request();
            auto vBuf = v.request();
            auto dBuf = d.request();

            if (gBuf.ndim != 3 || vBuf.ndim != 3 || dBuf.ndim != 3)
                throw std::runtime_error("mingru_forward: inputs must be 3D");

            const uint32_t batchSize = static_cast<uint32_t>(gBuf.shape[0]);
            const uint32_t seqLen    = static_cast<uint32_t>(gBuf.shape[1]);
            const uint32_t hiddenDim = static_cast<uint32_t>(gBuf.shape[2]);

            MinGruParams p{batchSize, seqLen, hiddenDim};

            py::array_t<float> result(gBuf.shape);
            auto rBuf = result.request();

            {
                py::gil_scoped_release release;
                minGruForward(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(gBuf.ptr),
                    static_cast<const float*>(vBuf.ptr),
                    static_cast<const float*>(dBuf.ptr),
                    static_cast<float*>(rBuf.ptr), p);
            }

            return result;
        },
        py::arg("device"), py::arg("g"), py::arg("v"), py::arg("d"),
        "Fused MinGRU forward: fuses activations and causal scan");

    m.def(
        "mingru_backward",
        [](GrillyCoreContext& ctx,
           py::array_t<float> grad_h, py::array_t<float> g,
           py::array_t<float> v, py::array_t<float> d,
           py::array_t<float> h) -> py::dict {
            auto dhBuf = grad_h.request();
            auto gBuf  = g.request();
            auto vBuf  = v.request();
            auto dBuf  = d.request();
            auto hBuf  = h.request();

            const uint32_t batchSize = static_cast<uint32_t>(dhBuf.shape[0]);
            const uint32_t seqLen    = static_cast<uint32_t>(dhBuf.shape[1]);
            const uint32_t hiddenDim = static_cast<uint32_t>(dhBuf.shape[2]);

            MinGruParams p{batchSize, seqLen, hiddenDim};

            py::array_t<float> gradG(dhBuf.shape);
            py::array_t<float> gradV(dhBuf.shape);
            py::array_t<float> gradD(dhBuf.shape);

            const void* dhPtr = dhBuf.ptr;
            const void* gPtr  = gBuf.ptr;
            const void* vPtr  = vBuf.ptr;
            const void* dPtr  = dBuf.ptr;
            const void* hPtr  = hBuf.ptr;
            
            void* ggPtr = gradG.request().ptr;
            void* gvPtr = gradV.request().ptr;
            void* gdPtr = gradD.request().ptr;

            {
                py::gil_scoped_release release;
                minGruBackward(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(dhPtr),
                    static_cast<const float*>(gPtr),
                    static_cast<const float*>(vPtr),
                    static_cast<const float*>(dPtr),
                    static_cast<const float*>(hPtr),
                    static_cast<float*>(ggPtr),
                    static_cast<float*>(gvPtr),
                    static_cast<float*>(gdPtr), p);
            }

            py::dict res;
            res["grad_g"] = gradG;
            res["grad_v"] = gradV;
            res["grad_d"] = gradD;
            return res;
        },
        py::arg("device"), py::arg("grad_h"), py::arg("g"),
        py::arg("v"), py::arg("d"), py::arg("h"),
        "Fused MinGRU backward: returns grad_g, grad_v, grad_d");
}
