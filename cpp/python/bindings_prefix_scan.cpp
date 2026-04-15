/// bindings_prefix_scan.cpp — Causal Linear-RNN prefix scan op bindings.
///
/// Exposes ``prefix_scan_causal`` and ``prefix_scan_causal_backward`` to
/// Python. Used by ``CausalSequenceMixer`` in the VSA-LM model to replace
/// the causally-leaky ``h.mean(dim=1)`` pooling with a strictly causal
/// subgroup-scanned Linear-RNN.

#include "bindings_core.h"
#include "grilly/ops/prefix_scan.h"

void register_prefix_scan_ops(py::module_& m) {
    using namespace grilly::ops;

    m.def(
        "prefix_scan_causal",
        [](GrillyCoreContext& ctx,
           py::array_t<float> x, py::array_t<float> a) -> py::array_t<float> {
            auto xBuf = x.request();
            auto aBuf = a.request();

            if (xBuf.ndim != 3 || aBuf.ndim != 3)
                throw std::runtime_error(
                    "prefix_scan_causal: x and a must be 3D (batch, seq, dim)");
            if (xBuf.shape != aBuf.shape)
                throw std::runtime_error(
                    "prefix_scan_causal: x and a must have identical shape");

            const uint32_t batchSize = static_cast<uint32_t>(xBuf.shape[0]);
            const uint32_t seqLen    = static_cast<uint32_t>(xBuf.shape[1]);
            const uint32_t hiddenDim = static_cast<uint32_t>(xBuf.shape[2]);

            PrefixScanParams p{seqLen, hiddenDim, batchSize};

            py::array_t<float> result(xBuf.shape);
            auto rBuf = result.request();

            {
                py::gil_scoped_release release;
                prefixScanCausal(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(xBuf.ptr),
                    static_cast<const float*>(aBuf.ptr),
                    static_cast<float*>(rBuf.ptr), p);
            }

            return result;
        },
        py::arg("device"), py::arg("x"), py::arg("a"),
        "Causal Linear-RNN forward: h_t = a_t * h_{t-1} + x_t via subgroup scan");

    m.def(
        "prefix_scan_causal_backward",
        [](GrillyCoreContext& ctx,
           py::array_t<float> grad_h, py::array_t<float> a,
           py::array_t<float> h, py::array_t<float> x) -> py::dict {
            auto dhBuf = grad_h.request();
            auto aBuf  = a.request();
            auto hBuf  = h.request();
            auto xBuf  = x.request();

            if (dhBuf.ndim != 3)
                throw std::runtime_error(
                    "prefix_scan_causal_backward: grad_h must be 3D");

            const uint32_t batchSize = static_cast<uint32_t>(dhBuf.shape[0]);
            const uint32_t seqLen    = static_cast<uint32_t>(dhBuf.shape[1]);
            const uint32_t hiddenDim = static_cast<uint32_t>(dhBuf.shape[2]);

            PrefixScanParams p{seqLen, hiddenDim, batchSize};

            py::array_t<float> gradX(dhBuf.shape);
            py::array_t<float> gradA(dhBuf.shape);

            // CRITICAL: get the raw pointers BEFORE releasing the GIL.
            // ``py::array::request()`` touches Python reference counts and
            // needs the GIL held — calling it inside a
            // ``gil_scoped_release`` block deadlocks on internal locks.
            // The forward binding gets this right; the first version of
            // this backward binding didn't and hung inside the dispatcher
            // with zero trace output from the C++ side (since the C++ call
            // was never reached — the lambda deadlocked on request()).
            const void* dhPtr = dhBuf.ptr;
            const void* aPtr  = aBuf.ptr;
            const void* hPtr  = hBuf.ptr;
            const void* xPtr  = xBuf.ptr;
            void* gradXPtr = gradX.request().ptr;
            void* gradAPtr = gradA.request().ptr;

            {
                py::gil_scoped_release release;
                prefixScanCausalBackward(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(dhPtr),
                    static_cast<const float*>(aPtr),
                    static_cast<const float*>(hPtr),
                    static_cast<const float*>(xPtr),
                    static_cast<float*>(gradXPtr),
                    static_cast<float*>(gradAPtr), p);
            }

            py::dict result;
            result["grad_x"] = gradX;
            result["grad_a"] = gradA;
            return result;
        },
        py::arg("device"), py::arg("grad_h"), py::arg("a"),
        py::arg("h"), py::arg("x"),
        "Causal prefix scan backward: returns grad_x and grad_a");
}
