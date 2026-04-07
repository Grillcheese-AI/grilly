/// bindings_moqe_train.cpp — MoQE persistent-weight training ops.
///
/// Expert review fixes applied:
///   1. pool.upload is synchronous (memcpy to staging + vkCmdCopyBuffer + fence)
///      so weight_arrays lifetime is safe. Added waitIdle() as belt-and-suspenders.
///   2. backward_dx calls the C++ API function, not inline duplication.
///   3. GIL released during all GPU work.

#include "bindings_core.h"
#include "grilly/ops/moqe_train.h"

#include <vector>

using namespace grilly;

void register_moqe_train_ops(py::module_& m) {

    m.def("moqe_train_upload",
        [](GrillyCoreContext& ctx, py::list weight_arrays,
           uint32_t dModel, uint32_t nLayers, uint32_t maxSeq) -> int {

            std::vector<const float*> ptrs;
            std::vector<py::array_t<float>> arrays;
            for (auto& w : weight_arrays) {
                auto arr = w.cast<py::array_t<float>>();
                arrays.push_back(arr);
                auto wb = arr.request();
                require_c_contiguous_float(wb);
                ptrs.push_back(static_cast<const float*>(wb.ptr));
            }

            int handle = ops::moqe_train_upload(
                ctx.pool, ptrs, dModel, nLayers, maxSeq);

            // Belt-and-suspenders: ensure all DMA transfers complete
            // before Python GC can free the source numpy arrays.
            ctx.waitIdle();

            return handle;
        },
        py::arg("device"), py::arg("weights"),
        py::arg("d_model"), py::arg("n_layers"), py::arg("max_seq"),
        "Upload expert weights (W + W^T) to GPU permanently.");

    m.def("moqe_train_release",
        [](GrillyCoreContext& ctx, int handle) {
            ops::moqe_train_release(ctx.pool, handle);
        },
        py::arg("device"), py::arg("handle"),
        "Release all GPU buffers for a MoQE training cache.");

    m.def("moqe_train_update_expert",
        [](GrillyCoreContext& ctx, int handle,
           uint32_t layerIdx, int expertIdx, py::array_t<float> w) {
            auto& cache = ops::moqe_train_get_cache(handle);
            auto buf = w.request();
            require_c_contiguous_float(buf);
            {
                py::gil_scoped_release release;
                ops::moqe_train_update_expert(ctx.pool, cache, layerIdx, expertIdx,
                                               static_cast<const float*>(buf.ptr),
                                               cache.dModel);
                ctx.waitIdle();
            }
        },
        py::arg("device"), py::arg("handle"),
        py::arg("layer_idx"), py::arg("expert_idx"), py::arg("weights"));

    m.def("moqe_layer_forward",
        [](GrillyCoreContext& ctx, int handle, uint32_t layerIdx,
           py::array_t<float> x0, py::array_t<float> x1)
           -> std::pair<py::array_t<float>, py::array_t<float>> {

            auto& tc = ops::moqe_train_get_cache(handle);
            auto b0 = x0.request();
            auto b1 = x1.request();
            if (b0.size > 0)
                require_c_contiguous_float(b0);
            if (b1.size > 0)
                require_c_contiguous_float(b1);
            uint32_t n0 = static_cast<uint32_t>(b0.shape[0]);
            uint32_t n1 = static_cast<uint32_t>(b1.shape[0]);
            uint32_t d = tc.dModel;

            py::array_t<float> out0({(py::ssize_t)n0, (py::ssize_t)d});
            py::array_t<float> out1({(py::ssize_t)n1, (py::ssize_t)d});

            {
                py::gil_scoped_release release;
                ops::moqe_layer_forward_gpu(
                    ctx.batch, ctx.pool, ctx.cache, tc, layerIdx,
                    n0 > 0 ? static_cast<const float*>(b0.ptr) : nullptr, n0,
                    n1 > 0 ? static_cast<const float*>(b1.ptr) : nullptr, n1,
                    out0.mutable_data(), out1.mutable_data());
            }
            return {out0, out1};
        },
        py::arg("device"), py::arg("handle"), py::arg("layer_idx"),
        py::arg("x0"), py::arg("x1"),
        "Forward both experts, single GPU submit, barrier-free.");

    m.def("moqe_layer_backward_dx",
        [](GrillyCoreContext& ctx, int handle, uint32_t layerIdx,
           py::array_t<float> d0, py::array_t<float> d1)
           -> std::pair<py::array_t<float>, py::array_t<float>> {

            auto& tc = ops::moqe_train_get_cache(handle);
            auto b0 = d0.request();
            auto b1 = d1.request();
            if (b0.size > 0)
                require_c_contiguous_float(b0);
            if (b1.size > 0)
                require_c_contiguous_float(b1);
            uint32_t n0 = static_cast<uint32_t>(b0.shape[0]);
            uint32_t n1 = static_cast<uint32_t>(b1.shape[0]);
            uint32_t d = tc.dModel;

            py::array_t<float> dx0({(py::ssize_t)n0, (py::ssize_t)d});
            py::array_t<float> dx1({(py::ssize_t)n1, (py::ssize_t)d});

            {
                py::gil_scoped_release release;
                // Calls the C++ API — no inline duplication
                ops::moqe_layer_backward_dx_gpu(
                    ctx.batch, ctx.pool, ctx.cache, tc, layerIdx,
                    n0 > 0 ? static_cast<const float*>(b0.ptr) : nullptr, n0,
                    n1 > 0 ? static_cast<const float*>(b1.ptr) : nullptr, n1,
                    dx0.mutable_data(), dx1.mutable_data());
            }
            return {dx0, dx1};
        },
        py::arg("device"), py::arg("handle"), py::arg("layer_idx"),
        py::arg("d0"), py::arg("d1"),
        "Backward dx for both experts, single GPU submit, barrier-free.\n"
        "grad_W computed on CPU by caller (small nMasked).");
}
