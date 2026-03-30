/// bindings_perceiver.cpp — Perceiver IO cross-attention GPU binding.
///
/// Exposes perceiver_cross_attn_gpu(device, Q, K, V) → Tensor
/// Register-pinned Q, streaming K/V, online softmax. Zero LDS.

#include "bindings_core.h"
#include "grilly/ops/perceiver.h"

#include <cstring>
#include <vector>

using namespace grilly;

void register_perceiver_ops(py::module_& m) {
    using namespace grilly::nn;

    m.def("perceiver_cross_attn_gpu",
        [](GrillyCoreContext& ctx,
           py::array_t<float> Q_arr,
           py::array_t<float> K_arr,
           py::array_t<float> V_arr) -> Tensor {

            auto qBuf = Q_arr.request();
            auto kBuf = K_arr.request();
            auto vBuf = V_arr.request();

            // Q: (N, D), K: (M, D), V: (M, D)
            uint32_t seqN = static_cast<uint32_t>(qBuf.shape[0]);
            uint32_t headDim = static_cast<uint32_t>(qBuf.shape[qBuf.ndim - 1]);
            uint32_t seqM = static_cast<uint32_t>(kBuf.shape[0]);

            const float* qPtr = static_cast<const float*>(qBuf.ptr);
            const float* kPtr = static_cast<const float*>(kBuf.ptr);
            const float* vPtr = static_cast<const float*>(vBuf.ptr);

            // Output buffer
            std::vector<float> output(seqN * headDim);

            {
                // Release GIL during GPU work
                py::gil_scoped_release release;

                ops::perceiverEncode(ctx.batch, ctx.pool, ctx.cache,
                                     qPtr, kPtr, vPtr, output.data(),
                                     seqN, seqM, headDim);
            }

            // Create output numpy array
            py::array_t<float> out({(py::ssize_t)seqN, (py::ssize_t)headDim});
            std::memcpy(out.mutable_data(), output.data(),
                        output.size() * sizeof(float));
            return Tensor::from_numpy(out);
        },
        py::arg("device"), py::arg("Q"), py::arg("K"), py::arg("V"),
        "Perceiver IO cross-attention on GPU.\n\n"
        "1 thread = 1 latent token. Q pinned in registers, K/V streamed.\n"
        "Online softmax — O(1) VRAM regardless of input length M.\n\n"
        "Args:\n"
        "    device: GrillyCoreContext\n"
        "    Q: (N, D) float32 latent queries\n"
        "    K: (M, D) float32 input keys\n"
        "    V: (M, D) float32 input values\n"
        "Returns:\n"
        "    Tensor (N, D) — updated latents after cross-attention");
}
