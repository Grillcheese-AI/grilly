/// bindings_perceiver.cpp — Perceiver IO GPU bindings.
///
/// Two APIs:
///   1. perceiver_cross_attn_gpu(device, Q, K, V) — single cross-attention dispatch
///   2. perceiver_upload_weights / perceiver_encode — native batched encoder
///      (single command buffer for the entire pipeline, zero Python round-trips)

#include "bindings_core.h"
#include "grilly/ops/perceiver.h"
#include "grilly/ops/perceiver_encoder.h"

#include <cstring>
#include <vector>

using namespace grilly;

void register_perceiver_ops(py::module_& m) {
    using namespace grilly::nn;

    // ── Single cross-attention dispatch ──────────────────────────────

    m.def("perceiver_cross_attn_gpu",
        [](GrillyCoreContext& ctx,
           py::array_t<float> Q_arr,
           py::array_t<float> K_arr,
           py::array_t<float> V_arr) -> Tensor {

            auto qBuf = Q_arr.request();
            auto kBuf = K_arr.request();
            auto vBuf = V_arr.request();

            uint32_t seqN = static_cast<uint32_t>(qBuf.shape[0]);
            uint32_t headDim = static_cast<uint32_t>(qBuf.shape[qBuf.ndim - 1]);
            uint32_t seqM = static_cast<uint32_t>(kBuf.shape[0]);

            const float* qPtr = static_cast<const float*>(qBuf.ptr);
            const float* kPtr = static_cast<const float*>(kBuf.ptr);
            const float* vPtr = static_cast<const float*>(vBuf.ptr);

            std::vector<float> output(seqN * headDim);

            {
                py::gil_scoped_release release;
                ops::perceiverEncode(ctx.batch, ctx.pool, ctx.cache,
                                     qPtr, kPtr, vPtr, output.data(),
                                     seqN, seqM, headDim);
            }

            py::array_t<float> out({(py::ssize_t)seqN, (py::ssize_t)headDim});
            std::memcpy(out.mutable_data(), output.data(),
                        output.size() * sizeof(float));
            return Tensor::from_numpy(out);
        },
        py::arg("device"), py::arg("Q"), py::arg("K"), py::arg("V"),
        "Perceiver IO cross-attention on GPU (single dispatch).");

    // ── Native Batched Perceiver Encoder ─────────────────────────────

    m.def("perceiver_upload_weights",
        [](GrillyCoreContext& ctx, py::list weights,
           uint32_t nLatents, uint32_t dModel,
           uint32_t nHeads, uint32_t nLayers,
           uint32_t maxPatches) -> int {

            // Convert py::list of numpy arrays to vector<vector<float>>
            std::vector<std::vector<float>> cpp_weights;
            cpp_weights.reserve(py::len(weights));
            for (auto& w : weights) {
                auto arr = w.cast<py::array_t<float>>();
                auto buf = arr.request();
                const float* ptr = static_cast<const float*>(buf.ptr);
                cpp_weights.emplace_back(ptr, ptr + buf.size);
            }

            return ops::perceiver_upload_weights(
                ctx.pool, cpp_weights,
                nLatents, dModel, nHeads, nLayers, maxPatches);
        },
        py::arg("device"), py::arg("weights"),
        py::arg("n_latents"), py::arg("d_model"),
        py::arg("n_heads"), py::arg("n_layers"),
        py::arg("max_patches"),
        "Upload Perceiver weights to GPU and pre-allocate all working VRAM.\n\n"
        "Weight order per layer: W_Q_cross, W_K_cross, W_V_cross, "
        "W_Q_self, W_K_self, W_V_self.\n"
        "Then: base_latents, position_embeddings.\n"
        "Returns handle for perceiver_encode().");

    m.def("perceiver_encode",
        [](GrillyCoreContext& ctx, int handle,
           py::array_t<float> patches) -> Tensor {

            auto& pc = ops::perceiver_get_cache(handle);
            auto pBuf = patches.request();

            uint32_t nPatches = static_cast<uint32_t>(pBuf.shape[0]);
            const float* patchPtr = static_cast<const float*>(pBuf.ptr);

            std::vector<float> result;
            {
                py::gil_scoped_release release;
                result = ops::perceiver_encode_native(
                    ctx.batch, ctx.pool, ctx.cache, pc,
                    patchPtr, nPatches);
            }

            uint32_t D = pc.dModel;
            py::array_t<float> out({(py::ssize_t)D});
            std::memcpy(out.mutable_data(), result.data(), D * sizeof(float));
            return Tensor::from_numpy(out);
        },
        py::arg("device"), py::arg("handle"), py::arg("patches"),
        "Encode patches through full Perceiver pipeline (single GPU submit).\n\n"
        "Records the entire N-layer dispatch graph into ONE command buffer.\n"
        "Zero Python round-trips during inference. GIL released during GPU work.\n\n"
        "Args:\n"
        "    device: GrillyCoreContext\n"
        "    handle: int from perceiver_upload_weights()\n"
        "    patches: (M, d_model) float32 input patches\n"
        "Returns:\n"
        "    Tensor (d_model,) — mean-pooled latent representation");
}
