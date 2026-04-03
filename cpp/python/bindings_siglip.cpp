/// bindings_siglip.cpp — SigLIP2 full encoder: persistent weights + batched dispatch.
///
/// Optimized:
///   1. GIL released during GPU work (Python can prepare next frame)
///   2. Post-LN weights cached on CPU at upload time (zero PCIe per encode)
///   3. Cache-friendly mean pool (sequential memory access)
///   4. Removed unused bufNorm allocation
///   5. Persistent working buffers (descriptor set cache hits)

#include "bindings_core.h"
#include "grilly/ops/batched_ops.h"
#include "grilly/ops/linear.h"

#include <cmath>
#include <unordered_map>
#include <vector>
#include <cstring>

using namespace grilly;

struct SigLIPLayerGPU {
    GrillyBuffer qkv_w, qkv_b;
    GrillyBuffer out_w, out_b;
    GrillyBuffer ln1_w, ln1_b;
    GrillyBuffer ln2_w, ln2_b;
    GrillyBuffer mlp_w1, mlp_b1, mlp_w2, mlp_b2;
};

struct SigLIPWeightCache {
    std::vector<SigLIPLayerGPU> layers;
    GrillyBuffer post_ln_w_gpu, post_ln_b_gpu;

    // CPU-side copies of post-LN weights (avoid PCIe download per encode)
    std::vector<float> post_ln_w_cpu, post_ln_b_cpu;

    uint32_t numLayers = 0, seqLen = 0, hidden = 0;
    uint32_t numHeads = 0, headDim = 0, mlpDim = 0;

    // Persistent working buffers (padded to TILE multiples)
    GrillyBuffer bufX, bufQKV, bufAttn, bufMLP1, bufMLP2;
};

static std::unordered_map<int, SigLIPWeightCache> g_caches;
static int g_nextHandle = 1;

static GrillyBuffer uploadPersistent(BufferPool& pool, py::array_t<float> arr) {
    auto buf = arr.request();
    require_c_contiguous_float(buf);
    size_t bytes = buf.size * sizeof(float);
    GrillyBuffer gpuBuf = pool.acquire(bytes);
    pool.upload(gpuBuf, static_cast<const float*>(buf.ptr), bytes);
    return gpuBuf;
}

static void cpuLayerNorm(const float* x, const float* w, const float* b,
                         float* out, uint32_t seq, uint32_t dim) {
    for (uint32_t s = 0; s < seq; ++s) {
        const float* row = x + s * dim;
        float* orow = out + s * dim;
        float mean = 0.0f, var = 0.0f;
        for (uint32_t i = 0; i < dim; ++i) mean += row[i];
        mean /= dim;
        for (uint32_t i = 0; i < dim; ++i) { float d = row[i] - mean; var += d * d; }
        float inv = 1.0f / sqrtf(var / dim + 1e-6f);
        for (uint32_t i = 0; i < dim; ++i)
            orow[i] = (row[i] - mean) * inv * w[i] + b[i];
    }
}

void register_siglip_ops(py::module_& m) {
    using namespace grilly::nn;

    m.def("siglip_upload_weights",
        [](GrillyCoreContext& ctx, py::list weights,
           uint32_t numLayers, uint32_t seqLen, uint32_t hidden,
           uint32_t numHeads, uint32_t mlpDim) -> int {

            SigLIPWeightCache wc;
            wc.numLayers = numLayers;
            wc.seqLen = seqLen;
            wc.hidden = hidden;
            wc.numHeads = numHeads;
            wc.headDim = hidden / numHeads;
            wc.mlpDim = mlpDim;

            size_t idx = 0;
            wc.layers.resize(numLayers);
            for (uint32_t l = 0; l < numLayers; ++l) {
                auto& lw = wc.layers[l];
                lw.qkv_w  = uploadPersistent(ctx.pool, weights[idx++].cast<py::array_t<float>>());
                lw.qkv_b  = uploadPersistent(ctx.pool, weights[idx++].cast<py::array_t<float>>());
                lw.out_w   = uploadPersistent(ctx.pool, weights[idx++].cast<py::array_t<float>>());
                lw.out_b   = uploadPersistent(ctx.pool, weights[idx++].cast<py::array_t<float>>());
                lw.ln1_w   = uploadPersistent(ctx.pool, weights[idx++].cast<py::array_t<float>>());
                lw.ln1_b   = uploadPersistent(ctx.pool, weights[idx++].cast<py::array_t<float>>());
                lw.ln2_w   = uploadPersistent(ctx.pool, weights[idx++].cast<py::array_t<float>>());
                lw.ln2_b   = uploadPersistent(ctx.pool, weights[idx++].cast<py::array_t<float>>());
                lw.mlp_w1  = uploadPersistent(ctx.pool, weights[idx++].cast<py::array_t<float>>());
                lw.mlp_b1  = uploadPersistent(ctx.pool, weights[idx++].cast<py::array_t<float>>());
                lw.mlp_w2  = uploadPersistent(ctx.pool, weights[idx++].cast<py::array_t<float>>());
                lw.mlp_b2  = uploadPersistent(ctx.pool, weights[idx++].cast<py::array_t<float>>());
            }

            // Upload post-LN to GPU AND keep CPU copy (fix #2: no PCIe per encode)
            auto postW = weights[idx++].cast<py::array_t<float>>();
            auto postB = weights[idx++].cast<py::array_t<float>>();
            wc.post_ln_w_gpu = uploadPersistent(ctx.pool, postW);
            wc.post_ln_b_gpu = uploadPersistent(ctx.pool, postB);

            auto pwBuf = postW.request();
            auto pbBuf = postB.request();
            wc.post_ln_w_cpu.assign(static_cast<float*>(pwBuf.ptr),
                                    static_cast<float*>(pwBuf.ptr) + pwBuf.size);
            wc.post_ln_b_cpu.assign(static_cast<float*>(pbBuf.ptr),
                                    static_cast<float*>(pbBuf.ptr) + pbBuf.size);

            // Persistent working buffers padded to 32 multiples
            uint32_t S_pad = (seqLen + 31) & ~31u;
            uint32_t H_pad = (hidden + 31) & ~31u;
            uint32_t H3_pad = (hidden * 3 + 31) & ~31u;
            uint32_t M_pad = (mlpDim + 31) & ~31u;

            wc.bufX    = ctx.pool.acquire(size_t(S_pad) * H_pad * sizeof(float));
            wc.bufQKV  = ctx.pool.acquire(size_t(S_pad) * H3_pad * sizeof(float));
            wc.bufAttn = ctx.pool.acquire(size_t(S_pad) * H_pad * sizeof(float));
            wc.bufMLP1 = ctx.pool.acquire(size_t(S_pad) * M_pad * sizeof(float));
            wc.bufMLP2 = ctx.pool.acquire(size_t(S_pad) * H_pad * sizeof(float));

            int handle = g_nextHandle++;
            g_caches[handle] = std::move(wc);
            return handle;
        },
        py::arg("device"), py::arg("weights"),
        py::arg("num_layers"), py::arg("seq_len"), py::arg("hidden"),
        py::arg("num_heads"), py::arg("mlp_dim"));

    m.def("siglip_encode",
        [](GrillyCoreContext& ctx, int handle, py::array_t<float> patches) -> Tensor {
            auto it = g_caches.find(handle);
            if (it == g_caches.end())
                throw std::runtime_error("Invalid SigLIP weight handle");

            auto& wc = it->second;
            auto pBuf = patches.request();
            require_c_contiguous_float(pBuf);
            const uint32_t S = wc.seqLen;
            const uint32_t H = wc.hidden;
            const uint32_t H3 = H * 3;
            const uint32_t M = wc.mlpDim;
            const size_t seqBytes = size_t(S) * H * sizeof(float);

            // Extract pointer before releasing GIL
            const float* patchPtr = static_cast<const float*>(pBuf.ptr);

            // ── FIX #1: Release GIL during GPU work ──
            py::gil_scoped_release release;

            // Upload patches (persistent buffers — no acquire needed)
            ctx.pool.upload(wc.bufX, patchPtr, seqBytes);

            // ── BATCH ALL 12 LAYERS IN ONE SUBMIT ──
            ctx.batch.begin();

            for (uint32_t l = 0; l < wc.numLayers; ++l) {
                const auto& lw = wc.layers[l];

                ops::batchedFusedLnLinear(ctx.batch, ctx.cache,
                    wc.bufX, lw.ln1_w, lw.ln1_b, lw.qkv_w, lw.qkv_b,
                    wc.bufQKV, S);
                ctx.batch.barrier();

                ops::batchedTiledLinear(ctx.batch, ctx.cache,
                    wc.bufQKV, lw.out_w, &lw.out_b, wc.bufAttn,
                    S, H3, H);
                ctx.batch.barrier();

                ops::batchedAdd(ctx.batch, ctx.cache, wc.bufX, wc.bufAttn, S * H);
                ctx.batch.barrier();

                ops::batchedFusedLnLinear(ctx.batch, ctx.cache,
                    wc.bufX, lw.ln2_w, lw.ln2_b, lw.mlp_w1, lw.mlp_b1,
                    wc.bufMLP1, S);
                ctx.batch.barrier();

                ops::batchedGelu(ctx.batch, ctx.cache, wc.bufMLP1, wc.bufMLP1, S * M);
                ctx.batch.barrier();

                ops::batchedTiledLinear(ctx.batch, ctx.cache,
                    wc.bufMLP1, lw.mlp_w2, &lw.mlp_b2, wc.bufMLP2,
                    S, M, H);
                ctx.batch.barrier();

                ops::batchedAdd(ctx.batch, ctx.cache, wc.bufX, wc.bufMLP2, S * H);
                ctx.batch.barrier();
            }

            ctx.batch.submit();  // ONE fence wait

            // Download result
            std::vector<float> x(S * H);
            ctx.pool.download(wc.bufX, x.data(), seqBytes);

            // ── FIX #2: Use CPU-cached post-LN weights (zero PCIe) ──
            std::vector<float> normed(S * H);
            cpuLayerNorm(x.data(), wc.post_ln_w_cpu.data(), wc.post_ln_b_cpu.data(),
                         normed.data(), S, H);

            // ── FIX #3: Cache-friendly mean pool (outer=S, inner=H) ──
            std::vector<float> emb(H, 0.0f);
            for (uint32_t s = 0; s < S; ++s) {
                for (uint32_t i = 0; i < H; ++i) {
                    emb[i] += normed[s * H + i];
                }
            }
            for (uint32_t i = 0; i < H; ++i) emb[i] /= S;

            // L2 normalize
            float norm = 0.0f;
            for (uint32_t i = 0; i < H; ++i) norm += emb[i] * emb[i];
            norm = sqrtf(norm + 1e-8f);
            for (uint32_t i = 0; i < H; ++i) emb[i] /= norm;

            // ── FIX #1: Reacquire GIL for Python object creation ──
            py::gil_scoped_acquire acquire;

            py::array_t<float> out({(py::ssize_t)H});
            std::memcpy(out.mutable_data(), emb.data(), H * sizeof(float));
            return Tensor::from_numpy(out);
        },
        py::arg("device"), py::arg("handle"), py::arg("patches"));
}
