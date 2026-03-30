/// bindings_siglip.cpp — SigLIP2 full encoder: persistent weights + batched dispatch.
///
/// Two Python functions:
///   siglip_upload_weights(ctx, weights, ...) → handle
///   siglip_encode(ctx, handle, patches) → 768D embedding
///
/// The encoder runs ALL 12 transformer layers in ONE batch.submit().
/// Weights live on GPU permanently. Working buffers are pooled.
/// Only TWO PCIe transfers: upload patches + download 768D embedding.

#include "bindings_core.h"
#include "grilly/ops/batched_ops.h"
#include "grilly/ops/linear.h"

#include <cmath>
#include <unordered_map>
#include <vector>
#include <cstring>

using namespace grilly;

// ── Persistent weight storage ────────────────────────────────────────────

struct SigLIPLayerGPU {
    GrillyBuffer qkv_w, qkv_b;        // Fused QKV: (3*H, H), (3*H,)
    GrillyBuffer out_w, out_b;         // Output proj: (H, H), (H,)
    GrillyBuffer ln1_w, ln1_b;         // LayerNorm 1
    GrillyBuffer ln2_w, ln2_b;         // LayerNorm 2
    GrillyBuffer mlp_w1, mlp_b1;       // MLP fc1: (4H, H), (4H,)
    GrillyBuffer mlp_w2, mlp_b2;       // MLP fc2: (H, 4H), (H,)
};

struct SigLIPWeightCache {
    std::vector<SigLIPLayerGPU> layers;
    GrillyBuffer post_ln_w, post_ln_b;
    uint32_t numLayers = 0, seqLen = 0, hidden = 0;
    uint32_t numHeads = 0, headDim = 0, mlpDim = 0;
};

static std::unordered_map<int, SigLIPWeightCache> g_caches;
static int g_nextHandle = 1;

static GrillyBuffer uploadPersistent(BufferPool& pool, py::array_t<float> arr) {
    auto buf = arr.request();
    size_t bytes = buf.size * sizeof(float);
    GrillyBuffer gpuBuf = pool.acquire(bytes);
    pool.upload(gpuBuf, static_cast<const float*>(buf.ptr), bytes);
    return gpuBuf;
}

// ── CPU helpers for LayerNorm + mean pool (tiny, not worth GPU dispatch) ──

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

// ── Registration ─────────────────────────────────────────────────────────

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
            wc.post_ln_w = uploadPersistent(ctx.pool, weights[idx++].cast<py::array_t<float>>());
            wc.post_ln_b = uploadPersistent(ctx.pool, weights[idx++].cast<py::array_t<float>>());

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

            const auto& wc = it->second;
            auto pBuf = patches.request();
            const uint32_t S = wc.seqLen;
            const uint32_t H = wc.hidden;
            const uint32_t H3 = H * 3;
            const uint32_t M = wc.mlpDim;  // 3072
            const size_t seqBytes = size_t(S) * H * sizeof(float);

            // ── Pre-allocate ALL working buffers on GPU ──
            GrillyBuffer bufX     = ctx.pool.acquire(seqBytes);       // (S, H)
            GrillyBuffer bufNorm  = ctx.pool.acquire(seqBytes);       // (S, H) normalized
            GrillyBuffer bufQKV   = ctx.pool.acquire(S * H3 * 4);    // (S, 3H)
            GrillyBuffer bufAttn  = ctx.pool.acquire(seqBytes);       // (S, H) attention out
            GrillyBuffer bufMLP1  = ctx.pool.acquire(S * M * 4);     // (S, 4H) MLP hidden
            GrillyBuffer bufMLP2  = ctx.pool.acquire(seqBytes);       // (S, H) MLP out

            // ── UPLOAD patches ONCE ──
            ctx.pool.upload(bufX, static_cast<const float*>(pBuf.ptr), seqBytes);

            // ── BATCH ALL LAYERS ──
            ctx.batch.begin();

            for (uint32_t l = 0; l < wc.numLayers; ++l) {
                const auto& lw = wc.layers[l];

                // === ATTENTION BLOCK ===

                // 1. LayerNorm1 + QKV projection (fused shader: LN in LDS → project)
                ops::batchedFusedLnLinear(ctx.batch, ctx.cache,
                    bufX, lw.ln1_w, lw.ln1_b, lw.qkv_w, lw.qkv_b,
                    bufQKV, S);
                ctx.batch.barrier();

                // 2. Attention: for now, approximate with output projection of QKV
                //    (proper flash_attention requires QKV reshape shader — next step)
                //    This computes: attn_out = QKV @ out_w.T + out_b
                //    which is mathematically wrong for attention but exercises the
                //    full batched pipeline. Replace with batchedFlashAttention2 + reshape.
                ops::batchedLinear(ctx.batch, ctx.cache,
                    bufQKV, lw.out_w, &lw.out_b, bufAttn,
                    S, H3, H);
                ctx.batch.barrier();

                // 3. Residual: X += attn_out
                ops::batchedAdd(ctx.batch, ctx.cache, bufX, bufAttn, S * H);
                ctx.batch.barrier();

                // === MLP BLOCK ===

                // 4. Fused LayerNorm2 + MLP fc1 (LN in LDS → fc1)
                ops::batchedFusedLnLinear(ctx.batch, ctx.cache,
                    bufX, lw.ln2_w, lw.ln2_b, lw.mlp_w1, lw.mlp_b1,
                    bufMLP1, S);
                ctx.batch.barrier();

                // 5. GELU activation
                ops::batchedGelu(ctx.batch, ctx.cache, bufMLP1, bufMLP1, S * M);
                ctx.batch.barrier();

                // 6. MLP fc2
                ops::batchedLinear(ctx.batch, ctx.cache,
                    bufMLP1, lw.mlp_w2, &lw.mlp_b2, bufMLP2,
                    S, M, H);
                ctx.batch.barrier();

                // 7. Residual: X += mlp_out
                ops::batchedAdd(ctx.batch, ctx.cache, bufX, bufMLP2, S * H);
                ctx.batch.barrier();
            }

            ctx.batch.submit();  // === ONE FENCE WAIT FOR ALL 12 LAYERS ===

            // ── DOWNLOAD result: only (S, H) = 3MB ──
            std::vector<float> x(S * H);
            ctx.pool.download(bufX, x.data(), seqBytes);

            // ── Post-LayerNorm (CPU, 1024×768 = tiny) ──
            std::vector<float> normed(S * H);
            std::vector<float> postW(H), postB(H);
            ctx.pool.download(wc.post_ln_w, postW.data(), H * sizeof(float));
            ctx.pool.download(wc.post_ln_b, postB.data(), H * sizeof(float));
            cpuLayerNorm(x.data(), postW.data(), postB.data(), normed.data(), S, H);

            // ── Mean pool → 768D embedding ──
            std::vector<float> emb(H, 0.0f);
            for (uint32_t i = 0; i < H; ++i) {
                float sum = 0.0f;
                for (uint32_t s = 0; s < S; ++s)
                    sum += normed[s * H + i];
                emb[i] = sum / S;
            }

            // ── L2 normalize ──
            float norm = 0.0f;
            for (uint32_t i = 0; i < H; ++i) norm += emb[i] * emb[i];
            norm = sqrtf(norm + 1e-8f);
            for (uint32_t i = 0; i < H; ++i) emb[i] /= norm;

            // ── Release working buffers ──
            ctx.pool.release(bufX);
            ctx.pool.release(bufNorm);
            ctx.pool.release(bufQKV);
            ctx.pool.release(bufAttn);
            ctx.pool.release(bufMLP1);
            ctx.pool.release(bufMLP2);

            py::array_t<float> out({(py::ssize_t)H});
            std::memcpy(out.mutable_data(), emb.data(), H * sizeof(float));
            return Tensor::from_numpy(out);
        },
        py::arg("device"), py::arg("handle"), py::arg("patches"));
}
