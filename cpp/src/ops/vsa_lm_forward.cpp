/// vsa_lm_forward.cpp — fused VSA-LM forward (AdditionLinear FFN + MindForge LoRA) on GPU,
/// Eigen backward on CPU.
///
/// FFN layers use the addition-linear shader (L1 distance, no matmul).
/// Output projection uses fnn-linear (standard matmul).
/// MindForge LoRA adapters are forged on CPU (tiny).

#include "grilly/ops/vsa_lm_forward.h"
#include "grilly/ops/batched_ops.h"

#include <Eigen/Dense>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace grilly {
namespace ops {

namespace {

static std::unordered_map<int, VsaLmHandleCache> g_vsa;
static int g_next_vsa = 1;

static void transpose_vd(const float* w, float* wt, uint32_t v, uint32_t d) {
    for (uint32_t r = 0; r < v; ++r)
        for (uint32_t c = 0; c < d; ++c)
            wt[c * v + r] = w[r * d + c];
}

struct AddLinPush {
    uint32_t batch_size;
    uint32_t in_features;
    uint32_t out_features;
    uint32_t use_bias;
};

struct LayerNormParams {
    uint32_t batch_size;
    uint32_t seq_len;
    uint32_t features;
    float    eps;
    uint32_t pass_type;
};

}  // namespace

VsaLmHandleCache& vsa_lm_get_cache(int handle) {
    auto it = g_vsa.find(handle);
    if (it == g_vsa.end())
        throw std::runtime_error("Invalid vsa_lm handle");
    return it->second;
}

void vsa_lm_release(BufferPool& pool, int handle) {
    auto it = g_vsa.find(handle);
    if (it == g_vsa.end())
        return;
    VsaLmHandleCache& h = it->second;

    pool.release(h.embedW);
    pool.release(h.posW);
    pool.release(h.outW);
    pool.release(h.outWt);

    for (auto& lw : h.layers) {
        pool.release(lw.ffnUpW);
        pool.release(lw.ffnUpB);
        pool.release(lw.ffnDownW);
        pool.release(lw.ffnDownB);
        pool.release(lw.lnGamma);
        pool.release(lw.lnBeta);
    }

    pool.release(h.bufIds);
    pool.release(h.bufPosSlice);
    pool.release(h.bufX);
    pool.release(h.bufLnOut);
    pool.release(h.bufFfnUp);
    pool.release(h.bufSign);
    pool.release(h.bufFfnDown);
    pool.release(h.bufLogits);
    pool.release(h.bufLnMean);
    pool.release(h.bufLnVar);

    for (auto& b : h.bufActivations)
        pool.release(b);

    g_vsa.erase(it);
}

int vsa_lm_upload(BufferPool& pool,
                  uint32_t vocab, uint32_t d, uint32_t d_ffn, uint32_t max_seq,
                  const float* embed_w, const float* pos_w,
                  const std::vector<const float*>& ffn_up_ws,
                  const std::vector<const float*>& ffn_up_bs,
                  const std::vector<const float*>& ffn_down_ws,
                  const std::vector<const float*>& ffn_down_bs,
                  const std::vector<const float*>& ln_gammas,
                  const std::vector<const float*>& ln_betas,
                  const float* out_w,
                  uint32_t n_layers) {

    if (n_layers == 0 || d == 0 || d_ffn == 0 || vocab == 0 || max_seq == 0)
        throw std::runtime_error("vsa_lm_upload: invalid dimensions");
    if (ffn_up_ws.size() != n_layers || ffn_up_bs.size() != n_layers ||
        ffn_down_ws.size() != n_layers || ffn_down_bs.size() != n_layers ||
        ln_gammas.size() != n_layers || ln_betas.size() != n_layers)
        throw std::runtime_error("vsa_lm_upload: list length mismatch");

    VsaLmHandleCache h;
    h.vocab   = vocab;
    h.d       = d;
    h.dFfn    = d_ffn;
    h.maxSeq  = max_seq;
    h.nLayers = n_layers;

    // Embedding
    size_t embed_bytes = size_t(vocab) * d * sizeof(float);
    h.cpu_embed.resize(vocab * d);
    std::memcpy(h.cpu_embed.data(), embed_w, embed_bytes);
    h.embedW = pool.acquire(embed_bytes);
    pool.upload(h.embedW, embed_w, embed_bytes);

    // Positional
    size_t pos_bytes = size_t(max_seq) * d * sizeof(float);
    h.cpu_pos.resize(max_seq * d);
    std::memcpy(h.cpu_pos.data(), pos_w, pos_bytes);
    h.posW = pool.acquire(pos_bytes);
    pool.upload(h.posW, pos_w, pos_bytes);

    // Output projection
    size_t out_bytes = size_t(vocab) * d * sizeof(float);
    h.cpu_out_w.resize(vocab * d);
    std::memcpy(h.cpu_out_w.data(), out_w, out_bytes);
    h.outW = pool.acquire(out_bytes);
    pool.upload(h.outW, out_w, out_bytes);
    std::vector<float> out_wt(d * vocab);
    transpose_vd(out_w, out_wt.data(), vocab, d);
    h.outWt = pool.acquire(out_wt.size() * sizeof(float));
    pool.upload(h.outWt, out_wt.data(), out_wt.size() * sizeof(float));

    // Per-layer weights
    h.layers.resize(n_layers);
    h.cpu_ffn_up_w.resize(n_layers);
    h.cpu_ffn_up_b.resize(n_layers);
    h.cpu_ffn_down_w.resize(n_layers);
    h.cpu_ffn_down_b.resize(n_layers);
    h.cpu_ln_gamma.resize(n_layers);
    h.cpu_ln_beta.resize(n_layers);

    for (uint32_t l = 0; l < n_layers; ++l) {
        auto& lw = h.layers[l];

        size_t up_w_bytes   = size_t(d_ffn) * d * sizeof(float);
        size_t up_b_bytes   = size_t(d_ffn) * sizeof(float);
        size_t down_w_bytes = size_t(d) * d_ffn * sizeof(float);
        size_t down_b_bytes = size_t(d) * sizeof(float);
        size_t ln_bytes     = size_t(d) * sizeof(float);

        lw.ffnUpW = pool.acquire(up_w_bytes);
        pool.upload(lw.ffnUpW, ffn_up_ws[l], up_w_bytes);
        lw.ffnUpB = pool.acquire(up_b_bytes);
        pool.upload(lw.ffnUpB, ffn_up_bs[l], up_b_bytes);

        lw.ffnDownW = pool.acquire(down_w_bytes);
        pool.upload(lw.ffnDownW, ffn_down_ws[l], down_w_bytes);
        lw.ffnDownB = pool.acquire(down_b_bytes);
        pool.upload(lw.ffnDownB, ffn_down_bs[l], down_b_bytes);

        lw.lnGamma = pool.acquire(ln_bytes);
        pool.upload(lw.lnGamma, ln_gammas[l], ln_bytes);
        lw.lnBeta = pool.acquire(ln_bytes);
        pool.upload(lw.lnBeta, ln_betas[l], ln_bytes);

        h.cpu_ffn_up_w[l].resize(d_ffn * d);
        std::memcpy(h.cpu_ffn_up_w[l].data(), ffn_up_ws[l], up_w_bytes);
        h.cpu_ffn_up_b[l].resize(d_ffn);
        std::memcpy(h.cpu_ffn_up_b[l].data(), ffn_up_bs[l], up_b_bytes);
        h.cpu_ffn_down_w[l].resize(d * d_ffn);
        std::memcpy(h.cpu_ffn_down_w[l].data(), ffn_down_ws[l], down_w_bytes);
        h.cpu_ffn_down_b[l].resize(d);
        std::memcpy(h.cpu_ffn_down_b[l].data(), ffn_down_bs[l], down_b_bytes);
        h.cpu_ln_gamma[l].resize(d);
        std::memcpy(h.cpu_ln_gamma[l].data(), ln_gammas[l], ln_bytes);
        h.cpu_ln_beta[l].resize(d);
        std::memcpy(h.cpu_ln_beta[l].data(), ln_betas[l], ln_bytes);
    }

    // Working buffers
    size_t sd       = size_t(max_seq) * d * sizeof(float);
    size_t s_dffn   = size_t(max_seq) * d_ffn * sizeof(float);
    size_t logit_sz = size_t(max_seq) * vocab * sizeof(float);

    h.bufIds     = pool.acquire(max_seq * sizeof(uint32_t));
    h.bufPosSlice= pool.acquire(sd);
    h.bufX       = pool.acquire(sd);
    h.bufLnOut   = pool.acquire(sd);
    h.bufFfnUp   = pool.acquire(s_dffn);
    h.bufSign    = pool.acquire(s_dffn);
    h.bufFfnDown = pool.acquire(sd);
    h.bufLogits  = pool.acquire(logit_sz);
    h.bufLnMean  = pool.acquire(max_seq * sizeof(float));
    h.bufLnVar   = pool.acquire(max_seq * sizeof(float));

    h.bufActivations.resize(n_layers + 1);
    for (uint32_t l = 0; l <= n_layers; ++l)
        h.bufActivations[l] = pool.acquire(sd);

    int hid = g_next_vsa++;
    g_vsa[hid] = std::move(h);
    return hid;
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper: record addition-linear dispatch into batch (no begin/submit)
// ═══════════════════════════════════════════════════════════════════════════

static void batchedAdditionLinear(CommandBatch& batch, PipelineCache& cache,
                                  const GrillyBuffer& input, const GrillyBuffer& weight,
                                  const GrillyBuffer& bias, GrillyBuffer& output,
                                  uint32_t S, uint32_t d_in, uint32_t d_out) {

    PipelineEntry pipe = cache.getOrCreate("addition-linear", 4, sizeof(AddLinPush));

    size_t inBytes  = size_t(S) * d_in * sizeof(float);
    size_t wBytes   = size_t(d_out) * d_in * sizeof(float);
    size_t bBytes   = size_t(d_out) * sizeof(float);
    size_t outBytes = size_t(S) * d_out * sizeof(float);

    std::vector<VkDescriptorBufferInfo> bufs = {
        {input.handle,  0, inBytes},
        {weight.handle, 0, wBytes},
        {bias.handle,   0, bBytes},
        {output.handle, 0, outBytes},
    };
    VkDescriptorSet desc = cache.allocDescriptorSet("addition-linear", bufs);

    AddLinPush push{S, d_in, d_out, 1};
    uint32_t total = S * d_out;
    uint32_t gx = (total + 255) / 256;

    batch.dispatch(pipe.pipeline, pipe.layout, desc, gx, 1, 1,
                   &push, sizeof(push));
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper: record sign activation into batch
// ═══════════════════════════════════════════════════════════════════════════

static void batchedSignActivation(CommandBatch& batch, PipelineCache& cache,
                                  const GrillyBuffer& input, GrillyBuffer& output,
                                  uint32_t totalElements) {

    PipelineEntry pipe = cache.getOrCreate("sign-activation", 2, sizeof(uint32_t));

    size_t bytes = size_t(totalElements) * sizeof(float);
    std::vector<VkDescriptorBufferInfo> bufs = {
        {input.handle,  0, bytes},
        {output.handle, 0, bytes},
    };
    VkDescriptorSet desc = cache.allocDescriptorSet("sign-activation", bufs);

    uint32_t push = totalElements;
    uint32_t gx = (totalElements + 255) / 256;
    batch.dispatch(pipe.pipeline, pipe.layout, desc, gx, 1, 1,
                   &push, sizeof(push));
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper: record 3-pass layernorm into batch (pre-allocated mean/var bufs)
// ═══════════════════════════════════════════════════════════════════════════

static void batchedLayerNorm(CommandBatch& batch, PipelineCache& cache,
                             const GrillyBuffer& input, GrillyBuffer& output,
                             const GrillyBuffer& gamma, const GrillyBuffer& beta,
                             GrillyBuffer& meanBuf, GrillyBuffer& varBuf,
                             uint32_t S, uint32_t features) {

    PipelineEntry pipe = cache.getOrCreate("fnn-layernorm", 6, sizeof(LayerNormParams));

    size_t elemBytes  = size_t(S) * features * sizeof(float);
    size_t paramBytes = size_t(features) * sizeof(float);
    size_t statBytes  = size_t(S) * sizeof(float);

    std::vector<VkDescriptorBufferInfo> bufs = {
        {input.handle,   0, elemBytes},
        {output.handle,  0, elemBytes},
        {gamma.handle,   0, paramBytes},
        {beta.handle,    0, paramBytes},
        {meanBuf.handle, 0, statBytes},
        {varBuf.handle,  0, statBytes},
    };
    VkDescriptorSet desc = cache.allocDescriptorSet("fnn-layernorm", bufs);

    // batch_size=1, seq_len=S
    LayerNormParams p0{1, S, features, 1e-5f, 0};
    uint32_t gxPos = (S + 255) / 256;
    batch.dispatch(pipe.pipeline, pipe.layout, desc, gxPos, 1, 1, &p0, sizeof(p0));
    batch.barrier();

    LayerNormParams p1{1, S, features, 1e-5f, 1};
    batch.dispatch(pipe.pipeline, pipe.layout, desc, gxPos, 1, 1, &p1, sizeof(p1));
    batch.barrier();

    LayerNormParams p2{1, S, features, 1e-5f, 2};
    uint32_t gxAll = (S * features + 255) / 256;
    batch.dispatch(pipe.pipeline, pipe.layout, desc, gxAll, 1, 1, &p2, sizeof(p2));
}

// ═══════════════════════════════════════════════════════════════════════════
// vsa_lm_forward_gpu
// ═══════════════════════════════════════════════════════════════════════════

void vsa_lm_forward_gpu(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                        VsaLmHandleCache& h, const int32_t* input_ids,
                        uint32_t seq_len, float* logits_out) {

    if (seq_len == 0 || seq_len > h.maxSeq)
        throw std::runtime_error("vsa_lm_forward: invalid seq_len");

    uint32_t S = seq_len;
    uint32_t d = h.d;
    uint32_t dF = h.dFfn;
    uint32_t V = h.vocab;

    // Upload token IDs and position slice
    pool.upload(h.bufIds, reinterpret_cast<const float*>(input_ids),
                S * sizeof(uint32_t));
    pool.upload(h.bufPosSlice, h.cpu_pos.data(), S * d * sizeof(float));

    // Phase 1: embedding lookup + position add
    batch.begin();
    batchedEmbeddingLookup(batch, cache, h.bufIds, h.embedW, h.bufX, 1, S, h.vocab, d);
    batch.barrier();
    batchedAdd(batch, cache, h.bufX, h.bufPosSlice, S * d);
    batch.barrier();
    batch.copyBuffer(h.bufX, h.bufActivations[0], S * d * sizeof(float));
    batch.submitDeferred();
    batch.waitForCompletion();

    // Phase 2: per-layer forward
    for (uint32_t l = 0; l < h.nLayers; ++l) {
        auto& lw = h.layers[l];

        batch.begin();

        // (a) LayerNorm
        batchedLayerNorm(batch, cache, h.bufX, h.bufLnOut,
                         lw.lnGamma, lw.lnBeta,
                         h.bufLnMean, h.bufLnVar, S, d);
        batch.barrier();

        // (d) Addition-linear up: (S, d) → (S, d_ffn)
        batchedAdditionLinear(batch, cache, h.bufLnOut, lw.ffnUpW, lw.ffnUpB,
                              h.bufFfnUp, S, d, dF);
        batch.barrier();

        // (e) Sign activation
        batchedSignActivation(batch, cache, h.bufFfnUp, h.bufSign, S * dF);
        batch.barrier();

        // (f) Addition-linear down: (S, d_ffn) → (S, d)
        batchedAdditionLinear(batch, cache, h.bufSign, lw.ffnDownW, lw.ffnDownB,
                              h.bufFfnDown, S, dF, d);
        batch.barrier();

        // (h) Residual: x += ffn_down
        batchedAdd(batch, cache, h.bufX, h.bufFfnDown, S * d);
        batch.barrier();

        // (i) Save activation
        batch.copyBuffer(h.bufX, h.bufActivations[l + 1], S * d * sizeof(float));

        batch.submitDeferred();
        batch.waitForCompletion();
    }

    // Phase 3: output projection — x @ out_w.T / sqrt(d)
    batch.begin();
    batchedLinear(batch, cache, h.bufX, h.outW, nullptr, h.bufLogits, S, d, V);
    batch.submitDeferred();
    batch.waitForCompletion();

    // Download and scale
    std::vector<float> raw(S * V);
    pool.download(h.bufLogits, raw.data(), S * V * sizeof(float));
    float scale = 1.0f / std::sqrt(static_cast<float>(d));
    for (size_t i = 0; i < size_t(S) * V; ++i)
        logits_out[i] = raw[i] * scale;
}

// ═══════════════════════════════════════════════════════════════════════════
// vsa_lm_backward_cpu — Eigen-based backward matching moe_backward_cpu pattern.
//
// AdditionLinear backward:
//   grad_input[row, k] = -sum_col( grad_out[row, col] * sign(W[col, k] - x[row, k]) )
//   grad_W[col, k]     = -sum_row( grad_out[row, col] * sign(W[col, k] - x[row, k]) )
//   grad_b[col]        = sum_row( grad_out[row, col] )
// ═══════════════════════════════════════════════════════════════════════════

VsaLmGradients vsa_lm_backward_cpu(const VsaLmHandleCache& h,
                                   const int32_t* input_ids, uint32_t seq_len,
                                   const float* grad_logits) {

    uint32_t S = seq_len;
    uint32_t d = h.d;
    uint32_t dF = h.dFfn;
    uint32_t V = h.vocab;
    uint32_t L = h.nLayers;

    if (S == 0 || S > h.maxSeq)
        throw std::runtime_error("vsa_lm_backward: invalid seq_len");

    using RM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // Output projection backward: dx = grad_logits @ out_w * scale
    float scale = 1.0f / std::sqrt(static_cast<float>(d));
    Eigen::Map<const RM> GL(grad_logits, S, V);
    Eigen::Map<const RM> OW(h.cpu_out_w.data(), V, d);

    RM scaledGL = GL * scale;  // (S, V) scaled
    RM dx = scaledGL * OW;     // (S, d)

    VsaLmGradients out;

    // grad_out_w = scaledGL.T @ x_final
    // Need final activation — download it. We stored it as bufActivations[L] during forward.
    // For CPU backward we'll recompute from saved CPU mirrors.
    // Actually let's replay the forward on CPU to get activations.

    // CPU forward replay for activations
    std::vector<std::vector<float>> acts(L + 1);
    std::vector<std::vector<float>> ln_outs(L);
    std::vector<std::vector<float>> ffn_ups(L);    // after addition-linear up
    std::vector<std::vector<float>> sign_outs(L);  // after sign activation

    // Initial: embed + pos
    acts[0].resize(S * d);
    for (uint32_t s = 0; s < S; ++s) {
        int32_t tok = input_ids[s];
        for (uint32_t j = 0; j < d; ++j) {
            float e = (tok >= 0 && static_cast<uint32_t>(tok) < V)
                      ? h.cpu_embed[tok * d + j] : 0.f;
            float p = h.cpu_pos[s * d + j];
            acts[0][s * d + j] = e + p;
        }
    }

    for (uint32_t l = 0; l < L; ++l) {
        const auto& uw = h.cpu_ffn_up_w[l];
        const auto& ub = h.cpu_ffn_up_b[l];
        const auto& dw = h.cpu_ffn_down_w[l];
        const auto& db = h.cpu_ffn_down_b[l];
        const auto& gm = h.cpu_ln_gamma[l];
        const auto& bt = h.cpu_ln_beta[l];

        // LayerNorm
        ln_outs[l].resize(S * d);
        for (uint32_t s = 0; s < S; ++s) {
            float mean = 0.f;
            for (uint32_t j = 0; j < d; ++j)
                mean += acts[l][s * d + j];
            mean /= float(d);
            float var = 0.f;
            for (uint32_t j = 0; j < d; ++j) {
                float diff = acts[l][s * d + j] - mean;
                var += diff * diff;
            }
            var /= float(d);
            float inv_std = 1.0f / std::sqrt(var + 1e-5f);
            for (uint32_t j = 0; j < d; ++j) {
                float norm = (acts[l][s * d + j] - mean) * inv_std;
                ln_outs[l][s * d + j] = gm[j] * norm + bt[j];
            }
        }

        // Addition-linear up: (S, d) → (S, d_ffn)
        ffn_ups[l].resize(S * dF);
        for (uint32_t s = 0; s < S; ++s) {
            for (uint32_t o = 0; o < dF; ++o) {
                float dist = 0.f;
                for (uint32_t k = 0; k < d; ++k)
                    dist += std::abs(uw[o * d + k] - ln_outs[l][s * d + k]);
                ffn_ups[l][s * dF + o] = -dist + ub[o];
            }
        }

        // Sign activation
        sign_outs[l].resize(S * dF);
        for (size_t i = 0; i < S * dF; ++i)
            sign_outs[l][i] = (ffn_ups[l][i] > 0.f) ? 1.f : -1.f;

        // Addition-linear down: (S, d_ffn) → (S, d)
        acts[l + 1].resize(S * d);
        for (uint32_t s = 0; s < S; ++s) {
            for (uint32_t o = 0; o < d; ++o) {
                float dist = 0.f;
                for (uint32_t k = 0; k < dF; ++k)
                    dist += std::abs(dw[o * dF + k] - sign_outs[l][s * dF + k]);
                float ffn_val = -dist + db[o];
                acts[l + 1][s * d + o] = acts[l][s * d + o] + ffn_val;
            }
        }
    }

    // grad_out_w = scaledGL.T @ x_final  → (V, d)
    Eigen::Map<const RM> XF(acts[L].data(), S, d);
    RM GOW = scaledGL.transpose() * XF;
    out.grad_out_w.resize(V * d);
    std::memcpy(out.grad_out_w.data(), GOW.data(), V * d * sizeof(float));

    // Back-propagate through layers in reverse
    out.grad_ffn_up_w.resize(L);
    out.grad_ffn_up_b.resize(L);
    out.grad_ffn_down_w.resize(L);
    out.grad_ffn_down_b.resize(L);
    out.grad_ln_gamma.resize(L);
    out.grad_ln_beta.resize(L);

    // dx is currently (S, d)
    for (int32_t l = L - 1; l >= 0; --l) {
        const auto& uw = h.cpu_ffn_up_w[l];
        const auto& dw = h.cpu_ffn_down_w[l];

        // dx passes through residual: grad to addition-linear down is dx
        // Addition-linear down backward:
        //   grad_sign[s, k] = -sum_o( dx[s, o] * sign(dw[o, k] - sign_out[s, k]) )
        //   grad_dw[o, k]   = -sum_s( dx[s, o] * sign(dw[o, k] - sign_out[s, k]) )
        //   grad_db[o]      = sum_s( dx[s, o] )

        out.grad_ffn_down_w[l].assign(d * dF, 0.f);
        out.grad_ffn_down_b[l].assign(d, 0.f);
        std::vector<float> grad_sign(S * dF, 0.f);

        for (uint32_t s = 0; s < S; ++s) {
            for (uint32_t o = 0; o < d; ++o) {
                float go = dx(s, o);
                out.grad_ffn_down_b[l][o] += go;
                for (uint32_t k = 0; k < dF; ++k) {
                    float sgn = (dw[o * dF + k] > sign_outs[l][s * dF + k]) ? 1.f :
                                (dw[o * dF + k] < sign_outs[l][s * dF + k]) ? -1.f : 0.f;
                    out.grad_ffn_down_w[l][o * dF + k] += -go * sgn;
                    grad_sign[s * dF + k] += -go * sgn;
                }
            }
        }

        // Sign activation backward: grad_ffn_up = grad_sign * 0 (sign is flat)
        // Actually sign subgradient: d/dx sign(x) = 0 almost everywhere.
        // But for training we use STE (straight-through estimator):
        // grad_ffn_up = grad_sign (pass through)
        std::vector<float>& grad_ffn_up = grad_sign;  // STE

        // Addition-linear up backward:
        //   grad_ln[s, k] = -sum_o( grad_up[s, o] * sign(uw[o, k] - ln_out[s, k]) )
        //   grad_uw[o, k] = -sum_s( grad_up[s, o] * sign(uw[o, k] - ln_out[s, k]) )
        //   grad_ub[o]    = sum_s( grad_up[s, o] )

        out.grad_ffn_up_w[l].assign(dF * d, 0.f);
        out.grad_ffn_up_b[l].assign(dF, 0.f);
        std::vector<float> grad_ln(S * d, 0.f);

        for (uint32_t s = 0; s < S; ++s) {
            for (uint32_t o = 0; o < dF; ++o) {
                float gu = grad_ffn_up[s * dF + o];
                out.grad_ffn_up_b[l][o] += gu;
                for (uint32_t k = 0; k < d; ++k) {
                    float sgn = (uw[o * d + k] > ln_outs[l][s * d + k]) ? 1.f :
                                (uw[o * d + k] < ln_outs[l][s * d + k]) ? -1.f : 0.f;
                    out.grad_ffn_up_w[l][o * d + k] += -gu * sgn;
                    grad_ln[s * d + k] += -gu * sgn;
                }
            }
        }

        // LayerNorm backward (simplified — gamma * grad_ln_out passed to LN backward)
        out.grad_ln_gamma[l].assign(d, 0.f);
        out.grad_ln_beta[l].assign(d, 0.f);

        const auto& gm = h.cpu_ln_gamma[l];

        // Recompute mean/invstd
        std::vector<float> means(S), inv_stds(S);
        for (uint32_t s = 0; s < S; ++s) {
            float m = 0.f;
            for (uint32_t j = 0; j < d; ++j)
                m += acts[l][s * d + j];
            m /= float(d);
            means[s] = m;
            float v = 0.f;
            for (uint32_t j = 0; j < d; ++j) {
                float diff = acts[l][s * d + j] - m;
                v += diff * diff;
            }
            v /= float(d);
            inv_stds[s] = 1.0f / std::sqrt(v + 1e-5f);
        }

        // grad_beta = sum_s(grad_ln), grad_gamma = sum_s(grad_ln * norm)
        for (uint32_t s = 0; s < S; ++s) {
            for (uint32_t j = 0; j < d; ++j) {
                float norm = (acts[l][s * d + j] - means[s]) * inv_stds[s];
                out.grad_ln_beta[l][j] += grad_ln[s * d + j];
                out.grad_ln_gamma[l][j] += grad_ln[s * d + j] * norm;
            }
        }

        // Backprop through layernorm to get grad w.r.t. input
        // Using full layernorm backward formula
        RM grad_x_ln(S, d);
        for (uint32_t s = 0; s < S; ++s) {
            float is = inv_stds[s];
            float m = means[s];

            // dl_dxhat = grad_ln * gamma
            Eigen::VectorXf dl_dxhat(d);
            for (uint32_t j = 0; j < d; ++j)
                dl_dxhat[j] = grad_ln[s * d + j] * gm[j];

            float sum1 = dl_dxhat.sum();
            float sum2 = 0.f;
            for (uint32_t j = 0; j < d; ++j) {
                float xhat = (acts[l][s * d + j] - m) * is;
                sum2 += dl_dxhat[j] * xhat;
            }

            for (uint32_t j = 0; j < d; ++j) {
                float xhat = (acts[l][s * d + j] - m) * is;
                grad_x_ln(s, j) = is * (dl_dxhat[j] - sum1 / d - xhat * sum2 / d);
            }
        }

        // Residual: dx for next layer = dx (through residual) + grad_x_ln
        RM new_dx = dx + grad_x_ln;
        dx = new_dx;
    }

    // Embedding gradient: scatter-add dx
    out.grad_embed.assign(V * d, 0.f);
    out.grad_pos.assign(h.maxSeq * d, 0.f);
    for (uint32_t s = 0; s < S; ++s) {
        int32_t tok = input_ids[s];
        if (tok >= 0 && static_cast<uint32_t>(tok) < V) {
            for (uint32_t j = 0; j < d; ++j)
                out.grad_embed[tok * d + j] += dx(s, j);
        }
        for (uint32_t j = 0; j < d; ++j)
            out.grad_pos[s * d + j] = dx(s, j);
    }

    return out;
}

}  // namespace ops
}  // namespace grilly
