/// moe_forward.cpp — fused MoE forward on GPU (router/blend CPU) + CPU backward.
#include "grilly/ops/moe_forward.h"
#include "grilly/ops/batched_ops.h"

#include <Eigen/Dense>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace grilly {
namespace ops {

namespace {

static std::unordered_map<int, MoeHandleCache> g_moe;
static int g_next_moe = 1;

static void transpose_dd(const float* w, float* wt, uint32_t d) {
    for (uint32_t r = 0; r < d; ++r)
        for (uint32_t c = 0; c < d; ++c)
            wt[c * d + r] = w[r * d + c];
}

static void transpose_vd(const float* w, float* wt, uint32_t v, uint32_t d) {
    for (uint32_t r = 0; r < v; ++r)
        for (uint32_t c = 0; c < d; ++c)
            wt[c * v + r] = w[r * d + c];
}

static Eigen::VectorXf softmax_vec(const Eigen::VectorXf& x) {
    float m = x.maxCoeff();
    Eigen::VectorXf e = (x.array() - m).exp();
    return e / e.sum();
}

static Eigen::VectorXf softmax_grad(const Eigen::VectorXf& p,
                                    const Eigen::VectorXf& grad_p) {
    float s = p.dot(grad_p);
    return p.cwiseProduct(grad_p - Eigen::VectorXf::Constant(p.size(), s));
}

}  // namespace

MoeHandleCache& moe_get_cache(int handle) {
    auto it = g_moe.find(handle);
    if (it == g_moe.end())
        throw std::runtime_error("Invalid moe handle");
    return it->second;
}

void moe_release(BufferPool& pool, int handle) {
    auto it = g_moe.find(handle);
    if (it == g_moe.end())
        return;
    MoeHandleCache& h = it->second;

    pool.release(h.embedW);
    pool.release(h.posW);
    pool.release(h.outW);
    pool.release(h.outWt);
    for (auto& lw : h.layers) {
        pool.release(lw.routerW);
        pool.release(lw.routerB);
        for (uint32_t e = 0; e < lw.expertW.size(); ++e) {
            pool.release(lw.expertW[e]);
            pool.release(lw.expertWt[e]);
        }
        pool.release(lw.expertPacked);
    }
    pool.release(h.bufIds);
    pool.release(h.bufPosSlice);
    pool.release(h.bufX);
    for (auto& b : h.bufExpertOut)
        pool.release(b);
    pool.release(h.bufBlended);
    pool.release(h.bufLogits);

    g_moe.erase(it);
}

static void upload_dd_pair(BufferPool& pool, GrillyBuffer& wbuf, GrillyBuffer& wtbuf,
                           const float* w, uint32_t d) {
    size_t bytes = size_t(d) * d * sizeof(float);
    wbuf = pool.acquire(bytes);
    pool.upload(wbuf, w, bytes);
    std::vector<float> wt(d * d);
    transpose_dd(w, wt.data(), d);
    wtbuf = pool.acquire(bytes);
    pool.upload(wtbuf, wt.data(), bytes);
}

int moe_upload(BufferPool& pool,
               uint32_t vocab_size, uint32_t d_model, uint32_t max_seq,
               const float* embed_w, const float* pos_w,
               const std::vector<const float*>& expert_ws,
               const std::vector<const float*>& router_ws,
               const std::vector<const float*>& router_bs,
               const float* out_w,
               uint32_t n_layers, uint32_t n_experts) {

    if (n_layers == 0 || n_experts == 0 || d_model == 0 || vocab_size == 0 || max_seq == 0)
        throw std::runtime_error("moe_upload: invalid dimensions");
    if (expert_ws.size() != size_t(n_layers) * n_experts)
        throw std::runtime_error("moe_upload: expert_ws length mismatch");
    if (router_ws.size() != n_layers || router_bs.size() != n_layers)
        throw std::runtime_error("moe_upload: router list length mismatch");

    MoeHandleCache h;
    h.vocab = vocab_size;
    h.d = d_model;
    h.maxSeq = max_seq;
    h.nLayers = n_layers;
    h.nExperts = n_experts;

    size_t embed_bytes = size_t(vocab_size) * d_model * sizeof(float);
    h.cpu_embed.assign(embed_bytes / sizeof(float), 0.f);
    std::memcpy(h.cpu_embed.data(), embed_w, embed_bytes);
    h.embedW = pool.acquire(embed_bytes);
    pool.upload(h.embedW, embed_w, embed_bytes);

    size_t pos_bytes = size_t(max_seq) * d_model * sizeof(float);
    h.cpu_pos.assign(pos_bytes / sizeof(float), 0.f);
    std::memcpy(h.cpu_pos.data(), pos_w, pos_bytes);
    h.posW = pool.acquire(pos_bytes);
    pool.upload(h.posW, pos_w, pos_bytes);

    size_t out_bytes = size_t(vocab_size) * d_model * sizeof(float);
    h.cpu_out_w.assign(out_bytes / sizeof(float), 0.f);
    std::memcpy(h.cpu_out_w.data(), out_w, out_bytes);
    h.outW = pool.acquire(out_bytes);
    pool.upload(h.outW, out_w, out_bytes);
    std::vector<float> out_wt(d_model * vocab_size);
    transpose_vd(out_w, out_wt.data(), vocab_size, d_model);
    h.outWt = pool.acquire(out_wt.size() * sizeof(float));
    pool.upload(h.outWt, out_wt.data(), out_wt.size() * sizeof(float));

    h.cpu_expert_w.resize(size_t(n_layers) * n_experts);
    h.layers.resize(n_layers);
    for (uint32_t l = 0; l < n_layers; ++l) {
        auto& lw = h.layers[l];
        size_t rbytes = size_t(n_experts) * d_model * sizeof(float);
        lw.routerW = pool.acquire(rbytes);
        pool.upload(lw.routerW, router_ws[l], rbytes);
        lw.routerB = pool.acquire(n_experts * sizeof(float));
        pool.upload(lw.routerB, router_bs[l], n_experts * sizeof(float));

        h.cpu_router_w.emplace_back(n_experts * d_model);
        std::memcpy(h.cpu_router_w.back().data(), router_ws[l], rbytes);
        h.cpu_router_b.emplace_back(n_experts);
        std::memcpy(h.cpu_router_b.back().data(), router_bs[l],
                    n_experts * sizeof(float));

        lw.expertW.resize(n_experts);
        lw.expertWt.resize(n_experts);
        // Pack all experts contiguously for fused shader
        size_t packed_size = size_t(n_experts) * d_model * d_model * sizeof(float);
        std::vector<float> packed(n_experts * d_model * d_model);
        for (uint32_t e = 0; e < n_experts; ++e) {
            const float* w = expert_ws[l * n_experts + e];
            upload_dd_pair(pool, lw.expertW[e], lw.expertWt[e], w, d_model);
            h.cpu_expert_w[l * n_experts + e].assign(d_model * d_model, 0.f);
            std::memcpy(h.cpu_expert_w[l * n_experts + e].data(), w,
                        d_model * d_model * sizeof(float));
            std::memcpy(packed.data() + e * d_model * d_model, w,
                        d_model * d_model * sizeof(float));
        }
        lw.expertPacked = pool.acquire(packed_size);
        pool.upload(lw.expertPacked, packed.data(), packed_size);
    }

    size_t seq_d = size_t(max_seq) * d_model * sizeof(float);
    h.bufIds = pool.acquire(max_seq * sizeof(uint32_t));
    h.bufPosSlice = pool.acquire(seq_d);
    h.bufX = pool.acquire(seq_d);
    h.bufBlended = pool.acquire(seq_d);

    // Activation buffers for backward (n_layers + 1: input to each layer + final)
    h.bufActivations.resize(n_layers + 1);
    for (uint32_t l = 0; l <= n_layers; ++l)
        h.bufActivations[l] = pool.acquire(seq_d);
    h.fwd_router_weights.resize(n_layers);

    h.bufExpertOut.resize(n_experts);
    for (uint32_t e = 0; e < n_experts; ++e)
        h.bufExpertOut[e] = pool.acquire(seq_d);
    h.bufLogits = pool.acquire(size_t(max_seq) * vocab_size * sizeof(float));

    int hid = g_next_moe++;
    g_moe[hid] = std::move(h);
    return hid;
}

void moe_update_weights(BufferPool& pool, MoeHandleCache& h,
                        const float* embed_w, const float* pos_w,
                        const std::vector<const float*>& expert_ws,
                        const std::vector<const float*>& router_ws,
                        const std::vector<const float*>& router_bs,
                        const float* out_w) {

    uint32_t V = h.vocab;
    uint32_t d = h.d;
    uint32_t max_seq = h.maxSeq;
    uint32_t L = h.nLayers;
    uint32_t E = h.nExperts;

    size_t embed_bytes = size_t(V) * d * sizeof(float);
    std::memcpy(h.cpu_embed.data(), embed_w, embed_bytes);
    pool.upload(h.embedW, embed_w, embed_bytes);

    size_t pos_bytes = size_t(max_seq) * d * sizeof(float);
    std::memcpy(h.cpu_pos.data(), pos_w, pos_bytes);
    pool.upload(h.posW, pos_w, pos_bytes);

    size_t out_bytes = size_t(V) * d * sizeof(float);
    std::memcpy(h.cpu_out_w.data(), out_w, out_bytes);
    pool.upload(h.outW, out_w, out_bytes);
    std::vector<float> out_wt(d * V);
    transpose_vd(out_w, out_wt.data(), V, d);
    pool.upload(h.outWt, out_wt.data(), out_wt.size() * sizeof(float));

    for (uint32_t l = 0; l < L; ++l) {
        auto& lw = h.layers[l];
        size_t rbytes = size_t(E) * d * sizeof(float);
        pool.upload(lw.routerW, router_ws[l], rbytes);
        pool.upload(lw.routerB, router_bs[l], E * sizeof(float));
        std::memcpy(h.cpu_router_w[l].data(), router_ws[l], rbytes);
        std::memcpy(h.cpu_router_b[l].data(), router_bs[l], E * sizeof(float));

        size_t packed_size = size_t(E) * d * d * sizeof(float);
        std::vector<float> packed(E * d * d);
        for (uint32_t e = 0; e < E; ++e) {
            const float* w = expert_ws[l * E + e];
            size_t dd = size_t(d) * d * sizeof(float);
            pool.upload(lw.expertW[e], w, dd);
            std::vector<float> wt(d * d);
            transpose_dd(w, wt.data(), d);
            pool.upload(lw.expertWt[e], wt.data(), dd);
            std::memcpy(h.cpu_expert_w[l * E + e].data(), w, dd);
            std::memcpy(packed.data() + e * d * d, w, dd);
        }
        pool.upload(lw.expertPacked, packed.data(), packed_size);
    }
}

void moe_forward_gpu(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                     MoeHandleCache& h, const int32_t* input_ids, uint32_t seq_len,
                     float* logits_out) {

    if (seq_len == 0 || seq_len > h.maxSeq)
        throw std::runtime_error("moe_forward: invalid seq_len");

    uint32_t S = seq_len;
    uint32_t d = h.d;
    uint32_t V = h.vocab;

    pool.upload(h.bufIds, reinterpret_cast<const float*>(input_ids),
                S * sizeof(int32_t));
    pool.upload(h.bufPosSlice, h.cpu_pos.data(), S * d * sizeof(float));

    batch.begin();
    batchedEmbeddingLookup(batch, cache, h.bufIds, h.embedW, h.bufX, 1, S, h.vocab, d);
    batch.barrier();
    batchedAdd(batch, cache, h.bufX, h.bufPosSlice, S * d);
    batch.submitDeferred();
    batch.waitForCompletion();

    size_t packed_size = size_t(h.nExperts) * d * d * sizeof(float);
    bool has_router = cache.hasShader("moe-router");
    // Prefer vec4 path, fall back to scalar fused, then legacy
    bool has_vec4 = cache.hasShader("moe-layer-fused-vec4") && (d % 4 == 0);
    bool has_fused = has_vec4 || cache.hasShader("moe-layer-fused");

    if (has_fused && has_router && h.nExperts == 4) {
        // ══════════════════════════════════════════════════════════════
        // ALL-GPU path: router + experts in ONE command buffer submission
        // ZERO CPU round-trips between layers. ONE fence wait total.
        // Vec4 path: 4x memory bandwidth via 128-bit loads.
        // ══════════════════════════════════════════════════════════════

        size_t scratch_size = (d + h.nExperts + h.nExperts) * sizeof(float);
        GrillyBuffer bufScratch = pool.acquire(scratch_size);
        uint32_t weights_offset = d + h.nExperts;

        struct RouterPush { uint32_t seq_len, d_model, n_experts, pass; };
        struct FusedPush { uint32_t seq_len, d_model, n_experts, weights_offset; };

        PipelineEntry routerPipe = cache.getOrCreate("moe-router", 4, sizeof(RouterPush));

        // Choose vec4 or scalar fused shader
        const char* fused_name = has_vec4 ? "moe-layer-fused-vec4" : "moe-layer-fused";
        PipelineEntry fusedPipe = cache.getOrCreate(fused_name, 4, sizeof(FusedPush));

        batch.begin();

        // Save initial activation
        batch.copyBuffer(h.bufX, h.bufActivations[0], S * d * sizeof(float));
        batch.barrier();

        for (uint32_t l = 0; l < h.nLayers; ++l) {
            auto& layer = h.layers[l];

            // Router: 3 passes (mean, logits, softmax)
            for (uint32_t pass = 0; pass < 3; ++pass) {
                std::vector<VkDescriptorBufferInfo> rbufs = {
                    {h.bufX.handle, 0, S * d * sizeof(float)},
                    {layer.routerW.handle, 0, h.nExperts * d * sizeof(float)},
                    {layer.routerB.handle, 0, h.nExperts * sizeof(float)},
                    {bufScratch.handle, 0, scratch_size},
                };
                VkDescriptorSet rdesc = cache.allocDescriptorSet("moe-router", rbufs);
                RouterPush rp{S, d, h.nExperts, pass};
                uint32_t wg = (pass == 0) ? (d + 255) / 256 :
                              (pass == 1) ? (h.nExperts + 255) / 256 : 1;
                batch.dispatch(routerPipe.pipeline, routerPipe.layout, rdesc,
                               wg, 1, 1, &rp, sizeof(rp));
                batch.barrier();
            }

            // Fused expert layer (vec4 or scalar)
            {
                FusedPush fp{S, d, h.nExperts, weights_offset};
                std::vector<VkDescriptorBufferInfo> fbufs = {
                    {h.bufX.handle, 0, S * d * sizeof(float)},
                    {layer.expertPacked.handle, 0, packed_size},
                    {h.bufBlended.handle, 0, S * d * sizeof(float)},
                    {bufScratch.handle, 0, scratch_size},
                };
                VkDescriptorSet fdesc = cache.allocDescriptorSet(fused_name, fbufs);

                uint32_t gx, gy;
                if (has_vec4) {
                    gx = ((d / 4) + 15) / 16;  // vec4 columns
                    gy = (S + 15) / 16;
                } else {
                    gx = (d + 15) / 16;
                    gy = (S + 15) / 16;
                }
                batch.dispatch(fusedPipe.pipeline, fusedPipe.layout, fdesc,
                               gx, gy, 1, &fp, sizeof(fp));
                batch.barrier();
            }

            std::swap(h.bufX, h.bufBlended);

            // Save activation for backward
            batch.copyBuffer(h.bufX, h.bufActivations[l + 1], S * d * sizeof(float));
            batch.barrier();
        }

        // Output projection (same command buffer!)
        batchedLinear(batch, cache, h.bufX, h.outW, nullptr, h.bufLogits, S, d, V);

        // ONE submit, ONE fence wait for entire forward pass
        batch.submitDeferred();
        batch.waitForCompletion();

        pool.release(bufScratch);
    } else {
        // ══════════════════════════════════════════════════════════════
        // Legacy CPU-router path (fallback)
        // ══════════════════════════════════════════════════════════════
        std::vector<float> x_cpu(S * d);
        std::vector<float> x_mean(d);

        for (uint32_t l = 0; l < h.nLayers; ++l) {
            pool.download(h.bufX, x_cpu.data(), S * d * sizeof(float));
            for (uint32_t j = 0; j < d; ++j) {
                float s = 0.f;
                for (uint32_t i = 0; i < S; ++i)
                    s += x_cpu[i * d + j];
                x_mean[j] = s / float(S);
            }

            Eigen::Map<Eigen::VectorXf> xmean(x_mean.data(), d);
            Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Wr(
                h.cpu_router_w[l].data(), h.nExperts, d);
            Eigen::Map<const Eigen::VectorXf> b(h.cpu_router_b[l].data(), h.nExperts);
            Eigen::VectorXf logits_v = Wr * xmean + b;
            Eigen::VectorXf p = softmax_vec(logits_v);

            auto& layer = h.layers[l];
            batch.begin();
            for (uint32_t e = 0; e < h.nExperts; ++e) {
                batchedLinear(batch, cache, h.bufX, layer.expertW[e], nullptr,
                              h.bufExpertOut[e], S, d, d);
            }
            batch.submitDeferred();
            batch.waitForCompletion();

            std::vector<float> expert_flat(S * d * h.nExperts);
            for (uint32_t e = 0; e < h.nExperts; ++e)
                pool.download(h.bufExpertOut[e], expert_flat.data() + e * S * d,
                              S * d * sizeof(float));

            std::vector<float> blended(S * d, 0.f);
            for (uint32_t e = 0; e < h.nExperts; ++e) {
                float pe = p[e];
                for (size_t i = 0; i < S * d; ++i)
                    blended[i] += pe * expert_flat[e * S * d + i];
            }
            pool.upload(h.bufBlended, blended.data(), S * d * sizeof(float));

            batch.begin();
            batchedAdd(batch, cache, h.bufX, h.bufBlended, S * d);
            batch.submitDeferred();
            batch.waitForCompletion();
        }
    }

    size_t log_bytes = S * V * sizeof(float);
    batch.begin();
    batchedLinear(batch, cache, h.bufX, h.outW, nullptr, h.bufLogits, S, d, V);
    batch.submitDeferred();
    batch.waitForCompletion();
    pool.download(h.bufLogits, logits_out, log_bytes);
}

MoeGradients moe_backward_cpu(const MoeHandleCache& h,
                              const int32_t* input_ids, uint32_t seq_len,
                              const float* grad_logits) {

    uint32_t S = seq_len;
    uint32_t d = h.d;
    uint32_t V = h.vocab;
    uint32_t L = h.nLayers;
    uint32_t E = h.nExperts;
    if (S == 0 || S > h.maxSeq)
        throw std::runtime_error("moe_backward: invalid seq_len");

    using RowMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    Eigen::MatrixXf X = Eigen::MatrixXf::Zero(S, d);
    for (uint32_t s = 0; s < S; ++s) {
        int32_t tok = input_ids[s];
        if (tok < 0 || static_cast<uint32_t>(tok) >= V)
            throw std::runtime_error("moe_backward: token id out of range");
        for (uint32_t j = 0; j < d; ++j) {
            X(s, j) = h.cpu_embed[tok * d + j] + h.cpu_pos[s * d + j];
        }
    }

    std::vector<Eigen::MatrixXf> Xs;
    Xs.reserve(L + 1);
    Xs.push_back(X);

    struct LayerTrace {
        Eigen::VectorXf p;
        Eigen::VectorXf xmean;
        std::vector<Eigen::MatrixXf> Y;
    };
    std::vector<LayerTrace> trace(L);

    for (uint32_t l = 0; l < L; ++l) {
        Eigen::MatrixXf& cur = Xs.back();
        Eigen::VectorXf xmean = cur.colwise().mean().transpose();

        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Wr(
            h.cpu_router_w[l].data(), E, d);
        Eigen::Map<const Eigen::VectorXf> b(h.cpu_router_b[l].data(), E);
        Eigen::VectorXf logits = Wr * xmean + b;
        Eigen::VectorXf p = softmax_vec(logits);

        Eigen::MatrixXf blend = Eigen::MatrixXf::Zero(S, d);
        trace[l].Y.resize(E);
        for (uint32_t e = 0; e < E; ++e) {
            Eigen::Map<const RowMajor> We(h.cpu_expert_w[l * E + e].data(), d, d);
            trace[l].Y[e] = cur * We.transpose();
            blend += p[e] * trace[l].Y[e];
        }
        trace[l].p = p;
        trace[l].xmean = xmean;

        Xs.push_back(cur + blend);
    }

    Eigen::MatrixXf Xfinal = Xs.back();

    Eigen::Map<const RowMajor> Glog(grad_logits, S, V);
    Eigen::Map<const RowMajor> Wo(h.cpu_out_w.data(), V, d);

    MoeGradients out;
    out.grad_out_w.resize(V * d);
    Eigen::MatrixXf grad_Wo_mat = Glog.transpose() * Xfinal;
    Eigen::Map<RowMajor> gwo(out.grad_out_w.data(), V, d);
    gwo = grad_Wo_mat;

    out.grad_router_w.resize(L);
    out.grad_router_b.resize(L);
    out.grad_experts.resize(L * E);
    for (auto& v : out.grad_router_w)
        v.assign(E * d, 0.f);
    for (auto& v : out.grad_router_b)
        v.assign(E, 0.f);
    for (auto& v : out.grad_experts)
        v.assign(d * d, 0.f);

    // logits = Xfinal * Wo^T where Wo is (V, d), so grad_X = grad_logits * Wo.
    Eigen::MatrixXf g = Glog * Wo;

    for (int li = static_cast<int>(L) - 1; li >= 0; --li) {
        uint32_t l = static_cast<uint32_t>(li);
        const Eigen::MatrixXf& cur = Xs[l];
        const LayerTrace& tr = trace[l];

        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Wr(
            h.cpu_router_w[l].data(), E, d);

        Eigen::MatrixXf grad_blend = g;
        Eigen::MatrixXf grad_Xl = Eigen::MatrixXf::Zero(S, d);
        grad_Xl += grad_blend;

        Eigen::VectorXf grad_p(E);
        for (uint32_t e = 0; e < E; ++e)
            grad_p[e] = (tr.Y[e].array() * grad_blend.array()).sum();

        Eigen::VectorXf grad_logits_r = softmax_grad(tr.p, grad_p);
        Eigen::VectorXf grad_xm = Wr.transpose() * grad_logits_r;

        for (uint32_t s = 0; s < S; ++s)
            grad_Xl.row(s) += (1.0f / float(S)) * grad_xm.transpose();

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> gWr(
            out.grad_router_w[l].data(), E, d);
        gWr = grad_logits_r * tr.xmean.transpose();

        Eigen::Map<Eigen::VectorXf> gb(out.grad_router_b[l].data(), E);
        gb = grad_logits_r;

        for (uint32_t e = 0; e < E; ++e) {
            Eigen::MatrixXf grad_Y = tr.p[e] * grad_blend;
            Eigen::Map<const RowMajor> We(h.cpu_expert_w[l * E + e].data(), d, d);
            grad_Xl += grad_Y * We;

            Eigen::MatrixXf grad_We = grad_Y.transpose() * cur;
            std::memcpy(out.grad_experts[l * E + e].data(), grad_We.data(),
                        d * d * sizeof(float));
        }

        g = grad_Xl;
    }

    out.grad_embed.assign(h.vocab * d, 0.f);
    out.grad_pos.assign(h.maxSeq * d, 0.f);

    for (uint32_t s = 0; s < S; ++s) {
        int32_t tok = input_ids[s];
        for (uint32_t j = 0; j < d; ++j) {
            out.grad_embed[tok * d + j] += g(s, j);
            out.grad_pos[s * d + j] = g(s, j);
        }
    }

    return out;
}

MoeGradients moe_backward_gpu(CommandBatch& batch, BufferPool& pool, PipelineCache& cache,
                              MoeHandleCache& h, const int32_t* input_ids, uint32_t seq_len,
                              const float* grad_logits) {

    // GPU backward shaders exist but Eigen CPU is faster for now
    // (strided memory access pattern in backward shader kills GPU perf).
    // TODO: transpose expert weights for backward-friendly layout.
    return moe_backward_cpu(h, input_ids, seq_len, grad_logits);

    bool has_bwd_vec4 = cache.hasShader("moe-layer-backward-vec4") && (h.d % 4 == 0);
    bool has_bwd = has_bwd_vec4 || cache.hasShader("moe-layer-backward");
    bool has_gw = cache.hasShader("moe-layer-grad-weight");

    if (!has_bwd)
        return moe_backward_cpu(h, input_ids, seq_len, grad_logits);

    uint32_t S = seq_len;
    uint32_t d = h.d;
    uint32_t V = h.vocab;
    uint32_t L = h.nLayers;
    uint32_t E = h.nExperts;
    size_t sd = S * d * sizeof(float);
    size_t packed_size = size_t(E) * d * d * sizeof(float);

    // Re-compute router weights on CPU (tiny)
    std::vector<std::vector<float>> router_p(L);
    for (uint32_t l = 0; l < L; ++l) {
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Wr(
            h.cpu_router_w[l].data(), E, d);
        Eigen::Map<const Eigen::VectorXf> b(h.cpu_router_b[l].data(), E);

        std::vector<float> act(S * d);
        pool.download(h.bufActivations[l], act.data(), sd);
        Eigen::VectorXf xmean = Eigen::VectorXf::Zero(d);
        for (uint32_t i = 0; i < S; ++i)
            for (uint32_t j = 0; j < d; ++j)
                xmean[j] += act[i * d + j];
        xmean /= float(S);

        Eigen::VectorXf logits_v = Wr * xmean + b;
        Eigen::VectorXf p = softmax_vec(logits_v);
        router_p[l].resize(E);
        for (uint32_t e = 0; e < E; ++e)
            router_p[l][e] = p[e];
    }

    // Upload grad_logits
    GrillyBuffer bufGL = pool.acquire(S * V * sizeof(float));
    pool.upload(bufGL, grad_logits, S * V * sizeof(float));

    // Output projection backward: dx = grad_logits @ out_w (using fnn-linear)
    GrillyBuffer bufDx = pool.acquire(sd);
    batch.begin();
    batchedLinear(batch, cache, bufGL, h.outWt, nullptr, bufDx, S, V, d);
    batch.submitDeferred();
    batch.waitForCompletion();

    // grad_out_w = grad_logits.T @ x_final (CPU — vocab-sized, not worth GPU)
    std::vector<float> gl_cpu(S * V);
    pool.download(bufGL, gl_cpu.data(), S * V * sizeof(float));
    std::vector<float> xfinal(S * d);
    pool.download(h.bufActivations[L], xfinal.data(), sd);

    MoeGradients out;
    out.grad_out_w.resize(V * d, 0.f);
    {
        using RM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        Eigen::Map<const RM> GL(gl_cpu.data(), S, V);
        Eigen::Map<const RM> XF(xfinal.data(), S, d);
        RM GOW = GL.transpose() * XF;
        std::memcpy(out.grad_out_w.data(), GOW.data(), V * d * sizeof(float));
    }

    // Per-layer backward on GPU
    const char* bwd_name = has_bwd_vec4 ? "moe-layer-backward-vec4" : "moe-layer-backward";
    struct BwdPush { uint32_t seq_len, d_model, n_experts; float w0, w1, w2, w3; };

    PipelineEntry bwdPipe = cache.getOrCreate(bwd_name, 3, sizeof(BwdPush));

    GrillyBuffer bufGradIn = pool.acquire(sd);
    GrillyBuffer bufGradW = pool.acquire(packed_size);

    out.grad_experts.resize(L * E);
    out.grad_router_w.resize(L);
    out.grad_router_b.resize(L);

    bool has_gw_shader = has_gw;
    PipelineEntry gwPipe{};
    if (has_gw_shader)
        gwPipe = cache.getOrCreate("moe-layer-grad-weight", 3, sizeof(BwdPush));

    for (int32_t l = L - 1; l >= 0; --l) {
        auto& layer = h.layers[l];
        float w0 = router_p[l][0], w1 = router_p[l][1];
        float w2 = router_p[l][2], w3 = router_p[l][3];
        BwdPush bp{S, d, E, w0, w1, w2, w3};

        batch.begin();

        // grad_input via backward shader
        {
            std::vector<VkDescriptorBufferInfo> bufs = {
                {bufDx.handle, 0, sd},
                {layer.expertPacked.handle, 0, packed_size},
                {bufGradIn.handle, 0, sd},
            };
            VkDescriptorSet desc = cache.allocDescriptorSet(bwd_name, bufs);

            uint32_t gx, gy;
            if (has_bwd_vec4) {
                gx = ((d / 4) + 31) / 32;  // vec4 columns, 32-wide workgroup
                gy = (S + 7) / 8;           // 8-high workgroup
            } else {
                gx = (d + 15) / 16;
                gy = (S + 15) / 16;
            }
            batch.dispatch(bwdPipe.pipeline, bwdPipe.layout, desc, gx, gy, 1, &bp, sizeof(bp));
            batch.barrier();
        }

        // grad_weight via grad-weight shader (if available)
        if (has_gw_shader) {
            std::vector<VkDescriptorBufferInfo> bufs = {
                {bufDx.handle, 0, sd},
                {h.bufActivations[l].handle, 0, sd},
                {bufGradW.handle, 0, packed_size},
            };
            VkDescriptorSet desc = cache.allocDescriptorSet("moe-layer-grad-weight", bufs);
            uint32_t gx = (d + 15) / 16;
            uint32_t gy = (d + 15) / 16;
            batch.dispatch(gwPipe.pipeline, gwPipe.layout, desc, gx, gy, 1, &bp, sizeof(bp));
        }

        batch.submitDeferred();
        batch.waitForCompletion();

        // Download grad_W
        if (has_gw_shader) {
            std::vector<float> gw_packed(E * d * d);
            pool.download(bufGradW, gw_packed.data(), packed_size);
            for (uint32_t e = 0; e < E; ++e) {
                out.grad_experts[l * E + e].assign(d * d, 0.f);
                std::memcpy(out.grad_experts[l * E + e].data(),
                            gw_packed.data() + e * d * d, d * d * sizeof(float));
            }
        } else {
            // CPU fallback for grad_W
            std::vector<float> dx_cpu(S * d);
            pool.download(bufDx, dx_cpu.data(), sd);
            std::vector<float> act(S * d);
            pool.download(h.bufActivations[l], act.data(), sd);
            using RM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
            Eigen::Map<const RM> DX(dx_cpu.data(), S, d);
            Eigen::Map<const RM> ACT(act.data(), S, d);
            RM GW = DX.transpose() * ACT;
            for (uint32_t e = 0; e < E; ++e) {
                out.grad_experts[l * E + e].resize(d * d);
                float pe = router_p[l][e];
                for (size_t i = 0; i < d * d; ++i)
                    out.grad_experts[l * E + e][i] = pe * GW.data()[i];
            }
        }

        // Router gradient (CPU, tiny — skip for now, zero placeholder)
        out.grad_router_w[l].assign(E * d, 0.f);
        out.grad_router_b[l].assign(E, 0.f);

        // Swap for next layer
        std::swap(bufDx, bufGradIn);
    }

    // Embedding gradient: scatter-add dx
    std::vector<float> dx_final(S * d);
    pool.download(bufDx, dx_final.data(), sd);

    out.grad_embed.assign(V * d, 0.f);
    out.grad_pos.assign(h.maxSeq * d, 0.f);
    for (uint32_t s = 0; s < S; ++s) {
        int32_t tok = input_ids[s];
        if (tok >= 0 && static_cast<uint32_t>(tok) < V)
            for (uint32_t j = 0; j < d; ++j)
                out.grad_embed[tok * d + j] += dx_final[s * d + j];
        for (uint32_t j = 0; j < d; ++j)
            out.grad_pos[s * d + j] = dx_final[s * d + j];
    }

    pool.release(bufGL);
    pool.release(bufDx);
    pool.release(bufGradIn);
    pool.release(bufGradW);

    return out;
}

}  // namespace ops
}  // namespace grilly
