/// bindings_vsa_lm.cpp — fused VSA-LM upload / forward / backward / release.
#include "bindings_core.h"
#include "grilly/ops/vsa_lm_forward.h"

#include <cstring>
#include <mutex>
#include <vector>

using namespace grilly;

void register_vsa_lm_ops(py::module_& m) {

    m.def(
        "vsa_lm_upload",
        [](GrillyCoreContext& ctx,
           py::array_t<float> embed_w, py::array_t<float> pos_w,
           py::list ffn_up_patterns, py::list ffn_up_biases,
           py::list ffn_down_patterns, py::list ffn_down_biases,
           py::list ln_gammas, py::list ln_betas,
           py::array_t<float> out_w,
           uint32_t n_layers, uint32_t d_model, uint32_t d_ffn) -> int {

            auto eb = embed_w.request();
            auto pb = pos_w.request();
            auto ob = out_w.request();
            require_c_contiguous_float(eb);
            require_c_contiguous_float(pb);
            require_c_contiguous_float(ob);

            if (eb.ndim != 2 || pb.ndim != 2 || ob.ndim != 2)
                throw std::runtime_error("vsa_lm_upload: embed/pos/out must be 2-D");
            uint32_t vocab = static_cast<uint32_t>(eb.shape[0]);
            uint32_t d = static_cast<uint32_t>(eb.shape[1]);
            if (d != d_model)
                throw std::runtime_error("vsa_lm_upload: embed_w dim-1 must match d_model");
            uint32_t max_seq = static_cast<uint32_t>(pb.shape[0]);

            auto extract_list = [](py::list& lst, uint32_t n,
                                   std::vector<const float*>& ptrs,
                                   std::vector<py::array_t<float>>& keep) {
                for (uint32_t i = 0; i < n; ++i) {
                    auto arr = lst[i].cast<py::array_t<float>>();
                    auto buf = arr.request();
                    require_c_contiguous_float(buf);
                    keep.push_back(arr);
                    ptrs.push_back(static_cast<const float*>(buf.ptr));
                }
            };

            std::vector<const float*> up_w_ptrs, up_b_ptrs, dn_w_ptrs, dn_b_ptrs;
            std::vector<const float*> gm_ptrs, bt_ptrs;
            std::vector<py::array_t<float>> keep1, keep2, keep3, keep4, keep5, keep6;

            extract_list(ffn_up_patterns,  n_layers, up_w_ptrs, keep1);
            extract_list(ffn_up_biases,    n_layers, up_b_ptrs, keep2);
            extract_list(ffn_down_patterns, n_layers, dn_w_ptrs, keep3);
            extract_list(ffn_down_biases,  n_layers, dn_b_ptrs, keep4);
            extract_list(ln_gammas,        n_layers, gm_ptrs,   keep5);
            extract_list(ln_betas,         n_layers, bt_ptrs,   keep6);

            int handle = ops::vsa_lm_upload(
                ctx.pool, vocab, d, d_ffn, max_seq,
                static_cast<const float*>(eb.ptr),
                static_cast<const float*>(pb.ptr),
                up_w_ptrs, up_b_ptrs, dn_w_ptrs, dn_b_ptrs,
                gm_ptrs, bt_ptrs,
                static_cast<const float*>(ob.ptr),
                n_layers);

            ctx.waitIdle();
            return handle;
        },
        py::arg("device"), py::arg("embed_w"), py::arg("pos_w"),
        py::arg("ffn_up_patterns"), py::arg("ffn_up_biases"),
        py::arg("ffn_down_patterns"), py::arg("ffn_down_biases"),
        py::arg("ln_gammas"), py::arg("ln_betas"),
        py::arg("out_w"),
        py::arg("n_layers"), py::arg("d_model"), py::arg("d_ffn"),
        "Upload VSA-LM weights to GPU; returns opaque integer handle.");

    m.def(
        "vsa_lm_release",
        [](GrillyCoreContext& ctx, int handle) {
            ops::vsa_lm_release(ctx.pool, handle);
        },
        py::arg("device"), py::arg("handle"),
        "Free GPU buffers for a VSA-LM handle.");

    m.def(
        "vsa_lm_forward",
        [](GrillyCoreContext& ctx, int handle, py::array_t<int32_t> input_ids)
            -> py::array_t<float> {
            auto& h = ops::vsa_lm_get_cache(handle);
            auto ib = input_ids.request();
            require_c_contiguous_int32(ib);
            if (ib.ndim != 1)
                throw std::runtime_error("vsa_lm_forward: input_ids must be 1-D");
            uint32_t seq_len = static_cast<uint32_t>(ib.shape[0]);

            py::array_t<float> logits({static_cast<py::ssize_t>(seq_len),
                                       static_cast<py::ssize_t>(h.vocab)});
            {
                py::gil_scoped_release release;
                std::lock_guard<std::mutex> lock(ctx.ctx_mutex);
                ops::vsa_lm_forward_gpu(ctx.batch, ctx.pool, ctx.cache, h,
                                        static_cast<const int32_t*>(ib.ptr),
                                        seq_len, logits.mutable_data());
            }
            return logits;
        },
        py::arg("device"), py::arg("handle"), py::arg("input_ids"),
        "Run full VSA-LM forward on GPU. Returns (seq_len, vocab) float32.");

    m.def(
        "vsa_lm_backward",
        [](GrillyCoreContext& ctx, int handle, py::array_t<int32_t> input_ids,
           py::array_t<float> grad_logits) -> py::dict {
            auto& h = ops::vsa_lm_get_cache(handle);
            auto ib = input_ids.request();
            auto gb = grad_logits.request();
            require_c_contiguous_int32(ib);
            require_c_contiguous_float(gb);
            if (ib.ndim != 1)
                throw std::runtime_error("vsa_lm_backward: input_ids must be 1-D");
            if (gb.ndim != 2)
                throw std::runtime_error("vsa_lm_backward: grad_logits must be 2-D");
            uint32_t seq_len = static_cast<uint32_t>(ib.shape[0]);
            uint32_t V = h.vocab;
            uint32_t d = h.d;
            uint32_t dF = h.dFfn;
            uint32_t L = h.nLayers;

            if (static_cast<uint32_t>(gb.shape[0]) != seq_len ||
                static_cast<uint32_t>(gb.shape[1]) != V)
                throw std::runtime_error("vsa_lm_backward: grad_logits shape mismatch");

            ops::VsaLmGradients grads;
            {
                py::gil_scoped_release release;
                grads = ops::vsa_lm_backward_cpu(
                    h, static_cast<const int32_t*>(ib.ptr), seq_len,
                    static_cast<const float*>(gb.ptr));
            }

            py::dict result;

            // grad_embed (vocab, d)
            py::array_t<float> ge({static_cast<py::ssize_t>(V),
                                   static_cast<py::ssize_t>(d)});
            std::memcpy(ge.mutable_data(), grads.grad_embed.data(),
                        grads.grad_embed.size() * sizeof(float));
            result["grad_embed"] = ge;

            // grad_pos (max_seq, d)
            py::array_t<float> gp({static_cast<py::ssize_t>(h.maxSeq),
                                   static_cast<py::ssize_t>(d)});
            std::memcpy(gp.mutable_data(), grads.grad_pos.data(),
                        grads.grad_pos.size() * sizeof(float));
            result["grad_pos"] = gp;

            // grad_out_w (vocab, d)
            py::array_t<float> gow({static_cast<py::ssize_t>(V),
                                    static_cast<py::ssize_t>(d)});
            std::memcpy(gow.mutable_data(), grads.grad_out_w.data(),
                        grads.grad_out_w.size() * sizeof(float));
            result["grad_out_w"] = gow;

            // Per-layer gradients as lists
            py::list g_up_w, g_up_b, g_dn_w, g_dn_b, g_ln_g, g_ln_b;
            for (uint32_t l = 0; l < L; ++l) {
                py::array_t<float> uw({static_cast<py::ssize_t>(dF),
                                       static_cast<py::ssize_t>(d)});
                std::memcpy(uw.mutable_data(), grads.grad_ffn_up_w[l].data(),
                            dF * d * sizeof(float));
                g_up_w.append(uw);

                py::array_t<float> ub({static_cast<py::ssize_t>(dF)});
                std::memcpy(ub.mutable_data(), grads.grad_ffn_up_b[l].data(),
                            dF * sizeof(float));
                g_up_b.append(ub);

                py::array_t<float> dw({static_cast<py::ssize_t>(d),
                                       static_cast<py::ssize_t>(dF)});
                std::memcpy(dw.mutable_data(), grads.grad_ffn_down_w[l].data(),
                            d * dF * sizeof(float));
                g_dn_w.append(dw);

                py::array_t<float> db({static_cast<py::ssize_t>(d)});
                std::memcpy(db.mutable_data(), grads.grad_ffn_down_b[l].data(),
                            d * sizeof(float));
                g_dn_b.append(db);

                py::array_t<float> lg({static_cast<py::ssize_t>(d)});
                std::memcpy(lg.mutable_data(), grads.grad_ln_gamma[l].data(),
                            d * sizeof(float));
                g_ln_g.append(lg);

                py::array_t<float> lb({static_cast<py::ssize_t>(d)});
                std::memcpy(lb.mutable_data(), grads.grad_ln_beta[l].data(),
                            d * sizeof(float));
                g_ln_b.append(lb);
            }
            result["grad_ffn_up_w"]   = g_up_w;
            result["grad_ffn_up_b"]   = g_up_b;
            result["grad_ffn_down_w"] = g_dn_w;
            result["grad_ffn_down_b"] = g_dn_b;
            result["grad_ln_gamma"]   = g_ln_g;
            result["grad_ln_beta"]    = g_ln_b;

            return result;
        },
        py::arg("device"), py::arg("handle"), py::arg("input_ids"),
        py::arg("grad_logits"),
        "CPU backward for VSA-LM. Returns dict with all gradient arrays.");

    m.def(
        "vsa_lm_update_weights",
        [](GrillyCoreContext& ctx, int handle,
           py::array_t<float> embed_w, py::array_t<float> pos_w,
           py::list ffn_up_patterns, py::list ffn_up_biases,
           py::list ffn_down_patterns, py::list ffn_down_biases,
           py::list ln_gammas, py::list ln_betas,
           py::array_t<float> out_w) {

            auto& h = ops::vsa_lm_get_cache(handle);
            auto eb = embed_w.request();
            auto pb = pos_w.request();
            auto ob = out_w.request();
            require_c_contiguous_float(eb);
            require_c_contiguous_float(pb);
            require_c_contiguous_float(ob);

            uint32_t d  = h.d;
            uint32_t dF = h.dFfn;
            uint32_t V  = h.vocab;
            uint32_t L  = h.nLayers;

            // Re-upload embedding
            size_t embed_bytes = size_t(V) * d * sizeof(float);
            std::memcpy(h.cpu_embed.data(), static_cast<const float*>(eb.ptr), embed_bytes);
            ctx.pool.upload(h.embedW, static_cast<const float*>(eb.ptr), embed_bytes);

            // Re-upload pos
            size_t pos_bytes = size_t(h.maxSeq) * d * sizeof(float);
            std::memcpy(h.cpu_pos.data(), static_cast<const float*>(pb.ptr), pos_bytes);
            ctx.pool.upload(h.posW, static_cast<const float*>(pb.ptr), pos_bytes);

            // Re-upload out_w + transpose
            size_t out_bytes = size_t(V) * d * sizeof(float);
            std::memcpy(h.cpu_out_w.data(), static_cast<const float*>(ob.ptr), out_bytes);
            ctx.pool.upload(h.outW, static_cast<const float*>(ob.ptr), out_bytes);
            std::vector<float> out_wt(d * V);
            for (uint32_t r = 0; r < V; ++r)
                for (uint32_t c = 0; c < d; ++c)
                    out_wt[c * V + r] = h.cpu_out_w[r * d + c];
            ctx.pool.upload(h.outWt, out_wt.data(), out_wt.size() * sizeof(float));

            // Per-layer
            for (uint32_t l = 0; l < L; ++l) {
                auto& lw = h.layers[l];
                auto upw  = ffn_up_patterns[l].cast<py::array_t<float>>();
                auto upb  = ffn_up_biases[l].cast<py::array_t<float>>();
                auto dnw  = ffn_down_patterns[l].cast<py::array_t<float>>();
                auto dnb  = ffn_down_biases[l].cast<py::array_t<float>>();
                auto gm   = ln_gammas[l].cast<py::array_t<float>>();
                auto bt   = ln_betas[l].cast<py::array_t<float>>();

                auto upwb = upw.request();
                auto upbb = upb.request();
                auto dnwb = dnw.request();
                auto dnbb = dnb.request();
                auto gmb  = gm.request();
                auto btb  = bt.request();

                size_t uw_bytes = size_t(dF) * d * sizeof(float);
                size_t ub_bytes = size_t(dF) * sizeof(float);
                size_t dw_bytes = size_t(d) * dF * sizeof(float);
                size_t db_bytes = size_t(d) * sizeof(float);
                size_t ln_bytes = size_t(d) * sizeof(float);

                ctx.pool.upload(lw.ffnUpW, static_cast<const float*>(upwb.ptr), uw_bytes);
                ctx.pool.upload(lw.ffnUpB, static_cast<const float*>(upbb.ptr), ub_bytes);
                ctx.pool.upload(lw.ffnDownW, static_cast<const float*>(dnwb.ptr), dw_bytes);
                ctx.pool.upload(lw.ffnDownB, static_cast<const float*>(dnbb.ptr), db_bytes);
                ctx.pool.upload(lw.lnGamma, static_cast<const float*>(gmb.ptr), ln_bytes);
                ctx.pool.upload(lw.lnBeta, static_cast<const float*>(btb.ptr), ln_bytes);

                std::memcpy(h.cpu_ffn_up_w[l].data(), static_cast<const float*>(upwb.ptr), uw_bytes);
                std::memcpy(h.cpu_ffn_up_b[l].data(), static_cast<const float*>(upbb.ptr), ub_bytes);
                std::memcpy(h.cpu_ffn_down_w[l].data(), static_cast<const float*>(dnwb.ptr), dw_bytes);
                std::memcpy(h.cpu_ffn_down_b[l].data(), static_cast<const float*>(dnbb.ptr), db_bytes);
                std::memcpy(h.cpu_ln_gamma[l].data(), static_cast<const float*>(gmb.ptr), ln_bytes);
                std::memcpy(h.cpu_ln_beta[l].data(), static_cast<const float*>(btb.ptr), ln_bytes);
            }

            ctx.waitIdle();
        },
        py::arg("device"), py::arg("handle"),
        py::arg("embed_w"), py::arg("pos_w"),
        py::arg("ffn_up_patterns"), py::arg("ffn_up_biases"),
        py::arg("ffn_down_patterns"), py::arg("ffn_down_biases"),
        py::arg("ln_gammas"), py::arg("ln_betas"),
        py::arg("out_w"),
        "Re-upload VSA-LM weights after optimizer step.");
}
