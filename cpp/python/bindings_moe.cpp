/// bindings_moe.cpp — fused MoE upload / forward / backward / weight update.
#include "bindings_core.h"
#include "grilly/ops/moe_forward.h"

#include <cstring>
#include <mutex>
#include <vector>

using namespace grilly;

void register_moe_ops(py::module_& m) {

    m.def(
        "moe_upload",
        [](GrillyCoreContext& ctx, py::array_t<float> embed_w, py::array_t<float> pos_w,
           py::list expert_ws, py::list router_ws, py::list router_bs, py::array_t<float> out_w,
           uint32_t n_layers, uint32_t n_experts) -> int {
            auto eb = embed_w.request();
            auto pb = pos_w.request();
            auto ob = out_w.request();
            require_c_contiguous_float(eb);
            require_c_contiguous_float(pb);
            require_c_contiguous_float(ob);

            if (eb.ndim != 2 || pb.ndim != 2 || ob.ndim != 2)
                throw std::runtime_error("moe_upload: embed/pos/out must be 2-D");
            uint32_t vocab = static_cast<uint32_t>(eb.shape[0]);
            uint32_t d = static_cast<uint32_t>(eb.shape[1]);
            if (static_cast<uint32_t>(ob.shape[1]) != d)
                throw std::runtime_error("moe_upload: out_w second dim must match d");
            if (static_cast<uint32_t>(ob.shape[0]) != vocab)
                throw std::runtime_error("moe_upload: out_w first dim must match vocab");
            uint32_t max_seq = static_cast<uint32_t>(pb.shape[0]);
            if (static_cast<uint32_t>(pb.shape[1]) != d)
                throw std::runtime_error("moe_upload: pos_w second dim must match d");

            std::vector<const float*> exp_ptrs;
            std::vector<py::array_t<float>> exp_keep;
            for (auto& item : expert_ws) {
                auto arr = item.cast<py::array_t<float>>();
                auto wb = arr.request();
                require_c_contiguous_float(wb);
                if (wb.ndim != 2 || static_cast<uint32_t>(wb.shape[0]) != d ||
                    static_cast<uint32_t>(wb.shape[1]) != d)
                    throw std::runtime_error("moe_upload: each expert must be (d, d)");
                exp_keep.push_back(arr);
                exp_ptrs.push_back(static_cast<const float*>(wb.ptr));
            }

            std::vector<const float*> rW_ptrs;
            std::vector<const float*> rb_ptrs;
            std::vector<py::array_t<float>> rW_keep;
            std::vector<py::array_t<float>> rb_keep;
            for (uint32_t l = 0; l < n_layers; ++l) {
                auto rw = router_ws[l].cast<py::array_t<float>>();
                auto rb = router_bs[l].cast<py::array_t<float>>();
                auto rwb = rw.request();
                auto rbb = rb.request();
                require_c_contiguous_float(rwb);
                require_c_contiguous_float(rbb);
                if (rwb.ndim != 2 || static_cast<uint32_t>(rwb.shape[0]) != n_experts ||
                    static_cast<uint32_t>(rwb.shape[1]) != d)
                    throw std::runtime_error("moe_upload: router_w must be (n_experts, d)");
                if (rbb.ndim != 1 || static_cast<uint32_t>(rbb.shape[0]) != n_experts)
                    throw std::runtime_error("moe_upload: router_b must be (n_experts,)");
                rW_keep.push_back(rw);
                rb_keep.push_back(rb);
                rW_ptrs.push_back(static_cast<const float*>(rwb.ptr));
                rb_ptrs.push_back(static_cast<const float*>(rbb.ptr));
            }

            int handle = ops::moe_upload(
                ctx.pool, vocab, d, max_seq,
                static_cast<const float*>(eb.ptr),
                static_cast<const float*>(pb.ptr),
                exp_ptrs, rW_ptrs, rb_ptrs,
                static_cast<const float*>(ob.ptr),
                n_layers, n_experts);

            ctx.waitIdle();
            return handle;
        },
        py::arg("device"), py::arg("embed_w"), py::arg("pos_w"), py::arg("expert_ws"),
        py::arg("router_ws"), py::arg("router_bs"), py::arg("out_w"),
        py::arg("n_layers"), py::arg("n_experts"),
        "Upload MoE weights to GPU; returns opaque integer handle.");

    m.def(
        "moe_release",
        [](GrillyCoreContext& ctx, int handle) { ops::moe_release(ctx.pool, handle); },
        py::arg("device"), py::arg("handle"), "Free GPU buffers for a MoE handle.");

    m.def(
        "moe_forward",
        [](GrillyCoreContext& ctx, int handle, py::array_t<int32_t> input_ids)
            -> py::array_t<float> {
            auto& cache = ops::moe_get_cache(handle);
            auto ib = input_ids.request();
            require_c_contiguous_int32(ib);
            if (ib.ndim != 1)
                throw std::runtime_error("moe_forward: input_ids must be 1-D");
            uint32_t seq_len = static_cast<uint32_t>(ib.shape[0]);
            uint32_t V = cache.vocab;

            py::array_t<float> logits({static_cast<py::ssize_t>(seq_len),
                                       static_cast<py::ssize_t>(V)});

            {
                py::gil_scoped_release release;
                std::lock_guard<std::mutex> lock(ctx.ctx_mutex);
                ops::moe_forward_gpu(ctx.batch, ctx.pool, ctx.cache, cache,
                                     static_cast<const int32_t*>(ib.ptr), seq_len,
                                     logits.mutable_data());
            }
            return logits;
        },
        py::arg("device"), py::arg("handle"), py::arg("input_ids"),
        "Run full MoE forward on GPU (router/blend on CPU). Returns (seq_len, vocab).");

    m.def(
        "moe_update_weights",
        [](GrillyCoreContext& ctx, int handle, py::array_t<float> embed_w,
           py::array_t<float> pos_w, py::list expert_ws, py::list router_ws,
           py::list router_bs, py::array_t<float> out_w) {
            auto& h = ops::moe_get_cache(handle);
            auto eb = embed_w.request();
            auto pb = pos_w.request();
            auto ob = out_w.request();
            require_c_contiguous_float(eb);
            require_c_contiguous_float(pb);
            require_c_contiguous_float(ob);

            uint32_t L = h.nLayers;
            uint32_t E = h.nExperts;

            std::vector<const float*> exp_ptrs;
            std::vector<py::array_t<float>> exp_keep;
            for (auto& item : expert_ws) {
                auto arr = item.cast<py::array_t<float>>();
                auto wb = arr.request();
                require_c_contiguous_float(wb);
                exp_keep.push_back(arr);
                exp_ptrs.push_back(static_cast<const float*>(wb.ptr));
            }

            std::vector<const float*> rW_ptrs;
            std::vector<const float*> rb_ptrs;
            for (uint32_t l = 0; l < L; ++l) {
                auto rw = router_ws[l].cast<py::array_t<float>>();
                auto rb = router_bs[l].cast<py::array_t<float>>();
                auto rwb = rw.request();
                auto rbb = rb.request();
                require_c_contiguous_float(rwb);
                require_c_contiguous_float(rbb);
                rW_ptrs.push_back(static_cast<const float*>(rwb.ptr));
                rb_ptrs.push_back(static_cast<const float*>(rbb.ptr));
            }

            {
                py::gil_scoped_release release;
                std::lock_guard<std::mutex> lock(ctx.ctx_mutex);
                ops::moe_update_weights(
                    ctx.pool, h,
                    static_cast<const float*>(eb.ptr),
                    static_cast<const float*>(pb.ptr),
                    exp_ptrs, rW_ptrs, rb_ptrs,
                    static_cast<const float*>(ob.ptr));
                ctx.waitIdle();
            }
        },
        py::arg("device"), py::arg("handle"), py::arg("embed_w"), py::arg("pos_w"),
        py::arg("expert_ws"), py::arg("router_ws"), py::arg("router_bs"), py::arg("out_w"),
        "Re-upload weights in place after an optimizer step.");

    m.def(
        "moe_backward",
        [](GrillyCoreContext& ctx, int handle, py::array_t<int32_t> input_ids,
           py::array_t<float> grad_logits) -> py::dict {
            auto& h = ops::moe_get_cache(handle);
            auto ib = input_ids.request();
            auto gb = grad_logits.request();
            require_c_contiguous_int32(ib);
            require_c_contiguous_float(gb);
            if (ib.ndim != 1)
                throw std::runtime_error("moe_backward: input_ids must be 1-D");
            if (gb.ndim != 2)
                throw std::runtime_error("moe_backward: grad_logits must be 2-D");
            uint32_t seq_len = static_cast<uint32_t>(ib.shape[0]);
            uint32_t V = h.vocab;
            if (static_cast<uint32_t>(gb.shape[0]) != seq_len ||
                static_cast<uint32_t>(gb.shape[1]) != V)
                throw std::runtime_error("moe_backward: grad_logits shape mismatch");

            ops::MoeGradients grads;
            {
                py::gil_scoped_release release;
                std::lock_guard<std::mutex> lock(ctx.ctx_mutex);
                grads = ops::moe_backward_gpu(
                    ctx.batch, ctx.pool, ctx.cache, h,
                    static_cast<const int32_t*>(ib.ptr), seq_len,
                    static_cast<const float*>(gb.ptr));
            }

            py::dict d;
            py::array_t<float> grad_embed({static_cast<py::ssize_t>(h.vocab),
                                           static_cast<py::ssize_t>(h.d)});
            std::memcpy(grad_embed.mutable_data(), grads.grad_embed.data(),
                        grads.grad_embed.size() * sizeof(float));
            d["grad_embed"] = grad_embed;

            py::array_t<float> grad_pos({static_cast<py::ssize_t>(h.maxSeq),
                                         static_cast<py::ssize_t>(h.d)});
            std::memcpy(grad_pos.mutable_data(), grads.grad_pos.data(),
                        grads.grad_pos.size() * sizeof(float));
            d["grad_pos"] = grad_pos;

            py::list ge;
            for (size_t i = 0; i < grads.grad_experts.size(); ++i) {
                py::array_t<float> gex({static_cast<py::ssize_t>(h.d),
                                        static_cast<py::ssize_t>(h.d)});
                std::memcpy(gex.mutable_data(), grads.grad_experts[i].data(),
                            h.d * h.d * sizeof(float));
                ge.append(gex);
            }
            d["grad_experts"] = ge;

            py::list grw;
            py::list grb;
            for (uint32_t l = 0; l < h.nLayers; ++l) {
                py::array_t<float> gw({static_cast<py::ssize_t>(h.nExperts),
                                       static_cast<py::ssize_t>(h.d)});
                std::memcpy(gw.mutable_data(), grads.grad_router_w[l].data(),
                            h.nExperts * h.d * sizeof(float));
                grw.append(gw);
                py::array_t<float> gbv({static_cast<py::ssize_t>(h.nExperts)});
                std::memcpy(gbv.mutable_data(), grads.grad_router_b[l].data(),
                            h.nExperts * sizeof(float));
                grb.append(gbv);
            }
            d["grad_routers_W"] = grw;
            d["grad_routers_b"] = grb;

            py::array_t<float> gow({static_cast<py::ssize_t>(h.vocab),
                                    static_cast<py::ssize_t>(h.d)});
            std::memcpy(gow.mutable_data(), grads.grad_out_w.data(),
                        grads.grad_out_w.size() * sizeof(float));
            d["grad_out_w"] = gow;
            return d;
        },
        py::arg("device"), py::arg("handle"), py::arg("input_ids"),
        py::arg("grad_logits"),
        "Backward pass (GPU path when available, CPU fallback) using uploaded MoE weights.");
}
