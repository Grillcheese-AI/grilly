#include "bindings_core.h"
#include "grilly/ops/eggroll.h"

static pybind11::dict eggroll_gen_impl(GrillyCoreContext& ctx, uint32_t d_out, uint32_t d_in, uint32_t n_workers, uint32_t seed, float sigma) {
    pybind11::array_t<float> u_data({(pybind11::ssize_t)d_out, (pybind11::ssize_t)n_workers});
    pybind11::array_t<float> v_data({(pybind11::ssize_t)d_in, (pybind11::ssize_t)n_workers});
    auto u_req = u_data.request();
    auto v_req = v_data.request();
    
    grilly::ops::EggrollGenParams p = {d_out, d_in, n_workers, seed, sigma};
    {
        pybind11::gil_scoped_release release;
        grilly::ops::eggrollGenerate(ctx.batch, ctx.pool, ctx.cache, (float*)u_req.ptr, (float*)v_req.ptr, p);
    }
    
    pybind11::dict out_dict;
    out_dict["U"] = u_data;
    out_dict["V"] = v_data;
    return out_dict;
}

static void eggroll_upd_impl(GrillyCoreContext& ctx, pybind11::array_t<float> weights, pybind11::array_t<float> merit, pybind11::array_t<float> u_pool, pybind11::array_t<float> v_pool, pybind11::array_t<uint32_t> top_idx, pybind11::array_t<float> top_fit, float m_inc, float m_dec) {
    auto w_req = weights.request();
    auto m_req = merit.request();
    auto up_req = u_pool.request();
    auto vp_req = v_pool.request();
    auto idx_req = top_idx.request();
    auto fit_req = top_fit.request();
    
    uint32_t d_out = (uint32_t)w_req.shape[0];
    uint32_t d_in = (uint32_t)w_req.shape[1];
    uint32_t n_workers = (uint32_t)up_req.shape[1];
    uint32_t top_k = (uint32_t)idx_req.shape[0];
    
    grilly::ops::EggrollUpdateParams p = {d_out, d_in, top_k, n_workers, m_inc, m_dec};
    {
        pybind11::gil_scoped_release release;
        grilly::ops::eggrollUpdate(ctx.batch, ctx.pool, ctx.cache, (float*)w_req.ptr, (float*)m_req.ptr, (const float*)up_req.ptr, (const float*)vp_req.ptr, (const uint32_t*)idx_req.ptr, (const float*)fit_req.ptr, p);
    }
}

void register_eggroll_ops(pybind11::module_& m) {
    m.def("eggroll_generate", &eggroll_gen_impl);
    m.def("eggroll_update", &eggroll_upd_impl);
}
