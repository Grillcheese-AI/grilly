#include "bindings_core.h"
#include "grilly/ops/bandit.h"

static pybind11::dict solver_impl(GrillyCoreContext& ctx, pybind11::array_t<float> mu, pybind11::array_t<float> n, uint32_t iters, float delta) {
    auto m_info = mu.request();
    auto n_info = n.request();
    uint32_t k = (uint32_t)m_info.shape[0];
    uint32_t ni = (uint32_t)m_info.shape[1];
    grilly::ops::BanditParams p = {k, ni, iters, delta};
    pybind11::array_t<float> tw({(pybind11::ssize_t)k, (pybind11::ssize_t)ni});
    pybind11::array_t<uint32_t> sf({(pybind11::ssize_t)ni});
    auto tw_info = tw.request();
    auto sf_info = sf.request();
    {
        pybind11::gil_scoped_release release;
        grilly::ops::banditSolve(ctx.batch, ctx.pool, ctx.cache, (const float*)m_info.ptr, (const float*)n_info.ptr, (float*)tw_info.ptr, (uint32_t*)sf_info.ptr, p);
    }
    pybind11::dict r;
    r["target_w"] = tw;
    r["stop_flags"] = sf;
    return r;
}

void register_bandit_ops(pybind11::module_& m) {
    m.def("bandit_solve", &solver_impl);
}
