/// bindings_snn.cpp — SNN standalone op bindings.
///
/// Migrated from monolithic bindings.cpp to use Tensor-based I/O.

#include "bindings_core.h"
#include "grilly/ops/snn.h"
#include "grilly/ops/learning.h"

void register_snn_ops(py::module_& m) {
    using namespace grilly::nn;

    // ── LIF neuron step ──────────────────────────────────────────────────
    m.def(
        "lif_step",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input, py::array_t<float> v_mem,
           py::array_t<float> t_refrac,
           float dt, float tau_mem, float v_rest, float v_reset,
           float v_thresh, float r_mem,
           float t_refrac_period) -> py::dict {
            auto inBuf = input.request();
            require_c_contiguous_float(inBuf);
            uint32_t n = 1;
            for (int i = 0; i < inBuf.ndim; ++i)
                n *= static_cast<uint32_t>(inBuf.shape[i]);

            py::array_t<float> vMemOut(v_mem.request().shape);
            py::array_t<float> refracOut(t_refrac.request().shape);
            py::array_t<float> spikes(inBuf.shape);
            auto vmIn = v_mem.request();
            auto trIn = t_refrac.request();
            require_c_contiguous_float(vmIn);
            require_c_contiguous_float(trIn);
            auto vmOut = vMemOut.request();
            auto rfOut = refracOut.request();
            auto spOut = spikes.request();

            std::memcpy(vmOut.ptr, vmIn.ptr, n * sizeof(float));
            std::memcpy(rfOut.ptr, trIn.ptr, n * sizeof(float));

            grilly::ops::LIFParams p{n, dt, tau_mem, v_rest, v_reset,
                                     v_thresh, r_mem, t_refrac_period};

            {
                py::gil_scoped_release release;
                grilly::ops::lifStep(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(inBuf.ptr),
                    static_cast<float*>(vmOut.ptr),
                    static_cast<float*>(rfOut.ptr),
                    static_cast<float*>(spOut.ptr), p);
            }

            py::dict result;
            result["spikes"] = Tensor::from_numpy(spikes);
            result["v_mem"] = Tensor::from_numpy(vMemOut);
            result["t_refrac"] = Tensor::from_numpy(refracOut);
            return result;
        },
        py::arg("device"), py::arg("input"), py::arg("v_mem"),
        py::arg("t_refrac"),
        py::arg("dt") = 1.0f, py::arg("tau_mem") = 20.0f,
        py::arg("v_rest") = 0.0f, py::arg("v_reset") = 0.0f,
        py::arg("v_thresh") = 1.0f, py::arg("r_mem") = 1.0f,
        py::arg("t_refrac_period") = 0.0f,
        "GPU LIF neuron step");

    // ── SNN node forward (IF/LIF/PLIF) ──────────────────────────────────
    m.def(
        "snn_node_forward",
        [](GrillyCoreContext& ctx,
           py::array_t<float> x_in, py::array_t<float> v_mem,
           py::array_t<float> tau_param,
           uint32_t neuron_type, float tau, float v_threshold,
           float v_reset, uint32_t reset_mode,
           uint32_t decay_input) -> py::dict {
            auto xBuf = x_in.request();
            require_c_contiguous_float(xBuf);
            uint32_t n = 1;
            for (int i = 0; i < xBuf.ndim; ++i)
                n *= static_cast<uint32_t>(xBuf.shape[i]);

            py::array_t<float> vMemOut(v_mem.request().shape);
            py::array_t<float> spikes(xBuf.shape);
            py::array_t<float> hOut(xBuf.shape);
            auto vmIn = v_mem.request();
            require_c_contiguous_float(vmIn);
            auto tpIn = tau_param.request();
            require_c_contiguous_float(tpIn);
            auto vmOut = vMemOut.request();
            auto spOut = spikes.request();
            auto hR = hOut.request();

            std::memcpy(vmOut.ptr, vmIn.ptr, n * sizeof(float));

            grilly::ops::SNNNodeForwardParams p{
                n, neuron_type, tau, v_threshold, v_reset,
                reset_mode, decay_input};

            {
                py::gil_scoped_release release;
                grilly::ops::snnNodeForward(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(xBuf.ptr),
                    static_cast<float*>(vmOut.ptr),
                    static_cast<float*>(spOut.ptr),
                    static_cast<float*>(hR.ptr),
                    static_cast<const float*>(tpIn.ptr), p);
            }

            py::dict result;
            result["spikes"] = Tensor::from_numpy(spikes);
            result["v_mem"] = Tensor::from_numpy(vMemOut);
            result["h_out"] = Tensor::from_numpy(hOut);
            return result;
        },
        py::arg("device"), py::arg("x_in"), py::arg("v_mem"),
        py::arg("tau_param"),
        py::arg("neuron_type") = 1,
        py::arg("tau") = 2.0f, py::arg("v_threshold") = 1.0f,
        py::arg("v_reset") = 0.0f, py::arg("reset_mode") = 0,
        py::arg("decay_input") = 0,
        "GPU SNN node forward (IF/LIF/PLIF)");

    // ── SNN node backward (surrogate gradient) ──────────────────────────
    m.def(
        "snn_node_backward",
        [](GrillyCoreContext& ctx,
           py::array_t<float> grad_spike, py::array_t<float> h_cache,
           float alpha, uint32_t surrogate_type,
           float v_threshold) -> Tensor {
            auto gBuf = grad_spike.request();
            require_c_contiguous_float(gBuf);
            uint32_t n = 1;
            for (int i = 0; i < gBuf.ndim; ++i)
                n *= static_cast<uint32_t>(gBuf.shape[i]);

            py::array_t<float> gradX(gBuf.shape);
            auto hBuf = h_cache.request();
            require_c_contiguous_float(hBuf);
            auto gxBuf = gradX.request();

            grilly::ops::SNNNodeBackwardParams p{
                n, alpha, surrogate_type, v_threshold};

            {
                py::gil_scoped_release release;
                grilly::ops::snnNodeBackward(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(gBuf.ptr),
                    static_cast<const float*>(hBuf.ptr),
                    static_cast<float*>(gxBuf.ptr), p);
            }

            return Tensor::from_numpy(gradX);
        },
        py::arg("device"), py::arg("grad_spike"), py::arg("h_cache"),
        py::arg("alpha") = 2.0f, py::arg("surrogate_type") = 0,
        py::arg("v_threshold") = 1.0f,
        "GPU SNN node backward (surrogate gradient)");

    // ── Hebbian learning ─────────────────────────────────────────────────
    m.def(
        "hebbian_learning",
        [](GrillyCoreContext& ctx,
           py::array_t<float> pre, py::array_t<float> post,
           py::array_t<float> weights,
           uint32_t batch_size, uint32_t time_steps,
           float learning_rate,
            float weight_decay) -> Tensor {
            auto preBuf = pre.request();
            auto postBuf = post.request();
            auto wBuf = weights.request();
            require_c_contiguous_float(preBuf);
            require_c_contiguous_float(postBuf);
            require_c_contiguous_float(wBuf);

            uint32_t pre_dim = static_cast<uint32_t>(
                preBuf.shape[preBuf.ndim - 1]);
            uint32_t post_dim = static_cast<uint32_t>(
                postBuf.shape[postBuf.ndim - 1]);

            py::array_t<float> wOut(wBuf.shape);
            auto woBuf = wOut.request();
            std::memcpy(woBuf.ptr, wBuf.ptr,
                        size_t(pre_dim) * post_dim * sizeof(float));

            grilly::ops::HebbianParams p{batch_size, time_steps,
                                         pre_dim, post_dim,
                                         learning_rate, weight_decay};

            {
                py::gil_scoped_release release;
                grilly::ops::hebbianLearning(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(preBuf.ptr),
                    static_cast<const float*>(postBuf.ptr),
                    static_cast<float*>(woBuf.ptr), p);
            }

            return Tensor::from_numpy(wOut);
        },
        py::arg("device"), py::arg("pre"), py::arg("post"),
        py::arg("weights"),
        py::arg("batch_size") = 1, py::arg("time_steps") = 1,
        py::arg("learning_rate") = 0.01f,
        py::arg("weight_decay") = 0.0f,
        "GPU Hebbian learning rule");

    // ── STDP learning ────────────────────────────────────────────────────
    m.def(
        "stdp_learning",
        [](GrillyCoreContext& ctx,
           py::array_t<float> pre, py::array_t<float> post,
           py::array_t<float> weights,
           py::array_t<float> pre_trace, py::array_t<float> post_trace,
           uint32_t batch_size, uint32_t time_steps,
           float lr_pot, float lr_dep,
           float trace_decay) -> py::dict {
            auto preBuf = pre.request();
            auto postBuf = post.request();
            auto wBuf = weights.request();
            require_c_contiguous_float(preBuf);
            require_c_contiguous_float(postBuf);
            require_c_contiguous_float(wBuf);

            uint32_t pre_dim = static_cast<uint32_t>(
                preBuf.shape[preBuf.ndim - 1]);
            uint32_t post_dim = static_cast<uint32_t>(
                postBuf.shape[postBuf.ndim - 1]);

            py::array_t<float> wOut(wBuf.shape);
            py::array_t<float> preTraceOut(
                pre_trace.request().shape);
            py::array_t<float> postTraceOut(
                post_trace.request().shape);
            auto ptIn = pre_trace.request();
            auto pstIn = post_trace.request();
            require_c_contiguous_float(ptIn);
            require_c_contiguous_float(pstIn);
            auto woBuf = wOut.request();
            auto ptoBuf = preTraceOut.request();
            auto pstoBuf = postTraceOut.request();

            std::memcpy(woBuf.ptr, wBuf.ptr,
                        size_t(pre_dim) * post_dim * sizeof(float));
            std::memcpy(ptoBuf.ptr, ptIn.ptr,
                        size_t(batch_size) * pre_dim * sizeof(float));
            std::memcpy(pstoBuf.ptr, pstIn.ptr,
                        size_t(batch_size) * post_dim * sizeof(float));

            grilly::ops::STDPParams p{batch_size, time_steps,
                                      pre_dim, post_dim,
                                      lr_pot, lr_dep, trace_decay, 0};

            {
                py::gil_scoped_release release;
                grilly::ops::stdpLearning(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(preBuf.ptr),
                    static_cast<const float*>(postBuf.ptr),
                    static_cast<float*>(woBuf.ptr),
                    static_cast<float*>(ptoBuf.ptr),
                    static_cast<float*>(pstoBuf.ptr), p);
            }

            py::dict result;
            result["weights"] = Tensor::from_numpy(wOut);
            result["pre_trace"] = Tensor::from_numpy(preTraceOut);
            result["post_trace"] = Tensor::from_numpy(postTraceOut);
            return result;
        },
        py::arg("device"), py::arg("pre"), py::arg("post"),
        py::arg("weights"), py::arg("pre_trace"),
        py::arg("post_trace"),
        py::arg("batch_size") = 1, py::arg("time_steps") = 1,
        py::arg("lr_pot") = 0.01f, py::arg("lr_dep") = 0.01f,
        py::arg("trace_decay") = 0.95f,
        "GPU STDP learning (2-pass: traces then weights)");

    // ── Synapse filter ───────────────────────────────────────────────────
    m.def(
        "synapse_filter",
        [](GrillyCoreContext& ctx,
           py::array_t<float> x_in,
           py::array_t<float> y_state,
           float decay) -> Tensor {
            auto xBuf = x_in.request();
            require_c_contiguous_float(xBuf);
            uint32_t n = 1;
            for (int i = 0; i < xBuf.ndim; ++i)
                n *= static_cast<uint32_t>(xBuf.shape[i]);

            py::array_t<float> yOut(y_state.request().shape);
            auto ysIn = y_state.request();
            require_c_contiguous_float(ysIn);
            auto yoBuf = yOut.request();
            std::memcpy(yoBuf.ptr, ysIn.ptr, n * sizeof(float));

            grilly::ops::SynapseFilterParams p{n, decay};

            {
                py::gil_scoped_release release;
                grilly::ops::synapseFilter(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(xBuf.ptr),
                    static_cast<float*>(yoBuf.ptr), p);
            }

            return Tensor::from_numpy(yOut);
        },
        py::arg("device"), py::arg("x_in"), py::arg("y_state"),
        py::arg("decay") = 0.95f,
        "GPU exponential synapse filter");

    // ── GIF neuron step ──────────────────────────────────────────────────
    m.def(
        "gif_neuron_step",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input, py::array_t<float> v_mem,
           py::array_t<float> i_adapt,
           py::array_t<float> g_input, py::array_t<float> g_forget,
           py::array_t<float> t_refrac,
           py::array_t<float> t_last_spike,
           float dt, float current_time,
           float tau_mem, float v_rest, float v_reset, float v_thresh,
           float r_mem, float tau_adapt, float delta_adapt,
           float b_adapt, float tau_gate, float gate_strength,
            float t_refrac_period) -> py::dict {
            auto inBuf = input.request();
            require_c_contiguous_float(inBuf);
            uint32_t n = 1;
            for (int i = 0; i < inBuf.ndim; ++i)
                n *= static_cast<uint32_t>(inBuf.shape[i]);

            py::array_t<float> vMemOut(v_mem.request().shape);
            py::array_t<float> iAdaptOut(i_adapt.request().shape);
            py::array_t<float> gInputOut(g_input.request().shape);
            py::array_t<float> gForgetOut(g_forget.request().shape);
            py::array_t<float> refracOut(t_refrac.request().shape);
            py::array_t<float> spikes(inBuf.shape);
            py::array_t<float> tLastOut(
                t_last_spike.request().shape);

            auto vmIn = v_mem.request();
            auto iaIn = i_adapt.request();
            auto giIn = g_input.request();
            auto gfIn = g_forget.request();
            auto trIn = t_refrac.request();
            auto tlsIn = t_last_spike.request();
            require_c_contiguous_float(vmIn);
            require_c_contiguous_float(iaIn);
            require_c_contiguous_float(giIn);
            require_c_contiguous_float(gfIn);
            require_c_contiguous_float(trIn);
            require_c_contiguous_float(tlsIn);

            auto vmOut = vMemOut.request();
            auto iaOut = iAdaptOut.request();
            auto giOut = gInputOut.request();
            auto gfOut = gForgetOut.request();
            auto trOut = refracOut.request();
            auto spOut = spikes.request();
            auto tlsOut = tLastOut.request();

            std::memcpy(vmOut.ptr, vmIn.ptr, n * sizeof(float));
            std::memcpy(iaOut.ptr, iaIn.ptr, n * sizeof(float));
            std::memcpy(giOut.ptr, giIn.ptr, n * sizeof(float));
            std::memcpy(gfOut.ptr, gfIn.ptr, n * sizeof(float));
            std::memcpy(trOut.ptr, trIn.ptr, n * sizeof(float));
            std::memcpy(tlsOut.ptr, tlsIn.ptr, n * sizeof(float));

            grilly::ops::GIFParams p{
                n, dt, current_time, tau_mem, v_rest, v_reset,
                v_thresh, r_mem, tau_adapt, delta_adapt, b_adapt,
                tau_gate, gate_strength, t_refrac_period};

            {
                py::gil_scoped_release release;
                grilly::ops::gifNeuronStep(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(inBuf.ptr),
                    static_cast<float*>(vmOut.ptr),
                    static_cast<float*>(iaOut.ptr),
                    static_cast<float*>(giOut.ptr),
                    static_cast<float*>(gfOut.ptr),
                    static_cast<float*>(trOut.ptr),
                    static_cast<float*>(spOut.ptr),
                    static_cast<float*>(tlsOut.ptr), p);
            }

            py::dict result;
            result["spikes"] = Tensor::from_numpy(spikes);
            result["v_mem"] = Tensor::from_numpy(vMemOut);
            result["i_adapt"] = Tensor::from_numpy(iAdaptOut);
            result["g_input"] = Tensor::from_numpy(gInputOut);
            result["g_forget"] = Tensor::from_numpy(gForgetOut);
            result["t_refrac"] = Tensor::from_numpy(refracOut);
            result["t_last_spike"] = Tensor::from_numpy(tLastOut);
            return result;
        },
        py::arg("device"), py::arg("input"), py::arg("v_mem"),
        py::arg("i_adapt"), py::arg("g_input"), py::arg("g_forget"),
        py::arg("t_refrac"), py::arg("t_last_spike"),
        py::arg("dt") = 1.0f, py::arg("current_time") = 0.0f,
        py::arg("tau_mem") = 20.0f, py::arg("v_rest") = 0.0f,
        py::arg("v_reset") = 0.0f, py::arg("v_thresh") = 1.0f,
        py::arg("r_mem") = 1.0f, py::arg("tau_adapt") = 100.0f,
        py::arg("delta_adapt") = 0.1f, py::arg("b_adapt") = 0.0f,
        py::arg("tau_gate") = 50.0f,
        py::arg("gate_strength") = 1.0f,
        py::arg("t_refrac_period") = 0.0f,
        "GPU GIF (Generalized Integrate-and-Fire) neuron step");

    // ── Event-driven sparse synaptic scatter ─────────────────────────────
    m.def(
        "spike_scatter",
        [](GrillyCoreContext& ctx,
           py::array_t<float> fired_idx,
           py::array_t<float> fired_count,
           py::array_t<float> weights,
           uint32_t n) -> Tensor {
            auto idxBuf = fired_idx.request();
            auto cntBuf = fired_count.request();
            auto wBuf = weights.request();
            require_c_contiguous_float(idxBuf);
            require_c_contiguous_float(cntBuf);
            require_c_contiguous_float(wBuf);

            uint32_t nFired = static_cast<uint32_t>(idxBuf.size);

            py::array_t<float> iAcc(std::vector<py::ssize_t>{
                static_cast<py::ssize_t>(n)});
            auto accBuf = iAcc.request();
            std::memset(accBuf.ptr, 0, size_t(n) * sizeof(float));

            grilly::ops::SpikeScatterParams p{n};

            {
                py::gil_scoped_release release;
                grilly::ops::spikeScatter(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(idxBuf.ptr),
                    static_cast<const float*>(cntBuf.ptr),
                    static_cast<const float*>(wBuf.ptr),
                    static_cast<float*>(accBuf.ptr),
                    nFired, p);
            }

            return Tensor::from_numpy(iAcc);
        },
        py::arg("device"), py::arg("fired_idx"), py::arg("fired_count"),
        py::arg("weights"), py::arg("n"),
        "GPU event-driven sparse synaptic scatter (Tier-0 measurement op)");

    // ── Resident-weight benchmark loop ───────────────────────────────────
    m.def(
        "resident_bench",
        [](GrillyCoreContext& ctx, uint32_t mode,
           py::array_t<float> fired_idx, py::array_t<float> fired_count,
           py::array_t<float> spikes, py::array_t<float> weights,
           uint32_t n, uint32_t iters, uint32_t batched) -> Tensor {
            auto idxBuf = fired_idx.request();
            auto cntBuf = fired_count.request();
            auto spkBuf = spikes.request();
            auto wBuf = weights.request();
            require_c_contiguous_float(idxBuf);
            require_c_contiguous_float(cntBuf);
            require_c_contiguous_float(spkBuf);
            require_c_contiguous_float(wBuf);
            uint32_t nFired = static_cast<uint32_t>(idxBuf.size);

            py::array_t<float> iAcc(std::vector<py::ssize_t>{
                static_cast<py::ssize_t>(n)});
            auto accBuf = iAcc.request();
            std::memset(accBuf.ptr, 0, size_t(n) * sizeof(float));

            {
                py::gil_scoped_release release;
                grilly::ops::residentBench(
                    ctx.batch, ctx.pool, ctx.cache, mode,
                    static_cast<const float*>(idxBuf.ptr),
                    static_cast<const float*>(cntBuf.ptr),
                    static_cast<const float*>(spkBuf.ptr),
                    static_cast<const float*>(wBuf.ptr),
                    static_cast<float*>(accBuf.ptr),
                    nFired, n, iters, batched);
            }
            return Tensor::from_numpy(iAcc);
        },
        py::arg("device"), py::arg("mode"), py::arg("fired_idx"),
        py::arg("fired_count"), py::arg("spikes"), py::arg("weights"),
        py::arg("n"), py::arg("iters"), py::arg("batched") = 0,
        "Resident-W benchmark loop (mode 0=scatter, 1=dense; batched=single submit)");

    // ── Batched event-driven propagation (production primitive) ──────────
    m.def(
        "spike_propagate_batch",
        [](GrillyCoreContext& ctx,
           py::array_t<float> fired_idx, py::array_t<float> fired_offsets,
           py::array_t<float> fired_counts, py::array_t<float> weights,
           py::array_t<float> fired_vals,
           uint32_t n_in, uint32_t n_out, uint32_t m_vecs) -> Tensor {
            auto idxBuf = fired_idx.request();
            auto offBuf = fired_offsets.request();
            auto cntBuf = fired_counts.request();
            auto wBuf = weights.request();
            auto valBuf = fired_vals.request();
            require_c_contiguous_float(idxBuf);
            require_c_contiguous_float(offBuf);
            require_c_contiguous_float(cntBuf);
            require_c_contiguous_float(wBuf);
            require_c_contiguous_float(valBuf);
            uint32_t nFiredTotal = static_cast<uint32_t>(idxBuf.size);

            py::array_t<float> out(std::vector<py::ssize_t>{
                static_cast<py::ssize_t>(m_vecs),
                static_cast<py::ssize_t>(n_out)});
            auto oBuf = out.request();

            {
                py::gil_scoped_release release;
                grilly::ops::spikePropagateBatch(
                    ctx.batch, ctx.pool, ctx.cache,
                    static_cast<const float*>(idxBuf.ptr),
                    static_cast<const float*>(offBuf.ptr),
                    static_cast<const float*>(cntBuf.ptr),
                    static_cast<const float*>(wBuf.ptr),
                    static_cast<float*>(oBuf.ptr),
                    static_cast<const float*>(valBuf.ptr),
                    nFiredTotal, n_in, n_out, m_vecs);
            }
            return Tensor::from_numpy(out);
        },
        py::arg("device"), py::arg("fired_idx"), py::arg("fired_offsets"),
        py::arg("fired_counts"), py::arg("weights"), py::arg("fired_vals"),
        py::arg("n_in"), py::arg("n_out"), py::arg("m"),
        "Batched event-driven sparse propagation (M vectors, resident W, 1 submit)");
}
