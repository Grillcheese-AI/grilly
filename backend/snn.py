"""
Spiking Neural Network (SNN) operations for Vulkan backend.
"""

import struct

import numpy as np

from .base import VULKAN_AVAILABLE, BufferMixin

if VULKAN_AVAILABLE:
    from vulkan import *


class VulkanSNN(BufferMixin):
    """SNN operations: LIF neurons and learning rules"""

    def __init__(self, core, pipelines):
        """Initialize with VulkanCore and VulkanPipelines instances"""
        self.core = core
        self.pipelines = pipelines

    def lif_step(self, input_current, membrane, refractory, dt=0.001, tau_mem=20.0, v_thresh=1.0):
        """Run LIF shader on GPU"""
        n = len(input_current)

        buf_in = self._acquire_buffer(input_current.nbytes)
        buf_mem = self._acquire_buffer(membrane.nbytes)
        buf_ref = self._acquire_buffer(refractory.nbytes)
        buf_out = self._acquire_buffer(n * 4)

        try:
            self._upload_buffer(buf_in, input_current.astype(np.float32))
            self._upload_buffer(buf_mem, membrane.astype(np.float32))
            self._upload_buffer(buf_ref, refractory.astype(np.float32))

            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "lif-neuron", 4, push_constant_size=32
            )

            descriptor_set = self.pipelines._create_descriptor_set(
                desc_layout,
                [
                    (self._get_buffer_handle(buf_in), input_current.nbytes),
                    (self._get_buffer_handle(buf_mem), membrane.nbytes),
                    (self._get_buffer_handle(buf_ref), refractory.nbytes),
                    (self._get_buffer_handle(buf_out), n * 4),
                ],
            )

            push_constants = struct.pack("Ifffffff", n, dt, tau_mem, 0.0, 0.0, v_thresh, 1.0, 2.0)

            workgroups = (n + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            membrane_out = self._download_buffer(buf_mem, membrane.nbytes)
            refractory_out = self._download_buffer(buf_ref, refractory.nbytes)
            spikes_out = self._download_buffer(buf_out, n * 4)

            vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])

            return membrane_out, refractory_out, spikes_out
        finally:
            self._release_buffers([buf_in, buf_mem, buf_ref, buf_out])

    def hebbian_learning(
        self, pre_activations, post_activations, weights, learning_rate=0.01, weight_decay=0.0
    ):
        """
        Apply Hebbian learning rule: dW = eta * <pre * post> - lambda * W
        """
        batch_size, time_steps, pre_dim = pre_activations.shape
        _, _, post_dim = post_activations.shape

        pre_flat = pre_activations.astype(np.float32).flatten()
        post_flat = post_activations.astype(np.float32).flatten()
        weights_flat = weights.astype(np.float32).flatten()

        buf_pre = self._acquire_buffer(pre_flat.nbytes)
        buf_post = self._acquire_buffer(post_flat.nbytes)
        buf_weights = self._acquire_buffer(weights_flat.nbytes)

        try:
            self._upload_buffer(buf_pre, pre_flat)
            self._upload_buffer(buf_post, post_flat)
            self._upload_buffer(buf_weights, weights_flat)

            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "hebbian-learning", 3, push_constant_size=32
            )

            descriptor_set = self.pipelines._create_descriptor_set(
                desc_layout,
                [
                    (self._get_buffer_handle(buf_pre), pre_flat.nbytes),
                    (self._get_buffer_handle(buf_post), post_flat.nbytes),
                    (self._get_buffer_handle(buf_weights), weights_flat.nbytes),
                ],
            )

            push_constants = struct.pack(
                "IIIIff", batch_size, time_steps, pre_dim, post_dim, learning_rate, weight_decay
            )

            workgroups_x = (pre_dim + 15) // 16
            workgroups_y = (post_dim + 15) // 16
            self.core._dispatch_compute(
                pipeline,
                pipeline_layout,
                descriptor_set,
                workgroups_x,
                push_constants,
                workgroups_y,
                1,
            )

            weights_out = self._download_buffer(buf_weights, weights_flat.nbytes, np.float32)
            weights_out = weights_out[: post_dim * pre_dim].reshape(post_dim, pre_dim)

            vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])

            return weights_out
        finally:
            self._release_buffers([buf_pre, buf_post, buf_weights])

    def stdp_learning(
        self,
        pre_activations,
        post_activations,
        weights,
        pre_trace,
        post_trace,
        lr_potentiation=0.01,
        lr_depression=0.01,
        trace_decay=0.9,
    ):
        """
        Apply STDP learning rule with eligibility traces
        """
        batch_size, time_steps, pre_dim = pre_activations.shape
        _, _, post_dim = post_activations.shape

        pre_flat = pre_activations.astype(np.float32).flatten()
        post_flat = post_activations.astype(np.float32).flatten()
        weights_flat = weights.astype(np.float32).flatten()
        pre_trace_flat = pre_trace.astype(np.float32).flatten()
        post_trace_flat = post_trace.astype(np.float32).flatten()

        buf_pre = self._acquire_buffer(pre_flat.nbytes)
        buf_post = self._acquire_buffer(post_flat.nbytes)
        buf_weights = self._acquire_buffer(weights_flat.nbytes)
        buf_pre_trace = self._acquire_buffer(pre_trace_flat.nbytes)
        buf_post_trace = self._acquire_buffer(post_trace_flat.nbytes)

        try:
            self._upload_buffer(buf_pre, pre_flat)
            self._upload_buffer(buf_post, post_flat)
            self._upload_buffer(buf_weights, weights_flat)
            self._upload_buffer(buf_pre_trace, pre_trace_flat)
            self._upload_buffer(buf_post_trace, post_trace_flat)

            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "stdp-learning", 5, push_constant_size=32
            )

            descriptor_set = self.pipelines._create_descriptor_set(
                desc_layout,
                [
                    (self._get_buffer_handle(buf_pre), pre_flat.nbytes),
                    (self._get_buffer_handle(buf_post), post_flat.nbytes),
                    (self._get_buffer_handle(buf_weights), weights_flat.nbytes),
                    (self._get_buffer_handle(buf_pre_trace), pre_trace_flat.nbytes),
                    (self._get_buffer_handle(buf_post_trace), post_trace_flat.nbytes),
                ],
            )

            # Pass 1: Update traces
            push_constants = struct.pack(
                "IIIIfffI",
                batch_size,
                time_steps,
                pre_dim,
                post_dim,
                lr_potentiation,
                lr_depression,
                trace_decay,
                0,
            )
            workgroups_x = (max(pre_dim, post_dim) + 15) // 16
            workgroups_y = (batch_size + 15) // 16
            self.core._dispatch_compute(
                pipeline,
                pipeline_layout,
                descriptor_set,
                workgroups_x,
                push_constants,
                workgroups_y,
                1,
            )

            pre_trace_out = self._download_buffer(buf_pre_trace, pre_trace_flat.nbytes, np.float32)
            post_trace_out = self._download_buffer(
                buf_post_trace, post_trace_flat.nbytes, np.float32
            )
            pre_trace_out = pre_trace_out[: batch_size * pre_dim].reshape(batch_size, pre_dim)
            post_trace_out = post_trace_out[: batch_size * post_dim].reshape(batch_size, post_dim)

            self._upload_buffer(buf_pre_trace, pre_trace_out.flatten())
            self._upload_buffer(buf_post_trace, post_trace_out.flatten())

            # Pass 2: Update weights
            push_constants = struct.pack(
                "IIIIfffI",
                batch_size,
                time_steps,
                pre_dim,
                post_dim,
                lr_potentiation,
                lr_depression,
                trace_decay,
                1,
            )
            workgroups_x = (pre_dim + 15) // 16
            workgroups_y = (post_dim + 15) // 16
            self.core._dispatch_compute(
                pipeline,
                pipeline_layout,
                descriptor_set,
                workgroups_x,
                push_constants,
                workgroups_y,
                1,
            )

            weights_out = self._download_buffer(buf_weights, weights_flat.nbytes, np.float32)
            weights_out = weights_out[: post_dim * pre_dim].reshape(post_dim, pre_dim)
            pre_trace_out = self._download_buffer(buf_pre_trace, pre_trace_flat.nbytes, np.float32)
            post_trace_out = self._download_buffer(
                buf_post_trace, post_trace_flat.nbytes, np.float32
            )
            pre_trace_out = pre_trace_out[: batch_size * pre_dim].reshape(batch_size, pre_dim)
            post_trace_out = post_trace_out[: batch_size * post_dim].reshape(batch_size, post_dim)

            vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])

            return weights_out, pre_trace_out, post_trace_out
        finally:
            self._release_buffers([buf_pre, buf_post, buf_weights, buf_pre_trace, buf_post_trace])

    def synapse_filter(self, x_in, y_state, decay):
        """GPU-accelerated exponential decay synaptic filter.

        Computes: y_state[i] = y_state[i] * decay + x_in[i]

        Args:
            x_in: Input at current timestep (flattened float32 array)
            y_state: Filter state from previous timestep (same shape as x_in)
            decay: Decay factor = exp(-1/tau)

        Returns:
            Updated y_state array
        """
        x_flat = x_in.astype(np.float32).flatten()
        y_flat = y_state.astype(np.float32).flatten()
        n = len(x_flat)

        buf_x = self._acquire_buffer(x_flat.nbytes)
        buf_y = self._acquire_buffer(y_flat.nbytes)

        try:
            self._upload_buffer(buf_x, x_flat)
            self._upload_buffer(buf_y, y_flat)

            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "snn-synapse-filter", 2, push_constant_size=8
            )

            descriptor_set = self.pipelines._create_descriptor_set(
                desc_layout,
                [
                    (self._get_buffer_handle(buf_x), x_flat.nbytes),
                    (self._get_buffer_handle(buf_y), y_flat.nbytes),
                ],
            )

            push_constants = struct.pack("If", n, float(decay))

            workgroups = (n + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            y_out = self._download_buffer(buf_y, y_flat.nbytes, np.float32)

            vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])

            return y_out[: n].reshape(x_in.shape)
        finally:
            self._release_buffers([buf_x, buf_y])

    def snn_node_forward(self, x_in, v_mem, neuron_type=0, tau=2.0,
                         v_threshold=1.0, v_reset=0.0, reset_mode=0,
                         decay_input=False, tau_param=None):
        """GPU-accelerated SNN neuron forward pass (IF/LIF/PLIF).

        Args:
            x_in: Input current (flattened float32)
            v_mem: Membrane potential state (same shape)
            neuron_type: 0=IF, 1=LIF, 2=PLIF
            tau: Time constant (LIF/PLIF default)
            v_threshold: Spike threshold
            v_reset: Reset voltage (-1e9 signals soft reset)
            reset_mode: 0=hard reset, 1=soft reset
            decay_input: If True, divide input by tau (physics). If False
                (default), add input at full strength (practical).
            tau_param: Per-neuron tau array for PLIF (optional)

        Returns:
            (spikes, updated_v_mem, h_cache) tuple
        """
        x_flat = x_in.astype(np.float32).flatten()
        v_flat = v_mem.astype(np.float32).flatten()
        n = len(x_flat)

        if tau_param is None:
            tau_param = np.full(1, tau, dtype=np.float32)
        tau_flat = tau_param.astype(np.float32).flatten()

        buf_x = self._acquire_buffer(x_flat.nbytes)
        buf_v = self._acquire_buffer(v_flat.nbytes)
        buf_s = self._acquire_buffer(n * 4)
        buf_h = self._acquire_buffer(n * 4)
        buf_tau = self._acquire_buffer(tau_flat.nbytes)

        try:
            self._upload_buffer(buf_x, x_flat)
            self._upload_buffer(buf_v, v_flat)
            self._upload_buffer(buf_tau, tau_flat)

            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "snn-node-forward", 5, push_constant_size=28
            )

            descriptor_set = self.pipelines._create_descriptor_set(
                desc_layout,
                [
                    (self._get_buffer_handle(buf_x), x_flat.nbytes),
                    (self._get_buffer_handle(buf_v), v_flat.nbytes),
                    (self._get_buffer_handle(buf_s), n * 4),
                    (self._get_buffer_handle(buf_h), n * 4),
                    (self._get_buffer_handle(buf_tau), tau_flat.nbytes),
                ],
            )

            push_constants = struct.pack(
                "IIfffII", n, neuron_type, tau, v_threshold, v_reset,
                reset_mode, 1 if decay_input else 0
            )

            workgroups = (n + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            spikes = self._download_buffer(buf_s, n * 4, np.float32)[:n].reshape(x_in.shape)
            v_out = self._download_buffer(buf_v, v_flat.nbytes, np.float32)[:n].reshape(x_in.shape)
            h_out = self._download_buffer(buf_h, n * 4, np.float32)[:n].reshape(x_in.shape)

            vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])

            return spikes, v_out, h_out
        finally:
            self._release_buffers([buf_x, buf_v, buf_s, buf_h, buf_tau])

    def snn_node_backward(self, grad_spike, h_cache, alpha=2.0, surrogate_type=0, v_threshold=1.0):
        """GPU-accelerated SNN backward pass with surrogate gradients.

        Args:
            grad_spike: Upstream gradient w.r.t. spikes
            h_cache: Pre-spike membrane potential from forward
            alpha: Surrogate function sharpness
            surrogate_type: 0=ATan, 1=Sigmoid, 2=FastSigmoid
            v_threshold: Spike threshold

        Returns:
            grad_x: Gradient w.r.t. input
        """
        gs_flat = grad_spike.astype(np.float32).flatten()
        h_flat = h_cache.astype(np.float32).flatten()
        n = len(gs_flat)

        buf_gs = self._acquire_buffer(gs_flat.nbytes)
        buf_h = self._acquire_buffer(h_flat.nbytes)
        buf_gx = self._acquire_buffer(n * 4)

        try:
            self._upload_buffer(buf_gs, gs_flat)
            self._upload_buffer(buf_h, h_flat)

            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "snn-node-backward", 3, push_constant_size=16
            )

            descriptor_set = self.pipelines._create_descriptor_set(
                desc_layout,
                [
                    (self._get_buffer_handle(buf_gs), gs_flat.nbytes),
                    (self._get_buffer_handle(buf_h), h_flat.nbytes),
                    (self._get_buffer_handle(buf_gx), n * 4),
                ],
            )

            push_constants = struct.pack("IfIf", n, alpha, surrogate_type, v_threshold)

            workgroups = (n + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            grad_x = self._download_buffer(buf_gx, n * 4, np.float32)[:n].reshape(grad_spike.shape)

            vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])

            return grad_x
        finally:
            self._release_buffers([buf_gs, buf_h, buf_gx])

    def gif_neuron_step(
        self,
        input_current: np.ndarray,
        membrane_potential: np.ndarray,
        adaptation_current: np.ndarray,
        input_gate: np.ndarray,
        forget_gate: np.ndarray,
        refractory_state: np.ndarray,
        last_spike_time: np.ndarray,
        dt: float = 0.001,
        current_time: float = 0.0,
        tau_mem: float = 20.0,
        v_rest: float = 0.0,
        v_reset: float = 0.0,
        v_thresh: float = 1.0,
        r_mem: float = 1.0,
        tau_adapt: float = 100.0,
        delta_adapt: float = 0.1,
        b_adapt: float = 0.02,
        tau_gate: float = 10.0,
        gate_strength: float = 1.0,
        t_refrac_period: float = 2.0,
    ) -> tuple:
        """
        GPU-accelerated GIF (Generalized Integrate-and-Fire) neuron step

        GIF neurons have gated dynamics similar to LSTM, allowing for
        adaptive integration and memory retention.

        Args:
            input_current: Input current for each neuron [n_neurons]
            membrane_potential: Current membrane potential [n_neurons]
            adaptation_current: Adaptation current state [n_neurons]
            input_gate: Input gate state [n_neurons]
            forget_gate: Forget gate state [n_neurons]
            refractory_state: Refractory counter [n_neurons]
            last_spike_time: Time of last spike [n_neurons]
            dt: Time step
            current_time: Current simulation time
            tau_mem: Membrane time constant
            v_rest: Resting potential
            v_reset: Reset potential
            v_thresh: Spike threshold
            r_mem: Membrane resistance
            tau_adapt: Adaptation time constant
            delta_adapt: Adaptation increment per spike
            b_adapt: Adaptation coupling strength
            tau_gate: Gate time constant
            gate_strength: Gate modulation strength
            t_refrac_period: Refractory period duration

        Returns:
            Tuple of (spikes, updated_membrane, updated_adaptation, updated_input_gate, updated_forget_gate, updated_refractory, updated_last_spike_time)
        """
        n_neurons = len(input_current)

        I_in = input_current.astype(np.float32).flatten()
        V_mem = membrane_potential.astype(np.float32).flatten()
        I_adapt = adaptation_current.astype(np.float32).flatten()
        g_input = input_gate.astype(np.float32).flatten()
        g_forget = forget_gate.astype(np.float32).flatten()
        t_refrac = refractory_state.astype(np.float32).flatten()
        t_last = last_spike_time.astype(np.float32).flatten()
        spikes = np.zeros(n_neurons, dtype=np.float32)

        buf_I = self._acquire_buffer(I_in.nbytes)
        buf_V = self._acquire_buffer(V_mem.nbytes)
        buf_Ia = self._acquire_buffer(I_adapt.nbytes)
        buf_gi = self._acquire_buffer(g_input.nbytes)
        buf_gf = self._acquire_buffer(g_forget.nbytes)
        buf_tref = self._acquire_buffer(t_refrac.nbytes)
        buf_spikes = self._acquire_buffer(spikes.nbytes)
        buf_tlast = self._acquire_buffer(t_last.nbytes)

        try:
            self._upload_buffer(buf_I, I_in)
            self._upload_buffer(buf_V, V_mem)
            self._upload_buffer(buf_Ia, I_adapt)
            self._upload_buffer(buf_gi, g_input)
            self._upload_buffer(buf_gf, g_forget)
            self._upload_buffer(buf_tref, t_refrac)
            self._upload_buffer(buf_tlast, t_last)

            if "gif-neuron" not in self.shaders:
                raise RuntimeError(
                    "gif-neuron shader not compiled. "
                    "Run: glslc -fshader-stage=compute shaders/gif-neuron.glsl -o shaders/spv/gif-neuron.spv"
                )

            num_bindings = 8
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "gif-neuron", num_bindings, push_constant_size=64
            )

            descriptor_set = self.pipelines.get_cached_descriptor_set(
                "gif-neuron",
                [
                    (self._get_buffer_handle(buf_I), I_in.nbytes),
                    (self._get_buffer_handle(buf_V), V_mem.nbytes),
                    (self._get_buffer_handle(buf_Ia), I_adapt.nbytes),
                    (self._get_buffer_handle(buf_gi), g_input.nbytes),
                    (self._get_buffer_handle(buf_gf), g_forget.nbytes),
                    (self._get_buffer_handle(buf_tref), t_refrac.nbytes),
                    (self._get_buffer_handle(buf_spikes), spikes.nbytes),
                    (self._get_buffer_handle(buf_tlast), t_last.nbytes),
                ],
            )

            push_constants = struct.pack(
                "Ifffffffffffff",
                n_neurons,
                dt,
                current_time,
                tau_mem,
                v_rest,
                v_reset,
                v_thresh,
                r_mem,
                tau_adapt,
                delta_adapt,
                b_adapt,
                tau_gate,
                gate_strength,
                t_refrac_period,
            )

            workgroups = (n_neurons + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            updated_V = self._download_buffer(buf_V, V_mem.nbytes, np.float32)
            updated_Ia = self._download_buffer(buf_Ia, I_adapt.nbytes, np.float32)
            updated_gi = self._download_buffer(buf_gi, g_input.nbytes, np.float32)
            updated_gf = self._download_buffer(buf_gf, g_forget.nbytes, np.float32)
            updated_tref = self._download_buffer(buf_tref, t_refrac.nbytes, np.float32)
            updated_spikes = self._download_buffer(buf_spikes, spikes.nbytes, np.float32)
            updated_tlast = self._download_buffer(buf_tlast, t_last.nbytes, np.float32)

            return (
                updated_spikes,
                updated_V,
                updated_Ia,
                updated_gi,
                updated_gf,
                updated_tref,
                updated_tlast,
            )
        finally:
            self._release_buffers(
                [buf_I, buf_V, buf_Ia, buf_gi, buf_gf, buf_tref, buf_spikes, buf_tlast]
            )
