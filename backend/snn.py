"""
Spiking Neural Network (SNN) operations for Vulkan backend.
"""

import numpy as np
import struct
from .base import VULKAN_AVAILABLE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT

if VULKAN_AVAILABLE:
    from vulkan import *


class VulkanSNN:
    """SNN operations: LIF neurons and learning rules"""
    
    def __init__(self, core, pipelines):
        """Initialize with VulkanCore and VulkanPipelines instances"""
        self.core = core
        self.pipelines = pipelines
    
    def lif_step(self, input_current, membrane, refractory, 
                 dt=0.001, tau_mem=20.0, v_thresh=1.0):
        """Run LIF shader on GPU"""
        n = len(input_current)
        
        # Create buffers
        buf_in, mem_in = self.core._create_buffer(
            input_current.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        buf_mem, mem_mem = self.core._create_buffer(
            membrane.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        buf_ref, mem_ref = self.core._create_buffer(
            refractory.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        buf_out, mem_out = self.core._create_buffer(
            n * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        
        # Upload data
        self.core._upload_buffer(buf_in, mem_in, input_current.astype(np.float32))
        self.core._upload_buffer(buf_mem, mem_mem, membrane.astype(np.float32))
        self.core._upload_buffer(buf_ref, mem_ref, refractory.astype(np.float32))
        
        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'lif-neuron', 4, push_constant_size=32
        )
        
        # Create descriptor set
        descriptor_set = self.pipelines._create_descriptor_set(
            desc_layout,
            [
                (buf_in, input_current.nbytes),
                (buf_mem, membrane.nbytes),
                (buf_ref, refractory.nbytes),
                (buf_out, n * 4)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack(
            'Ifffffff',
            n, dt, tau_mem, 0.0, 0.0, v_thresh, 1.0, 2.0
        )
        
        # Dispatch
        workgroups = (n + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )
        
        # Download results
        membrane_out = self.core._download_buffer(mem_mem, membrane.nbytes)
        refractory_out = self.core._download_buffer(mem_ref, refractory.nbytes)
        spikes_out = self.core._download_buffer(mem_out, n * 4)
        
        # Free descriptor set
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_in, None)
        vkDestroyBuffer(self.core.device, buf_mem, None)
        vkDestroyBuffer(self.core.device, buf_ref, None)
        vkDestroyBuffer(self.core.device, buf_out, None)
        vkFreeMemory(self.core.device, mem_in, None)
        vkFreeMemory(self.core.device, mem_mem, None)
        vkFreeMemory(self.core.device, mem_ref, None)
        vkFreeMemory(self.core.device, mem_out, None)
        
        return membrane_out, refractory_out, spikes_out
    
    def hebbian_learning(self, pre_activations, post_activations, weights,
                         learning_rate=0.01, weight_decay=0.0):
        """
        Apply Hebbian learning rule: ΔW = η * <pre * post> - λ * W
        """
        batch_size, time_steps, pre_dim = pre_activations.shape
        _, _, post_dim = post_activations.shape
        
        # Flatten activations
        pre_flat = pre_activations.astype(np.float32).flatten()
        post_flat = post_activations.astype(np.float32).flatten()
        weights_flat = weights.astype(np.float32).flatten()
        
        # Create buffers
        buf_pre, mem_pre = self.core._create_buffer(pre_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_post, mem_post = self.core._create_buffer(post_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_weights, mem_weights = self.core._create_buffer(weights_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(buf_pre, mem_pre, pre_flat)
        self.core._upload_buffer(buf_post, mem_post, post_flat)
        self.core._upload_buffer(buf_weights, mem_weights, weights_flat)
        
        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'hebbian-learning', 3, push_constant_size=32
        )
        
        # Create descriptor set
        descriptor_set = self.pipelines._create_descriptor_set(
            desc_layout,
            [
                (buf_pre, pre_flat.nbytes),
                (buf_post, post_flat.nbytes),
                (buf_weights, weights_flat.nbytes)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack(
            'IIIIff', batch_size, time_steps, pre_dim, post_dim, learning_rate, weight_decay
        )
        
        # Dispatch
        workgroups_x = (pre_dim + 15) // 16
        workgroups_y = (post_dim + 15) // 16
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups_x, push_constants, workgroups_y, 1
        )
        
        # Download results
        weights_out = self.core._download_buffer(mem_weights, weights_flat.nbytes, dtype=np.float32)
        weights_out = weights_out[:post_dim * pre_dim].reshape(post_dim, pre_dim)
        
        # Free descriptor set
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_pre, None)
        vkDestroyBuffer(self.core.device, buf_post, None)
        vkDestroyBuffer(self.core.device, buf_weights, None)
        vkFreeMemory(self.core.device, mem_pre, None)
        vkFreeMemory(self.core.device, mem_post, None)
        vkFreeMemory(self.core.device, mem_weights, None)
        
        return weights_out
    
    def stdp_learning(self, pre_activations, post_activations, weights,
                      pre_trace, post_trace,
                      lr_potentiation=0.01, lr_depression=0.01, trace_decay=0.9):
        """
        Apply STDP learning rule with eligibility traces
        """
        batch_size, time_steps, pre_dim = pre_activations.shape
        _, _, post_dim = post_activations.shape
        
        # Flatten activations
        pre_flat = pre_activations.astype(np.float32).flatten()
        post_flat = post_activations.astype(np.float32).flatten()
        weights_flat = weights.astype(np.float32).flatten()
        pre_trace_flat = pre_trace.astype(np.float32).flatten()
        post_trace_flat = post_trace.astype(np.float32).flatten()
        
        # Create buffers
        buf_pre, mem_pre = self.core._create_buffer(pre_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_post, mem_post = self.core._create_buffer(post_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_weights, mem_weights = self.core._create_buffer(weights_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_pre_trace, mem_pre_trace = self.core._create_buffer(pre_trace_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_post_trace, mem_post_trace = self.core._create_buffer(post_trace_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(buf_pre, mem_pre, pre_flat)
        self.core._upload_buffer(buf_post, mem_post, post_flat)
        self.core._upload_buffer(buf_weights, mem_weights, weights_flat)
        self.core._upload_buffer(buf_pre_trace, mem_pre_trace, pre_trace_flat)
        self.core._upload_buffer(buf_post_trace, mem_post_trace, post_trace_flat)
        
        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'stdp-learning', 5, push_constant_size=32
        )
        
        # Create descriptor set
        descriptor_set = self.pipelines._create_descriptor_set(
            desc_layout,
            [
                (buf_pre, pre_flat.nbytes),
                (buf_post, post_flat.nbytes),
                (buf_weights, weights_flat.nbytes),
                (buf_pre_trace, pre_trace_flat.nbytes),
                (buf_post_trace, post_trace_flat.nbytes)
            ]
        )
        
        # Pass 1: Update traces
        push_constants = struct.pack(
            'IIIIfffI', batch_size, time_steps, pre_dim, post_dim,
            lr_potentiation, lr_depression, trace_decay, 0
        )
        workgroups_x = (max(pre_dim, post_dim) + 15) // 16
        workgroups_y = (batch_size + 15) // 16
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups_x, push_constants, workgroups_y, 1
        )
        
        # Download updated traces
        pre_trace_out = self.core._download_buffer(mem_pre_trace, pre_trace_flat.nbytes, dtype=np.float32)
        post_trace_out = self.core._download_buffer(mem_post_trace, post_trace_flat.nbytes, dtype=np.float32)
        pre_trace_out = pre_trace_out[:batch_size * pre_dim].reshape(batch_size, pre_dim)
        post_trace_out = post_trace_out[:batch_size * post_dim].reshape(batch_size, post_dim)
        
        # Upload updated traces back
        self.core._upload_buffer(buf_pre_trace, mem_pre_trace, pre_trace_out.flatten())
        self.core._upload_buffer(buf_post_trace, mem_post_trace, post_trace_out.flatten())
        
        # Pass 2: Update weights
        push_constants = struct.pack(
            'IIIIfffI', batch_size, time_steps, pre_dim, post_dim,
            lr_potentiation, lr_depression, trace_decay, 1
        )
        workgroups_x = (pre_dim + 15) // 16
        workgroups_y = (post_dim + 15) // 16
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups_x, push_constants, workgroups_y, 1
        )
        
        # Download results
        weights_out = self.core._download_buffer(mem_weights, weights_flat.nbytes, dtype=np.float32)
        weights_out = weights_out[:post_dim * pre_dim].reshape(post_dim, pre_dim)
        pre_trace_out = self.core._download_buffer(mem_pre_trace, pre_trace_flat.nbytes, dtype=np.float32)
        post_trace_out = self.core._download_buffer(mem_post_trace, post_trace_flat.nbytes, dtype=np.float32)
        pre_trace_out = pre_trace_out[:batch_size * pre_dim].reshape(batch_size, pre_dim)
        post_trace_out = post_trace_out[:batch_size * post_dim].reshape(batch_size, post_dim)
        
        # Free descriptor set
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_pre, None)
        vkDestroyBuffer(self.core.device, buf_post, None)
        vkDestroyBuffer(self.core.device, buf_weights, None)
        vkDestroyBuffer(self.core.device, buf_pre_trace, None)
        vkDestroyBuffer(self.core.device, buf_post_trace, None)
        vkFreeMemory(self.core.device, mem_pre, None)
        vkFreeMemory(self.core.device, mem_post, None)
        vkFreeMemory(self.core.device, mem_weights, None)
        vkFreeMemory(self.core.device, mem_pre_trace, None)
        vkFreeMemory(self.core.device, mem_post_trace, None)
        
        return weights_out, pre_trace_out, post_trace_out
    
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
        t_refrac_period: float = 2.0
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
        
        # Ensure all arrays are same size and float32
        I_in = input_current.astype(np.float32).flatten()
        V_mem = membrane_potential.astype(np.float32).flatten()
        I_adapt = adaptation_current.astype(np.float32).flatten()
        g_input = input_gate.astype(np.float32).flatten()
        g_forget = forget_gate.astype(np.float32).flatten()
        t_refrac = refractory_state.astype(np.float32).flatten()
        t_last = last_spike_time.astype(np.float32).flatten()
        
        # Output spikes
        spikes = np.zeros(n_neurons, dtype=np.float32)
        
        # Create buffers
        buf_I, mem_I = self.core._create_buffer(I_in.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_V, mem_V = self.core._create_buffer(V_mem.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_Ia, mem_Ia = self.core._create_buffer(I_adapt.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_gi, mem_gi = self.core._create_buffer(g_input.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_gf, mem_gf = self.core._create_buffer(g_forget.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_tref, mem_tref = self.core._create_buffer(t_refrac.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_spikes, mem_spikes = self.core._create_buffer(spikes.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_tlast, mem_tlast = self.core._create_buffer(t_last.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(buf_I, mem_I, I_in)
        self.core._upload_buffer(buf_V, mem_V, V_mem)
        self.core._upload_buffer(buf_Ia, mem_Ia, I_adapt)
        self.core._upload_buffer(buf_gi, mem_gi, g_input)
        self.core._upload_buffer(buf_gf, mem_gf, g_forget)
        self.core._upload_buffer(buf_tref, mem_tref, t_refrac)
        self.core._upload_buffer(buf_tlast, mem_tlast, t_last)
        
        # Check if shader is available
        if 'gif-neuron' not in self.shaders:
            raise RuntimeError(
                "gif-neuron shader not compiled. "
                "Run: glslc -fshader-stage=compute shaders/gif-neuron.glsl -o shaders/spv/gif-neuron.spv"
            )
        
        # Get or create pipeline
        num_bindings = 8  # I_input, V_mem, I_adapt, g_input, g_forget, t_refrac, spikes, t_last_spike
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'gif-neuron', num_bindings, push_constant_size=64
        )
        
        # Create descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'gif-neuron',
            [
                (buf_I, I_in.nbytes),
                (buf_V, V_mem.nbytes),
                (buf_Ia, I_adapt.nbytes),
                (buf_gi, g_input.nbytes),
                (buf_gf, g_forget.nbytes),
                (buf_tref, t_refrac.nbytes),
                (buf_spikes, spikes.nbytes),
                (buf_tlast, t_last.nbytes)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack(
            'Ifffffffffffff',
            n_neurons,
            dt, current_time,
            tau_mem, v_rest, v_reset, v_thresh, r_mem,
            tau_adapt, delta_adapt, b_adapt,
            tau_gate, gate_strength,
            t_refrac_period
        )
        
        # Dispatch
        workgroups = (n_neurons + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )
        
        # Download results
        updated_V = self.core._download_buffer(mem_V, V_mem.nbytes, dtype=np.float32)
        updated_Ia = self.core._download_buffer(mem_Ia, I_adapt.nbytes, dtype=np.float32)
        updated_gi = self.core._download_buffer(mem_gi, g_input.nbytes, dtype=np.float32)
        updated_gf = self.core._download_buffer(mem_gf, g_forget.nbytes, dtype=np.float32)
        updated_tref = self.core._download_buffer(mem_tref, t_refrac.nbytes, dtype=np.float32)
        updated_spikes = self.core._download_buffer(mem_spikes, spikes.nbytes, dtype=np.float32)
        updated_tlast = self.core._download_buffer(mem_tlast, t_last.nbytes, dtype=np.float32)
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_I, None)
        vkDestroyBuffer(self.core.device, buf_V, None)
        vkDestroyBuffer(self.core.device, buf_Ia, None)
        vkDestroyBuffer(self.core.device, buf_gi, None)
        vkDestroyBuffer(self.core.device, buf_gf, None)
        vkDestroyBuffer(self.core.device, buf_tref, None)
        vkDestroyBuffer(self.core.device, buf_spikes, None)
        vkDestroyBuffer(self.core.device, buf_tlast, None)
        vkFreeMemory(self.core.device, mem_I, None)
        vkFreeMemory(self.core.device, mem_V, None)
        vkFreeMemory(self.core.device, mem_Ia, None)
        vkFreeMemory(self.core.device, mem_gi, None)
        vkFreeMemory(self.core.device, mem_gf, None)
        vkFreeMemory(self.core.device, mem_tref, None)
        vkFreeMemory(self.core.device, mem_spikes, None)
        vkFreeMemory(self.core.device, mem_tlast, None)
        
        return (
            updated_spikes,
            updated_V,
            updated_Ia,
            updated_gi,
            updated_gf,
            updated_tref,
            updated_tlast
        )

