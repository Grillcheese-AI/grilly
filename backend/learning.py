"""
Vulkan Learning Operations Module

GPU-accelerated learning operations:
- Fisher Information / EWC (Elastic Weight Consolidation)
- NLMS (Normalized Least Mean Squares) adaptive filtering
- Whitening transforms
- Bridge operations (continuous ↔ spike)
- Domain routing for mixture of experts
- Optimizer updates (Adam, SGD, etc.)
"""

import numpy as np
import struct
from .base import VULKAN_AVAILABLE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT

if VULKAN_AVAILABLE:
    from vulkan import *


class VulkanLearning:
    """GPU-accelerated learning operations"""
    
    def __init__(self, core, pipelines, shaders):
        """Initialize with VulkanCore, VulkanPipelines, and shaders dict"""
        self.core = core
        self.pipelines = pipelines
        self.shaders = shaders
    
    # ==================== Fisher / EWC ====================
    
    def fisher_info_update(
        self,
        gradients: np.ndarray,
        fisher: np.ndarray,
        momentum: float = 0.9,
        use_ema: bool = True,
        reset: bool = False
    ) -> np.ndarray:
        """
        Update Fisher information estimate from gradients.
        
        Fisher information F = E[∇log p(θ)²] ≈ mean(gradient²)
        
        Args:
            gradients: Parameter gradients [num_params]
            fisher: Current Fisher information [num_params]
            momentum: EMA momentum for running estimate
            use_ema: Use exponential moving average
            reset: Reset Fisher before accumulation
            
        Returns:
            Updated Fisher information
        """
        num_params = len(gradients)
        grads_flat = gradients.astype(np.float32).flatten()
        fisher_flat = fisher.astype(np.float32).flatten()
        
        # Create buffers
        buf_grads, mem_grads = self.core._create_buffer(grads_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_fisher, mem_fisher = self.core._create_buffer(fisher_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload
        self.core._upload_buffer(buf_grads, mem_grads, grads_flat)
        self.core._upload_buffer(buf_fisher, mem_fisher, fisher_flat)
        
        # Get pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'fisher-info', 2, push_constant_size=16
        )
        
        # Create descriptor set
        descriptor_set = self.pipelines._create_descriptor_set(
            desc_layout,
            [(buf_grads, grads_flat.nbytes), (buf_fisher, fisher_flat.nbytes)]
        )
        
        # Pack push constants
        push_constants = struct.pack(
            'IfII', num_params, momentum, 1 if use_ema else 0, 1 if reset else 0
        )
        
        # Dispatch
        workgroups = (num_params + 255) // 256
        self.core._dispatch_compute(pipeline, pipeline_layout, descriptor_set, workgroups, push_constants)
        
        # Download
        result = self.core._download_buffer(mem_fisher, fisher_flat.nbytes, dtype=np.float32)
        
        # Cleanup
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])
        vkDestroyBuffer(self.core.device, buf_grads, None)
        vkDestroyBuffer(self.core.device, buf_fisher, None)
        vkFreeMemory(self.core.device, mem_grads, None)
        vkFreeMemory(self.core.device, mem_fisher, None)
        
        return result[:num_params]
    
    def ewc_penalty(
        self,
        current_params: np.ndarray,
        optimal_params: np.ndarray,
        fisher: np.ndarray,
        lambda_ewc: float = 1000.0
    ) -> np.ndarray:
        """
        Compute EWC penalty for continual learning.
        
        Penalty = (λ/2) * Σ F_i * (θ_i - θ*_i)²
        
        Args:
            current_params: Current parameters [num_params]
            optimal_params: Optimal params from previous task [num_params]
            fisher: Fisher information [num_params]
            lambda_ewc: Regularization strength
            
        Returns:
            Per-parameter penalty [num_params] (sum for total penalty)
        """
        num_params = len(current_params)
        current = current_params.astype(np.float32).flatten()
        optimal = optimal_params.astype(np.float32).flatten()
        fisher_flat = fisher.astype(np.float32).flatten()
        penalty = np.zeros(num_params, dtype=np.float32)
        
        # Create buffers
        buf_current, mem_current = self.core._create_buffer(current.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_optimal, mem_optimal = self.core._create_buffer(optimal.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_fisher, mem_fisher = self.core._create_buffer(fisher_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_penalty, mem_penalty = self.core._create_buffer(penalty.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload
        self.core._upload_buffer(buf_current, mem_current, current)
        self.core._upload_buffer(buf_optimal, mem_optimal, optimal)
        self.core._upload_buffer(buf_fisher, mem_fisher, fisher_flat)
        
        # Get pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'fisher-ewc-penalty', 4, push_constant_size=8
        )
        
        # Create descriptor set
        descriptor_set = self.pipelines._create_descriptor_set(
            desc_layout,
            [
                (buf_current, current.nbytes),
                (buf_optimal, optimal.nbytes),
                (buf_fisher, fisher_flat.nbytes),
                (buf_penalty, penalty.nbytes)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack('If', num_params, lambda_ewc)
        
        # Dispatch
        workgroups = (num_params + 255) // 256
        self.core._dispatch_compute(pipeline, pipeline_layout, descriptor_set, workgroups, push_constants)
        
        # Download
        result = self.core._download_buffer(mem_penalty, penalty.nbytes, dtype=np.float32)
        
        # Cleanup
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])
        for buf in [buf_current, buf_optimal, buf_fisher, buf_penalty]:
            vkDestroyBuffer(self.core.device, buf, None)
        for mem in [mem_current, mem_optimal, mem_fisher, mem_penalty]:
            vkFreeMemory(self.core.device, mem, None)
        
        return result[:num_params]
    
    def natural_gradient(
        self,
        gradients: np.ndarray,
        fisher: np.ndarray,
        learning_rate: float = 0.001,
        epsilon: float = 1e-8
    ) -> np.ndarray:
        """
        Apply natural gradient scaling using Fisher information.
        
        Natural gradient: ∇_nat = F^(-1) * ∇ ≈ ∇ / (F + ε)
        
        Args:
            gradients: Raw gradients [num_params]
            fisher: Fisher information [num_params]
            learning_rate: Base learning rate
            epsilon: Stability constant
            
        Returns:
            Scaled gradients for parameter update
        """
        num_params = len(gradients)
        grads = gradients.astype(np.float32).flatten()
        fisher_flat = fisher.astype(np.float32).flatten()
        scaled = np.zeros(num_params, dtype=np.float32)
        
        # Create buffers
        buf_grads, mem_grads = self.core._create_buffer(grads.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_fisher, mem_fisher = self.core._create_buffer(fisher_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_scaled, mem_scaled = self.core._create_buffer(scaled.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload
        self.core._upload_buffer(buf_grads, mem_grads, grads)
        self.core._upload_buffer(buf_fisher, mem_fisher, fisher_flat)
        
        # Get pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'fisher-natural-gradient', 3, push_constant_size=12
        )
        
        # Create descriptor set
        descriptor_set = self.pipelines._create_descriptor_set(
            desc_layout,
            [
                (buf_grads, grads.nbytes),
                (buf_fisher, fisher_flat.nbytes),
                (buf_scaled, scaled.nbytes)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack('Iff', num_params, learning_rate, epsilon)
        
        # Dispatch
        workgroups = (num_params + 255) // 256
        self.core._dispatch_compute(pipeline, pipeline_layout, descriptor_set, workgroups, push_constants)
        
        # Download
        result = self.core._download_buffer(mem_scaled, scaled.nbytes, dtype=np.float32)
        
        # Cleanup
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])
        for buf in [buf_grads, buf_fisher, buf_scaled]:
            vkDestroyBuffer(self.core.device, buf, None)
        for mem in [mem_grads, mem_fisher, mem_scaled]:
            vkFreeMemory(self.core.device, mem, None)
        
        return result[:num_params]
    
    # ==================== NLMS Adaptive Filtering ====================
    
    def nlms_predict(
        self,
        features: np.ndarray,
        weights: np.ndarray,
        bias: float = 0.0
    ) -> np.ndarray:
        """
        NLMS prediction: y = w · x + b
        
        Args:
            features: Input features [batch, n_features] or [n_features]
            weights: Filter weights [n_features]
            bias: Bias term
            
        Returns:
            Predictions [batch] or scalar
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        batch_size, n_features = features.shape
        x = features.astype(np.float32).flatten()
        w = weights.astype(np.float32).flatten()
        b = np.array([bias], dtype=np.float32)
        preds = np.zeros(batch_size, dtype=np.float32)
        
        # Create buffers
        buf_x, mem_x = self.core._create_buffer(x.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_w, mem_w = self.core._create_buffer(w.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_b, mem_b = self.core._create_buffer(b.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_pred, mem_pred = self.core._create_buffer(preds.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload
        self.core._upload_buffer(buf_x, mem_x, x)
        self.core._upload_buffer(buf_w, mem_w, w)
        self.core._upload_buffer(buf_b, mem_b, b)
        
        # Get pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'nlms-predict', 4, push_constant_size=8
        )
        
        # Create descriptor set
        descriptor_set = self.pipelines._create_descriptor_set(
            desc_layout,
            [
                (buf_x, x.nbytes),
                (buf_w, w.nbytes),
                (buf_b, b.nbytes),
                (buf_pred, preds.nbytes)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack('II', batch_size, n_features)
        
        # Dispatch
        workgroups = (batch_size + 255) // 256
        self.core._dispatch_compute(pipeline, pipeline_layout, descriptor_set, workgroups, push_constants)
        
        # Download
        result = self.core._download_buffer(mem_pred, preds.nbytes, dtype=np.float32)
        
        # Cleanup
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])
        for buf in [buf_x, buf_w, buf_b, buf_pred]:
            vkDestroyBuffer(self.core.device, buf, None)
        for mem in [mem_x, mem_w, mem_b, mem_pred]:
            vkFreeMemory(self.core.device, mem, None)
        
        return result[0] if batch_size == 1 else result[:batch_size]
    
    def nlms_update(
        self,
        features: np.ndarray,
        prediction: float,
        target: float,
        weights: np.ndarray,
        bias: float = 0.0,
        learning_rate: float = 0.5,
        mu_decay: float = 0.99995,
        mu_min: float = 0.1,
        epsilon: float = 1e-6
    ) -> tuple:
        """
        NLMS weight update with learning rate decay.
        
        Update: w = w + (μ * error * x) / ||x||²
        
        Args:
            features: Input features [n_features]
            prediction: Current prediction
            target: Target value
            weights: Current weights [n_features]
            bias: Current bias
            learning_rate: Current learning rate (μ)
            mu_decay: Learning rate decay factor
            mu_min: Minimum learning rate
            epsilon: Normalization constant
            
        Returns:
            Tuple of (updated_weights, updated_bias, updated_lr, error)
        """
        n_features = len(features)
        x = features.astype(np.float32).flatten()
        w = weights.astype(np.float32).flatten()
        y_pred = np.array([prediction], dtype=np.float32)
        y_true = np.array([target], dtype=np.float32)
        b = np.array([bias], dtype=np.float32)
        mu = np.array([learning_rate], dtype=np.float32)
        error_out = np.zeros(1, dtype=np.float32)
        
        # Create buffers
        buf_x, mem_x = self.core._create_buffer(x.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_pred, mem_pred = self.core._create_buffer(y_pred.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_true, mem_true = self.core._create_buffer(y_true.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_w, mem_w = self.core._create_buffer(w.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_b, mem_b = self.core._create_buffer(b.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_mu, mem_mu = self.core._create_buffer(mu.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_err, mem_err = self.core._create_buffer(error_out.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload
        self.core._upload_buffer(buf_x, mem_x, x)
        self.core._upload_buffer(buf_pred, mem_pred, y_pred)
        self.core._upload_buffer(buf_true, mem_true, y_true)
        self.core._upload_buffer(buf_w, mem_w, w)
        self.core._upload_buffer(buf_b, mem_b, b)
        self.core._upload_buffer(buf_mu, mem_mu, mu)
        
        # Get pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'nlms-update', 7, push_constant_size=20
        )
        
        # Create descriptor set
        descriptor_set = self.pipelines._create_descriptor_set(
            desc_layout,
            [
                (buf_x, x.nbytes),
                (buf_pred, y_pred.nbytes),
                (buf_true, y_true.nbytes),
                (buf_w, w.nbytes),
                (buf_b, b.nbytes),
                (buf_mu, mu.nbytes),
                (buf_err, error_out.nbytes)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack('Iffff', n_features, mu_decay, mu_min, 0.1, epsilon)
        
        # Dispatch
        workgroups = (n_features + 255) // 256
        self.core._dispatch_compute(pipeline, pipeline_layout, descriptor_set, workgroups, push_constants)
        
        # Download results
        w_out = self.core._download_buffer(mem_w, w.nbytes, dtype=np.float32)[:n_features]
        b_out = self.core._download_buffer(mem_b, b.nbytes, dtype=np.float32)[0]
        mu_out = self.core._download_buffer(mem_mu, mu.nbytes, dtype=np.float32)[0]
        err_out = self.core._download_buffer(mem_err, error_out.nbytes, dtype=np.float32)[0]
        
        # Cleanup
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])
        for buf in [buf_x, buf_pred, buf_true, buf_w, buf_b, buf_mu, buf_err]:
            vkDestroyBuffer(self.core.device, buf, None)
        for mem in [mem_x, mem_pred, mem_true, mem_w, mem_b, mem_mu, mem_err]:
            vkFreeMemory(self.core.device, mem, None)
        
        return w_out, b_out, mu_out, err_out
    
    # ==================== Whitening Transform ====================
    
    def whitening_transform(
        self,
        data: np.ndarray,
        running_mean: np.ndarray,
        running_var: np.ndarray,
        momentum: float = 0.01,
        epsilon: float = 1e-6
    ) -> tuple:
        """
        Apply whitening transform with running statistics.
        
        Output = (x - μ) / sqrt(σ² + ε)
        
        Args:
            data: Input data [batch, dim] or [dim]
            running_mean: Running mean [dim]
            running_var: Running variance [dim]
            momentum: EMA momentum for stats update
            epsilon: Stability constant
            
        Returns:
            Tuple of (whitened_data, updated_mean, updated_var)
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        batch_size, dim = data.shape
        x = data.astype(np.float32).flatten()
        mu = running_mean.astype(np.float32).flatten()
        var = running_var.astype(np.float32).flatten()
        output = np.zeros_like(x)
        
        # Create buffers
        buf_x, mem_x = self.core._create_buffer(x.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_mu, mem_mu = self.core._create_buffer(mu.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_var, mem_var = self.core._create_buffer(var.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self.core._create_buffer(output.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload
        self.core._upload_buffer(buf_x, mem_x, x)
        self.core._upload_buffer(buf_mu, mem_mu, mu)
        self.core._upload_buffer(buf_var, mem_var, var)
        
        # Get pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'whitening-transform', 4, push_constant_size=16
        )
        
        # Create descriptor set
        descriptor_set = self.pipelines._create_descriptor_set(
            desc_layout,
            [
                (buf_x, x.nbytes),
                (buf_mu, mu.nbytes),
                (buf_var, var.nbytes),
                (buf_out, output.nbytes)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack('IIff', batch_size, dim, momentum, epsilon)
        
        # Dispatch
        workgroups = (batch_size * dim + 255) // 256
        self.core._dispatch_compute(pipeline, pipeline_layout, descriptor_set, workgroups, push_constants)
        
        # Download
        result = self.core._download_buffer(mem_out, output.nbytes, dtype=np.float32)
        mu_out = self.core._download_buffer(mem_mu, mu.nbytes, dtype=np.float32)
        var_out = self.core._download_buffer(mem_var, var.nbytes, dtype=np.float32)
        
        # Cleanup
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])
        for buf in [buf_x, buf_mu, buf_var, buf_out]:
            vkDestroyBuffer(self.core.device, buf, None)
        for mem in [mem_x, mem_mu, mem_var, mem_out]:
            vkFreeMemory(self.core.device, mem, None)
        
        result = result.reshape(batch_size, dim)
        return result, mu_out[:dim], var_out[:dim]
    
    # ==================== Bridge Operations ====================
    
    def continuous_to_spikes(
        self,
        features: np.ndarray,
        num_timesteps: int = 10,
        encoding_type: int = 0,
        projection_weights: np.ndarray = None,
        projection_bias: np.ndarray = None,
        random_seed: int = None
    ) -> np.ndarray:
        """
        Convert continuous features to spike trains.
        
        Encoding types:
            0 = Poisson: spike probability = sigmoid(feature)
            1 = Temporal: spike at time proportional to value
        
        Args:
            features: Input features [batch, input_dim] or [input_dim]
            num_timesteps: Number of time steps
            encoding_type: 0=poisson, 1=temporal
            projection_weights: Optional projection [spike_dim, input_dim]
            projection_bias: Optional bias [spike_dim]
            random_seed: Seed for Poisson encoding
            
        Returns:
            Spike trains [batch, time, spike_dim]
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        batch_size, input_dim = features.shape
        use_projection = projection_weights is not None
        spike_dim = projection_weights.shape[0] if use_projection else input_dim
        
        x = features.astype(np.float32).flatten()
        
        # Generate random numbers for Poisson
        if random_seed is not None:
            np.random.seed(random_seed)
        random_nums = np.random.random(batch_size * num_timesteps * spike_dim).astype(np.float32)
        
        # Prepare projection
        if use_projection:
            W = projection_weights.astype(np.float32).flatten()
            b = projection_bias.astype(np.float32).flatten() if projection_bias is not None else np.zeros(spike_dim, dtype=np.float32)
        else:
            W = np.zeros(1, dtype=np.float32)
            b = np.zeros(1, dtype=np.float32)
        
        temp = np.zeros(batch_size * spike_dim, dtype=np.float32)
        spikes = np.zeros(batch_size * num_timesteps * spike_dim, dtype=np.float32)
        
        # Create buffers
        buf_x, mem_x = self.core._create_buffer(x.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_spikes, mem_spikes = self.core._create_buffer(spikes.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_W, mem_W = self.core._create_buffer(W.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_b, mem_b = self.core._create_buffer(b.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_rand, mem_rand = self.core._create_buffer(random_nums.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_temp, mem_temp = self.core._create_buffer(temp.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload
        self.core._upload_buffer(buf_x, mem_x, x)
        self.core._upload_buffer(buf_W, mem_W, W)
        self.core._upload_buffer(buf_b, mem_b, b)
        self.core._upload_buffer(buf_rand, mem_rand, random_nums)
        
        # Get pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'bridge-continuous-to-spike', 6, push_constant_size=28
        )
        
        # Create descriptor set
        descriptor_set = self.pipelines._create_descriptor_set(
            desc_layout,
            [
                (buf_x, x.nbytes),
                (buf_spikes, spikes.nbytes),
                (buf_W, W.nbytes),
                (buf_b, b.nbytes),
                (buf_rand, random_nums.nbytes),
                (buf_temp, temp.nbytes)
            ]
        )
        
        # Pass 1: Project features
        push_constants = struct.pack(
            'IIIIIII',
            batch_size, num_timesteps, input_dim, spike_dim,
            encoding_type, 1 if use_projection else 0, 0  # pass_type=0
        )
        workgroups = (batch_size * spike_dim + 255) // 256
        self.core._dispatch_compute(pipeline, pipeline_layout, descriptor_set, workgroups, push_constants)
        
        # Pass 2: Encode to spikes
        push_constants = struct.pack(
            'IIIIIII',
            batch_size, num_timesteps, input_dim, spike_dim,
            encoding_type, 1 if use_projection else 0, 1  # pass_type=1
        )
        workgroups = (batch_size * num_timesteps * spike_dim + 255) // 256
        self.core._dispatch_compute(pipeline, pipeline_layout, descriptor_set, workgroups, push_constants)
        
        # Download
        result = self.core._download_buffer(mem_spikes, spikes.nbytes, dtype=np.float32)
        
        # Cleanup
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])
        for buf in [buf_x, buf_spikes, buf_W, buf_b, buf_rand, buf_temp]:
            vkDestroyBuffer(self.core.device, buf, None)
        for mem in [mem_x, mem_spikes, mem_W, mem_b, mem_rand, mem_temp]:
            vkFreeMemory(self.core.device, mem, None)
        
        return result.reshape(batch_size, num_timesteps, spike_dim)
    
    def spikes_to_continuous(
        self,
        spikes: np.ndarray,
        encoding_type: int = 0,
        time_window: int = 5,
        temporal_weights: np.ndarray = None,
        projection_weights: np.ndarray = None,
        projection_bias: np.ndarray = None
    ) -> np.ndarray:
        """
        Convert spike trains to continuous features.
        
        Encoding types:
            0 = Rate: mean firing rate over time window
            1 = Temporal: exponentially weighted average
            2 = Phase: simplified phase encoding
        
        Args:
            spikes: Spike trains [batch, time, spike_dim]
            encoding_type: 0=rate, 1=temporal, 2=phase
            time_window: Window for rate encoding
            temporal_weights: Weights for temporal encoding [time]
            projection_weights: Optional projection [output_dim, spike_dim]
            projection_bias: Optional bias [output_dim]
            
        Returns:
            Continuous features [batch, output_dim]
        """
        batch_size, total_time, spike_dim = spikes.shape
        use_projection = projection_weights is not None
        output_dim = projection_weights.shape[0] if use_projection else spike_dim
        
        spike_data = spikes.astype(np.float32).flatten()
        
        # Temporal weights
        if temporal_weights is not None:
            tw = temporal_weights.astype(np.float32).flatten()
        else:
            # Default exponential decay
            tw = np.exp(-np.arange(total_time) / (time_window + 1e-6)).astype(np.float32)
        
        # Projection
        if use_projection:
            W = projection_weights.astype(np.float32).flatten()
            b = projection_bias.astype(np.float32).flatten() if projection_bias is not None else np.zeros(output_dim, dtype=np.float32)
        else:
            W = np.zeros(1, dtype=np.float32)
            b = np.zeros(1, dtype=np.float32)
        
        temp = np.zeros(batch_size * spike_dim, dtype=np.float32)
        output = np.zeros(batch_size * output_dim, dtype=np.float32)
        
        # Create buffers
        buf_spikes, mem_spikes = self.core._create_buffer(spike_data.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self.core._create_buffer(output.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_tw, mem_tw = self.core._create_buffer(tw.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_W, mem_W = self.core._create_buffer(W.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_b, mem_b = self.core._create_buffer(b.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_temp, mem_temp = self.core._create_buffer(temp.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload
        self.core._upload_buffer(buf_spikes, mem_spikes, spike_data)
        self.core._upload_buffer(buf_tw, mem_tw, tw)
        self.core._upload_buffer(buf_W, mem_W, W)
        self.core._upload_buffer(buf_b, mem_b, b)
        
        # Get pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'bridge-spike-to-continuous', 6, push_constant_size=32
        )
        
        # Create descriptor set
        descriptor_set = self.pipelines._create_descriptor_set(
            desc_layout,
            [
                (buf_spikes, spike_data.nbytes),
                (buf_out, output.nbytes),
                (buf_tw, tw.nbytes),
                (buf_W, W.nbytes),
                (buf_b, b.nbytes),
                (buf_temp, temp.nbytes)
            ]
        )
        
        # Pass 1: Encode
        push_constants = struct.pack(
            'IIIIIIII',
            batch_size, total_time, spike_dim, output_dim,
            time_window, encoding_type, 1 if use_projection else 0, 0  # pass_type=0
        )
        workgroups = (batch_size * spike_dim + 255) // 256
        self.core._dispatch_compute(pipeline, pipeline_layout, descriptor_set, workgroups, push_constants)
        
        # Pass 2: Project (if needed)
        if use_projection:
            push_constants = struct.pack(
                'IIIIIIII',
                batch_size, total_time, spike_dim, output_dim,
                time_window, encoding_type, 1, 1  # pass_type=1
            )
            workgroups = (batch_size * output_dim + 255) // 256
            self.core._dispatch_compute(pipeline, pipeline_layout, descriptor_set, workgroups, push_constants)
            result = self.core._download_buffer(mem_out, output.nbytes, dtype=np.float32)
        else:
            result = self.core._download_buffer(mem_temp, temp.nbytes, dtype=np.float32)
        
        # Cleanup
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])
        for buf in [buf_spikes, buf_out, buf_tw, buf_W, buf_b, buf_temp]:
            vkDestroyBuffer(self.core.device, buf, None)
        for mem in [mem_spikes, mem_out, mem_tw, mem_W, mem_b, mem_temp]:
            vkFreeMemory(self.core.device, mem, None)
        
        output_dim_actual = output_dim if use_projection else spike_dim
        return result[:batch_size * output_dim_actual].reshape(batch_size, output_dim_actual)
    
    # ==================== Domain Routing ====================
    
    def domain_route(
        self,
        domain_probs: np.ndarray,
        expert_weights: np.ndarray,
        top_k: int = 2,
        routing_mode: int = 1
    ) -> tuple:
        """
        Route inputs to experts based on domain probabilities.
        
        Args:
            domain_probs: Domain probabilities [batch, num_domains]
            expert_weights: Domain-expert weights [num_domains, num_experts]
            top_k: Number of experts to select
            routing_mode: 0=weighted_sum, 1=top_k_selection
            
        Returns:
            Tuple of (routing_weights, selected_experts)
        """
        if domain_probs.ndim == 1:
            domain_probs = domain_probs.reshape(1, -1)
        
        batch_size, num_domains = domain_probs.shape
        num_experts = expert_weights.shape[1]
        
        probs = domain_probs.astype(np.float32).flatten()
        weights = expert_weights.astype(np.float32).flatten()
        routing = np.zeros(batch_size * num_experts, dtype=np.float32)
        selected = np.zeros(batch_size * top_k, dtype=np.uint32)
        
        # Create buffers
        buf_probs, mem_probs = self.core._create_buffer(probs.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_weights, mem_weights = self.core._create_buffer(weights.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_routing, mem_routing = self.core._create_buffer(routing.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_selected, mem_selected = self.core._create_buffer(selected.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload
        self.core._upload_buffer(buf_probs, mem_probs, probs)
        self.core._upload_buffer(buf_weights, mem_weights, weights)
        
        # Get pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'domain-router', 4, push_constant_size=20
        )
        
        # Create descriptor set
        descriptor_set = self.pipelines._create_descriptor_set(
            desc_layout,
            [
                (buf_probs, probs.nbytes),
                (buf_weights, weights.nbytes),
                (buf_routing, routing.nbytes),
                (buf_selected, selected.nbytes)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack('IIIII', batch_size, num_domains, num_experts, top_k, routing_mode)
        
        # Dispatch
        workgroups = (batch_size + 255) // 256
        self.core._dispatch_compute(pipeline, pipeline_layout, descriptor_set, workgroups, push_constants)
        
        # Download
        routing_out = self.core._download_buffer(mem_routing, routing.nbytes, dtype=np.float32)
        selected_out = self.core._download_buffer(mem_selected, selected.nbytes, dtype=np.uint32)
        
        # Cleanup
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])
        for buf in [buf_probs, buf_weights, buf_routing, buf_selected]:
            vkDestroyBuffer(self.core.device, buf, None)
        for mem in [mem_probs, mem_weights, mem_routing, mem_selected]:
            vkFreeMemory(self.core.device, mem, None)
        
        return (
            routing_out.reshape(batch_size, num_experts),
            selected_out.reshape(batch_size, top_k)
        )
    
    # ==================== Embedding Operations ====================
    
    def embedding_lookup(
        self,
        token_ids: np.ndarray,
        embedding_table: np.ndarray
    ) -> np.ndarray:
        """
        GPU-accelerated embedding lookup.
        
        Args:
            token_ids: Token IDs [batch, seq_len] or [seq_len]
            embedding_table: Embedding matrix [vocab_size, embedding_dim]
            
        Returns:
            Embeddings [batch, seq_len, embedding_dim]
        """
        if token_ids.ndim == 1:
            token_ids = token_ids.reshape(1, -1)
        
        batch_size, seq_len = token_ids.shape
        vocab_size, embedding_dim = embedding_table.shape
        
        tokens = token_ids.astype(np.uint32).flatten()
        embeddings = embedding_table.astype(np.float32).flatten()
        output = np.zeros(batch_size * seq_len * embedding_dim, dtype=np.float32)
        
        # Create buffers
        buf_tokens, mem_tokens = self.core._create_buffer(tokens.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_emb, mem_emb = self.core._create_buffer(embeddings.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self.core._create_buffer(output.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload
        self.core._upload_buffer(buf_tokens, mem_tokens, tokens)
        self.core._upload_buffer(buf_emb, mem_emb, embeddings)
        
        # Get pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'embedding-lookup', 3, push_constant_size=16
        )
        
        # Create descriptor set
        descriptor_set = self.pipelines._create_descriptor_set(
            desc_layout,
            [
                (buf_tokens, tokens.nbytes),
                (buf_emb, embeddings.nbytes),
                (buf_out, output.nbytes)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack('IIII', batch_size, seq_len, vocab_size, embedding_dim)
        
        # Dispatch
        workgroups = (batch_size * seq_len + 255) // 256
        self.core._dispatch_compute(pipeline, pipeline_layout, descriptor_set, workgroups, push_constants)
        
        # Download
        result = self.core._download_buffer(mem_out, output.nbytes, dtype=np.float32)
        
        # Cleanup
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])
        for buf in [buf_tokens, buf_emb, buf_out]:
            vkDestroyBuffer(self.core.device, buf, None)
        for mem in [mem_tokens, mem_emb, mem_out]:
            vkFreeMemory(self.core.device, mem, None)
        
        return result.reshape(batch_size, seq_len, embedding_dim)
    
    def embedding_backward(
        self,
        grad_output: np.ndarray,
        token_ids: np.ndarray,
        vocab_size: int,
        embedding_dim: int
    ) -> np.ndarray:
        """
        GPU-accelerated embedding backward pass.
        
        Accumulates gradients into embedding table using atomic operations.
        
        Args:
            grad_output: Gradient w.r.t. output (batch, seq_len, embedding_dim)
            token_ids: Token IDs (batch, seq_len) or (seq_len)
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            
        Returns:
            Gradient w.r.t. embedding table (vocab_size, embedding_dim)
        """
        if token_ids.ndim == 1:
            token_ids = token_ids.reshape(1, -1)
        
        batch_size, seq_len = token_ids.shape
        
        # Flatten arrays
        tokens = token_ids.astype(np.uint32).flatten()
        grad_flat = grad_output.astype(np.float32).flatten()
        grad_weight = np.zeros(vocab_size * embedding_dim, dtype=np.float32)
        
        # Check if shader is available
        if 'embedding-backward' not in self.shaders:
            # CPU fallback: accumulate gradients
            for i, token_id in enumerate(tokens):
                if 0 <= token_id < vocab_size:
                    start_idx = int(token_id) * embedding_dim
                    grad_start = i * embedding_dim
                    grad_weight[start_idx:start_idx + embedding_dim] += grad_flat[grad_start:grad_start + embedding_dim]
            return grad_weight.reshape(vocab_size, embedding_dim)
        
        # GPU implementation using atomic operations
        # Create buffers
        buf_tokens, mem_tokens = self.core._create_buffer(tokens.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_grad, mem_grad = self.core._create_buffer(grad_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_grad_weight, mem_grad_weight = self.core._create_buffer(grad_weight.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload
        self.core._upload_buffer(buf_tokens, mem_tokens, tokens)
        self.core._upload_buffer(buf_grad, mem_grad, grad_flat)
        # Initialize grad_weight to zeros (already done)
        
        # Get pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'embedding-backward', 3, push_constant_size=16
        )
        
        # Create descriptor set
        descriptor_set = self.pipelines._create_descriptor_set(
            desc_layout,
            [
                (buf_tokens, tokens.nbytes),
                (buf_grad, grad_flat.nbytes),
                (buf_grad_weight, grad_weight.nbytes)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack('IIII', batch_size, seq_len, vocab_size, embedding_dim)
        
        # Dispatch
        workgroups = (batch_size * seq_len + 255) // 256
        self.core._dispatch_compute(pipeline, pipeline_layout, descriptor_set, workgroups, push_constants)
        
        # Download
        result = self.core._download_buffer(mem_grad_weight, grad_weight.nbytes, dtype=np.float32)
        
        # Cleanup
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])
        for buf in [buf_tokens, buf_grad, buf_grad_weight]:
            vkDestroyBuffer(self.core.device, buf, None)
        for mem in [mem_tokens, mem_grad, mem_grad_weight]:
            vkFreeMemory(self.core.device, mem, None)
        
        return result.reshape(vocab_size, embedding_dim)
    
    # ==================== Optimizer Updates ====================
    
    def adam_update(
        self,
        weights: np.ndarray,
        gradients: np.ndarray,
        moment1: np.ndarray,
        moment2: np.ndarray,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        beta1_t: float = 0.0,
        beta2_t: float = 0.0,
        clear_grad: bool = False
    ) -> tuple:
        """
        GPU-accelerated Adam optimizer update.
        
        Uses: adam-update.glsl
        
        Args:
            weights: Parameter weights to update (any shape, flattened)
            gradients: Parameter gradients (same shape as weights)
            moment1: First moment estimate (exp_avg) (same shape as weights)
            moment2: Second moment estimate (exp_avg_sq) (same shape as weights)
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment (default: 0.9)
            beta2: Exponential decay rate for second moment (default: 0.999)
            epsilon: Small value for numerical stability (default: 1e-8)
            beta1_t: beta1^t for bias correction
            beta2_t: beta2^t for bias correction
            clear_grad: Whether to clear gradients after update (default: False)
        
        Returns:
            (updated_weights, updated_moment1, updated_moment2)
        """
        # Check if shader is available
        if 'adam-update' not in self.shaders:
            # CPU fallback
            moment1_new = beta1 * moment1 + (1.0 - beta1) * gradients
            moment2_new = beta2 * moment2 + (1.0 - beta2) * gradients * gradients
            m_hat = moment1_new / (1.0 - beta1_t) if beta1_t < 1.0 else moment1_new
            v_hat = moment2_new / (1.0 - beta2_t) if beta2_t < 1.0 else moment2_new
            weights_new = weights - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            return weights_new, moment1_new, moment2_new
        
        # GPU implementation
        weights_flat = weights.astype(np.float32).flatten()
        grad_flat = gradients.astype(np.float32).flatten()
        m1_flat = moment1.astype(np.float32).flatten()
        m2_flat = moment2.astype(np.float32).flatten()
        
        total_weights = len(weights_flat)
        
        # Verify all arrays have same size
        if not (len(grad_flat) == len(m1_flat) == len(m2_flat) == total_weights):
            raise ValueError("All arrays must have the same size")
        
        # Create buffers
        buf_weights, mem_weights = self.core._create_buffer(weights_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_grad, mem_grad = self.core._create_buffer(grad_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_m1, mem_m1 = self.core._create_buffer(m1_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_m2, mem_m2 = self.core._create_buffer(m2_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(buf_weights, mem_weights, weights_flat)
        self.core._upload_buffer(buf_grad, mem_grad, grad_flat)
        self.core._upload_buffer(buf_m1, mem_m1, m1_flat)
        self.core._upload_buffer(buf_m2, mem_m2, m2_flat)
        
        # Get or create pipeline
        # Push constants: total_weights(uint), lr(float), beta1(float), beta2(float), 
        # epsilon(float), beta1_t(float), beta2_t(float), clear_grad(uint)
        # Total: 4 + 4*5 + 4 = 28 bytes
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'adam-update', 4, push_constant_size=28
        )
        
        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'adam-update',
            [
                (buf_weights, weights_flat.nbytes),
                (buf_grad, grad_flat.nbytes),
                (buf_m1, m1_flat.nbytes),
                (buf_m2, m2_flat.nbytes)
            ]
        )
        
        # Pack push constants
        # Order in shader: total_weights(uint), lr(float), beta1(float), beta2(float),
        # epsilon(float), beta1_t(float), beta2_t(float), clear_grad(uint)
        push_constants = struct.pack('IfffffIf',
            total_weights,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            beta1_t,
            beta2_t,
            1 if clear_grad else 0
        )
        
        # Dispatch
        workgroups = (total_weights + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )
        
        # Download results
        weights_updated = self.core._download_buffer(mem_weights, weights_flat.nbytes, dtype=np.float32)
        m1_updated = self.core._download_buffer(mem_m1, m1_flat.nbytes, dtype=np.float32)
        m2_updated = self.core._download_buffer(mem_m2, m2_flat.nbytes, dtype=np.float32)
        
        # Reshape to original shape
        weights_updated = weights_updated[:total_weights].reshape(weights.shape)
        m1_updated = m1_updated[:total_weights].reshape(moment1.shape)
        m2_updated = m2_updated[:total_weights].reshape(moment2.shape)
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_weights, None)
        vkDestroyBuffer(self.core.device, buf_grad, None)
        vkDestroyBuffer(self.core.device, buf_m1, None)
        vkDestroyBuffer(self.core.device, buf_m2, None)
        vkFreeMemory(self.core.device, mem_weights, None)
        vkFreeMemory(self.core.device, mem_grad, None)
        vkFreeMemory(self.core.device, mem_m1, None)
        vkFreeMemory(self.core.device, mem_m2, None)
        
        return weights_updated, m1_updated, m2_updated
