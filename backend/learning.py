"""
Vulkan Learning Operations Module

GPU-accelerated learning operations:
- Fisher Information / EWC (Elastic Weight Consolidation)
- NLMS (Normalized Least Mean Squares) adaptive filtering
- Whitening transforms
- Bridge operations (continuous <-> spike)
- Domain routing for mixture of experts
- Optimizer updates (Adam, SGD, etc.)
"""

import os
import struct

import numpy as np

from .base import VULKAN_AVAILABLE, BufferMixin

if VULKAN_AVAILABLE:
    from vulkan import *


class VulkanLearning(BufferMixin):
    """GPU-accelerated learning operations"""

    def __init__(self, core, pipelines, shaders):
        """Initialize with VulkanCore, VulkanPipelines, and shaders dict"""
        self.core = core
        self.pipelines = pipelines
        self.shaders = shaders

    def _free_descriptor_set(self, descriptor_set) -> None:
        if descriptor_set is None or not VULKAN_AVAILABLE:
            return
        try:
            vkFreeDescriptorSets(
                self.core.device,
                self.core.descriptor_pool,
                1,
                [descriptor_set],
            )
        except Exception:
            pass

    # ==================== SSM Fused Math ====================

    def ssm_fused_math(
        self,
        gate: np.ndarray,
        value: np.ndarray,
        decay: np.ndarray,
        *,
        return_gpu_tensor: bool = False,
    ) -> np.ndarray:
        """
        Fused selective-scan recurrence for SSM blocks.

        Computes, per feature lane:
            state_t = decay * state_{t-1} + (1 - decay) * value_t
            out_t = state_t * gate_t

        Args:
            gate:  [batch, seq_len, features]
            value: [batch, seq_len, features]
            decay: [features] or broadcastable shape
            return_gpu_tensor: Return VulkanTensor without download

        Returns:
            scan output with shape [batch, seq_len, features]
        """
        gate_np = np.asarray(gate, dtype=np.float32)
        value_np = np.asarray(value, dtype=np.float32)
        if gate_np.shape != value_np.shape or gate_np.ndim != 3:
            raise ValueError("ssm_fused_math expects gate/value with shape [batch, seq_len, features]")

        batch_size, seq_len, features = gate_np.shape
        if batch_size <= 0 or seq_len <= 0 or features <= 0:
            return np.zeros_like(gate_np, dtype=np.float32)

        decay_np = np.asarray(decay, dtype=np.float32).reshape(-1)
        if decay_np.size != features:
            if decay_np.size == 1:
                decay_np = np.full((features,), float(decay_np[0]), dtype=np.float32)
            else:
                raise ValueError(
                    f"ssm_fused_math decay size mismatch: got {decay_np.size}, expected {features}"
                )
        decay_np = np.clip(decay_np, 1e-4, 1.0 - 1e-4).astype(np.float32, copy=False)

        # CPU fallback if shader is missing.
        if "ssm-fused-math" not in self.shaders:
            d = decay_np.reshape(1, 1, features)
            keep = (1.0 - d).astype(np.float32, copy=False)
            state = np.zeros((batch_size, features), dtype=np.float32)
            out = np.zeros_like(gate_np, dtype=np.float32)
            for t in range(seq_len):
                state = (d[:, 0, :] * state) + (keep[:, 0, :] * value_np[:, t, :])
                out[:, t, :] = state * gate_np[:, t, :]
            return out

        gate_flat = np.ascontiguousarray(gate_np.reshape(-1), dtype=np.float32)
        value_flat = np.ascontiguousarray(value_np.reshape(-1), dtype=np.float32)
        decay_flat = np.ascontiguousarray(decay_np.reshape(-1), dtype=np.float32)
        out_nbytes = gate_flat.nbytes

        buf_gate = self._acquire_buffer(gate_flat.nbytes)
        buf_value = self._acquire_buffer(value_flat.nbytes)
        buf_decay = self._acquire_buffer(decay_flat.nbytes)
        buf_out = self._acquire_buffer(out_nbytes)
        descriptor_set = None

        try:
            self._upload_buffer(buf_gate, gate_flat)
            self._upload_buffer(buf_value, value_flat)
            self._upload_buffer(buf_decay, decay_flat)

            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "ssm-fused-math", 4, push_constant_size=20
            )
            descriptor_set = self.pipelines._create_descriptor_set(
                desc_layout,
                [
                    (self._get_buffer_handle(buf_gate), gate_flat.nbytes),
                    (self._get_buffer_handle(buf_value), value_flat.nbytes),
                    (self._get_buffer_handle(buf_decay), decay_flat.nbytes),
                    (self._get_buffer_handle(buf_out), out_nbytes),
                ],
            )

            push_constants = struct.pack(
                "IIIff",
                int(batch_size),
                int(seq_len),
                int(features),
                1e-4,
                float(1.0 - 1e-4),
            )

            total_lanes = int(batch_size * features)
            workgroups = (total_lanes + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            if return_gpu_tensor:
                return self._wrap_output_tensor(buf_out, (batch_size, seq_len, features))

            out = self._download_buffer(buf_out, out_nbytes, np.float32)
            return out.reshape(batch_size, seq_len, features)
        finally:
            self._free_descriptor_set(descriptor_set)
            self._release_buffers([buf_gate, buf_value, buf_decay])
            if not return_gpu_tensor:
                self._release_buffer(buf_out)

    def ssm_fused_uv(
        self,
        uv: np.ndarray,
        decay: np.ndarray,
        *,
        attention_mask: np.ndarray | None = None,
        return_gpu_tensor: bool = False,
    ) -> np.ndarray:
        """
        Fused selective scan from projected UV tensor.

        Args:
            uv: [batch, seq_len, 2 * features] (numpy or VulkanTensor)
            decay: [features]
            attention_mask: optional [batch, seq_len]
            return_gpu_tensor: return VulkanTensor if True
        """
        uv_shape = tuple(int(x) for x in uv.shape)
        if len(uv_shape) != 3 or uv_shape[2] % 2 != 0:
            raise ValueError("ssm_fused_uv expects uv shape [batch, seq_len, 2 * features]")
        batch_size, seq_len, uv_dim = uv_shape
        features = uv_dim // 2
        if batch_size <= 0 or seq_len <= 0 or features <= 0:
            return np.zeros((batch_size, seq_len, features), dtype=np.float32)

        decay_np = np.asarray(decay, dtype=np.float32).reshape(-1)
        if decay_np.size != features:
            if decay_np.size == 1:
                decay_np = np.full((features,), float(decay_np[0]), dtype=np.float32)
            else:
                raise ValueError(f"ssm_fused_uv decay mismatch: got {decay_np.size}, expected {features}")
        decay_np = np.clip(decay_np, 1e-4, 1.0 - 1e-4).astype(np.float32, copy=False)

        if "ssm-fused-uv" not in self.shaders:
            # CPU fallback path.
            uv_np = np.asarray(uv, dtype=np.float32)
            gate_raw, value_raw = np.split(uv_np, 2, axis=-1)
            gate = 1.0 / (1.0 + np.exp(-np.clip(gate_raw, -20.0, 20.0)))
            value_act = np.tanh(np.clip(value_raw, -20.0, 20.0))
            if attention_mask is not None:
                m = np.asarray(attention_mask, dtype=np.float32)
                if m.ndim == 2:
                    m = m[:, :, None]
                gate = gate * m
                value_act = value_act * m
            d = decay_np.reshape(1, 1, features)
            keep = (1.0 - d).astype(np.float32, copy=False)
            state = np.zeros((batch_size, features), dtype=np.float32)
            out = np.zeros((batch_size, seq_len, features), dtype=np.float32)
            for t in range(seq_len):
                state = (d[:, 0, :] * state) + (keep[:, 0, :] * value_act[:, t, :])
                out[:, t, :] = state * gate[:, t, :]
            return out

        uv_nbytes = int(batch_size * seq_len * uv_dim * 4)
        out_nbytes = int(batch_size * seq_len * features * 4)
        mask_arr = None
        has_mask = 0
        if attention_mask is not None:
            mask_arr = np.asarray(attention_mask, dtype=np.float32).reshape(batch_size, seq_len)
            has_mask = 1
        else:
            mask_arr = np.ones((1,), dtype=np.float32)

        buf_uv, release_uv = self._prepare_input(uv, size=uv_nbytes)
        buf_decay = self._acquire_buffer(decay_np.nbytes)
        buf_mask = self._acquire_buffer(mask_arr.nbytes)
        buf_out = self._acquire_buffer(out_nbytes)
        descriptor_set = None

        try:
            self._upload_buffer(buf_decay, decay_np)
            self._upload_buffer(buf_mask, np.ascontiguousarray(mask_arr, dtype=np.float32).reshape(-1))

            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "ssm-fused-uv", 4, push_constant_size=24
            )
            descriptor_set = self.pipelines._create_descriptor_set(
                desc_layout,
                [
                    (self._get_buffer_handle(buf_uv), uv_nbytes),
                    (self._get_buffer_handle(buf_decay), decay_np.nbytes),
                    (self._get_buffer_handle(buf_mask), mask_arr.nbytes),
                    (self._get_buffer_handle(buf_out), out_nbytes),
                ],
            )
            push_constants = struct.pack(
                "IIIIff",
                int(batch_size),
                int(seq_len),
                int(features),
                int(has_mask),
                1e-4,
                float(1.0 - 1e-4),
            )
            total_lanes = int(batch_size * features)
            workgroups = (total_lanes + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            if return_gpu_tensor:
                return self._wrap_output_tensor(buf_out, (batch_size, seq_len, features))

            out = self._download_buffer(buf_out, out_nbytes, np.float32)
            return out.reshape(batch_size, seq_len, features)
        finally:
            self._free_descriptor_set(descriptor_set)
            if release_uv:
                self._release_buffer(buf_uv)
            self._release_buffers([buf_decay, buf_mask])
            if not return_gpu_tensor:
                self._release_buffer(buf_out)

    # ==================== SSD Chunk Scan ====================

    def ssd_chunk_scan(
        self,
        input_data,
        decay: np.ndarray,
        init_state,
        chunk_size: int,
        *,
        is_backward: bool = False,
        return_gpu_tensor: bool = False,
    ):
        """
        SSD within-chunk scan (bidirectional linear recurrence).

        Forward:  s[j] = decay * s[j-1] + u[j],  j = 0..C-1
        Backward: g[j] = decay * g[j+1] + gs[j], j = C-1..0

        Args:
            input_data: (batch_chunks, chunk_size, features) - scan input
                        Accepts numpy array or VulkanTensor.
            decay: (features,) - per-feature decay
            init_state: (batch_chunks, features) - initial state per chunk
                        Accepts numpy array or VulkanTensor.
            chunk_size: C - timesteps per chunk
            is_backward: if True, reverse scan direction
            return_gpu_tensor: return VulkanTensor if True

        Returns:
            output: (batch_chunks, chunk_size, features) - scan output
            carry_out: (batch_chunks, features) - final state per chunk
        """
        input_shape = tuple(int(x) for x in input_data.shape)
        if len(input_shape) != 3:
            raise ValueError(f"ssd_chunk_scan expects 3D input, got shape {input_shape}")
        batch_chunks, cs, features = input_shape
        if cs != chunk_size:
            raise ValueError(f"chunk_size mismatch: input has {cs}, expected {chunk_size}")

        decay_np = np.asarray(decay, dtype=np.float32).reshape(-1)
        if decay_np.size != features:
            raise ValueError(f"decay size {decay_np.size} != features {features}")
        decay_np = np.clip(decay_np, 1e-4, 1.0 - 1e-4).astype(np.float32, copy=False)

        # CPU fallback if shader is missing
        if "ssd-scan-chunks" not in self.shaders:
            inp = np.asarray(input_data, dtype=np.float32)
            init = np.asarray(init_state, dtype=np.float32)
            d = decay_np.reshape(1, features)
            out = np.zeros_like(inp, dtype=np.float32)
            carry = np.zeros((batch_chunks, features), dtype=np.float32)
            if not is_backward:
                for bc in range(batch_chunks):
                    state = init[bc].copy()
                    for j in range(chunk_size):
                        state = d[0] * state + inp[bc, j]
                        out[bc, j] = state
                    carry[bc] = state
            else:
                for bc in range(batch_chunks):
                    state = init[bc].copy()
                    for j in range(chunk_size - 1, -1, -1):
                        state = d[0] * state + inp[bc, j]
                        out[bc, j] = state
                    carry[bc] = state
            return out, carry

        out_nbytes = int(batch_chunks * chunk_size * features * 4)
        carry_nbytes = int(batch_chunks * features * 4)

        buf_input, release_input = self._prepare_input(input_data, size=out_nbytes)
        buf_decay = self._acquire_buffer(decay_np.nbytes)
        buf_init, release_init = self._prepare_input(init_state, size=carry_nbytes)
        buf_out = self._acquire_buffer(out_nbytes)
        buf_carry = self._acquire_buffer(carry_nbytes)
        descriptor_set = None

        try:
            self._upload_buffer(buf_decay, np.ascontiguousarray(decay_np))

            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "ssd-scan-chunks", 5, push_constant_size=16
            )
            descriptor_set = self.pipelines._create_descriptor_set(
                desc_layout,
                [
                    (self._get_buffer_handle(buf_input), out_nbytes),
                    (self._get_buffer_handle(buf_decay), decay_np.nbytes),
                    (self._get_buffer_handle(buf_init), carry_nbytes),
                    (self._get_buffer_handle(buf_out), out_nbytes),
                    (self._get_buffer_handle(buf_carry), carry_nbytes),
                ],
            )

            push_constants = struct.pack(
                "IIII",
                int(batch_chunks),
                int(chunk_size),
                int(features),
                int(1 if is_backward else 0),
            )

            total_lanes = int(batch_chunks * features)
            workgroups = (total_lanes + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            if return_gpu_tensor:
                out_vt = self._wrap_output_tensor(buf_out, (batch_chunks, chunk_size, features))
                carry_np = self._download_buffer(buf_carry, carry_nbytes, np.float32)
                carry_np = carry_np.reshape(batch_chunks, features)
                return out_vt, carry_np

            out = self._download_buffer(buf_out, out_nbytes, np.float32)
            out = out.reshape(batch_chunks, chunk_size, features)
            carry = self._download_buffer(buf_carry, carry_nbytes, np.float32)
            carry = carry.reshape(batch_chunks, features)
            return out, carry
        finally:
            self._free_descriptor_set(descriptor_set)
            if release_input:
                self._release_buffer(buf_input)
            if release_init:
                self._release_buffer(buf_init)
            self._release_buffers([buf_decay])
            if not return_gpu_tensor:
                self._release_buffer(buf_out)
            self._release_buffer(buf_carry)

    # ==================== Fisher / EWC ====================

    def fisher_info_update(
        self,
        gradients: np.ndarray,
        fisher: np.ndarray,
        momentum: float = 0.9,
        use_ema: bool = True,
        reset: bool = False,
    ) -> np.ndarray:
        """
        Update Fisher information estimate from gradients.

        Fisher information F = E[nabla log p(theta)^2] ~ mean(gradient^2)

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

        # Acquire buffers
        buf_grads = self._acquire_buffer(grads_flat.nbytes)
        buf_fisher = self._acquire_buffer(fisher_flat.nbytes)
        descriptor_set = None

        try:
            # Upload
            self._upload_buffer(buf_grads, grads_flat)
            self._upload_buffer(buf_fisher, fisher_flat)

            # Get pipeline
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "fisher-info", 2, push_constant_size=16
            )

            # Create descriptor set
            descriptor_set = self.pipelines._create_descriptor_set(
                desc_layout,
                [
                    (self._get_buffer_handle(buf_grads), grads_flat.nbytes),
                    (self._get_buffer_handle(buf_fisher), fisher_flat.nbytes),
                ],
            )

            # Pack push constants
            push_constants = struct.pack(
                "IfII", num_params, momentum, 1 if use_ema else 0, 1 if reset else 0
            )

            # Dispatch
            workgroups = (num_params + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            # Download
            result = self._download_buffer(buf_fisher, fisher_flat.nbytes, np.float32)

            return result[:num_params]
        finally:
            self._free_descriptor_set(descriptor_set)
            self._release_buffers([buf_grads, buf_fisher])

    def ewc_penalty(
        self,
        current_params: np.ndarray,
        optimal_params: np.ndarray,
        fisher: np.ndarray,
        lambda_ewc: float = 1000.0,
    ) -> np.ndarray:
        """
        Compute EWC penalty for continual learning.

        Penalty = (lambda/2) * sum F_i * (theta_i - theta*_i)^2

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

        # Acquire buffers
        buf_current = self._acquire_buffer(current.nbytes)
        buf_optimal = self._acquire_buffer(optimal.nbytes)
        buf_fisher = self._acquire_buffer(fisher_flat.nbytes)
        buf_penalty = self._acquire_buffer(penalty.nbytes)
        descriptor_set = None

        try:
            # Upload
            self._upload_buffer(buf_current, current)
            self._upload_buffer(buf_optimal, optimal)
            self._upload_buffer(buf_fisher, fisher_flat)

            # Get pipeline
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "fisher-ewc-penalty", 4, push_constant_size=8
            )

            # Create descriptor set
            descriptor_set = self.pipelines._create_descriptor_set(
                desc_layout,
                [
                    (self._get_buffer_handle(buf_current), current.nbytes),
                    (self._get_buffer_handle(buf_optimal), optimal.nbytes),
                    (self._get_buffer_handle(buf_fisher), fisher_flat.nbytes),
                    (self._get_buffer_handle(buf_penalty), penalty.nbytes),
                ],
            )

            # Pack push constants
            push_constants = struct.pack("If", num_params, lambda_ewc)

            # Dispatch
            workgroups = (num_params + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            # Download
            result = self._download_buffer(buf_penalty, penalty.nbytes, np.float32)

            return result[:num_params]
        finally:
            self._free_descriptor_set(descriptor_set)
            self._release_buffers([buf_current, buf_optimal, buf_fisher, buf_penalty])

    def natural_gradient(
        self,
        gradients: np.ndarray,
        fisher: np.ndarray,
        learning_rate: float = 0.001,
        epsilon: float = 1e-8,
    ) -> np.ndarray:
        """
        Apply natural gradient scaling using Fisher information.

        Natural gradient: nabla_nat = F^(-1) * nabla ~ nabla / (F + epsilon)

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

        # Acquire buffers
        buf_grads = self._acquire_buffer(grads.nbytes)
        buf_fisher = self._acquire_buffer(fisher_flat.nbytes)
        buf_scaled = self._acquire_buffer(scaled.nbytes)
        descriptor_set = None

        try:
            # Upload
            self._upload_buffer(buf_grads, grads)
            self._upload_buffer(buf_fisher, fisher_flat)

            # Get pipeline
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "fisher-natural-gradient", 3, push_constant_size=12
            )

            # Create descriptor set
            descriptor_set = self.pipelines._create_descriptor_set(
                desc_layout,
                [
                    (self._get_buffer_handle(buf_grads), grads.nbytes),
                    (self._get_buffer_handle(buf_fisher), fisher_flat.nbytes),
                    (self._get_buffer_handle(buf_scaled), scaled.nbytes),
                ],
            )

            # Pack push constants
            push_constants = struct.pack("Iff", num_params, learning_rate, epsilon)

            # Dispatch
            workgroups = (num_params + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            # Download
            result = self._download_buffer(buf_scaled, scaled.nbytes, np.float32)

            return result[:num_params]
        finally:
            self._free_descriptor_set(descriptor_set)
            self._release_buffers([buf_grads, buf_fisher, buf_scaled])

    # ==================== NLMS Adaptive Filtering ====================

    def nlms_predict(
        self, features: np.ndarray, weights: np.ndarray, bias: float = 0.0
    ) -> np.ndarray:
        """
        NLMS prediction: y = w . x + b

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

        # Acquire buffers
        buf_x = self._acquire_buffer(x.nbytes)
        buf_w = self._acquire_buffer(w.nbytes)
        buf_b = self._acquire_buffer(b.nbytes)
        buf_pred = self._acquire_buffer(preds.nbytes)
        descriptor_set = None

        try:
            # Upload
            self._upload_buffer(buf_x, x)
            self._upload_buffer(buf_w, w)
            self._upload_buffer(buf_b, b)

            # Get pipeline
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "nlms-predict", 4, push_constant_size=8
            )

            # Create descriptor set
            descriptor_set = self.pipelines._create_descriptor_set(
                desc_layout,
                [
                    (self._get_buffer_handle(buf_x), x.nbytes),
                    (self._get_buffer_handle(buf_w), w.nbytes),
                    (self._get_buffer_handle(buf_b), b.nbytes),
                    (self._get_buffer_handle(buf_pred), preds.nbytes),
                ],
            )

            # Pack push constants
            push_constants = struct.pack("II", batch_size, n_features)

            # Dispatch
            workgroups = (batch_size + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            # Download
            result = self._download_buffer(buf_pred, preds.nbytes, np.float32)

            return result[0] if batch_size == 1 else result[:batch_size]
        finally:
            self._free_descriptor_set(descriptor_set)
            self._release_buffers([buf_x, buf_w, buf_b, buf_pred])

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
        epsilon: float = 1e-6,
    ) -> tuple:
        """
        NLMS weight update with learning rate decay.

        Update: w = w + (mu * error * x) / ||x||^2

        Args:
            features: Input features [n_features]
            prediction: Current prediction
            target: Target value
            weights: Current weights [n_features]
            bias: Current bias
            learning_rate: Current learning rate (mu)
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

        # Acquire buffers
        buf_x = self._acquire_buffer(x.nbytes)
        buf_pred = self._acquire_buffer(y_pred.nbytes)
        buf_true = self._acquire_buffer(y_true.nbytes)
        buf_w = self._acquire_buffer(w.nbytes)
        buf_b = self._acquire_buffer(b.nbytes)
        buf_mu = self._acquire_buffer(mu.nbytes)
        buf_err = self._acquire_buffer(error_out.nbytes)
        descriptor_set = None

        try:
            # Upload
            self._upload_buffer(buf_x, x)
            self._upload_buffer(buf_pred, y_pred)
            self._upload_buffer(buf_true, y_true)
            self._upload_buffer(buf_w, w)
            self._upload_buffer(buf_b, b)
            self._upload_buffer(buf_mu, mu)

            # Get pipeline
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "nlms-update", 7, push_constant_size=20
            )

            # Create descriptor set
            descriptor_set = self.pipelines._create_descriptor_set(
                desc_layout,
                [
                    (self._get_buffer_handle(buf_x), x.nbytes),
                    (self._get_buffer_handle(buf_pred), y_pred.nbytes),
                    (self._get_buffer_handle(buf_true), y_true.nbytes),
                    (self._get_buffer_handle(buf_w), w.nbytes),
                    (self._get_buffer_handle(buf_b), b.nbytes),
                    (self._get_buffer_handle(buf_mu), mu.nbytes),
                    (self._get_buffer_handle(buf_err), error_out.nbytes),
                ],
            )

            # Pack push constants
            push_constants = struct.pack("Iffff", n_features, mu_decay, mu_min, 0.1, epsilon)

            # Dispatch
            workgroups = (n_features + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            # Download results
            w_out = self._download_buffer(buf_w, w.nbytes, np.float32)[:n_features]
            b_out = self._download_buffer(buf_b, b.nbytes, np.float32)[0]
            mu_out = self._download_buffer(buf_mu, mu.nbytes, np.float32)[0]
            err_out = self._download_buffer(buf_err, error_out.nbytes, np.float32)[0]

            return w_out, b_out, mu_out, err_out
        finally:
            self._free_descriptor_set(descriptor_set)
            self._release_buffers([buf_x, buf_pred, buf_true, buf_w, buf_b, buf_mu, buf_err])

    # ==================== Whitening Transform ====================

    def whitening_transform(
        self,
        data: np.ndarray,
        running_mean: np.ndarray,
        running_var: np.ndarray,
        momentum: float = 0.01,
        epsilon: float = 1e-6,
    ) -> tuple:
        """
        Apply whitening transform with running statistics.

        Output = (x - mu) / sqrt(sigma^2 + epsilon)

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

        # Acquire buffers
        buf_x = self._acquire_buffer(x.nbytes)
        buf_mu = self._acquire_buffer(mu.nbytes)
        buf_var = self._acquire_buffer(var.nbytes)
        buf_out = self._acquire_buffer(output.nbytes)
        descriptor_set = None

        try:
            # Upload
            self._upload_buffer(buf_x, x)
            self._upload_buffer(buf_mu, mu)
            self._upload_buffer(buf_var, var)

            # Get pipeline
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "whitening-transform", 4, push_constant_size=16
            )

            # Create descriptor set
            descriptor_set = self.pipelines._create_descriptor_set(
                desc_layout,
                [
                    (self._get_buffer_handle(buf_x), x.nbytes),
                    (self._get_buffer_handle(buf_mu), mu.nbytes),
                    (self._get_buffer_handle(buf_var), var.nbytes),
                    (self._get_buffer_handle(buf_out), output.nbytes),
                ],
            )

            # Pack push constants
            push_constants = struct.pack("IIff", batch_size, dim, momentum, epsilon)

            # Dispatch
            workgroups = (batch_size * dim + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            # Download
            result = self._download_buffer(buf_out, output.nbytes, np.float32)
            mu_out = self._download_buffer(buf_mu, mu.nbytes, np.float32)
            var_out = self._download_buffer(buf_var, var.nbytes, np.float32)

            result = result.reshape(batch_size, dim)
            return result, mu_out[:dim], var_out[:dim]
        finally:
            self._free_descriptor_set(descriptor_set)
            self._release_buffers([buf_x, buf_mu, buf_var, buf_out])

    # ==================== Bridge Operations ====================

    def continuous_to_spikes(
        self,
        features: np.ndarray,
        num_timesteps: int = 10,
        encoding_type: int = 0,
        projection_weights: np.ndarray = None,
        projection_bias: np.ndarray = None,
        random_seed: int = None,
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
            b = (
                projection_bias.astype(np.float32).flatten()
                if projection_bias is not None
                else np.zeros(spike_dim, dtype=np.float32)
            )
        else:
            W = np.zeros(1, dtype=np.float32)
            b = np.zeros(1, dtype=np.float32)

        temp = np.zeros(batch_size * spike_dim, dtype=np.float32)
        spikes = np.zeros(batch_size * num_timesteps * spike_dim, dtype=np.float32)

        # Acquire buffers
        buf_x = self._acquire_buffer(x.nbytes)
        buf_spikes = self._acquire_buffer(spikes.nbytes)
        buf_W = self._acquire_buffer(W.nbytes)
        buf_b = self._acquire_buffer(b.nbytes)
        buf_rand = self._acquire_buffer(random_nums.nbytes)
        buf_temp = self._acquire_buffer(temp.nbytes)
        descriptor_set = None

        try:
            # Upload
            self._upload_buffer(buf_x, x)
            self._upload_buffer(buf_W, W)
            self._upload_buffer(buf_b, b)
            self._upload_buffer(buf_rand, random_nums)

            # Get pipeline
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "bridge-continuous-to-spike", 6, push_constant_size=28
            )

            # Create descriptor set
            descriptor_set = self.pipelines._create_descriptor_set(
                desc_layout,
                [
                    (self._get_buffer_handle(buf_x), x.nbytes),
                    (self._get_buffer_handle(buf_spikes), spikes.nbytes),
                    (self._get_buffer_handle(buf_W), W.nbytes),
                    (self._get_buffer_handle(buf_b), b.nbytes),
                    (self._get_buffer_handle(buf_rand), random_nums.nbytes),
                    (self._get_buffer_handle(buf_temp), temp.nbytes),
                ],
            )

            # Pass 1: Project features
            push_constants = struct.pack(
                "IIIIIII",
                batch_size,
                num_timesteps,
                input_dim,
                spike_dim,
                encoding_type,
                1 if use_projection else 0,
                0,  # pass_type=0
            )
            workgroups = (batch_size * spike_dim + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            # Pass 2: Encode to spikes
            push_constants = struct.pack(
                "IIIIIII",
                batch_size,
                num_timesteps,
                input_dim,
                spike_dim,
                encoding_type,
                1 if use_projection else 0,
                1,  # pass_type=1
            )
            workgroups = (batch_size * num_timesteps * spike_dim + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            # Download
            result = self._download_buffer(buf_spikes, spikes.nbytes, np.float32)

            return result.reshape(batch_size, num_timesteps, spike_dim)
        finally:
            self._free_descriptor_set(descriptor_set)
            self._release_buffers([buf_x, buf_spikes, buf_W, buf_b, buf_rand, buf_temp])

    def spikes_to_continuous(
        self,
        spikes: np.ndarray,
        encoding_type: int = 0,
        time_window: int = 5,
        temporal_weights: np.ndarray = None,
        projection_weights: np.ndarray = None,
        projection_bias: np.ndarray = None,
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
            b = (
                projection_bias.astype(np.float32).flatten()
                if projection_bias is not None
                else np.zeros(output_dim, dtype=np.float32)
            )
        else:
            W = np.zeros(1, dtype=np.float32)
            b = np.zeros(1, dtype=np.float32)

        temp = np.zeros(batch_size * spike_dim, dtype=np.float32)
        output = np.zeros(batch_size * output_dim, dtype=np.float32)

        # Acquire buffers
        buf_spikes = self._acquire_buffer(spike_data.nbytes)
        buf_out = self._acquire_buffer(output.nbytes)
        buf_tw = self._acquire_buffer(tw.nbytes)
        buf_W = self._acquire_buffer(W.nbytes)
        buf_b = self._acquire_buffer(b.nbytes)
        buf_temp = self._acquire_buffer(temp.nbytes)
        descriptor_set = None

        try:
            # Upload
            self._upload_buffer(buf_spikes, spike_data)
            self._upload_buffer(buf_tw, tw)
            self._upload_buffer(buf_W, W)
            self._upload_buffer(buf_b, b)

            # Get pipeline
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "bridge-spike-to-continuous", 6, push_constant_size=32
            )

            # Create descriptor set
            descriptor_set = self.pipelines._create_descriptor_set(
                desc_layout,
                [
                    (self._get_buffer_handle(buf_spikes), spike_data.nbytes),
                    (self._get_buffer_handle(buf_out), output.nbytes),
                    (self._get_buffer_handle(buf_tw), tw.nbytes),
                    (self._get_buffer_handle(buf_W), W.nbytes),
                    (self._get_buffer_handle(buf_b), b.nbytes),
                    (self._get_buffer_handle(buf_temp), temp.nbytes),
                ],
            )

            # Pass 1: Encode
            push_constants = struct.pack(
                "IIIIIIII",
                batch_size,
                total_time,
                spike_dim,
                output_dim,
                time_window,
                encoding_type,
                1 if use_projection else 0,
                0,  # pass_type=0
            )
            workgroups = (batch_size * spike_dim + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            # Pass 2: Project (if needed)
            if use_projection:
                push_constants = struct.pack(
                    "IIIIIIII",
                    batch_size,
                    total_time,
                    spike_dim,
                    output_dim,
                    time_window,
                    encoding_type,
                    1,
                    1,  # pass_type=1
                )
                workgroups = (batch_size * output_dim + 255) // 256
                self.core._dispatch_compute(
                    pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
                )
                result = self._download_buffer(buf_out, output.nbytes, np.float32)
            else:
                result = self._download_buffer(buf_temp, temp.nbytes, np.float32)

            output_dim_actual = output_dim if use_projection else spike_dim
            return result[: batch_size * output_dim_actual].reshape(batch_size, output_dim_actual)
        finally:
            self._free_descriptor_set(descriptor_set)
            self._release_buffers([buf_spikes, buf_out, buf_tw, buf_W, buf_b, buf_temp])

    # ==================== Domain Routing ====================

    def domain_route(
        self,
        domain_probs: np.ndarray,
        expert_weights: np.ndarray,
        top_k: int = 2,
        routing_mode: int = 1,
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

        # Acquire buffers
        buf_probs = self._acquire_buffer(probs.nbytes)
        buf_weights = self._acquire_buffer(weights.nbytes)
        buf_routing = self._acquire_buffer(routing.nbytes)
        buf_selected = self._acquire_buffer(selected.nbytes)
        descriptor_set = None

        try:
            # Upload
            self._upload_buffer(buf_probs, probs)
            self._upload_buffer(buf_weights, weights)

            # Get pipeline
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "domain-router", 4, push_constant_size=20
            )

            # Create descriptor set
            descriptor_set = self.pipelines._create_descriptor_set(
                desc_layout,
                [
                    (self._get_buffer_handle(buf_probs), probs.nbytes),
                    (self._get_buffer_handle(buf_weights), weights.nbytes),
                    (self._get_buffer_handle(buf_routing), routing.nbytes),
                    (self._get_buffer_handle(buf_selected), selected.nbytes),
                ],
            )

            # Pack push constants
            push_constants = struct.pack(
                "IIIII", batch_size, num_domains, num_experts, top_k, routing_mode
            )

            # Dispatch
            workgroups = (batch_size + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            # Download
            routing_out = self._download_buffer(buf_routing, routing.nbytes, np.float32)
            selected_out = self._download_buffer(buf_selected, selected.nbytes, np.uint32)

            return (
                routing_out.reshape(batch_size, num_experts),
                selected_out.reshape(batch_size, top_k),
            )
        finally:
            self._free_descriptor_set(descriptor_set)
            self._release_buffers([buf_probs, buf_weights, buf_routing, buf_selected])

    # ==================== Embedding Operations ====================

    def embedding_lookup(
        self,
        token_ids: np.ndarray,
        embedding_table: np.ndarray,
        *,
        return_gpu_tensor: bool = False,
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

        # Preserve identity for weight-cache hits whenever possible.
        if embedding_table.dtype != np.float32 or not embedding_table.flags.c_contiguous:
            embedding_table = np.ascontiguousarray(embedding_table, dtype=np.float32)

        batch_size, seq_len = token_ids.shape
        vocab_size, embedding_dim = embedding_table.shape

        tokens = np.ascontiguousarray(token_ids.astype(np.uint32, copy=False)).reshape(-1)
        output = np.zeros(batch_size * seq_len * embedding_dim, dtype=np.float32)

        # Acquire buffers
        buf_tokens = self._acquire_buffer(tokens.nbytes)
        buf_out = self._acquire_buffer(output.nbytes)
        # Cache embedding tables on GPU to avoid re-uploading full vocabulary each step.
        buf_emb, _ = self._get_or_upload_weight(embedding_table)
        descriptor_set = None

        try:
            # Upload
            self._upload_buffer(buf_tokens, tokens)

            shader_name = "embedding-lookup"
            tiled_min_bytes = int(
                os.getenv("GRILLY_EMBEDDING_TILED_MIN_BYTES", str(64 * 1024 * 1024))
            )
            if (
                "embedding-lookup-tiled" in self.shaders
                and embedding_table.nbytes >= tiled_min_bytes
            ):
                shader_name = "embedding-lookup-tiled"

            # Get pipeline
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                shader_name, 3, push_constant_size=16
            )

            # Create descriptor set
            descriptor_set = self.pipelines._create_descriptor_set(
                desc_layout,
                [
                    (self._get_buffer_handle(buf_tokens), tokens.nbytes),
                    (self._get_buffer_handle(buf_emb), embedding_table.nbytes),
                    (self._get_buffer_handle(buf_out), output.nbytes),
                ],
            )

            # Pack push constants
            push_constants = struct.pack("IIII", batch_size, seq_len, vocab_size, embedding_dim)

            # Dispatch
            total_tokens = batch_size * seq_len
            if shader_name == "embedding-lookup-tiled":
                workgroups_x = (total_tokens + 15) // 16
                workgroups_y = (embedding_dim + 15) // 16
                self.core._dispatch_compute(
                    pipeline,
                    pipeline_layout,
                    descriptor_set,
                    workgroups_x,
                    push_constants,
                    workgroup_y=workgroups_y,
                )
            else:
                workgroups = (total_tokens + 255) // 256
                self.core._dispatch_compute(
                    pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
                )

            if return_gpu_tensor:
                return self._wrap_output_tensor(buf_out, (batch_size, seq_len, embedding_dim))

            # Download
            result = self._download_buffer(buf_out, output.nbytes, np.float32)
            return result.reshape(batch_size, seq_len, embedding_dim)
        finally:
            self._free_descriptor_set(descriptor_set)
            self._release_buffer(buf_tokens)
            if not return_gpu_tensor:
                self._release_buffer(buf_out)

    def embedding_backward(
        self, grad_output: np.ndarray, token_ids: np.ndarray, vocab_size: int, embedding_dim: int
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
        if "embedding-backward" not in self.shaders:
            # CPU fallback: accumulate gradients
            for i, token_id in enumerate(tokens):
                if 0 <= token_id < vocab_size:
                    start_idx = int(token_id) * embedding_dim
                    grad_start = i * embedding_dim
                    grad_weight[start_idx : start_idx + embedding_dim] += grad_flat[
                        grad_start : grad_start + embedding_dim
                    ]
            return grad_weight.reshape(vocab_size, embedding_dim)

        # GPU implementation using atomic operations
        # Acquire buffers
        buf_tokens = self._acquire_buffer(tokens.nbytes)
        buf_grad = self._acquire_buffer(grad_flat.nbytes)
        buf_grad_weight = self._acquire_buffer(grad_weight.nbytes)
        descriptor_set = None

        try:
            # Upload
            self._upload_buffer(buf_tokens, tokens)
            self._upload_buffer(buf_grad, grad_flat)
            # Initialize grad_weight to zeros (already done)

            # Get pipeline
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "embedding-backward", 3, push_constant_size=16
            )

            # Create descriptor set
            descriptor_set = self.pipelines._create_descriptor_set(
                desc_layout,
                [
                    (self._get_buffer_handle(buf_tokens), tokens.nbytes),
                    (self._get_buffer_handle(buf_grad), grad_flat.nbytes),
                    (self._get_buffer_handle(buf_grad_weight), grad_weight.nbytes),
                ],
            )

            # Pack push constants
            push_constants = struct.pack("IIII", batch_size, seq_len, vocab_size, embedding_dim)

            # Dispatch
            workgroups = (batch_size * seq_len + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            # Download
            result = self._download_buffer(buf_grad_weight, grad_weight.nbytes, np.float32)

            return result.reshape(vocab_size, embedding_dim)
        finally:
            self._free_descriptor_set(descriptor_set)
            self._release_buffers([buf_tokens, buf_grad, buf_grad_weight])

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
        clear_grad: bool = False,
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
        if "adam-update" not in self.shaders:
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

        # Acquire buffers
        buf_weights = self._acquire_buffer(weights_flat.nbytes)
        buf_grad = self._acquire_buffer(grad_flat.nbytes)
        buf_m1 = self._acquire_buffer(m1_flat.nbytes)
        buf_m2 = self._acquire_buffer(m2_flat.nbytes)

        try:
            # Upload data
            self._upload_buffer(buf_weights, weights_flat)
            self._upload_buffer(buf_grad, grad_flat)
            self._upload_buffer(buf_m1, m1_flat)
            self._upload_buffer(buf_m2, m2_flat)

            # Get or create pipeline
            # Push constants: total_weights(uint), lr(float), beta1(float), beta2(float),
            # epsilon(float), beta1_t(float), beta2_t(float), clear_grad(uint)
            # Total: 4 + 6*4 + 4 = 32 bytes.
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "adam-update", 4, push_constant_size=32
            )

            # Get cached descriptor set
            descriptor_set = self.pipelines.get_cached_descriptor_set(
                "adam-update",
                [
                    (self._get_buffer_handle(buf_weights), weights_flat.nbytes),
                    (self._get_buffer_handle(buf_grad), grad_flat.nbytes),
                    (self._get_buffer_handle(buf_m1), m1_flat.nbytes),
                    (self._get_buffer_handle(buf_m2), m2_flat.nbytes),
                ],
            )

            # Pack push constants
            # Order in shader: total_weights(uint), lr(float), beta1(float), beta2(float),
            # epsilon(float), beta1_t(float), beta2_t(float), clear_grad(uint)
            push_constants = struct.pack(
                "IffffffI",
                total_weights,
                learning_rate,
                beta1,
                beta2,
                epsilon,
                beta1_t,
                beta2_t,
                1 if clear_grad else 0,
            )

            # Dispatch
            workgroups = (total_weights + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            # Download results
            weights_updated = self._download_buffer(buf_weights, weights_flat.nbytes, np.float32)
            m1_updated = self._download_buffer(buf_m1, m1_flat.nbytes, np.float32)
            m2_updated = self._download_buffer(buf_m2, m2_flat.nbytes, np.float32)

            # Reshape to original shape
            weights_updated = weights_updated[:total_weights].reshape(weights.shape)
            m1_updated = m1_updated[:total_weights].reshape(moment1.shape)
            m2_updated = m2_updated[:total_weights].reshape(moment2.shape)

            return weights_updated, m1_updated, m2_updated
        finally:
            self._release_buffers([buf_weights, buf_grad, buf_m1, buf_m2])

    def adamw_update(
        self,
        weights: np.ndarray,
        gradients: np.ndarray,
        moment1: np.ndarray,
        moment2: np.ndarray,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01,
        beta1_t: float = 0.0,
        beta2_t: float = 0.0,
        clear_grad: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        GPU-accelerated AdamW optimizer update.

        Uses: adamw-update.glsl

        Args:
            weights: Parameter weights (any shape)
            gradients: Parameter gradients (same shape as weights)
            moment1: First moment estimate
            moment2: Second moment estimate
            learning_rate: Learning rate
            beta1: Exponential decay for first moment
            beta2: Exponential decay for second moment
            epsilon: Numerical stability epsilon
            weight_decay: Decoupled weight decay coefficient
            beta1_t: beta1**t for bias correction
            beta2_t: beta2**t for bias correction
            clear_grad: Whether to clear gradients in shader

        Returns:
            (updated_weights, updated_moment1, updated_moment2)
        """
        if "adamw-update" not in self.shaders:
            # CPU fallback
            moment1_new = beta1 * moment1 + (1.0 - beta1) * gradients
            moment2_new = beta2 * moment2 + (1.0 - beta2) * gradients * gradients
            m_hat = moment1_new / (1.0 - beta1_t) if beta1_t < 1.0 else moment1_new
            v_hat = moment2_new / (1.0 - beta2_t) if beta2_t < 1.0 else moment2_new
            decayed = weights * (1.0 - learning_rate * weight_decay)
            weights_new = decayed - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            return weights_new, moment1_new, moment2_new

        weights_flat = np.asarray(weights, dtype=np.float32).reshape(-1)
        grad_flat = np.asarray(gradients, dtype=np.float32).reshape(-1)
        m1_flat = np.asarray(moment1, dtype=np.float32).reshape(-1)
        m2_flat = np.asarray(moment2, dtype=np.float32).reshape(-1)

        total_weights = int(weights_flat.size)
        if not (grad_flat.size == m1_flat.size == m2_flat.size == total_weights):
            raise ValueError("All arrays must have the same size")

        buf_weights = self._acquire_buffer(weights_flat.nbytes)
        buf_grad = self._acquire_buffer(grad_flat.nbytes)
        buf_m1 = self._acquire_buffer(m1_flat.nbytes)
        buf_m2 = self._acquire_buffer(m2_flat.nbytes)

        try:
            self._upload_buffer(buf_weights, weights_flat)
            self._upload_buffer(buf_grad, grad_flat)
            self._upload_buffer(buf_m1, m1_flat)
            self._upload_buffer(buf_m2, m2_flat)

            # Push constants: uint + 7 floats + uint = 36 bytes.
            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "adamw-update", 4, push_constant_size=36
            )
            descriptor_set = self.pipelines.get_cached_descriptor_set(
                "adamw-update",
                [
                    (self._get_buffer_handle(buf_weights), weights_flat.nbytes),
                    (self._get_buffer_handle(buf_grad), grad_flat.nbytes),
                    (self._get_buffer_handle(buf_m1), m1_flat.nbytes),
                    (self._get_buffer_handle(buf_m2), m2_flat.nbytes),
                ],
            )
            push_constants = struct.pack(
                "IfffffffI",
                total_weights,
                float(learning_rate),
                float(beta1),
                float(beta2),
                float(epsilon),
                float(weight_decay),
                float(beta1_t),
                float(beta2_t),
                1 if clear_grad else 0,
            )
            workgroups = (total_weights + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            weights_updated = self._download_buffer(buf_weights, weights_flat.nbytes, np.float32)
            m1_updated = self._download_buffer(buf_m1, m1_flat.nbytes, np.float32)
            m2_updated = self._download_buffer(buf_m2, m2_flat.nbytes, np.float32)

            return (
                weights_updated[:total_weights].reshape(weights.shape),
                m1_updated[:total_weights].reshape(moment1.shape),
                m2_updated[:total_weights].reshape(moment2.shape),
            )
        finally:
            self._release_buffers([buf_weights, buf_grad, buf_m1, buf_m2])
