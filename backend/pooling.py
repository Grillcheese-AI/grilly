"""
GPU-accelerated pooling operations for embeddings.
Supports mean, max, and sum pooling with optional mask support.
"""

import numpy as np
import struct
from .base import VULKAN_AVAILABLE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT

if VULKAN_AVAILABLE:
    from vulkan import *


class VulkanPooling:
    """GPU-accelerated pooling operations"""
    
    def __init__(self, core, pipelines, shaders):
        self.core = core
        self.pipelines = pipelines
        self.shaders = shaders
    
    def mean_pool(self, embeddings: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        GPU-accelerated mean pooling with optional mask.
        
        Args:
            embeddings: Input embeddings (batch, seq_len, dim)
            mask: Optional mask (batch, seq_len) - 1.0 = keep, 0.0 = mask out
        
        Returns:
            Pooled embeddings (batch, dim)
        """
        data = embeddings.astype(np.float32)
        batch_size, seq_len, dim = data.shape
        
        data_flat = data.flatten()
        
        # Create buffers
        buf_in, mem_in = self.core._create_buffer(data_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self.core._create_buffer(batch_size * dim * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(buf_in, mem_in, data_flat)
        
        # Handle mask if provided
        if mask is not None:
            mask_flat = mask.astype(np.float32).flatten()
            buf_mask, mem_mask = self.core._create_buffer(mask_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
            self.core._upload_buffer(buf_mask, mem_mask, mask_flat)
            
            # Use embedding-pool shader with mask (it already supports masks)
            if 'embedding-pool' in self.shaders:
                pipeline, layout, desc_layout = self.pipelines.get_or_create_pipeline(
                    'embedding-pool', 3, push_constant_size=16
                )
                descriptor_set = self.pipelines.get_cached_descriptor_set(
                    'embedding-pool',
                    [
                        (buf_in, data_flat.nbytes),
                        (buf_mask, mask_flat.nbytes),
                        (buf_out, batch_size * dim * 4)
                    ]
                )
                push_constants = struct.pack('IIII', batch_size, seq_len, dim, 0)  # 0 = mean
                
                # Dispatch
                workgroups = (batch_size * dim + 255) // 256
                self.core._dispatch_compute(pipeline, layout, descriptor_set, workgroups, push_constants)
                
                # Download results
                result = self.core._download_buffer(mem_out, batch_size * dim * 4, dtype=np.float32)
                result = result[:batch_size * dim].reshape(batch_size, dim)
                
                # Cleanup
                vkDestroyBuffer(self.core.device, buf_in, None)
                vkDestroyBuffer(self.core.device, buf_mask, None)
                vkDestroyBuffer(self.core.device, buf_out, None)
                vkFreeMemory(self.core.device, mem_in, None)
                vkFreeMemory(self.core.device, mem_mask, None)
                vkFreeMemory(self.core.device, mem_out, None)
                
                return result
            else:
                # CPU fallback with mask (optimized)
                vkDestroyBuffer(self.core.device, buf_in, None)
                vkDestroyBuffer(self.core.device, buf_mask, None)
                vkDestroyBuffer(self.core.device, buf_out, None)
                vkFreeMemory(self.core.device, mem_in, None)
                vkFreeMemory(self.core.device, mem_mask, None)
                vkFreeMemory(self.core.device, mem_out, None)
                
                # Optimized CPU pooling with mask
                mask_expanded = mask[:, :, None]  # (batch, seq_len, 1)
                x_masked = data * mask_expanded
                mask_sum = mask_expanded.sum(axis=1)  # (batch, 1)
                return (x_masked.sum(axis=1) / (mask_sum + 1e-8)).astype(np.float32)
        else:
            # No mask - use standard embedding-pool shader with all-ones mask
            if 'embedding-pool' in self.shaders:
                # Create all-ones mask for no masking
                ones_mask = np.ones((batch_size, seq_len), dtype=np.float32)
                mask_flat = ones_mask.flatten()
                buf_mask, mem_mask = self.core._create_buffer(mask_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
                self.core._upload_buffer(buf_mask, mem_mask, mask_flat)
                
                pipeline, layout, desc_layout = self.pipelines.get_or_create_pipeline(
                    'embedding-pool', 3, push_constant_size=16
                )
                descriptor_set = self.pipelines.get_cached_descriptor_set(
                    'embedding-pool',
                    [
                        (buf_in, data_flat.nbytes),
                        (buf_mask, mask_flat.nbytes),
                        (buf_out, batch_size * dim * 4)
                    ]
                )
                push_constants = struct.pack('IIII', batch_size, seq_len, dim, 0)  # 0 = mean
                
                # Dispatch
                workgroups = (batch_size * dim + 255) // 256
                self.core._dispatch_compute(pipeline, layout, descriptor_set, workgroups, push_constants)
                
                # Download results
                result = self.core._download_buffer(mem_out, batch_size * dim * 4, dtype=np.float32)
                result = result[:batch_size * dim].reshape(batch_size, dim)
                
                # Cleanup
                vkDestroyBuffer(self.core.device, buf_in, None)
                vkDestroyBuffer(self.core.device, buf_mask, None)
                vkDestroyBuffer(self.core.device, buf_out, None)
                vkFreeMemory(self.core.device, mem_in, None)
                vkFreeMemory(self.core.device, mem_mask, None)
                vkFreeMemory(self.core.device, mem_out, None)
                
                return result
            else:
                # CPU fallback
                vkDestroyBuffer(self.core.device, buf_in, None)
                vkDestroyBuffer(self.core.device, buf_out, None)
                vkFreeMemory(self.core.device, mem_in, None)
                vkFreeMemory(self.core.device, mem_out, None)
                return data.mean(axis=1).astype(np.float32)

    def maxpool2d(self, x: np.ndarray, kernel_size, stride=None, padding=0, dilation=1):
        """
        2D max pooling forward pass.

        Args:
            x: Input (batch, channels, height, width)
            kernel_size: Kernel size (int or tuple)
            stride: Stride (defaults to kernel_size)
            padding: Padding (int or tuple)
            dilation: Dilation (int or tuple)

        Returns:
            output: Pooled output
            indices: Max indices for backward pass
        """
        x = np.asarray(x, dtype=np.float32)
        batch_size, channels, in_h, in_w = x.shape

        # Parse parameters
        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        sh, sw = (stride or kernel_size, stride or kernel_size) if isinstance(stride or kernel_size, int) else (stride or kernel_size)
        ph, pw = (padding, padding) if isinstance(padding, int) else padding
        dh, dw = (dilation, dilation) if isinstance(dilation, int) else dilation

        # Calculate output dimensions
        out_h = (in_h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        out_w = (in_w + 2 * pw - dw * (kw - 1) - 1) // sw + 1

        # Buffer sizes (in bytes)
        input_size = batch_size * channels * in_h * in_w * 4
        output_size = batch_size * channels * out_h * out_w * 4
        indices_size = batch_size * channels * out_h * out_w * 4  # uint

        # Create buffers
        buf_in, mem_in = self.core._create_buffer(input_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self.core._create_buffer(output_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_idx, mem_idx = self.core._create_buffer(indices_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)

        # Upload input
        self.core._upload_buffer(buf_in, mem_in, x.flatten())

        # Push constants (14 uints)
        push_data = struct.pack('IIIIIIIIIIIIII', batch_size, channels, in_h, in_w, out_h, out_w,
                                kh, kw, sh, sw, ph, pw, dh, dw)

        # Pipeline (14 uints = 56 bytes)
        pipeline, layout, _ = self.pipelines.get_or_create_pipeline('maxpool2d-forward', 3, push_constant_size=56)
        desc = self.pipelines.get_cached_descriptor_set('maxpool2d-forward',
                                                        [(buf_in, input_size), (buf_out, output_size), (buf_idx, indices_size)])

        # Dispatch
        gx = (out_w + 7) // 8
        gy = (out_h + 7) // 8
        gz = batch_size * channels
        self.core._dispatch_compute(pipeline, layout, desc, gx, push_data, gy, gz)

        # Download results
        output = self.core._download_buffer(mem_out, output_size, np.float32).reshape(batch_size, channels, out_h, out_w)
        indices = self.core._download_buffer(mem_idx, indices_size, np.uint32).reshape(batch_size, channels, out_h, out_w)

        # Cleanup
        for buf in [buf_in, buf_out, buf_idx]:
            vkDestroyBuffer(self.core.device, buf, None)
        for mem in [mem_in, mem_out, mem_idx]:
            vkFreeMemory(self.core.device, mem, None)

        return output, indices

    def maxpool2d_backward(self, grad_output: np.ndarray, indices: np.ndarray, input_shape):
        """
        2D max pooling backward pass.

        Args:
            grad_output: Gradient w.r.t output
            indices: Max indices from forward pass
            input_shape: Shape of original input

        Returns:
            Gradient w.r.t input
        """
        grad_output = np.asarray(grad_output, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.uint32)

        batch_size, channels, in_h, in_w = input_shape
        _, _, out_h, out_w = grad_output.shape
        output_size = grad_output.size
        input_size = batch_size * channels * in_h * in_w

        # Create buffers
        buf_grad_out, mem_grad_out = self.core._create_buffer(output_size * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_idx, mem_idx = self.core._create_buffer(output_size * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_grad_in, mem_grad_in = self.core._create_buffer(input_size * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)

        # Upload
        self.core._upload_buffer(buf_grad_out, mem_grad_out, grad_output.flatten())
        self.core._upload_buffer(buf_idx, mem_idx, indices.flatten())

        # Push constants (6 uints: batch, channels, in_h, in_w, out_h, out_w)
        push_data = struct.pack('IIIIII', batch_size, channels, in_h, in_w, out_h, out_w)

        # Pipeline
        pipeline, layout, _ = self.pipelines.get_or_create_pipeline('maxpool2d-backward', 3, push_constant_size=24)
        desc = self.pipelines.get_cached_descriptor_set('maxpool2d-backward',
                                                        [(buf_grad_out, output_size * 4), (buf_idx, output_size * 4), (buf_grad_in, input_size * 4)])

        # Dispatch over INPUT dimensions
        gx = (in_w + 7) // 8
        gy = (in_h + 7) // 8
        gz = batch_size * channels
        self.core._dispatch_compute(pipeline, layout, desc, gx, push_data, gy, gz)

        # Download
        grad_input = self.core._download_buffer(mem_grad_in, input_size * 4, np.float32).reshape(input_shape)

        # Cleanup
        for buf in [buf_grad_out, buf_idx, buf_grad_in]:
            vkDestroyBuffer(self.core.device, buf, None)
        for mem in [mem_grad_out, mem_idx, mem_grad_in]:
            vkFreeMemory(self.core.device, mem, None)

        return grad_input

    def avgpool2d(self, x: np.ndarray, kernel_size, stride=None, padding=0, count_include_pad=True):
        """
        2D average pooling forward pass.

        Args:
            x: Input (batch, channels, height, width)
            kernel_size: Kernel size (int or tuple)
            stride: Stride (defaults to kernel_size)
            padding: Padding (int or tuple)
            count_include_pad: Include padding in average

        Returns:
            Pooled output
        """
        x = np.asarray(x, dtype=np.float32)
        batch_size, channels, in_h, in_w = x.shape

        # Parse parameters
        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        sh, sw = (stride or kernel_size, stride or kernel_size) if isinstance(stride or kernel_size, int) else (stride or kernel_size)
        ph, pw = (padding, padding) if isinstance(padding, int) else padding

        # Calculate output dimensions
        out_h = (in_h + 2 * ph - kh) // sh + 1
        out_w = (in_w + 2 * pw - kw) // sw + 1

        # Buffer sizes
        input_size = batch_size * channels * in_h * in_w * 4
        output_size = batch_size * channels * out_h * out_w * 4

        # Create buffers
        buf_in, mem_in = self.core._create_buffer(input_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self.core._create_buffer(output_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)

        # Upload
        self.core._upload_buffer(buf_in, mem_in, x.flatten())

        # Push constants
        push_data = struct.pack('IIIIIIIIIIIII', batch_size, channels, in_h, in_w, out_h, out_w,
                                kh, kw, sh, sw, ph, pw, 1 if count_include_pad else 0)

        # Pipeline
        pipeline, layout, _ = self.pipelines.get_or_create_pipeline('avgpool2d-forward', 2, push_constant_size=52)
        desc = self.pipelines.get_cached_descriptor_set('avgpool2d-forward',
                                                        [(buf_in, input_size), (buf_out, output_size)])

        # Dispatch
        gx = (out_w + 7) // 8
        gy = (out_h + 7) // 8
        gz = batch_size * channels
        self.core._dispatch_compute(pipeline, layout, desc, gx, push_data, gy, gz)

        # Download
        output = self.core._download_buffer(mem_out, output_size, np.float32).reshape(batch_size, channels, out_h, out_w)

        # Cleanup
        for buf in [buf_in, buf_out]:
            vkDestroyBuffer(self.core.device, buf, None)
        for mem in [mem_in, mem_out]:
            vkFreeMemory(self.core.device, mem, None)

        return output

    def avgpool2d_backward(self, grad_output: np.ndarray, input_shape, kernel_size, stride=None, padding=0, count_include_pad=True):
        """
        2D average pooling backward pass.

        Args:
            grad_output: Gradient w.r.t output
            input_shape: Shape of original input
            kernel_size: Kernel size
            stride: Stride
            padding: Padding
            count_include_pad: Include padding in average

        Returns:
            Gradient w.r.t input
        """
        grad_output = np.asarray(grad_output, dtype=np.float32)
        batch_size, channels, in_h, in_w = input_shape
        _, _, out_h, out_w = grad_output.shape

        # Parse parameters
        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        sh, sw = (stride or kernel_size, stride or kernel_size) if isinstance(stride or kernel_size, int) else (stride or kernel_size)
        ph, pw = (padding, padding) if isinstance(padding, int) else padding

        # Buffer sizes
        output_size = batch_size * channels * out_h * out_w * 4
        input_size = batch_size * channels * in_h * in_w * 4

        # Create buffers
        buf_grad_out, mem_grad_out = self.core._create_buffer(output_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_grad_in, mem_grad_in = self.core._create_buffer(input_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)

        # Upload
        self.core._upload_buffer(buf_grad_out, mem_grad_out, grad_output.flatten())

        # Zero grad_input
        zeros = np.zeros(batch_size * channels * in_h * in_w, dtype=np.float32)
        self.core._upload_buffer(buf_grad_in, mem_grad_in, zeros)

        # Push constants
        push_data = struct.pack('IIIIIIIIIIIII', batch_size, channels, in_h, in_w, out_h, out_w,
                                kh, kw, sh, sw, ph, pw, 1 if count_include_pad else 0)

        # Debug
        import sys
        if hasattr(sys, '_called_from_test'):
            print(f"[AvgPool Backward] batch={batch_size}, ch={channels}, in={in_h}x{in_w}, out={out_h}x{out_w}, kernel={kh}x{kw}, stride={sh}x{sw}, pad={ph}x{pw}")
            gx_calc = (out_w + 7) // 8
            gy_calc = (out_h + 7) // 8
            gz_calc = batch_size * channels
            print(f"[AvgPool Backward] Dispatch: gx={gx_calc}, gy={gy_calc}, gz={gz_calc}")

        # Pipeline
        pipeline, layout, _ = self.pipelines.get_or_create_pipeline('avgpool2d-backward', 2, push_constant_size=52)
        desc = self.pipelines.get_cached_descriptor_set('avgpool2d-backward',
                                                        [(buf_grad_out, output_size), (buf_grad_in, input_size)])

        # Dispatch over INPUT dimensions (not output) to avoid race conditions
        gx = (in_w + 7) // 8
        gy = (in_h + 7) // 8
        gz = batch_size * channels
        self.core._dispatch_compute(pipeline, layout, desc, gx, push_data, gy, gz)

        # Download
        grad_input = self.core._download_buffer(mem_grad_in, input_size, np.float32).reshape(input_shape)

        # Cleanup
        for buf in [buf_grad_out, buf_grad_in]:
            vkDestroyBuffer(self.core.device, buf, None)
        for mem in [mem_grad_out, mem_grad_in]:
            vkFreeMemory(self.core.device, mem, None)

        return grad_input
