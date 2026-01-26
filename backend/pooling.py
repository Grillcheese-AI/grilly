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
