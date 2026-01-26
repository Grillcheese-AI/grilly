"""
Place and Time cell operations for Vulkan backend.
GPU-accelerated spatial and temporal encoding for hippocampal-inspired memory.
"""

import numpy as np
import struct
from .base import VULKAN_AVAILABLE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT

if VULKAN_AVAILABLE:
    from vulkan import *


class VulkanCells:
    """Place and Time cell operations for spatial/temporal encoding"""
    
    def __init__(self, core, pipelines, shaders):
        self.core = core
        self.pipelines = pipelines
        self.shaders = shaders
    
    def place_cell(self, agent_position, field_centers, field_width=1.0, max_rate=20.0, baseline_rate=0.1, spatial_dims=2):
        """
        Generate place cell firing rates based on agent position
        
        Args:
            agent_position: Current position (spatial_dims,)
            field_centers: Place field centers (n_neurons, spatial_dims)
            field_width: Place field width (default: 1.0)
            max_rate: Maximum firing rate in Hz (default: 20.0)
            baseline_rate: Baseline firing rate in Hz (default: 0.1)
            spatial_dims: Number of spatial dimensions (default: 2)
        
        Returns:
            Firing rates (n_neurons,)
        """
        pos = agent_position.astype(np.float32).flatten()[:spatial_dims]
        centers = field_centers.astype(np.float32)
        n_neurons = centers.shape[0]
        
        centers_flat = centers.flatten()
        
        # Create buffers
        buf_pos, mem_pos = self.core._create_buffer(pos.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_centers, mem_centers = self.core._create_buffer(centers_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_rates, mem_rates = self.core._create_buffer(n_neurons * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(buf_pos, mem_pos, pos)
        self.core._upload_buffer(buf_centers, mem_centers, centers_flat)
        
        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'place-cell', 3, push_constant_size=20
        )
        
        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'place-cell',
            [
                (buf_pos, pos.nbytes),
                (buf_centers, centers_flat.nbytes),
                (buf_rates, n_neurons * 4)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack('IIfff', n_neurons, spatial_dims, field_width, max_rate, baseline_rate)
        
        # Dispatch
        workgroups = (n_neurons + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )
        
        # Download results
        result = self.core._download_buffer(mem_rates, n_neurons * 4, dtype=np.float32)
        result = result[:n_neurons]
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_pos, None)
        vkDestroyBuffer(self.core.device, buf_centers, None)
        vkDestroyBuffer(self.core.device, buf_rates, None)
        vkFreeMemory(self.core.device, mem_pos, None)
        vkFreeMemory(self.core.device, mem_centers, None)
        vkFreeMemory(self.core.device, mem_rates, None)
        
        return result
    
    def time_cell(self, current_time, preferred_times, time_constant=1.0, max_rate=20.0, baseline_rate=0.1):
        """
        Generate time cell firing rates based on elapsed time
        
        Args:
            current_time: Current normalized time (0-1)
            preferred_times: Preferred firing times for each cell (n_neurons,)
            time_constant: Time field width (default: 1.0)
            max_rate: Maximum firing rate in Hz (default: 20.0)
            baseline_rate: Baseline firing rate in Hz (default: 0.1)
        
        Returns:
            Firing rates (n_neurons,)
        """
        time_arr = np.array([current_time], dtype=np.float32)
        prefs = preferred_times.astype(np.float32).flatten()
        n_neurons = len(prefs)
        
        # Create buffers
        buf_time, mem_time = self.core._create_buffer(time_arr.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_prefs, mem_prefs = self.core._create_buffer(prefs.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_rates, mem_rates = self.core._create_buffer(n_neurons * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_mem, mem_mem = self.core._create_buffer(n_neurons * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Initialize memory buffer
        self.core._upload_buffer(buf_mem, mem_mem, np.zeros(n_neurons, dtype=np.float32))
        
        # Upload data
        self.core._upload_buffer(buf_time, mem_time, time_arr)
        self.core._upload_buffer(buf_prefs, mem_prefs, prefs)
        
        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'time-cell', 4, push_constant_size=20
        )
        
        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'time-cell',
            [
                (buf_time, time_arr.nbytes),
                (buf_prefs, prefs.nbytes),
                (buf_rates, n_neurons * 4),
                (buf_mem, n_neurons * 4)
            ]
        )
        
        # Pack push constants (padding to match expected size)
        push_constants = struct.pack('Iffff', n_neurons, time_constant, max_rate, baseline_rate, 0.0)
        
        # Dispatch
        workgroups = (n_neurons + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )
        
        # Download results
        result = self.core._download_buffer(mem_rates, n_neurons * 4, dtype=np.float32)
        result = result[:n_neurons]
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_time, None)
        vkDestroyBuffer(self.core.device, buf_prefs, None)
        vkDestroyBuffer(self.core.device, buf_rates, None)
        vkDestroyBuffer(self.core.device, buf_mem, None)
        vkFreeMemory(self.core.device, mem_time, None)
        vkFreeMemory(self.core.device, mem_prefs, None)
        vkFreeMemory(self.core.device, mem_rates, None)
        vkFreeMemory(self.core.device, mem_mem, None)
        
        return result

