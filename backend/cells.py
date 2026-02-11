"""
Place and Time cell operations for Vulkan backend.
GPU-accelerated spatial and temporal encoding for hippocampal-inspired memory.
"""

import struct

import numpy as np

from .base import VULKAN_AVAILABLE, BufferMixin

if VULKAN_AVAILABLE:
    from vulkan import *


class VulkanCells(BufferMixin):
    """Place and Time cell operations for spatial/temporal encoding"""

    def __init__(self, core, pipelines, shaders):
        """Initialize the instance."""
        self.core = core
        self.pipelines = pipelines
        self.shaders = shaders

    def place_cell(
        self,
        agent_position,
        field_centers,
        field_width=1.0,
        max_rate=20.0,
        baseline_rate=0.1,
        spatial_dims=2,
    ):
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

        buf_pos = self._acquire_buffer(pos.nbytes)
        buf_centers = self._acquire_buffer(centers_flat.nbytes)
        buf_rates = self._acquire_buffer(n_neurons * 4)

        try:
            self._upload_buffer(buf_pos, pos)
            self._upload_buffer(buf_centers, centers_flat)

            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "place-cell", 3, push_constant_size=20
            )

            descriptor_set = self.pipelines.get_cached_descriptor_set(
                "place-cell",
                [
                    (self._get_buffer_handle(buf_pos), pos.nbytes),
                    (self._get_buffer_handle(buf_centers), centers_flat.nbytes),
                    (self._get_buffer_handle(buf_rates), n_neurons * 4),
                ],
            )

            push_constants = struct.pack(
                "IIfff", n_neurons, spatial_dims, field_width, max_rate, baseline_rate
            )

            workgroups = (n_neurons + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            result = self._download_buffer(buf_rates, n_neurons * 4, np.float32)
            return result[:n_neurons]
        finally:
            self._release_buffers([buf_pos, buf_centers, buf_rates])

    def time_cell(
        self, current_time, preferred_times, time_constant=1.0, max_rate=20.0, baseline_rate=0.1
    ):
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

        buf_time = self._acquire_buffer(time_arr.nbytes)
        buf_prefs = self._acquire_buffer(prefs.nbytes)
        buf_rates = self._acquire_buffer(n_neurons * 4)
        buf_mem = self._acquire_buffer(n_neurons * 4)

        try:
            self._upload_buffer(buf_mem, np.zeros(n_neurons, dtype=np.float32))
            self._upload_buffer(buf_time, time_arr)
            self._upload_buffer(buf_prefs, prefs)

            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                "time-cell", 4, push_constant_size=20
            )

            descriptor_set = self.pipelines.get_cached_descriptor_set(
                "time-cell",
                [
                    (self._get_buffer_handle(buf_time), time_arr.nbytes),
                    (self._get_buffer_handle(buf_prefs), prefs.nbytes),
                    (self._get_buffer_handle(buf_rates), n_neurons * 4),
                    (self._get_buffer_handle(buf_mem), n_neurons * 4),
                ],
            )

            push_constants = struct.pack(
                "Iffff", n_neurons, time_constant, max_rate, baseline_rate, 0.0
            )

            workgroups = (n_neurons + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set, workgroups, push_constants
            )

            result = self._download_buffer(buf_rates, n_neurons * 4, np.float32)
            return result[:n_neurons]
        finally:
            self._release_buffers([buf_time, buf_prefs, buf_rates, buf_mem])
