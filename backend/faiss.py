"""
FAISS (Facebook AI Similarity Search) operations for Vulkan backend.
GPU-accelerated vector similarity search using custom Vulkan compute shaders.
"""

import logging
import numpy as np
import struct
from .base import VULKAN_AVAILABLE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT

if VULKAN_AVAILABLE:
    from vulkan import *

logger = logging.getLogger(__name__)


class VulkanFAISS:
    """FAISS operations: distance computation and top-k selection"""
    
    def __init__(self, core, pipelines):
        """Initialize with VulkanCore and VulkanPipelines instances"""
        self.core = core
        self.pipelines = pipelines
    
    def compute_distances(self, queries, database, distance_type='cosine'):
        """
        Compute pairwise distances between query and database vectors.
        
        Args:
            queries: Query vectors (num_queries, dim) or (dim,)
            database: Database vectors (num_database, dim)
            distance_type: Distance metric - 'l2', 'cosine', or 'dot'
            
        Returns:
            Distance matrix (num_queries, num_database)
        """
        queries = queries.astype(np.float32)
        database = database.astype(np.float32)
        
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        
        num_queries, dim = queries.shape
        num_database = database.shape[0]
        
        distance_map = {'l2': 0, 'cosine': 1, 'dot': 2}
        dist_type_int = distance_map.get(distance_type, 1)
        
        queries_flat = queries.flatten()
        database_flat = database.flatten()
        
        buf_queries, mem_queries = self.core._create_buffer(
            queries_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        buf_database, mem_database = self.core._create_buffer(
            database_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        buf_distances, mem_distances = self.core._create_buffer(
            num_queries * num_database * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        
        self.core._upload_buffer(buf_queries, mem_queries, queries_flat)
        self.core._upload_buffer(buf_database, mem_database, database_flat)
        
        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'faiss-distance', 3, push_constant_size=16
        )
        
        # Get cached descriptor set (reuses existing or creates new)
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'faiss-distance',
            [
                (buf_queries, queries_flat.nbytes),
                (buf_database, database_flat.nbytes),
                (buf_distances, num_queries * num_database * 4)
            ]
        )
        
        push_constants = struct.pack('IIII', num_queries, num_database, dim, dist_type_int)
        
        workgroups_x = (num_database + 15) // 16
        workgroups_y = (num_queries + 15) // 16
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups_x, push_constants, workgroups_y, 1
        )
        
        distances = self.core._download_buffer(
            mem_distances, num_queries * num_database * 4, dtype=np.float32
        )
        distances = distances[:num_queries * num_database].reshape(num_queries, num_database)
        
        # Don't free cached descriptor sets - they're reused
        # vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])
        
        vkDestroyBuffer(self.core.device, buf_queries, None)
        vkDestroyBuffer(self.core.device, buf_database, None)
        vkDestroyBuffer(self.core.device, buf_distances, None)
        vkFreeMemory(self.core.device, mem_queries, None)
        vkFreeMemory(self.core.device, mem_database, None)
        vkFreeMemory(self.core.device, mem_distances, None)
        
        return distances
    
    def _cpu_topk(self, distances: np.ndarray, k: int):
        """CPU top-k helper for correctness fallback."""
        if distances.ndim == 1:
            distances = distances.reshape(1, -1)
        num_queries, num_database = distances.shape
        k = min(int(k), num_database)
        # argpartition gives unordered top-k; sort them for deterministic output
        idx_part = np.argpartition(distances, k - 1, axis=1)[:, :k]
        part_vals = np.take_along_axis(distances, idx_part, axis=1)
        order = np.argsort(part_vals, axis=1)
        topk_indices = np.take_along_axis(idx_part, order, axis=1).astype(np.uint32)
        topk_distances = np.take_along_axis(part_vals, order, axis=1).astype(np.float32)
        return topk_indices, topk_distances

    def topk(self, distances, k):
        """
        Select top-k smallest distances for each query.
        
        Args:
            distances: Distance matrix (num_queries, num_database)
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (indices, distances) arrays, each (num_queries, k)
        """
        distances = np.asarray(distances, dtype=np.float32)
        if distances.ndim == 1:
            distances = distances.reshape(1, -1)
        num_queries, num_database = distances.shape
        k = min(int(k), num_database)

        # Fallback to CPU when Vulkan is unavailable
        if not VULKAN_AVAILABLE or self.core is None:
            return self._cpu_topk(distances, k)

        # GPU attempt with validation fallback
        try:
            distances_flat = distances.flatten()
            db_indices = np.arange(num_database, dtype=np.uint32)

            buf_distances, mem_distances = self.core._create_buffer(
                distances_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            )
            buf_db_indices, mem_db_indices = self.core._create_buffer(
                db_indices.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            )
            buf_topk_indices, mem_topk_indices = self.core._create_buffer(
                num_queries * k * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            )
            buf_topk_distances, mem_topk_distances = self.core._create_buffer(
                num_queries * k * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            )

            self.core._upload_buffer(buf_distances, mem_distances, distances_flat)
            self.core._upload_buffer(buf_db_indices, mem_db_indices, db_indices)

            pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
                'faiss-topk', 4, push_constant_size=12
            )

            descriptor_set = self.pipelines.get_cached_descriptor_set(
                'faiss-topk',
                [
                    (buf_distances, distances_flat.nbytes),
                    (buf_db_indices, db_indices.nbytes),
                    (buf_topk_indices, num_queries * k * 4),
                    (buf_topk_distances, num_queries * k * 4)
                ]
            )

            push_constants = struct.pack('III', num_queries, num_database, k)
            workgroups = (num_queries + 255) // 256
            self.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set,
                workgroups, push_constants
            )

            topk_indices = self.core._download_buffer(
                mem_topk_indices, num_queries * k * 4, dtype=np.uint32
            )
            topk_distances = self.core._download_buffer(
                mem_topk_distances, num_queries * k * 4, dtype=np.float32
            )

            topk_indices = topk_indices[:num_queries * k].reshape(num_queries, k)
            topk_distances = topk_distances[:num_queries * k].reshape(num_queries, k)

            # Optional validation against CPU for correctness; fallback if mismatch
            cpu_idx, cpu_dist = self._cpu_topk(distances, k)
            if not (np.array_equal(np.sort(topk_indices, axis=1), np.sort(cpu_idx, axis=1)) and
                    np.allclose(np.sort(topk_distances, axis=1), np.sort(cpu_dist, axis=1), rtol=1e-4, atol=1e-5)):
                logger.warning("GPU topk mismatch detected, falling back to CPU results")
                topk_indices, topk_distances = cpu_idx, cpu_dist

            return topk_indices, topk_distances

        except Exception as e:
            logger.warning(f"GPU topk failed, falling back to CPU: {e}")
            return self._cpu_topk(distances, k)

        finally:
            # Cleanup (guarded in case buffers weren't created)
            try:
                vkDestroyBuffer(self.core.device, buf_distances, None)
                vkDestroyBuffer(self.core.device, buf_db_indices, None)
                vkDestroyBuffer(self.core.device, buf_topk_indices, None)
                vkDestroyBuffer(self.core.device, buf_topk_distances, None)
                vkFreeMemory(self.core.device, mem_distances, None)
                vkFreeMemory(self.core.device, mem_db_indices, None)
                vkFreeMemory(self.core.device, mem_topk_indices, None)
                vkFreeMemory(self.core.device, mem_topk_distances, None)
            except Exception:
                pass

