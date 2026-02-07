"""
VulkanVSA - GPU-accelerated Vector Symbolic Architecture operations.

Provides GPU implementations of VSA operations for high-performance
hyperdimensional computing.
"""

import numpy as np
from typing import List, Optional, Tuple
from ..core import VulkanCore
from ..pipelines import VulkanPipelines


class VulkanVSA:
    """
    GPU-accelerated VSA operations using Vulkan compute shaders.
    
    Provides:
    - bind_bipolar: Element-wise multiplication (O(d))
    - bind_bipolar_batch: Batch binding (O(B*d))
    - bundle: Superposition with majority voting (O(d))
    - bundle_batch: Batched superposition with majority voting (O(B*d))
    - similarity_batch: Parallel cosine similarity (O(V*d))
    - resonator_step: Codebook projection step (O(V*d))
    - circular_convolve: FFT-based HRR binding (O(d log d))
    """
    
    def __init__(self, core: VulkanCore):
        """
        Initialize VulkanVSA.
        
        Args:
            core: VulkanCore instance
        """
        self.core = core
        self.pipelines = VulkanPipelines(core)
        
        # Experimental shaders are now automatically loaded by VulkanCore
        # from shaders/experimental/spv directory. No need to load separately.
        # Create pipelines
        self._init_pipelines()
    
    def _init_pipelines(self):
        """Initialize compute pipelines."""
        # Pipelines are created on-demand via get_or_create_pipeline
        pass
    
    def bind_bipolar(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated bipolar binding (element-wise multiplication).
        
        Args:
            a: First bipolar vector
            b: Second bipolar vector
            
        Returns:
            Bound vector
        """
        from vulkan import VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        
        dim = len(a)
        assert len(b) == dim, "Vectors must have same dimension"
        
        # Check if shader available, otherwise fallback to CPU
        if 'vsa-bind' not in self.core.shaders:
            from grilly.experimental.vsa.ops import BinaryOps
            return BinaryOps.bind(a, b)
        
        # Create buffers
        a_buf, a_mem = self.core._create_buffer(dim * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        b_buf, b_mem = self.core._create_buffer(dim * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        result_buf, result_mem = self.core._create_buffer(dim * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(a_buf, a_mem, a.astype(np.float32))
        self.core._upload_buffer(b_buf, b_mem, b.astype(np.float32))
        
        # Get or create pipeline
        pipeline, layout, descriptor = self.pipelines.get_or_create_pipeline(
            'vsa-bind', num_buffers=3, push_constant_size=4
        )
        
        # Create descriptor set
        desc_set = self.pipelines._create_descriptor_set(
            descriptor,
            [(a_buf, dim * 4), (b_buf, dim * 4), (result_buf, dim * 4)]
        )
        
        # Push constants
        import struct
        push_consts = struct.pack('I', dim)
        
        # Dispatch
        workgroups = (dim + 255) // 256
        self.core._dispatch_compute(pipeline, layout, desc_set, workgroups, push_consts)
        
        # Download result
        result = self.core._download_buffer(result_mem, dim * 4, dtype=np.float32)
        
        # Cleanup
        from vulkan import vkDestroyBuffer, vkFreeMemory, vkFreeDescriptorSets
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [desc_set])
        vkDestroyBuffer(self.core.device, a_buf, None)
        vkDestroyBuffer(self.core.device, b_buf, None)
        vkDestroyBuffer(self.core.device, result_buf, None)
        vkFreeMemory(self.core.device, a_mem, None)
        vkFreeMemory(self.core.device, b_mem, None)
        vkFreeMemory(self.core.device, result_mem, None)
        
        return result

    def bind_bipolar_batch(self, a_batch: np.ndarray, b_batch: np.ndarray) -> np.ndarray:
        """
        Batch bipolar binding using a single GPU dispatch when available.

        Args:
            a_batch: Batch of bipolar vectors (batch, dim)
            b_batch: Batch of bipolar vectors (batch, dim)

        Returns:
            Bound vectors (batch, dim)
        """
        from vulkan import VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        import struct

        assert a_batch.shape == b_batch.shape, "Batch inputs must match shape"
        assert a_batch.ndim == 2, "Batch inputs must be 2D (batch, dim)"

        batch_size, dim = a_batch.shape

        # Check if shader available, otherwise fallback to loop
        if 'vsa-bind-batch' not in self.core.shaders:
            results = np.empty((batch_size, dim), dtype=np.float32)
            for i in range(batch_size):
                results[i] = self.bind_bipolar(a_batch[i], b_batch[i])
            return results

        total = batch_size * dim
        a_flat = a_batch.astype(np.float32).reshape(-1)
        b_flat = b_batch.astype(np.float32).reshape(-1)

        a_buf, a_mem = self.core._create_buffer(total * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        b_buf, b_mem = self.core._create_buffer(total * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        result_buf, result_mem = self.core._create_buffer(total * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)

        self.core._upload_buffer(a_buf, a_mem, a_flat)
        self.core._upload_buffer(b_buf, b_mem, b_flat)

        pipeline, layout, descriptor = self.pipelines.get_or_create_pipeline(
            'vsa-bind-batch', num_buffers=3, push_constant_size=4
        )

        desc_set = self.pipelines._create_descriptor_set(
            descriptor,
            [(a_buf, total * 4), (b_buf, total * 4), (result_buf, total * 4)]
        )

        push_consts = struct.pack('I', total)

        workgroups = (total + 255) // 256
        self.core._dispatch_compute(pipeline, layout, desc_set, workgroups, push_consts)

        result = self.core._download_buffer(result_mem, total * 4, dtype=np.float32)
        result = result.reshape(batch_size, dim)

        from vulkan import vkDestroyBuffer, vkFreeMemory, vkFreeDescriptorSets
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [desc_set])
        vkDestroyBuffer(self.core.device, a_buf, None)
        vkDestroyBuffer(self.core.device, b_buf, None)
        vkDestroyBuffer(self.core.device, result_buf, None)
        vkFreeMemory(self.core.device, a_mem, None)
        vkFreeMemory(self.core.device, b_mem, None)
        vkFreeMemory(self.core.device, result_mem, None)

        return result
    
    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """
        GPU-accelerated bundling (superposition with majority voting).
        
        Args:
            vectors: List of vectors to bundle
            
        Returns:
            Bundled vector
        """
        from vulkan import VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        import struct
        
        if not vectors:
            raise ValueError("Cannot bundle empty list")
        
        dim = len(vectors[0])
        num_vectors = len(vectors)
        
        # Check if shader available, otherwise fallback to CPU
        if 'vsa-bundle' not in self.core.shaders:
            from grilly.experimental.vsa.ops import BinaryOps
            return BinaryOps.bundle(vectors)
        
        # Flatten vectors
        vectors_flat = np.array(vectors, dtype=np.float32).flatten()
        
        # Create buffers
        vectors_buf, vectors_mem = self.core._create_buffer(num_vectors * dim * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        result_buf, result_mem = self.core._create_buffer(dim * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(vectors_buf, vectors_mem, vectors_flat)
        
        # Get or create pipeline
        pipeline, layout, descriptor = self.pipelines.get_or_create_pipeline(
            'vsa-bundle', num_buffers=2, push_constant_size=8
        )
        
        # Create descriptor set
        desc_set = self.pipelines._create_descriptor_set(
            descriptor,
            [(vectors_buf, num_vectors * dim * 4), (result_buf, dim * 4)]
        )
        
        # Push constants: dim, num_vectors
        push_consts = struct.pack('II', dim, num_vectors)
        
        # Dispatch
        workgroups = (dim + 255) // 256
        self.core._dispatch_compute(pipeline, layout, desc_set, workgroups, push_consts)
        
        # Download result
        result = self.core._download_buffer(result_mem, dim * 4, dtype=np.float32)
        
        # Cleanup
        from vulkan import vkDestroyBuffer, vkFreeMemory, vkFreeDescriptorSets
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [desc_set])
        vkDestroyBuffer(self.core.device, vectors_buf, None)
        vkDestroyBuffer(self.core.device, result_buf, None)
        vkFreeMemory(self.core.device, vectors_mem, None)
        vkFreeMemory(self.core.device, result_mem, None)
        
        return result

    def bundle_batch(self, vectors: np.ndarray) -> np.ndarray:
        """
        Batch bundling using a single GPU dispatch when available.

        Args:
            vectors: Array of shape (batch, num_vectors, dim)

        Returns:
            Bundled vectors of shape (batch, dim)
        """
        from vulkan import VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        import struct

        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim != 3:
            raise ValueError("vectors must have shape (batch, num_vectors, dim)")

        batch_size, num_vectors, dim = vecs.shape

        # Check if shader available, otherwise fallback to CPU
        if 'vsa-bundle-batch' not in self.core.shaders:
            from grilly.experimental.vsa.ops import BinaryOps
            return BinaryOps.bundle_batch(vecs)

        total = batch_size * dim
        vectors_flat = vecs.reshape(-1)

        vectors_buf, vectors_mem = self.core._create_buffer(
            batch_size * num_vectors * dim * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        result_buf, result_mem = self.core._create_buffer(total * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)

        self.core._upload_buffer(vectors_buf, vectors_mem, vectors_flat)

        pipeline, layout, descriptor = self.pipelines.get_or_create_pipeline(
            'vsa-bundle-batch', num_buffers=2, push_constant_size=12
        )

        desc_set = self.pipelines._create_descriptor_set(
            descriptor,
            [(vectors_buf, batch_size * num_vectors * dim * 4), (result_buf, total * 4)]
        )

        push_consts = struct.pack('III', dim, num_vectors, batch_size)

        workgroups = (total + 255) // 256
        self.core._dispatch_compute(pipeline, layout, desc_set, workgroups, push_consts)

        result = self.core._download_buffer(result_mem, total * 4, dtype=np.float32)
        result = result.reshape(batch_size, dim)

        from vulkan import vkDestroyBuffer, vkFreeMemory, vkFreeDescriptorSets
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [desc_set])
        vkDestroyBuffer(self.core.device, vectors_buf, None)
        vkDestroyBuffer(self.core.device, result_buf, None)
        vkFreeMemory(self.core.device, vectors_mem, None)
        vkFreeMemory(self.core.device, result_mem, None)

        return result
    
    def similarity_batch(
        self,
        query: np.ndarray,
        codebook: np.ndarray
    ) -> np.ndarray:
        """
        GPU-accelerated batch similarity computation.
        
        Args:
            query: Query vector of shape (dim,)
            codebook: Codebook vectors of shape (codebook_size, dim)
            
        Returns:
            Similarities of shape (codebook_size,)
        """
        from vulkan import VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        import struct
        
        dim = len(query)
        codebook_size = len(codebook)
        
        assert codebook.shape == (codebook_size, dim), "Invalid codebook shape"
        
        # Check if shader available, otherwise fallback to CPU
        if 'vsa-similarity-batch' not in self.core.shaders:
            from grilly.experimental.vsa.ops import BinaryOps
            return np.array([BinaryOps.similarity(query, vec) for vec in codebook])
        
        # Flatten codebook
        codebook_flat = codebook.flatten()
        
        # Create buffers
        query_buf, query_mem = self.core._create_buffer(dim * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        codebook_buf, codebook_mem = self.core._create_buffer(codebook_size * dim * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        similarities_buf, similarities_mem = self.core._create_buffer(codebook_size * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(query_buf, query_mem, query.astype(np.float32))
        self.core._upload_buffer(codebook_buf, codebook_mem, codebook_flat.astype(np.float32))
        
        # Get or create pipeline
        pipeline, layout, descriptor = self.pipelines.get_or_create_pipeline(
            'vsa-similarity-batch', num_buffers=3, push_constant_size=8
        )
        
        # Create descriptor set
        desc_set = self.pipelines._create_descriptor_set(
            descriptor,
            [
                (query_buf, dim * 4),
                (codebook_buf, codebook_size * dim * 4),
                (similarities_buf, codebook_size * 4)
            ]
        )
        
        # Push constants: dim, codebook_size
        push_consts = struct.pack('II', dim, codebook_size)
        
        # Dispatch: one workgroup per codebook vector
        workgroups = codebook_size
        self.core._dispatch_compute(pipeline, layout, desc_set, workgroups, push_consts)
        
        # Download result
        result = self.core._download_buffer(similarities_mem, codebook_size * 4, dtype=np.float32)
        
        # Cleanup
        from vulkan import vkDestroyBuffer, vkFreeMemory, vkFreeDescriptorSets
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [desc_set])
        vkDestroyBuffer(self.core.device, query_buf, None)
        vkDestroyBuffer(self.core.device, codebook_buf, None)
        vkDestroyBuffer(self.core.device, similarities_buf, None)
        vkFreeMemory(self.core.device, query_mem, None)
        vkFreeMemory(self.core.device, codebook_mem, None)
        vkFreeMemory(self.core.device, similarities_mem, None)
        
        return result

    def resonator_step(
        self,
        composite: np.ndarray,
        codebook: np.ndarray,
        other_estimates: Optional[List[np.ndarray]] = None
    ) -> Tuple[np.ndarray, int]:
        """
        GPU-accelerated resonator projection step.

        Args:
            composite: Composite vector to factorize
            codebook: Codebook vectors of shape (codebook_size, dim)
            other_estimates: Optional list of other factor estimates to unbind

        Returns:
            Tuple of (best_match_vector, best_index)
        """
        from vulkan import VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        import struct

        unbound = composite.copy()
        if other_estimates:
            from grilly.experimental.vsa.ops import BinaryOps
            for estimate in other_estimates:
                unbound = BinaryOps.unbind(unbound, estimate)

        dim = len(unbound)
        codebook_size = len(codebook)

        assert codebook.shape == (codebook_size, dim), "Invalid codebook shape"

        if 'vsa-resonator-step' not in self.core.shaders:
            # Fallback to similarity_batch (GPU) or CPU
            if 'vsa-similarity-batch' in self.core.shaders:
                sims = self.similarity_batch(unbound, codebook)
            else:
                sims = (codebook @ unbound) / float(dim)
            best_idx = int(np.argmax(sims))
            return codebook[best_idx].copy(), best_idx

        # Create buffers
        query_buf, query_mem = self.core._create_buffer(dim * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        codebook_buf, codebook_mem = self.core._create_buffer(codebook_size * dim * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        similarities_buf, similarities_mem = self.core._create_buffer(codebook_size * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)

        # Upload data
        self.core._upload_buffer(query_buf, query_mem, unbound.astype(np.float32))
        self.core._upload_buffer(codebook_buf, codebook_mem, codebook.astype(np.float32).reshape(-1))

        # Pipeline
        pipeline, layout, descriptor = self.pipelines.get_or_create_pipeline(
            'vsa-resonator-step', num_buffers=3, push_constant_size=8
        )

        desc_set = self.pipelines._create_descriptor_set(
            descriptor,
            [
                (query_buf, dim * 4),
                (codebook_buf, codebook_size * dim * 4),
                (similarities_buf, codebook_size * 4),
            ]
        )

        push_consts = struct.pack('II', dim, codebook_size)

        workgroups = codebook_size
        self.core._dispatch_compute(pipeline, layout, desc_set, workgroups, push_consts)

        sims = self.core._download_buffer(similarities_mem, codebook_size * 4, dtype=np.float32)

        from vulkan import vkDestroyBuffer, vkFreeMemory, vkFreeDescriptorSets
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [desc_set])
        vkDestroyBuffer(self.core.device, query_buf, None)
        vkDestroyBuffer(self.core.device, codebook_buf, None)
        vkDestroyBuffer(self.core.device, similarities_buf, None)
        vkFreeMemory(self.core.device, query_mem, None)
        vkFreeMemory(self.core.device, codebook_mem, None)
        vkFreeMemory(self.core.device, similarities_mem, None)

        best_idx = int(np.argmax(sims))
        return codebook[best_idx].copy(), best_idx
    
    def circular_convolve(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated circular convolution for HRR binding.
        
        Note: Full FFT implementation requires multiple shader passes.
        This is a placeholder that falls back to CPU FFT.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Convolved vector
        """
        # For now, fallback to CPU FFT
        # Full GPU FFT implementation would require multiple shader passes
        from grilly.experimental.vsa.ops import HolographicOps
        return HolographicOps.convolve(a, b)
