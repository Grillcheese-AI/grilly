"""
VulkanVSA - GPU-accelerated Vector Symbolic Architecture operations.

Provides GPU implementations of VSA operations for high-performance
hyperdimensional computing.
"""

import numpy as np
from typing import List, Optional
from ..core import VulkanCore
from ..pipelines import VulkanPipelines


class VulkanVSA:
    """
    GPU-accelerated VSA operations using Vulkan compute shaders.
    
    Provides:
    - bind_bipolar: Element-wise multiplication (O(d))
    - bundle: Superposition with majority voting (O(d))
    - similarity_batch: Parallel cosine similarity (O(V*d))
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
        
        # Load shaders
        self.shaders = self._load_shaders()
        
        # Register shaders with core so pipelines can find them
        for shader_name, shader_code in self.shaders.items():
            if shader_name not in self.core.shaders:
                self.core.shaders[shader_name] = shader_code
        
        # Create pipelines
        self._init_pipelines()
    
    def _load_shaders(self) -> dict:
        """Load VSA shaders from experimental directory."""
        shaders = {}
        experimental_dir = self.core.shader_dir / "experimental"
        experimental_spv_dir = experimental_dir / "spv"
        
        shader_names = [
            "vsa-bind",
            "vsa-bundle",
            "vsa-similarity-batch",
            "vsa-fft-convolve",
        ]
        
        # Try to load from SPV directory first (compiled shaders)
        if experimental_spv_dir.exists():
            for shader_name in shader_names:
                spv_path = experimental_spv_dir / f"{shader_name}.spv"
                if spv_path.exists():
                    with open(spv_path, 'rb') as f:
                        shaders[shader_name] = f.read()
        
        # Also check main SPV directory (fallback)
        main_spv_dir = self.core.shader_dir / "spv"
        if main_spv_dir.exists():
            for shader_name in shader_names:
                if shader_name not in shaders:
                    spv_path = main_spv_dir / f"{shader_name}.spv"
                    if spv_path.exists():
                        with open(spv_path, 'rb') as f:
                            shaders[shader_name] = f.read()
        
        # Note: If shaders are not found, operations will fallback to CPU
        return shaders
    
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
        if 'vsa-bind' not in self.shaders:
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
        if 'vsa-bundle' not in self.shaders:
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
        if 'vsa-similarity-batch' not in self.shaders:
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
