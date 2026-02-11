"""
VulkanVSA - GPU-accelerated Vector Symbolic Architecture operations.

Provides GPU implementations of VSA operations for high-performance
hyperdimensional computing.
"""

import numpy as np

from ..base import BufferMixin
from ..core import VulkanCore
from ..pipelines import VulkanPipelines


class VulkanVSA(BufferMixin):
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
        dim = len(a)
        assert len(b) == dim, "Vectors must have same dimension"

        # Check if shader available, otherwise fallback to CPU
        if "vsa-bind" not in self.core.shaders:
            from grilly.experimental.vsa.ops import BinaryOps

            return BinaryOps.bind(a, b)

        # Create buffers
        a_buf = self._acquire_buffer(dim * 4)
        b_buf = self._acquire_buffer(dim * 4)
        result_buf = self._acquire_buffer(dim * 4)

        try:
            # Upload data
            self._upload_buffer(a_buf, a.astype(np.float32))
            self._upload_buffer(b_buf, b.astype(np.float32))

            # Get or create pipeline
            pipeline, layout, descriptor = self.pipelines.get_or_create_pipeline(
                "vsa-bind", num_buffers=3, push_constant_size=4
            )

            # Create descriptor set
            desc_set = self.pipelines._create_descriptor_set(
                descriptor,
                [
                    (self._get_buffer_handle(a_buf), dim * 4),
                    (self._get_buffer_handle(b_buf), dim * 4),
                    (self._get_buffer_handle(result_buf), dim * 4),
                ],
            )

            # Push constants
            import struct

            push_consts = struct.pack("I", dim)

            # Dispatch
            workgroups = (dim + 255) // 256
            self.core._dispatch_compute(pipeline, layout, desc_set, workgroups, push_consts)

            # Download result
            result = self._download_buffer(result_buf, dim * 4, np.float32)

            # Free descriptor set
            from vulkan import vkFreeDescriptorSets

            vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [desc_set])

            return result
        finally:
            self._release_buffers([a_buf, b_buf, result_buf])

    def bind_bipolar_batch(self, a_batch: np.ndarray, b_batch: np.ndarray) -> np.ndarray:
        """
        Batch bipolar binding using a single GPU dispatch when available.

        Args:
            a_batch: Batch of bipolar vectors (batch, dim)
            b_batch: Batch of bipolar vectors (batch, dim)

        Returns:
            Bound vectors (batch, dim)
        """
        import struct

        assert a_batch.shape == b_batch.shape, "Batch inputs must match shape"
        assert a_batch.ndim == 2, "Batch inputs must be 2D (batch, dim)"

        batch_size, dim = a_batch.shape

        # Check if shader available, otherwise fallback to loop
        if "vsa-bind-batch" not in self.core.shaders:
            results = np.empty((batch_size, dim), dtype=np.float32)
            for i in range(batch_size):
                results[i] = self.bind_bipolar(a_batch[i], b_batch[i])
            return results

        total = batch_size * dim
        a_flat = a_batch.astype(np.float32).reshape(-1)
        b_flat = b_batch.astype(np.float32).reshape(-1)

        a_buf = self._acquire_buffer(total * 4)
        b_buf = self._acquire_buffer(total * 4)
        result_buf = self._acquire_buffer(total * 4)

        try:
            self._upload_buffer(a_buf, a_flat)
            self._upload_buffer(b_buf, b_flat)

            pipeline, layout, descriptor = self.pipelines.get_or_create_pipeline(
                "vsa-bind-batch", num_buffers=3, push_constant_size=4
            )

            desc_set = self.pipelines._create_descriptor_set(
                descriptor,
                [
                    (self._get_buffer_handle(a_buf), total * 4),
                    (self._get_buffer_handle(b_buf), total * 4),
                    (self._get_buffer_handle(result_buf), total * 4),
                ],
            )

            push_consts = struct.pack("I", total)

            workgroups = (total + 255) // 256
            self.core._dispatch_compute(pipeline, layout, desc_set, workgroups, push_consts)

            result = self._download_buffer(result_buf, total * 4, np.float32)
            result = result.reshape(batch_size, dim)

            from vulkan import vkFreeDescriptorSets

            vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [desc_set])

            return result
        finally:
            self._release_buffers([a_buf, b_buf, result_buf])

    def bundle(self, vectors: list[np.ndarray]) -> np.ndarray:
        """
        GPU-accelerated bundling (superposition with majority voting).

        Args:
            vectors: List of vectors to bundle

        Returns:
            Bundled vector
        """
        import struct

        if not vectors:
            raise ValueError("Cannot bundle empty list")

        dim = len(vectors[0])
        num_vectors = len(vectors)

        # Check if shader available, otherwise fallback to CPU
        if "vsa-bundle" not in self.core.shaders:
            from grilly.experimental.vsa.ops import BinaryOps

            return BinaryOps.bundle(vectors)

        # Flatten vectors
        vectors_flat = np.array(vectors, dtype=np.float32).flatten()

        # Create buffers
        vectors_buf = self._acquire_buffer(num_vectors * dim * 4)
        result_buf = self._acquire_buffer(dim * 4)

        try:
            # Upload data
            self._upload_buffer(vectors_buf, vectors_flat)

            # Get or create pipeline
            pipeline, layout, descriptor = self.pipelines.get_or_create_pipeline(
                "vsa-bundle", num_buffers=2, push_constant_size=8
            )

            # Create descriptor set
            desc_set = self.pipelines._create_descriptor_set(
                descriptor,
                [
                    (self._get_buffer_handle(vectors_buf), num_vectors * dim * 4),
                    (self._get_buffer_handle(result_buf), dim * 4),
                ],
            )

            # Push constants: dim, num_vectors
            push_consts = struct.pack("II", dim, num_vectors)

            # Dispatch
            workgroups = (dim + 255) // 256
            self.core._dispatch_compute(pipeline, layout, desc_set, workgroups, push_consts)

            # Download result
            result = self._download_buffer(result_buf, dim * 4, np.float32)

            # Free descriptor set
            from vulkan import vkFreeDescriptorSets

            vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [desc_set])

            return result
        finally:
            self._release_buffers([vectors_buf, result_buf])

    def bundle_batch(self, vectors: np.ndarray) -> np.ndarray:
        """
        Batch bundling using a single GPU dispatch when available.

        Args:
            vectors: Array of shape (batch, num_vectors, dim)

        Returns:
            Bundled vectors of shape (batch, dim)
        """
        import struct

        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim != 3:
            raise ValueError("vectors must have shape (batch, num_vectors, dim)")

        batch_size, num_vectors, dim = vecs.shape

        # Check if shader available, otherwise fallback to CPU
        if "vsa-bundle-batch" not in self.core.shaders:
            from grilly.experimental.vsa.ops import BinaryOps

            return BinaryOps.bundle_batch(vecs)

        total = batch_size * dim
        vectors_flat = vecs.reshape(-1)

        vectors_buf = self._acquire_buffer(batch_size * num_vectors * dim * 4)
        result_buf = self._acquire_buffer(total * 4)

        try:
            self._upload_buffer(vectors_buf, vectors_flat)

            pipeline, layout, descriptor = self.pipelines.get_or_create_pipeline(
                "vsa-bundle-batch", num_buffers=2, push_constant_size=12
            )

            desc_set = self.pipelines._create_descriptor_set(
                descriptor,
                [
                    (self._get_buffer_handle(vectors_buf), batch_size * num_vectors * dim * 4),
                    (self._get_buffer_handle(result_buf), total * 4),
                ],
            )

            push_consts = struct.pack("III", dim, num_vectors, batch_size)

            workgroups = (total + 255) // 256
            self.core._dispatch_compute(pipeline, layout, desc_set, workgroups, push_consts)

            result = self._download_buffer(result_buf, total * 4, np.float32)
            result = result.reshape(batch_size, dim)

            from vulkan import vkFreeDescriptorSets

            vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [desc_set])

            return result
        finally:
            self._release_buffers([vectors_buf, result_buf])

    def similarity_batch(self, query: np.ndarray, codebook: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated batch similarity computation.

        Args:
            query: Query vector of shape (dim,)
            codebook: Codebook vectors of shape (codebook_size, dim)

        Returns:
            Similarities of shape (codebook_size,)
        """
        import struct

        dim = len(query)
        codebook_size = len(codebook)

        assert codebook.shape == (codebook_size, dim), "Invalid codebook shape"

        # Check if shader available, otherwise fallback to CPU
        if "vsa-similarity-batch" not in self.core.shaders:
            from grilly.experimental.vsa.ops import BinaryOps

            return np.array([BinaryOps.similarity(query, vec) for vec in codebook])

        # Flatten codebook
        codebook_flat = codebook.flatten()

        # Create buffers
        query_buf = self._acquire_buffer(dim * 4)
        codebook_buf = self._acquire_buffer(codebook_size * dim * 4)
        similarities_buf = self._acquire_buffer(codebook_size * 4)

        try:
            # Upload data
            self._upload_buffer(query_buf, query.astype(np.float32))
            self._upload_buffer(codebook_buf, codebook_flat.astype(np.float32))

            # Get or create pipeline
            pipeline, layout, descriptor = self.pipelines.get_or_create_pipeline(
                "vsa-similarity-batch", num_buffers=3, push_constant_size=8
            )

            # Create descriptor set
            desc_set = self.pipelines._create_descriptor_set(
                descriptor,
                [
                    (self._get_buffer_handle(query_buf), dim * 4),
                    (self._get_buffer_handle(codebook_buf), codebook_size * dim * 4),
                    (self._get_buffer_handle(similarities_buf), codebook_size * 4),
                ],
            )

            # Push constants: dim, codebook_size
            push_consts = struct.pack("II", dim, codebook_size)

            # Dispatch: one workgroup per codebook vector
            workgroups = codebook_size
            self.core._dispatch_compute(pipeline, layout, desc_set, workgroups, push_consts)

            # Download result
            result = self._download_buffer(similarities_buf, codebook_size * 4, np.float32)

            # Free descriptor set
            from vulkan import vkFreeDescriptorSets

            vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [desc_set])

            return result
        finally:
            self._release_buffers([query_buf, codebook_buf, similarities_buf])

    # -------------------------------------------------------------------------
    # GEMM-based similarity (RDNA2-friendly)
    # -------------------------------------------------------------------------

    def similarity_matrix_gemm(
        self,
        queries: np.ndarray,
        codebook: np.ndarray,
        divide_by_dim: bool = True,
    ) -> np.ndarray:
        """
        Compute similarities using a tiled GEMM shader.

        Computes: scores[B, N] = queries[B, D] @ codebook[N, D]^T

        Args:
            queries: (B, D) float32
            codebook: (N, D) float32
            divide_by_dim: if True, divide dot by D (recommended for bipolar)

        Returns:
            scores: (B, N) float32
        """
        import struct

        queries = np.asarray(queries, dtype=np.float32)
        codebook = np.asarray(codebook, dtype=np.float32)

        assert queries.ndim == 2, "queries must be (B, D)"
        assert codebook.ndim == 2, "codebook must be (N, D)"

        B, D = queries.shape
        N, D2 = codebook.shape
        assert D == D2, "dimension mismatch"

        # Fallback if shader not available
        if "vsa-similarity-gemm" not in self.core.shaders:
            # CPU dot
            scores = queries @ codebook.T
            if divide_by_dim and D > 0:
                scores = scores / float(D)
            return scores.astype(np.float32)

        # Create buffers
        q_buf = self._acquire_buffer(B * D * 4)
        c_buf = self._acquire_buffer(N * D * 4)
        o_buf = self._acquire_buffer(B * N * 4)

        try:
            # Upload
            self._upload_buffer(q_buf, queries.reshape(-1))
            self._upload_buffer(c_buf, codebook.reshape(-1))

            # Pipeline
            pipeline, layout, descriptor = self.pipelines.get_or_create_pipeline(
                "vsa-similarity-gemm", num_buffers=3, push_constant_size=16
            )

            desc_set = self.pipelines._create_descriptor_set(
                descriptor,
                [
                    (self._get_buffer_handle(q_buf), B * D * 4),
                    (self._get_buffer_handle(c_buf), N * D * 4),
                    (self._get_buffer_handle(o_buf), B * N * 4),
                ],
            )

            flags = 1 if divide_by_dim else 0
            push_consts = struct.pack("IIII", B, N, D, flags)

            # Dispatch tiles
            wg_x = (N + 15) // 16
            wg_y = (B + 15) // 16
            self.core._dispatch_compute(
                pipeline, layout, desc_set, wg_x, push_consts, workgroup_y=wg_y
            )

            # Download
            out = self._download_buffer(o_buf, B * N * 4, np.float32).reshape(B, N)

            # Free descriptor set
            from vulkan import vkFreeDescriptorSets

            vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [desc_set])

            return out
        finally:
            self._release_buffers([q_buf, c_buf, o_buf])

    def similarity_batch_gemm(self, query: np.ndarray, codebook: np.ndarray) -> np.ndarray:
        """
        GEMM-based drop-in replacement for similarity_batch (single query).

        Args:
            query: (D,)
            codebook: (N, D)

        Returns:
            (N,) similarities
        """
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        scores = self.similarity_matrix_gemm(q, codebook, divide_by_dim=True)
        return scores[0]

    def similarity_topk_gemm(
        self,
        queries: np.ndarray,
        codebook: np.ndarray,
        top_k: int = 1,
        mask_value: float = -1e20,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Top-k routing on GPU: GEMM -> row-wise argmax (k rounds with masking).

        Args:
            queries: (B, D)
            codebook: (N, D)
            top_k: number of winners per row
            mask_value: value to write at selected positions between rounds

        Returns:
            idx: (B, top_k) uint32
            val: (B, top_k) float32
        """
        import struct as _struct

        queries = np.asarray(queries, dtype=np.float32)
        codebook = np.asarray(codebook, dtype=np.float32)
        assert queries.ndim == 2 and codebook.ndim == 2
        B, D = queries.shape
        N, D2 = codebook.shape
        assert D == D2
        top_k = int(top_k)
        assert top_k >= 1

        # Fallback (CPU)
        if (
            "vsa-similarity-gemm" not in self.core.shaders
            or "vsa-argmax-rows" not in self.core.shaders
            or "vsa-mask-selected" not in self.core.shaders
        ):
            scores = (queries @ codebook.T) / float(D) if D > 0 else (queries @ codebook.T)
            idx = np.argsort(-scores, axis=1)[:, :top_k].astype(np.uint32)
            val = np.take_along_axis(scores, idx.astype(np.int64), axis=1).astype(np.float32)
            return idx, val

        # --- Allocate similarity buffers (keep on GPU) ---
        q_buf = self._acquire_buffer(B * D * 4)
        c_buf = self._acquire_buffer(N * D * 4)
        s_buf = self._acquire_buffer(B * N * 4)
        idx_buf = self._acquire_buffer(B * 4)
        val_buf = self._acquire_buffer(B * 4)

        try:
            self._upload_buffer(q_buf, queries.reshape(-1))
            self._upload_buffer(c_buf, codebook.reshape(-1))

            # Similarity GEMM pipeline
            sim_pipe, sim_layout, sim_desc = self.pipelines.get_or_create_pipeline(
                "vsa-similarity-gemm", num_buffers=3, push_constant_size=16
            )
            sim_set = self.pipelines._create_descriptor_set(
                sim_desc,
                [
                    (self._get_buffer_handle(q_buf), B * D * 4),
                    (self._get_buffer_handle(c_buf), N * D * 4),
                    (self._get_buffer_handle(s_buf), B * N * 4),
                ],
            )
            flags = 1  # divide by D
            sim_pc = _struct.pack("IIII", B, N, D, flags)
            wg_x = (N + 15) // 16
            wg_y = (B + 15) // 16
            self.core._dispatch_compute(
                sim_pipe, sim_layout, sim_set, wg_x, sim_pc, workgroup_y=wg_y
            )

            # Argmax pipeline
            arg_pipe, arg_layout, arg_desc = self.pipelines.get_or_create_pipeline(
                "vsa-argmax-rows", num_buffers=3, push_constant_size=8
            )
            mask_pipe, mask_layout, mask_desc = self.pipelines.get_or_create_pipeline(
                "vsa-mask-selected", num_buffers=2, push_constant_size=12
            )

            idx_out = np.zeros((B, top_k), dtype=np.uint32)
            val_out = np.zeros((B, top_k), dtype=np.float32)

            for r in range(top_k):
                # Argmax
                arg_set = self.pipelines._create_descriptor_set(
                    arg_desc,
                    [
                        (self._get_buffer_handle(s_buf), B * N * 4),
                        (self._get_buffer_handle(idx_buf), B * 4),
                        (self._get_buffer_handle(val_buf), B * 4),
                    ],
                )
                arg_pc = _struct.pack("II", N, B)
                self.core._dispatch_compute(arg_pipe, arg_layout, arg_set, B, arg_pc)

                # Read back indices + values
                idx_round = self._download_buffer(idx_buf, B * 4, np.uint32)
                val_round = self._download_buffer(val_buf, B * 4, np.float32)
                idx_out[:, r] = idx_round
                val_out[:, r] = val_round

                # Mask selected for next round (except last)
                if r != top_k - 1:
                    mask_set = self.pipelines._create_descriptor_set(
                        mask_desc,
                        [
                            (self._get_buffer_handle(s_buf), B * N * 4),
                            (self._get_buffer_handle(idx_buf), B * 4),
                        ],
                    )
                    mask_pc = _struct.pack("IIf", N, B, float(mask_value))
                    self.core._dispatch_compute(mask_pipe, mask_layout, mask_set, B, mask_pc)

                    from vulkan import vkFreeDescriptorSets

                    vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [mask_set])

                from vulkan import vkFreeDescriptorSets

                vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [arg_set])

            # Free similarity descriptor set
            from vulkan import vkFreeDescriptorSets

            vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [sim_set])

            return idx_out, val_out
        finally:
            self._release_buffers([q_buf, c_buf, s_buf, idx_buf, val_buf])

    def resonator_step(
        self,
        composite: np.ndarray,
        codebook: np.ndarray,
        other_estimates: list[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        GPU-accelerated resonator projection step.

        Args:
            composite: Composite vector to factorize
            codebook: Codebook vectors of shape (codebook_size, dim)
            other_estimates: Optional list of other factor estimates to unbind

        Returns:
            Tuple of (best_match_vector, best_index)
        """
        import struct

        unbound = composite.copy()
        if other_estimates:
            from grilly.experimental.vsa.ops import BinaryOps

            for estimate in other_estimates:
                unbound = BinaryOps.unbind(unbound, estimate)

        dim = len(unbound)
        codebook_size = len(codebook)

        assert codebook.shape == (codebook_size, dim), "Invalid codebook shape"

        if "vsa-resonator-step" not in self.core.shaders:
            # Fallback to similarity_batch (GPU) or CPU
            if "vsa-similarity-batch" in self.core.shaders:
                sims = self.similarity_batch(unbound, codebook)
            else:
                sims = (codebook @ unbound) / float(dim)
            best_idx = int(np.argmax(sims))
            return codebook[best_idx].copy(), best_idx

        # Create buffers
        query_buf = self._acquire_buffer(dim * 4)
        codebook_buf = self._acquire_buffer(codebook_size * dim * 4)
        similarities_buf = self._acquire_buffer(codebook_size * 4)

        try:
            # Upload data
            self._upload_buffer(query_buf, unbound.astype(np.float32))
            self._upload_buffer(codebook_buf, codebook.astype(np.float32).reshape(-1))

            # Pipeline
            pipeline, layout, descriptor = self.pipelines.get_or_create_pipeline(
                "vsa-resonator-step", num_buffers=3, push_constant_size=8
            )

            desc_set = self.pipelines._create_descriptor_set(
                descriptor,
                [
                    (self._get_buffer_handle(query_buf), dim * 4),
                    (self._get_buffer_handle(codebook_buf), codebook_size * dim * 4),
                    (self._get_buffer_handle(similarities_buf), codebook_size * 4),
                ],
            )

            push_consts = struct.pack("II", dim, codebook_size)

            workgroups = codebook_size
            self.core._dispatch_compute(pipeline, layout, desc_set, workgroups, push_consts)

            sims = self._download_buffer(similarities_buf, codebook_size * 4, np.float32)

            # Free descriptor set
            from vulkan import vkFreeDescriptorSets

            vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [desc_set])

            best_idx = int(np.argmax(sims))
            return codebook[best_idx].copy(), best_idx
        finally:
            self._release_buffers([query_buf, codebook_buf, similarities_buf])

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
