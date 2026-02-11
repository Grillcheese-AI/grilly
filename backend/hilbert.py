"""
Vulkan GPU Acceleration for Hilbert Routing

Provides GPU-accelerated:
- Hilbert transform (via FFT)
- Complex similarity computation
- Batch operations

Integration: vulkan_backend/vulkan_hilbert.py

Author: Nick [Redacted]
Date: January 2026
"""

import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Try to import Vulkan compute
try:
    from .core import VulkanCore
    from .pipelines import VulkanPipelines

    VULKAN_AVAILABLE = True
except ImportError:
    VULKAN_AVAILABLE = False
    logger.warning("Vulkan backend not available, using CPU fallback")


class VulkanHilbert:
    """
    GPU-accelerated Hilbert space operations.

    Provides:
    - Complex similarity computation (hilbert_similarity.comp shader)
    - Batch Hilbert transform
    - Top-k selection with complex similarity

    Usage:
        hilbert_gpu = VulkanHilbert()

        # Convert to complex
        psi_queries = hilbert_gpu.hilbert_embed_batch(embeddings)

        # Compute similarities
        sims = hilbert_gpu.similarity_matrix(psi_queries, psi_keys)

        # Top-k search
        indices, scores = hilbert_gpu.topk(psi_query, psi_keys, k=5)
    """

    def __init__(self, device_index: int = 0, shader_dir: Path | None = None):
        """Initialize the instance."""

        self.device_index = device_index
        # Shader directory - use package-relative path
        if shader_dir is None:
            try:
                import grilly

                package_path = Path(grilly.__file__).parent
                shader_dir = package_path / "shaders"
            except (ImportError, AttributeError):
                shader_dir = Path(__file__).parent.parent / "shaders"
        self.shader_dir = shader_dir

        self.vulkan_core = None
        self.pipelines = None
        self.gpu_available = False

        self._init_vulkan()

        # Timing stats
        self.gpu_times: list[float] = []
        self.cpu_times: list[float] = []

    def _init_vulkan(self):
        """Initialize Vulkan compute."""
        if not VULKAN_AVAILABLE:
            logger.info("Running in CPU-only mode")
            return

        try:
            self.vulkan_core = VulkanCore(device_index=self.device_index)
            self.pipelines = VulkanPipelines(self.vulkan_core)

            # Compile shaders
            shader_path = self.shader_dir / "hilbert_similarity.comp"
            if shader_path.exists():
                self.pipelines.create_pipeline("hilbert_similarity", str(shader_path))
                logger.info("Hilbert similarity shader compiled")

            self.gpu_available = True
            logger.info(f"Vulkan Hilbert initialized on device {self.device_index}")

        except Exception as e:
            logger.warning(f"Failed to initialize Vulkan: {e}")
            self.gpu_available = False

    def hilbert_embed(
        self, x: np.ndarray, beta: float = 0.0, phi: float = 0.0, k: float = 10.0
    ) -> np.ndarray:
        """
        Convert real embedding to complex Hilbert space.

        GPU acceleration via FFT for Hilbert transform.
        """
        if self.gpu_available:
            return self._hilbert_embed_gpu(x, beta, phi, k)
        else:
            return self._hilbert_embed_cpu(x, beta, phi, k)

    def _hilbert_embed_cpu(self, x: np.ndarray, beta: float, phi: float, k: float) -> np.ndarray:
        """CPU Hilbert embedding."""
        from scipy.signal import hilbert

        n = len(x)
        mod_s = np.linalg.norm(x)

        if mod_s > 1e-8:
            x_norm = x / mod_s
        else:
            x_norm = x

        indices = np.arange(n, dtype=np.float32)
        freq_mod = np.sin((indices + beta) / k)
        signal = x_norm * freq_mod * mod_s

        analytic = hilbert(signal)
        psi = analytic * np.exp(1j * phi)
        psi = psi / (np.linalg.norm(psi) + 1e-8)

        return psi.astype(np.complex64)

    def _hilbert_embed_gpu(self, x: np.ndarray, beta: float, phi: float, k: float) -> np.ndarray:
        """GPU Hilbert embedding using FFT."""
        # For now, use CPU - FFT shader would be added later
        # GPU benefit is mainly in batch similarity computation
        return self._hilbert_embed_cpu(x, beta, phi, k)

    def hilbert_embed_batch(
        self, X: np.ndarray, beta: float = 0.0, phi: float = 0.0, k: float = 10.0
    ) -> np.ndarray:
        """Batch Hilbert embedding."""
        return np.array([self.hilbert_embed(x, beta, phi, k) for x in X])

    def similarity(self, psi1: np.ndarray, psi2: np.ndarray) -> float:
        """Compute complex similarity between two embeddings."""
        inner = np.vdot(psi1, psi2)
        norm1 = np.linalg.norm(psi1)
        norm2 = np.linalg.norm(psi2)

        if norm1 < 1e-12 or norm2 < 1e-12:
            return 0.0

        return float(np.abs(inner) / (norm1 * norm2))

    def similarity_matrix(self, psi_queries: np.ndarray, psi_keys: np.ndarray) -> np.ndarray:
        """
        Compute pairwise similarities between queries and keys.

        GPU-accelerated when available.

        Args:
            psi_queries: [N, dim] complex
            psi_keys: [M, dim] complex

        Returns:
            [N, M] similarity matrix
        """
        if self.gpu_available and len(psi_queries) * len(psi_keys) > 1000:
            return self._similarity_matrix_gpu(psi_queries, psi_keys)
        else:
            return self._similarity_matrix_cpu(psi_queries, psi_keys)

    def _similarity_matrix_cpu(self, psi_queries: np.ndarray, psi_keys: np.ndarray) -> np.ndarray:
        """CPU similarity matrix computation."""
        t0 = time.perf_counter()

        n_queries = len(psi_queries)
        n_keys = len(psi_keys)

        sims = np.zeros((n_queries, n_keys), dtype=np.float32)

        for i, psi_q in enumerate(psi_queries):
            for j, psi_k in enumerate(psi_keys):
                sims[i, j] = self.similarity(psi_q, psi_k)

        self.cpu_times.append(time.perf_counter() - t0)
        return sims

    def _similarity_matrix_gpu(self, psi_queries: np.ndarray, psi_keys: np.ndarray) -> np.ndarray:
        """GPU similarity matrix using Vulkan shader."""
        t0 = time.perf_counter()

        n_queries = len(psi_queries)
        n_keys = len(psi_keys)
        dim = psi_queries.shape[1]

        # Convert complex to interleaved real/imag
        queries_flat = np.zeros((n_queries, dim * 2), dtype=np.float32)
        keys_flat = np.zeros((n_keys, dim * 2), dtype=np.float32)

        queries_flat[:, 0::2] = psi_queries.real
        queries_flat[:, 1::2] = psi_queries.imag
        keys_flat[:, 0::2] = psi_keys.real
        keys_flat[:, 1::2] = psi_keys.imag

        # Allocate output
        output = np.zeros((n_queries, n_keys), dtype=np.float32)

        try:
            # Run GPU shader
            self.pipelines.run_pipeline(
                "hilbert_similarity",
                push_constants={
                    "num_queries": n_queries,
                    "num_keys": n_keys,
                    "dim": dim,
                    "temperature": 1.0,
                },
                buffers=[queries_flat.flatten(), keys_flat.flatten(), output.flatten()],
                workgroups=(n_queries * n_keys, 1, 1),
            )

            self.gpu_times.append(time.perf_counter() - t0)
            return output

        except Exception as e:
            logger.warning(f"GPU similarity failed, falling back to CPU: {e}")
            return self._similarity_matrix_cpu(psi_queries, psi_keys)

    def topk(
        self, psi_query: np.ndarray, psi_keys: np.ndarray, k: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find top-k most similar keys to query.

        Args:
            psi_query: [dim] complex query
            psi_keys: [N, dim] complex keys
            k: Number of results

        Returns:
            (indices, scores) of top-k matches
        """
        # Compute similarities
        sims = self.similarity_matrix(psi_query.reshape(1, -1), psi_keys)[0]

        # Top-k selection
        k = min(k, len(sims))
        indices = np.argsort(sims)[-k:][::-1]
        scores = sims[indices]

        return indices, scores

    def topk_batch(
        self, psi_queries: np.ndarray, psi_keys: np.ndarray, k: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch top-k search."""
        sims = self.similarity_matrix(psi_queries, psi_keys)

        k = min(k, sims.shape[1])
        indices = np.argsort(sims, axis=1)[:, -k:][:, ::-1]

        # Gather scores
        scores = np.take_along_axis(sims, indices, axis=1)

        return indices, scores

    def get_stats(self) -> dict:
        """Get performance statistics."""
        stats = {"gpu_available": self.gpu_available, "device_index": self.device_index}

        if self.gpu_times:
            stats["gpu_avg_ms"] = np.mean(self.gpu_times) * 1000
            stats["gpu_p99_ms"] = np.percentile(self.gpu_times, 99) * 1000

        if self.cpu_times:
            stats["cpu_avg_ms"] = np.mean(self.cpu_times) * 1000
            stats["cpu_p99_ms"] = np.percentile(self.cpu_times, 99) * 1000

        if self.gpu_times and self.cpu_times:
            stats["speedup"] = np.mean(self.cpu_times) / np.mean(self.gpu_times)

        return stats


# =============================================================================
# INTEGRATION WITH EXISTING VULKAN_FAISS
# =============================================================================


def patch_vulkan_faiss(vulkan_faiss_module):
    """
    Patch existing vulkan_faiss.py to use Hilbert similarity.

    Usage:
        from . import vulkan_faiss
        from .vulkan_hilbert import patch_vulkan_faiss

        patch_vulkan_faiss(vulkan_faiss)
    """
    original_search = vulkan_faiss_module.search
    hilbert_gpu = VulkanHilbert()

    def hilbert_search(
        query_embedding: np.ndarray,
        key_embeddings: np.ndarray,
        k: int = 5,
        use_hilbert: bool = True,
        **kwargs,
    ):
        """Run hilbert search."""

        if not use_hilbert:
            return original_search(query_embedding, key_embeddings, k, **kwargs)

        # Convert to Hilbert space
        psi_query = hilbert_gpu.hilbert_embed(query_embedding)
        psi_keys = hilbert_gpu.hilbert_embed_batch(key_embeddings)

        # Search
        indices, scores = hilbert_gpu.topk(psi_query, psi_keys, k)

        return indices, scores

    vulkan_faiss_module.search = hilbert_search
    vulkan_faiss_module._hilbert_gpu = hilbert_gpu

    logger.info("Patched vulkan_faiss with Hilbert similarity")


# =============================================================================
# TESTS
# =============================================================================


def benchmark():
    """Benchmark GPU vs CPU performance."""
    print("=" * 60)
    print("VULKAN HILBERT BENCHMARK")
    print("=" * 60)

    hilbert = VulkanHilbert()

    print(f"\nGPU available: {hilbert.gpu_available}")

    # Test sizes
    sizes = [(10, 100), (100, 1000), (100, 10000)]

    for n_queries, n_keys in sizes:
        print(f"\n--- {n_queries} queries × {n_keys} keys ---")

        # Generate test data
        np.random.seed(42)
        queries = np.random.randn(n_queries, 384).astype(np.float32)
        keys = np.random.randn(n_keys, 384).astype(np.float32)

        # Embed
        psi_queries = hilbert.hilbert_embed_batch(queries)
        psi_keys = hilbert.hilbert_embed_batch(keys)

        # Time CPU
        t0 = time.perf_counter()
        sims_cpu = hilbert._similarity_matrix_cpu(psi_queries, psi_keys)
        cpu_time = time.perf_counter() - t0

        # Time GPU if available
        if hilbert.gpu_available:
            t0 = time.perf_counter()
            sims_gpu = hilbert._similarity_matrix_gpu(psi_queries, psi_keys)
            gpu_time = time.perf_counter() - t0

            # Verify
            error = np.abs(sims_cpu - sims_gpu).max()

            print(f"  CPU: {cpu_time * 1000:.2f}ms")
            print(f"  GPU: {gpu_time * 1000:.2f}ms")
            print(f"  Speedup: {cpu_time / gpu_time:.1f}x")
            print(f"  Max error: {error:.2e}")
        else:
            print(f"  CPU: {cpu_time * 1000:.2f}ms")
            print("  GPU: N/A")


def test_correctness():
    """Test Hilbert operations correctness."""
    print("\n" + "=" * 60)
    print("CORRECTNESS TEST")
    print("=" * 60)

    hilbert = VulkanHilbert()

    np.random.seed(42)

    # Similar embeddings should have high similarity
    emb_a = np.random.randn(384).astype(np.float32)
    emb_b = emb_a + np.random.randn(384) * 0.1
    emb_c = np.random.randn(384).astype(np.float32)

    psi_a = hilbert.hilbert_embed(emb_a)
    psi_b = hilbert.hilbert_embed(emb_b)
    psi_c = hilbert.hilbert_embed(emb_c)

    sim_ab = hilbert.similarity(psi_a, psi_b)
    sim_ac = hilbert.similarity(psi_a, psi_c)

    print(f"\nSimilar embeddings:   {sim_ab:.3f}")
    print(f"Different embeddings: {sim_ac:.3f}")
    print(f"Ordering correct:     {sim_ab > sim_ac}")

    # Top-k test
    np.random.seed(42)
    query = np.random.randn(384).astype(np.float32)
    keys = np.random.randn(100, 384).astype(np.float32)

    psi_query = hilbert.hilbert_embed(query)
    psi_keys = hilbert.hilbert_embed_batch(keys)

    indices, scores = hilbert.topk(psi_query, psi_keys, k=5)

    print(f"\nTop-5 indices: {indices}")
    print(f"Top-5 scores:  {scores.round(3)}")


if __name__ == "__main__":
    test_correctness()
    benchmark()
