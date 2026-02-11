"""
Vector Symbolic Architecture (VSA) Operations.

Provides O(d) operations for hyperdimensional computing:
- Binding: Combine two vectors into a composite
- Unbinding: Recover a vector from a composite
- Bundling: Superposition of multiple vectors
- Similarity: Measure relatedness between vectors

Two operation classes are provided:
- BinaryOps: For bipolar (+1/-1) vectors - exact binding/unbinding
- HolographicOps: For continuous vectors using FFT - approximate binding/unbinding

Author: Grilly Team
Date: February 2026
"""

import numpy as np

# Stable hashing (BLAKE3) for deterministic string->vector
try:
    from utils.stable_hash import bipolar_from_key, stable_u32
except ModuleNotFoundError:
    try:
        from grilly.utils.stable_hash import bipolar_from_key, stable_u32  # type: ignore
    except Exception:
        stable_u32 = None  # type: ignore
        bipolar_from_key = None  # type: ignore


class BinaryOps:
    """
    Operations for bipolar (+1/-1) vectors.

    Bipolar vectors are efficient for hardware implementation and have
    exact inverse properties: bind(a, a) = identity, unbind = bind.

    All operations are O(d) and embarrassingly parallel.
    """

    @staticmethod
    def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Bind two bipolar vectors via element-wise multiplication.

        Properties:
            - Commutative: bind(a, b) == bind(b, a)
            - Associative: bind(bind(a, b), c) == bind(a, bind(b, c))
            - Self-inverse: bind(a, a) == identity (all ones)
            - Preserves bipolarity: output is +1/-1

        Args:
            a: First bipolar vector
            b: Second bipolar vector

        Returns:
            Bound bipolar vector
        """
        return (a * b).astype(np.float32)

    @staticmethod
    def bind_batch(a_batch: np.ndarray, b_batch: np.ndarray) -> np.ndarray:
        """
        Batch bind two sets of bipolar vectors via element-wise multiplication.

        Args:
            a_batch: Batch of vectors (batch, dim)
            b_batch: Batch of vectors (batch, dim)

        Returns:
            Bound vectors (batch, dim)
        """
        a_batch = np.asarray(a_batch)
        b_batch = np.asarray(b_batch)

        if a_batch.shape != b_batch.shape:
            raise ValueError("Batch inputs must have the same shape")
        if a_batch.ndim != 2:
            raise ValueError("Batch inputs must be 2D (batch, dim)")

        return (a_batch * b_batch).astype(np.float32)

    @staticmethod
    def unbind(composite: np.ndarray, known: np.ndarray) -> np.ndarray:
        """
        Unbind a known vector from a composite.

        For bipolar vectors, unbind is identical to bind since
        each element is its own inverse: x * x = 1.

        Args:
            composite: The composite vector
            known: The known factor to remove

        Returns:
            The recovered vector (approximately the original bound vector)
        """
        return (composite * known).astype(np.float32)

    @staticmethod
    def bundle(vectors: list[np.ndarray], normalize: bool = True) -> np.ndarray:
        """
        Bundle multiple vectors via majority voting.

        The result preserves similarity to each component, allowing
        multiple items to be stored in superposition.

        Args:
            vectors: List of vectors to bundle
            normalize: If True, apply sign function for bipolar output

        Returns:
            Bundled vector (bipolar if normalize=True)
        """
        if not vectors:
            raise ValueError("Cannot bundle empty list of vectors")

        result = np.sum(vectors, axis=0)

        if normalize:
            # Majority vote: sign of sum (with small epsilon to break ties)
            result = np.sign(result + 1e-8).astype(np.float32)

        return result

    @staticmethod
    def bundle_batch(vectors: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Bundle multiple batches of vectors via majority voting.

        Args:
            vectors: Array of shape (batch, num_vectors, dim)
            normalize: If True, apply sign function for bipolar output

        Returns:
            Bundled vectors of shape (batch, dim)
        """
        vecs = np.asarray(vectors, dtype=np.float32)

        if vecs.ndim != 3:
            raise ValueError("vectors must have shape (batch, num_vectors, dim)")

        result = np.sum(vecs, axis=1)

        if normalize:
            result = np.sign(result + 1e-8).astype(np.float32)

        return result

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        For bipolar vectors, this is equivalent to normalized Hamming distance.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Similarity in range [-1, 1]
        """
        return float(np.dot(a, b) / len(a))

    @staticmethod
    def similarity_batch(query: np.ndarray, codebook: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between a query and a codebook.

        Args:
            query: Query vector of shape (dim,)
            codebook: Codebook vectors of shape (num_vectors, dim)

        Returns:
            Similarities of shape (num_vectors,)
        """
        query = np.asarray(query, dtype=np.float32)
        codebook = np.asarray(codebook, dtype=np.float32)

        if query.ndim != 1:
            raise ValueError("query must be 1D (dim,)")
        if codebook.ndim != 2 or codebook.shape[1] != query.shape[0]:
            raise ValueError("codebook must have shape (num_vectors, dim)")

        return (codebook @ query) / float(query.shape[0])

    @staticmethod
    def random_bipolar(dim: int, seed: int | None = None) -> np.ndarray:
        """
        Generate a random bipolar vector (+1/-1).

        NOTE: Uses a local RNG when seed is provided to avoid mutating global numpy state.
        """
        if seed is None:
            return np.sign(np.random.randn(dim)).astype(np.float32)
        rng = np.random.RandomState(seed)
        return np.sign(rng.randn(dim)).astype(np.float32)

    @staticmethod
    def hash_to_bipolar(s: str, dim: int) -> np.ndarray:
        """
        Deterministically map a string to a bipolar (+1/-1) vector.

        IMPORTANT:
        - Does NOT use Python's built-in `hash()` (which is randomized per-process).
        - Uses BLAKE3 (preferred) via utils.stable_hash.bipolar_from_key.

        Falls back to a stable-u32 seed (still deterministic) if the helper is unavailable.
        """
        if bipolar_from_key is not None:
            return bipolar_from_key(s, dim, domain="grilly.vsa.binaryops")
        # fallback: stable u32 -> RNG (still deterministic, but depends on numpy RNG)
        if stable_u32 is None:
            seed = 0
        else:
            seed = stable_u32(s, domain="grilly.vsa.binaryops.seed")
        return BinaryOps.random_bipolar(dim, seed)


class HolographicOps:
    """
    Operations for continuous vectors using Holographic Reduced Representations (HRR).

    Uses circular convolution for binding and correlation for unbinding.
    Implemented via FFT for O(d log d) complexity.

    HRR preserves more information than binary binding but unbinding
    is approximate rather than exact.
    """

    @staticmethod
    def convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Circular convolution (binding) via frequency domain multiplication.

        Properties:
            - Commutative: convolve(a, b) == convolve(b, a)
            - Associative: convolve(convolve(a, b), c) == convolve(a, convolve(b, c))
            - Approximate inverse via correlate

        Args:
            a: First vector
            b: Second vector

        Returns:
            Convolved vector
        """
        return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b))).astype(np.float32)

    @staticmethod
    def convolve_batch(a_batch: np.ndarray, b_batch: np.ndarray) -> np.ndarray:
        """
        Batch circular convolution via frequency domain multiplication.

        Args:
            a_batch: Batch of vectors (batch, dim)
            b_batch: Batch of vectors (batch, dim)

        Returns:
            Convolved vectors (batch, dim)
        """
        a_batch = np.asarray(a_batch, dtype=np.float32)
        b_batch = np.asarray(b_batch, dtype=np.float32)

        if a_batch.shape != b_batch.shape:
            raise ValueError("Batch inputs must have the same shape")
        if a_batch.ndim != 2:
            raise ValueError("Batch inputs must be 2D (batch, dim)")

        fft_a = np.fft.fft(a_batch, axis=1)
        fft_b = np.fft.fft(b_batch, axis=1)
        return np.real(np.fft.ifft(fft_a * fft_b, axis=1)).astype(np.float32)

    @staticmethod
    def correlate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Circular correlation (unbinding) - approximate inverse of convolve.

        Given composite = convolve(x, key), correlate(composite, key) ≈ x.

        Args:
            a: The composite vector
            b: The known factor (key) to remove

        Returns:
            Approximate recovered vector
        """
        return np.real(np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b)))).astype(np.float32)

    @staticmethod
    def bundle(vectors: list[np.ndarray], normalize: bool = True) -> np.ndarray:
        """
        Bundle multiple vectors via element-wise sum.

        Args:
            vectors: List of vectors to bundle
            normalize: If True, normalize result to unit length

        Returns:
            Bundled vector
        """
        if not vectors:
            raise ValueError("Cannot bundle empty list of vectors")

        result = np.sum(vectors, axis=0).astype(np.float32)

        if normalize:
            norm = np.linalg.norm(result)
            if norm > 0:
                result = result / norm

        return result

    @staticmethod
    def bundle_batch(vectors: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Bundle multiple batches of vectors via element-wise sum.

        Args:
            vectors: Array of shape (batch, num_vectors, dim)
            normalize: If True, normalize each result to unit length

        Returns:
            Bundled vectors of shape (batch, dim)
        """
        vecs = np.asarray(vectors, dtype=np.float32)

        if vecs.ndim != 3:
            raise ValueError("vectors must have shape (batch, num_vectors, dim)")

        result = np.sum(vecs, axis=1).astype(np.float32)

        if normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            result = result / norms

        return result

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Similarity in range [-1, 1] for unit vectors
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def similarity_batch(query: np.ndarray, codebook: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between a query and a codebook.

        Args:
            query: Query vector of shape (dim,)
            codebook: Codebook vectors of shape (num_vectors, dim)

        Returns:
            Similarities of shape (num_vectors,)
        """
        query = np.asarray(query, dtype=np.float32)
        codebook = np.asarray(codebook, dtype=np.float32)

        if query.ndim != 1:
            raise ValueError("query must be 1D (dim,)")
        if codebook.ndim != 2 or codebook.shape[1] != query.shape[0]:
            raise ValueError("codebook must have shape (num_vectors, dim)")

        norm_query = np.linalg.norm(query)
        if norm_query == 0:
            return np.zeros(codebook.shape[0], dtype=np.float32)

        norms = np.linalg.norm(codebook, axis=1)
        norms = np.where(norms == 0, 1.0, norms)
        sims = (codebook @ query) / (norms * norm_query)
        return sims.astype(np.float32)

    @staticmethod
    def random_vector(dim: int, seed: int | None = None) -> np.ndarray:
        """
        Generate a random unit vector.

        NOTE: Uses a local RNG when seed is provided to avoid mutating global numpy state.
        """
        if seed is None:
            v = np.random.randn(dim).astype(np.float32)
            return v / np.linalg.norm(v)

        rng = np.random.RandomState(seed)
        v = rng.randn(dim).astype(np.float32)
        return v / np.linalg.norm(v)
