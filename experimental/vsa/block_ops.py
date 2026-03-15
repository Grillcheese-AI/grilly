"""
Sparse Block Code VSA Operations.

Implements the block-structured VSA from IBM's Neuro-Vector Symbolic Architectures
(NVSA). Vectors are partitioned into k blocks of length l, where each block is a
one-hot (discrete) or probability distribution (continuous) vector.

Key advantage over flat bipolar/holographic: binding via per-block circular
convolution preserves sparsity and magnitude exactly, preventing the gradient
explosion that occurs with flat HRR or the zero-gradient problem of bipolar sign().

Reference: Hersche et al., "Neuro-Vector-Symbolic Architecture for Solving
Raven's Progressive Matrices", 2023.

Author: Grilly Team
Date: March 2026
"""

import numpy as np

EPS = 1e-20


class BlockCodeOps:
    """
    Operations for sparse block code vectors shaped (k, l) where d = k * l.

    Each block is a probability distribution (sums to ~1 for continuous codes)
    or one-hot (for discrete codes). Binding is per-block circular convolution,
    which preserves this structure exactly.

    All operations work on arrays of shape (..., k, l) where leading dims are batch.
    """

    # ── Construction ──────────────────────────────────────────────

    @staticmethod
    def random_discrete(k: int, l: int, seed: int | None = None) -> np.ndarray:
        """
        Generate a random discrete block code vector (one-hot per block).

        Args:
            k: Number of blocks
            l: Length of each block
            seed: Optional RNG seed

        Returns:
            Block code vector of shape (k, l) with exactly one 1.0 per block
        """
        rng = np.random.default_rng(seed)
        v = np.zeros((k, l), dtype=np.float32)
        for block in range(k):
            v[block, rng.integers(0, l)] = 1.0
        return v

    @staticmethod
    def random_continuous(k: int, l: int, seed: int | None = None) -> np.ndarray:
        """
        Generate a random continuous block code vector (softmax per block).

        Args:
            k: Number of blocks
            l: Length of each block
            seed: Optional RNG seed

        Returns:
            Block code vector of shape (k, l) with each block summing to ~1
        """
        rng = np.random.default_rng(seed)
        logits = rng.standard_normal((k, l)).astype(np.float32)
        # softmax per block
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    @staticmethod
    def zero_vector(k: int, l: int) -> np.ndarray:
        """
        Generate the zero/identity element: first position hot in each block.

        Binding any vector with the zero vector returns the original.

        Args:
            k: Number of blocks
            l: Length of each block

        Returns:
            Identity block code of shape (k, l)
        """
        v = np.zeros((k, l), dtype=np.float32)
        v[:, 0] = 1.0
        return v

    @staticmethod
    def codebook_discrete(
        k: int, l: int, n: int, seed: int | None = None, orthogonal: bool = True
    ) -> np.ndarray:
        """
        Generate a codebook of n discrete block code vectors.

        If orthogonal=True (default), generates via successive binding from a
        seed element, ensuring algebraic closure and collision-free codes
        (IBM NVSA method). Falls back to random sampling if n > l (block length).

        Args:
            k: Number of blocks
            l: Length of each block
            n: Number of codebook entries
            seed: Optional RNG seed
            orthogonal: If True, use successive binding for orthogonality

        Returns:
            Codebook of shape (n, k, l)
        """
        rng = np.random.default_rng(seed)
        codebook = np.zeros((n, k, l), dtype=np.float32)

        if orthogonal and n <= l:
            # IBM method: zero vector + seed + successive bindings
            # Entry 0: identity (first position hot)
            codebook[0, :, 0] = 1.0

            if n > 1:
                # Entry 1: random one-hot per block (not position 0)
                for block in range(k):
                    idx = rng.integers(1, l)
                    codebook[1, block, idx] = 1.0

                # Remaining entries: successive binding with entry 1
                for i in range(2, n):
                    codebook[i] = BlockCodeOps.bind(codebook[i - 1], codebook[1])
        else:
            # Random sampling (may have collisions for large n)
            for i in range(n):
                for block in range(k):
                    codebook[i, block, rng.integers(0, l)] = 1.0

        return codebook

    # ── Binding (per-block circular convolution) ──────────────────

    @staticmethod
    def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Bind two block code vectors via per-block circular convolution.

        This is the key difference from flat HRR: convolution is done independently
        within each block, preserving sparsity. For one-hot inputs, the result is
        exactly one-hot (a circular shift).

        Args:
            a: Block code vector (..., k, l)
            b: Block code vector (..., k, l)

        Returns:
            Bound block code vector (..., k, l)
        """
        # Per-block circular convolution via FFT
        fft_a = np.fft.fft(a, axis=-1)
        fft_b = np.fft.fft(b, axis=-1)
        return np.real(np.fft.ifft(fft_a * fft_b, axis=-1)).astype(np.float32)

    @staticmethod
    def unbind(composite: np.ndarray, known: np.ndarray) -> np.ndarray:
        """
        Unbind a known vector from a composite via per-block circular correlation.

        Given composite = bind(x, key), unbind(composite, key) recovers x.
        For discrete (one-hot) inputs, recovery is exact.

        Args:
            composite: The composite block code vector (..., k, l)
            known: The known factor to remove (..., k, l)

        Returns:
            Recovered block code vector (..., k, l)
        """
        fft_c = np.fft.fft(composite, axis=-1)
        fft_k = np.fft.fft(known, axis=-1)
        return np.real(np.fft.ifft(fft_c * np.conj(fft_k), axis=-1)).astype(
            np.float32
        )

    @staticmethod
    def bind3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Bind three vectors: bind(bind(a, b), c)."""
        return BlockCodeOps.bind(BlockCodeOps.bind(a, b), c)

    @staticmethod
    def unbind3(
        composite: np.ndarray, known1: np.ndarray, known2: np.ndarray
    ) -> np.ndarray:
        """Unbind two known vectors: unbind(unbind(composite, known1), known2)."""
        return BlockCodeOps.unbind(BlockCodeOps.unbind(composite, known1), known2)

    # ── Bundling (superposition) ──────────────────────────────────

    @staticmethod
    def bundle(
        vectors: list[np.ndarray], normalize: bool = True
    ) -> np.ndarray:
        """
        Bundle multiple block code vectors via element-wise sum.

        Unlike bipolar majority voting, block code bundling preserves
        the probability structure: each block remains a valid distribution
        after normalization.

        Args:
            vectors: List of block code vectors, each (..., k, l)
            normalize: If True, normalize each block to sum to 1

        Returns:
            Bundled block code vector (..., k, l)
        """
        if not vectors:
            raise ValueError("Cannot bundle empty list of vectors")

        result = np.sum(vectors, axis=0).astype(np.float32)

        if normalize:
            result = BlockCodeOps._normalize_blocks(result)

        return result

    # ── Similarity ────────────────────────────────────────────────

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute similarity between two block code vectors.

        Uses the IBM NVSA formula: (1/k) * sum(a * b) over all blocks.
        For one-hot discrete codes, this equals the fraction of matching blocks.

        Args:
            a: Block code vector (k, l)
            b: Block code vector (k, l)

        Returns:
            Similarity in range [0, 1] for properly normalized block codes
        """
        k = a.shape[-2]
        return float(np.sum(a * b) / k)

    @staticmethod
    def similarity_batch(
        query: np.ndarray, codebook: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity between a query and all codebook entries.

        Args:
            query: Query vector (k, l)
            codebook: Codebook (n, k, l)

        Returns:
            Similarities (n,)
        """
        k = query.shape[-2]
        # Flatten to (k*l,) and (n, k*l) for matmul
        q_flat = query.reshape(-1)
        cb_flat = codebook.reshape(codebook.shape[0], -1)
        return (cb_flat @ q_flat / k).astype(np.float32)

    @staticmethod
    def similarity_zero(a: np.ndarray) -> float:
        """
        Compute similarity between a block code vector and the zero vector.

        The zero vector has position 0 hot in each block, so this is
        just the sum of first elements divided by k.

        Args:
            a: Block code vector (k, l)

        Returns:
            Similarity to zero vector
        """
        k = a.shape[-2]
        return float(np.sum(a[..., 0]) / k)

    # ── Probability space operations ──────────────────────────────

    @staticmethod
    def cosine_to_pmf(
        similarities: np.ndarray,
        temperature: float = 40.0,
        mode: str = "softmax",
    ) -> np.ndarray:
        """
        Convert similarity scores to a probability distribution.

        This is IBM's cosine2pmf — the critical step that keeps everything
        bounded and differentiable.

        Args:
            similarities: Raw similarity scores (n,)
            temperature: Softmax temperature (higher = sharper). Default 40.
            mode: 'softmax' or 'normalize' (L1 normalization)

        Returns:
            Probability distribution (n,) summing to 1
        """
        s = np.asarray(similarities, dtype=np.float64)  # fp64 for stability

        if mode == "softmax":
            scaled = s * temperature
            scaled -= scaled.max()  # numerical stability
            exp = np.exp(scaled)
            return (exp / exp.sum()).astype(np.float32)
        elif mode == "normalize":
            abs_s = np.abs(s)
            total = abs_s.sum()
            if total == 0:
                return np.ones_like(s, dtype=np.float32) / len(s)
            return (abs_s / total).astype(np.float32)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def pmf_to_vector(
        codebook: np.ndarray, pmf: np.ndarray
    ) -> np.ndarray:
        """
        Convert a probability distribution over codebook entries back to a vector.

        This is IBM's pmf2vec — weighted sum of codebook entries.

        Args:
            codebook: Codebook (n, k, l)
            pmf: Probability distribution (n,) or (batch, n)

        Returns:
            Weighted vector (k, l) or (batch, k, l)
        """
        n, k, l = codebook.shape
        pmf = np.asarray(pmf, dtype=np.float32)

        if pmf.ndim == 1:
            # (n,) @ (n, k*l) -> (k*l,) -> (k, l)
            cb_flat = codebook.reshape(n, -1)
            return (pmf @ cb_flat).reshape(k, l).astype(np.float32)
        else:
            # (batch, n) @ (n, k*l) -> (batch, k*l) -> (batch, k, l)
            cb_flat = codebook.reshape(n, -1)
            return (pmf @ cb_flat).reshape(-1, k, l).astype(np.float32)

    @staticmethod
    def project(
        query: np.ndarray,
        codebook: np.ndarray,
        temperature: float = 40.0,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Project a query onto a codebook: similarity -> softmax -> weighted sum.

        Full IBM pipeline: query -> similarities -> pmf -> vector.
        Returns the projected vector, the pmf, and the best-match index.

        Args:
            query: Query vector (k, l)
            codebook: Codebook (n, k, l)
            temperature: Softmax temperature

        Returns:
            (projected_vector, pmf, best_index)
        """
        sims = BlockCodeOps.similarity_batch(query, codebook)
        pmf = BlockCodeOps.cosine_to_pmf(sims, temperature=temperature)
        projected = BlockCodeOps.pmf_to_vector(codebook, pmf)
        best_idx = int(np.argmax(sims))
        return projected, pmf, best_idx

    # ── Positional encoding ───────────────────────────────────────

    @staticmethod
    def cyclic_shift(a: np.ndarray, n: int) -> np.ndarray:
        """
        Blockwise cyclic shift — shifts which block maps to which position.

        This is the block code equivalent of positional encoding: each position
        in a sequence gets a different cyclic shift, making the representation
        position-dependent.

        Args:
            a: Block code vector (..., k, l)
            n: Number of blocks to shift

        Returns:
            Shifted vector (..., k, l)
        """
        return np.roll(a, n, axis=-2)

    # ── Discretization ────────────────────────────────────────────

    @staticmethod
    def discretize(a: np.ndarray) -> np.ndarray:
        """
        Discretize a continuous block code to one-hot (argmax per block).

        Useful after bundling or unbinding to snap back to valid discrete codes.

        Args:
            a: Continuous block code vector (..., k, l)

        Returns:
            Discrete (one-hot) block code vector (..., k, l)
        """
        result = np.zeros_like(a)
        max_indices = np.argmax(a, axis=-1)
        # Handle arbitrary leading dimensions
        for idx in np.ndindex(a.shape[:-1]):
            result[idx + (max_indices[idx],)] = 1.0
        return result

    # ── Conversion helpers ────────────────────────────────────────

    @staticmethod
    def from_flat(v: np.ndarray, k: int) -> np.ndarray:
        """
        Reshape a flat vector (d,) to block code format (k, l).

        Args:
            v: Flat vector of dimension d = k * l
            k: Number of blocks

        Returns:
            Block code vector (k, l)
        """
        d = v.shape[-1]
        if d % k != 0:
            raise ValueError(f"Dimension {d} not divisible by k={k}")
        l = d // k
        return v.reshape(*v.shape[:-1], k, l)

    @staticmethod
    def to_flat(a: np.ndarray) -> np.ndarray:
        """
        Flatten a block code vector (k, l) to flat (d,) format.

        Args:
            a: Block code vector (..., k, l)

        Returns:
            Flat vector (..., d) where d = k * l
        """
        return a.reshape(*a.shape[:-2], -1)

    # ── Internal ──────────────────────────────────────────────────

    @staticmethod
    def _normalize_blocks(a: np.ndarray) -> np.ndarray:
        """Normalize each block to sum to 1 (L1 normalization per block)."""
        block_sums = a.sum(axis=-1, keepdims=True)
        block_sums = np.where(block_sums == 0, 1.0, block_sums)
        return (a / block_sums).astype(np.float32)
