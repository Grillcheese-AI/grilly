"""
Resonator Network for VSA Factorization.

A resonator network iteratively factorizes a composite vector into its
constituent factors by projecting onto codebooks and using the structure
of VSA binding to separate components.

Given: composite = bind(A, bind(B, C, ...))
Find: A, B, C from their respective codebooks

Key insight: To find factor A, unbind all OTHER estimated factors from
the composite, then project onto A's codebook. Iterate until convergence.

Author: Grilly Team
Date: February 2026
"""

from typing import Any

import numpy as np

from .ops import BinaryOps


class ResonatorNetwork:
    """Resonator network for factorizing composite vectors.

    The network iteratively estimates each factor by unbinding the other
    current estimates from the composite vector and projecting the result
    onto the target factor codebook.
    """

    def __init__(
        self,
        codebooks: dict[str, np.ndarray],
        max_iterations: int = 20,
        convergence_threshold: float = 0.95,
        vsa_backend: Any | None = None,
    ):
        """
        Initialize resonator network.

        Args:
            codebooks: Dict mapping factor_name -> (num_items, dim) array
                       Each codebook contains possible values for that factor
            max_iterations: Maximum resonator iterations
            convergence_threshold: Similarity threshold for convergence

        Raises:
            ValueError: If codebooks is empty
            AssertionError: If codebooks have different dimensions
        """
        if not codebooks:
            raise ValueError("Codebooks cannot be empty")

        self.codebooks = codebooks
        self.factor_names = list(codebooks.keys())
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        # Validate dimensions
        dims = [cb.shape[-1] for cb in codebooks.values()]
        assert len(set(dims)) == 1, "All codebooks must have same dimension"
        self.dim = dims[0]
        self.vsa_backend = vsa_backend

    def factorize(
        self, composite: np.ndarray, init_estimates: dict[str, np.ndarray] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, int], int]:
        """
        Factorize composite vector into components.

        Uses multiple random restarts to avoid local minima.

        Args:
            composite: The bound composite vector to factorize
            init_estimates: Optional initial estimates for each factor

        Returns:
            Tuple of:
                - estimates: Dict of factor_name -> estimated vector
                - indices: Dict of factor_name -> index in codebook
                - iterations: Number of iterations until convergence
        """
        if init_estimates is not None:
            # Use provided initialization
            return self._factorize_single_run(composite, init_estimates)

        # Multiple restarts to find best solution
        num_restarts = 10 if len(self.factor_names) > 1 else 1
        best_score = -np.inf
        best_result = None
        best_iterations = 0

        for restart in range(num_restarts):
            # Different initialization strategies
            if restart == 0:
                # First try: best matches to composite
                init = {name: self._init_estimate(name, composite) for name in self.factor_names}
            else:
                # Other tries: random initialization from each codebook
                init = {}
                for name in self.factor_names:
                    codebook = self.codebooks[name]
                    idx = np.random.randint(len(codebook))
                    init[name] = codebook[idx].copy()

            estimates, indices, iters = self._factorize_single_run(composite, init)

            # Score this solution by how well it reconstructs the composite
            reconstructed = np.ones(self.dim, dtype=np.float32)
            for name in self.factor_names:
                reconstructed = BinaryOps.bind(reconstructed, estimates[name])

            score = BinaryOps.similarity(reconstructed, composite)

            if score > best_score:
                best_score = score
                best_result = (estimates, indices)
                best_iterations = iters

            # Early termination if we found a perfect match
            if score > 0.99:
                break

        return best_result[0], best_result[1], best_iterations

    def _factorize_single_run(
        self, composite: np.ndarray, init_estimates: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], dict[str, int], int]:
        """Single factorization run with given initialization."""
        estimates = {k: v.copy() for k, v in init_estimates.items()}

        # Resonator dynamics
        for iteration in range(self.max_iterations):
            new_estimates = {}
            converged = True

            for name in self.factor_names:
                # Unbind all OTHER factors from composite
                unbound = composite.copy()
                for other_name, other_est in estimates.items():
                    if other_name != name:
                        unbound = BinaryOps.unbind(unbound, other_est)

                # Project onto codebook (find best match)
                codebook = self.codebooks[name]

                if self.vsa_backend is not None and hasattr(self.vsa_backend, "similarity_topk"):
                    idx, _val = self.vsa_backend.similarity_topk(unbound, codebook, top_k=1)

                    best_idx = int(idx.reshape(-1)[0])

                else:
                    similarities = codebook @ unbound  # (num_items,)

                    best_idx = int(np.argmax(similarities))

                best_match = codebook[best_idx].copy()

                # Check convergence
                old_sim = BinaryOps.similarity(estimates[name], best_match)
                if old_sim < self.convergence_threshold:
                    converged = False

                new_estimates[name] = best_match

            estimates = new_estimates

            if converged:
                break

        # Get final indices
        indices = {}
        for name in self.factor_names:
            codebook = self.codebooks[name]

            if self.vsa_backend is not None and hasattr(self.vsa_backend, "similarity_topk"):
                idx, _val = self.vsa_backend.similarity_topk(estimates[name], codebook, top_k=1)

                indices[name] = int(idx.reshape(-1)[0])

            else:
                similarities = codebook @ estimates[name]

                indices[name] = int(np.argmax(similarities))

        return estimates, indices, iteration + 1

    def _init_estimate(self, name: str, composite: np.ndarray = None) -> np.ndarray:
        """
        Initialize estimate for a factor.

        If composite is provided, uses best matching codebook entry.
        Otherwise uses random codebook entry.
        """
        codebook = self.codebooks[name]

        if composite is not None:
            # Find best matching item in codebook

            if self.vsa_backend is not None and hasattr(self.vsa_backend, "similarity_topk"):
                idx, _val = self.vsa_backend.similarity_topk(composite, codebook, top_k=1)

                best_idx = int(idx.reshape(-1)[0])

            else:
                similarities = codebook @ composite

                best_idx = int(np.argmax(similarities))

            return codebook[best_idx].copy()
        else:
            idx = np.random.randint(len(codebook))
            return codebook[idx].copy()

    def factorize_partial(
        self, composite: np.ndarray, known_factors: dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Extract unknown factor when some factors are known.

        If composite = bind(A, bind(B, Z)) and we know A and B:
        Z = unbind(unbind(composite, A), B) for bipolar vectors.

        This is O(d) - instant extraction!

        Args:
            composite: The composite vector
            known_factors: Dict of factor_name -> known factor vector

        Returns:
            The recovered unknown factor
        """
        result = composite.copy()
        for factor in known_factors.values():
            result = BinaryOps.unbind(result, factor)
        return result
