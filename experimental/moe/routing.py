"""
ResonatorMoE - Resonator-based Mixture of Experts routing.

Uses VSA operations to route queries to relevant experts.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from grilly.experimental.moe.relational import RelationalEncoder

from grilly.experimental.cognitive.capsule import CapsuleEncoder, cosine_similarity
from grilly.experimental.vsa.ops import BinaryOps


class ResonatorMoE:
    """
    Mixture of Experts using resonator-based routing.

    Routes queries to experts by computing similarity between
    query and expert vectors. Supports top-k expert selection
    and weighted combination.
    """

    def __init__(
        self,
        dim: int,
        experts: dict[str, Callable],
        expert_vectors: dict[str, np.ndarray] | None = None,
        expert_capsules: dict[str, np.ndarray] | None = None,
        capsule_encoder: CapsuleEncoder | None = None,
        capsule_weight: float = 0.3,
        vsa_backend: Any | None = None,
    ):
        """
        Initialize ResonatorMoE.

        Args:
            dim: Dimension of vectors
            experts: Dictionary mapping expert names to functions
            expert_vectors: Optional pre-computed expert vectors.
                          If None, generates random vectors for each expert.
        """
        self.dim = dim
        self.experts = experts

        # Generate or use provided expert vectors
        if expert_vectors is not None:
            self.expert_vectors = expert_vectors
        else:
            self.expert_vectors = {}
            for name in experts.keys():
                # Generate random bipolar vector for each expert
                self.expert_vectors[name] = BinaryOps.random_bipolar(dim)

        self.capsule_encoder = capsule_encoder
        if self.capsule_encoder is None and (expert_capsules is not None):
            self.capsule_encoder = CapsuleEncoder(input_dim=dim)

        self.expert_capsules = expert_capsules or {}
        if self.capsule_encoder is not None and not self.expert_capsules:
            for name, vec in self.expert_vectors.items():
                self.expert_capsules[name] = self.capsule_encoder.encode_vector(vec)

        self.capsule_weight = float(np.clip(capsule_weight, 0.0, 1.0))
        self.vsa_backend = vsa_backend
        self._expert_names = list(self.expert_vectors.keys())
        self._expert_matrix = (
            np.stack([self.expert_vectors[n] for n in self._expert_names], axis=0).astype(
                np.float32
            )
            if self._expert_names
            else None
        )

    def _combined_similarity(self, query: np.ndarray, expert_name: str) -> float:
        """Combine VSA and capsule similarity for routing."""
        expert_vec = self.expert_vectors[expert_name]
        vsa_sim = BinaryOps.similarity(query, expert_vec)

        if self.capsule_encoder is None:
            return vsa_sim

        expert_capsule = self.expert_capsules.get(expert_name)
        if expert_capsule is None:
            return vsa_sim

        query_capsule = self.capsule_encoder.encode_vector(query)
        cap_sim = cosine_similarity(query_capsule, expert_capsule)
        return (1.0 - self.capsule_weight) * vsa_sim + self.capsule_weight * cap_sim

    def route(self, query: np.ndarray, top_k: int = 1, threshold: float | None = None) -> list[str]:
        """
        Route query to top-k most similar experts.

        Args:
            query: Query vector of shape (dim,)
            top_k: Number of experts to select
            threshold: Optional minimum similarity threshold

        Returns:
            List of expert names, ordered by similarity (descending)
        """
        # Fast path: VSA-only routing on GPU (capsule_weight == 0)
        if (
            self.vsa_backend is not None
            and hasattr(self.vsa_backend, "similarity_topk")
            and self.capsule_weight <= 0.0
            and self._expert_matrix is not None
        ):
            k = int(min(top_k, len(self._expert_names)))
            idx, sims = self.vsa_backend.similarity_topk(
                query.astype(np.float32), self._expert_matrix, top_k=k
            )
            idx = idx.reshape(-1).tolist()
            sims = sims.reshape(-1).tolist()
            similarities = [(self._expert_names[i], float(s)) for i, s in zip(idx, sims)]
        else:
            # Compute similarities (CPU / combined)
            similarities = []
            for name, expert_vec in self.expert_vectors.items():
                sim = self._combined_similarity(query, name)
                similarities.append((name, sim))

        # Filter by threshold if provided
        if threshold is not None:
            similarities = [(n, s) for n, s in similarities if s >= threshold]

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k expert names
        return [name for name, _ in similarities[:top_k]]

    def get_weights(self, query: np.ndarray, normalize: bool = True) -> dict[str, float]:
        """
        Get expert weights based on query similarity.

        Args:
            query: Query vector
            normalize: If True, apply softmax normalization

        Returns:
            Dictionary mapping expert names to weights
        """
        # Compute raw similarities
        weights = {}
        for name, expert_vec in self.expert_vectors.items():
            sim = self._combined_similarity(query, name)
            # Convert similarity [-1, 1] to non-negative weight [0, 2]
            weights[name] = sim + 1.0

        # Apply softmax normalization if requested
        if normalize:
            exp_weights = {k: np.exp(v) for k, v in weights.items()}
            total = sum(exp_weights.values())
            weights = {k: v / total for k, v in exp_weights.items()}

        return weights

    def forward(self, x: np.ndarray, query: np.ndarray, top_k: int = 1) -> np.ndarray:
        """
        Forward pass through MoE: route query and apply selected experts.

        Args:
            x: Input tensor
            query: Query vector for routing
            top_k: Number of experts to use

        Returns:
            Combined expert outputs
        """
        # Route to top-k experts
        selected = self.route(query, top_k=top_k)

        if not selected:
            # No experts selected, return zeros
            return np.zeros_like(x)

        # Get weights for selected experts
        weights = self.get_weights(query, normalize=True)

        # Apply experts and combine
        outputs = []
        total_weight = 0.0

        for expert_name in selected:
            expert_fn = self.experts[expert_name]
            expert_output = expert_fn(x)
            weight = weights.get(expert_name, 0.0)

            outputs.append(expert_output * weight)
            total_weight += weight

        # Combine weighted outputs
        if total_weight > 0:
            result = sum(outputs) / total_weight
        else:
            result = sum(outputs)

        return result.astype(np.float32)

    @classmethod
    def from_realm_vectors(
        cls,
        dim: int,
        realm_expert_fns: dict[str, Callable],
        realm_vectors: dict[str, np.ndarray] | None = None,
    ) -> "ResonatorMoE":
        """
        Build a ResonatorMoE from SVC realm expert vectors.

        If ``realm_vectors`` is provided (e.g. bundled sentence prototypes
        from ``InstantLanguage.ingest_svc``), those are used as expert
        vectors.  Otherwise, deterministic hash-based bipolar vectors
        are generated for each realm name.

        Args:
            dim: Vector dimension.
            realm_expert_fns: Mapping from realm name to expert callable.
            realm_vectors: Optional pre-computed realm prototype vectors
                           (e.g. from ``SVCIngestionResult.realm_vectors``).

        Returns:
            A configured ResonatorMoE instance.
        """
        if realm_vectors is None:
            realm_vectors = {
                realm: BinaryOps.hash_to_bipolar(realm, dim) for realm in realm_expert_fns
            }
        return cls(dim=dim, experts=realm_expert_fns, expert_vectors=realm_vectors)


class RelationalMoE(ResonatorMoE):
    """
    RelationalMoE - MoE with relational expert codebook.

    Extends ResonatorMoE to use RelationalEncoder for creating
    expert vectors from relational concepts.
    """

    def __init__(
        self,
        dim: int,
        experts: dict[str, Callable],
        expert_relations: dict[str, tuple[str, str]],
        relational_encoder: Optional["RelationalEncoder"] = None,
    ):
        """
        Initialize RelationalMoE.

        Args:
            dim: Dimension of vectors
            experts: Dictionary mapping expert names to functions
            expert_relations: Dictionary mapping expert names to (source, target) tuples
                           representing the relation the expert handles
            relational_encoder: Optional RelationalEncoder instance.
                               If None, creates a new one.
        """
        from grilly.experimental.moe.relational import RelationalEncoder

        if relational_encoder is None:
            relational_encoder = RelationalEncoder(dim=dim)

        self.relational_encoder = relational_encoder
        self.expert_relations = expert_relations

        # Create expert vectors from relations
        expert_vectors = {}
        for expert_name, (source, target) in expert_relations.items():
            # Encode the target concept as the expert vector
            expert_vectors[expert_name] = relational_encoder.encode(target)

        # Initialize parent with computed expert vectors
        super().__init__(dim=dim, experts=experts, expert_vectors=expert_vectors)
