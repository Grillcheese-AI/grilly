"""
Capsule encoding utilities for cognitive modules.

Provides a lightweight capsule encoder and similarity helpers
without requiring the full capsule transformer stack.
"""

import numpy as np


class CapsuleEncoder:
    """
    Deterministic capsule encoder using a fixed random projection.

    Maps high-dimensional vectors to a lower capsule space with
    optional cognitive feature injection in the last dimensions.
    """

    def __init__(
        self, input_dim: int, capsule_dim: int = 32, semantic_dims: int = 28, seed: int = 9101
    ) -> None:
        """Initialize the instance."""

        if semantic_dims >= capsule_dim:
            raise ValueError("semantic_dims must be smaller than capsule_dim")

        self.input_dim = input_dim
        self.capsule_dim = capsule_dim
        self.semantic_dims = semantic_dims
        self.cognitive_dims = capsule_dim - semantic_dims

        rng = np.random.default_rng(seed)
        proj = rng.standard_normal((capsule_dim, input_dim)).astype(np.float32)
        norms = np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8
        self._proj = proj / norms

    def encode_vector(
        self, vec: np.ndarray, cognitive_features: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Encode a vector into capsule space.

        Args:
            vec: Input vector (input_dim,)
            cognitive_features: Optional cognitive features (cognitive_dims,)

        Returns:
            Capsule vector (capsule_dim,)
        """
        v = np.asarray(vec, dtype=np.float32)
        capsule = self._proj @ v

        if cognitive_features is not None:
            if hasattr(cognitive_features, "to_array"):
                feats = np.asarray(cognitive_features.to_array(), dtype=np.float32)
            else:
                feats = np.asarray(cognitive_features, dtype=np.float32)
            if feats.shape != (self.cognitive_dims,):
                raise ValueError("cognitive_features has incorrect shape")
            capsule[self.semantic_dims :] = feats

        semantic = capsule[: self.semantic_dims]
        norm_semantic = np.linalg.norm(semantic)
        if norm_semantic > 0:
            capsule[: self.semantic_dims] = semantic / norm_semantic

        norm_full = np.linalg.norm(capsule)
        if norm_full > 0:
            capsule = capsule / norm_full

        return capsule.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def batch_cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Batch cosine similarity between query and rows of matrix.

    Args:
        query: Query vector (dim,)
        matrix: Matrix (num_vectors, dim)

    Returns:
        Similarities (num_vectors,)
    """
    query = np.asarray(query, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)

    if query.ndim != 1 or matrix.ndim != 2:
        raise ValueError("Invalid shapes for batch cosine similarity")
    if matrix.shape[1] != query.shape[0]:
        raise ValueError("Matrix dimension does not match query size")

    norm_query = np.linalg.norm(query)
    if norm_query == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)

    norms = np.linalg.norm(matrix, axis=1)
    norms = np.where(norms == 0, 1.0, norms)
    sims = (matrix @ query) / (norms * norm_query)
    return sims.astype(np.float32)
