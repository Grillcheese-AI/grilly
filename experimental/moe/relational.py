"""
RelationalEncoder - Encode entities and relations using VSA.

Provides deterministic encoding of concepts with support for:
- Modality tags (text, image, audio, etc.)
- Polarity (positive/negative)
- Relation extraction (analogies, transformations)
"""

import hashlib

import numpy as np

from grilly.experimental.vsa.ops import BinaryOps


class RelationalEncoder:
    """
    Encodes entities and relations as high-dimensional vectors.

    Uses hash-based deterministic encoding to ensure same input
    produces same vector. Supports modality and polarity tags.
    """

    DEFAULT_DIM = 1024

    def __init__(self, dim: int = DEFAULT_DIM):
        """
        Initialize RelationalEncoder.

        Args:
            dim: Dimension of encoded vectors (default: 1024)
        """
        self.dim = dim
        self._cache: dict[str, np.ndarray] = {}

    def encode(
        self, concept: str, modality: str | None = None, polarity: str | None = None
    ) -> np.ndarray:
        """
        Encode a concept as a bipolar vector.

        Args:
            concept: Concept name to encode
            modality: Optional modality tag (e.g., "text", "image")
            polarity: Optional polarity ("positive" or "negative")

        Returns:
            Bipolar vector encoding the concept
        """
        # Create cache key
        key_parts = [concept]
        if modality:
            key_parts.append(f"mod:{modality}")
        if polarity:
            key_parts.append(f"pol:{polarity}")
        cache_key = "|".join(key_parts)

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Hash-based deterministic encoding
        hash_input = cache_key.encode("utf-8")
        hash_obj = hashlib.sha256(hash_input)
        hash_bytes = hash_obj.digest()

        # Generate bipolar vector from hash
        vector = np.zeros(self.dim, dtype=np.float32)

        # Use hash bytes to set vector elements
        for i in range(self.dim):
            byte_idx = i % len(hash_bytes)
            bit = (hash_bytes[byte_idx] >> (i % 8)) & 1
            vector[i] = 1.0 if bit else -1.0

        # Apply polarity flip if negative
        if polarity == "negative":
            vector = -vector

        self._cache[cache_key] = vector
        return vector

    def get_opposite(self, vector: np.ndarray) -> np.ndarray:
        """
        Get the opposite/negated version of a vector.

        For bipolar vectors, this is simply negation.

        Args:
            vector: Input vector

        Returns:
            Negated vector
        """
        return -vector

    def extract_relation(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Extract the transformation/relation from vector A to vector B.

        For bipolar vectors: relation = bind(a, b) = a * b

        Args:
            a: Source vector
            b: Target vector

        Returns:
            Relation vector representing transformation A→B
        """
        return BinaryOps.bind(a, b)

    def apply_relation(self, vector: np.ndarray, relation: np.ndarray) -> np.ndarray:
        """
        Apply a relation to a vector.

        For bipolar vectors: result = bind(vector, relation)

        Args:
            vector: Input vector
            relation: Relation to apply

        Returns:
            Transformed vector
        """
        return BinaryOps.bind(vector, relation)

    def encode_batch(self, concepts: list[str]) -> np.ndarray:
        """
        Encode multiple concepts efficiently.

        Args:
            concepts: List of concept names

        Returns:
            Array of shape (len(concepts), dim)
        """
        return np.array([self.encode(c) for c in concepts])

    def similarity_batch(self, query: np.ndarray, codebook: np.ndarray) -> np.ndarray:
        """
        Compute similarities between query and codebook vectors.

        Args:
            query: Query vector of shape (dim,)
            codebook: Codebook vectors of shape (n, dim)

        Returns:
            Similarities of shape (n,)
        """
        similarities = []
        for i in range(len(codebook)):
            sim = BinaryOps.similarity(query, codebook[i])
            similarities.append(sim)
        return np.array(similarities)
