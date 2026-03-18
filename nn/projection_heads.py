"""Multi-head subspace projection for structured embeddings.

Projects a base model's hidden state through specialized linear heads,
then concatenates to create a structured embedding with named subspaces.

Each subspace captures a different aspect of the input:
- semantic: general meaning (largest, most information)
- emotion: affective content
- intent: action/purpose
- context: syntactic/positional information

Subspaces can be queried independently for specialized retrieval.
"""

import numpy as np

from .linear import Linear
from .module import Module


class ProjectionHeads(Module):
    """Multi-head projection from base model hidden state to structured embedding.

    Args:
        input_dim: Base model hidden dimension (e.g., 768 for BERT)
        head_configs: dict mapping head_name -> output_dim
            Default: {"semantic": 256, "emotion": 128, "intent": 128, "context": 128}

    Usage:
        heads = ProjectionHeads(768)
        hidden = model(tokens)  # (batch, seq, 768)
        pooled = hidden.mean(axis=1)  # (batch, 768) mean pool
        structured = heads(pooled)  # (batch, 640)

        # Query specific subspace
        semantic_emb = heads.get_subspace(structured, "semantic")  # (batch, 256)
    """

    def __init__(self, input_dim, head_configs=None):
        """Initialize projection heads.

        Args:
            input_dim: Input feature dimension from base model hidden state.
            head_configs: dict mapping head name to output dimension.
        """
        super().__init__()
        if head_configs is None:
            head_configs = {
                "semantic": 256,
                "emotion": 128,
                "intent": 128,
                "context": 128,
            }
        self.input_dim = input_dim
        self.head_configs = head_configs
        self.output_dim = sum(head_configs.values())

        # Create projection heads and track byte offsets into the concat output
        self.heads = {}
        self._offsets = {}
        offset = 0
        for name, dim in head_configs.items():
            head = Linear(input_dim, dim)
            self.heads[name] = head
            self._modules[name] = head
            self._offsets[name] = (offset, offset + dim)
            offset += dim

    def forward(self, x):
        """Project input through all heads and concatenate.

        Args:
            x: (batch, input_dim) or (input_dim,) numpy array.

        Returns:
            (batch, output_dim) or (output_dim,) structured embedding.
        """
        parts = [self.heads[name](x) for name in self.head_configs]
        return np.concatenate(parts, axis=-1)

    def get_subspace(self, structured_embedding, head_name):
        """Extract a specific subspace from a structured embedding.

        Args:
            structured_embedding: (..., output_dim) array from forward().
            head_name: name of the head ("semantic", "emotion", etc.)

        Returns:
            (..., head_dim) subspace embedding.

        Raises:
            KeyError: If head_name is not a registered head.
        """
        if head_name not in self._offsets:
            raise KeyError(
                f"Unknown head '{head_name}'. Available heads: {list(self._offsets.keys())}"
            )
        start, end = self._offsets[head_name]
        return structured_embedding[..., start:end]

    def get_all_subspaces(self, structured_embedding):
        """Extract all subspaces as a dict.

        Args:
            structured_embedding: (..., output_dim) array from forward().

        Returns:
            dict mapping head name -> subspace array.
        """
        return {name: self.get_subspace(structured_embedding, name) for name in self.head_configs}

    def subspace_similarity(self, a, b, head_name):
        """Cosine similarity between two structured embeddings in a specific subspace.

        Args:
            a: (..., output_dim) structured embedding.
            b: (..., output_dim) structured embedding.
            head_name: which subspace to compare in.

        Returns:
            (...,) cosine similarity values in [-1, 1].
        """
        sa = self.get_subspace(a, head_name)
        sb = self.get_subspace(b, head_name)
        dot = np.sum(sa * sb, axis=-1)
        norm = np.linalg.norm(sa, axis=-1) * np.linalg.norm(sb, axis=-1)
        return dot / np.maximum(norm, 1e-8)

    def __repr__(self):
        """Return a human-readable summary of the projection heads."""
        heads_str = ", ".join(f"{k}={v}" for k, v in self.head_configs.items())
        return f"ProjectionHeads({self.input_dim} -> {self.output_dim}: {heads_str})"
