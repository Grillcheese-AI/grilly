"""Capsule embedding modules for semantic and cognitive representations."""

import numpy as np

from .capsule import CapsuleProject, DentateGyrus
from .module import Module
from .modules import LayerNorm


class CapsuleEmbedding(Module):
    """Capsule embedding pipeline from dense embeddings to compact capsules."""

    def __init__(
        self,
        embedding_dim: int = 384,
        capsule_dim: int = 32,
        semantic_dims: int = 28,
        use_dg: bool = False,
        dg_dim: int = 128,
        dg_sparsity: float = 0.02,
    ):
        """
        Initialize CapsuleEmbedding.

        Args:
            embedding_dim: Input embedding dimension (default: 384)
            capsule_dim: Capsule dimension (default: 32)
            semantic_dims: Number of semantic dimensions (default: 28)
            use_dg: Whether to use DentateGyrus expansion (default: False)
            dg_dim: DentateGyrus output dimension (default: 128)
            dg_sparsity: DentateGyrus sparsity level (default: 0.02)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.capsule_dim = capsule_dim
        self.semantic_dims = semantic_dims
        self.use_dg = use_dg

        # Capsule projection: embedding_dim → capsule_dim
        self.capsule_proj = CapsuleProject(embedding_dim, capsule_dim)
        self._modules["capsule_proj"] = self.capsule_proj

        # Optional DentateGyrus expansion
        if use_dg:
            self.dg = DentateGyrus(capsule_dim, dg_dim, dg_sparsity)
            self._modules["dg"] = self.dg
        else:
            self.dg = None

        # Layer normalization for stability
        self.norm = LayerNorm(capsule_dim)
        self._modules["norm"] = self.norm

    def forward(
        self, embeddings: np.ndarray, cognitive_features: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Forward pass - encode embeddings to capsule space.

        Args:
            embeddings: Input embeddings (batch, embedding_dim) or (batch, seq_len, embedding_dim)
            cognitive_features: Optional cognitive features (batch, 4) for last 4 dims

        Returns:
            Capsule vectors (batch, capsule_dim) or (batch, seq_len, capsule_dim)
            If use_dg=True: (batch, dg_dim) or (batch, seq_len, dg_dim)
        """
        # Project to capsule space
        capsule = self.capsule_proj(embeddings)  # (batch, capsule_dim)

        # Normalize semantic portion (first semantic_dims)
        if capsule.ndim == 2:
            semantic = capsule[:, : self.semantic_dims]
            norms = np.linalg.norm(semantic, axis=1, keepdims=True) + 1e-8
            capsule[:, : self.semantic_dims] = semantic / norms
        else:
            # (batch, seq_len, capsule_dim)
            semantic = capsule[:, :, : self.semantic_dims]
            norms = np.linalg.norm(semantic, axis=2, keepdims=True) + 1e-8
            capsule[:, :, : self.semantic_dims] = semantic / norms

        # Inject cognitive features if provided
        if cognitive_features is not None:
            if capsule.ndim == 2:
                capsule[:, self.semantic_dims :] = cognitive_features
            else:
                capsule[:, :, self.semantic_dims :] = cognitive_features[:, None, :]

        # Normalize entire capsule
        capsule_normalized = self.norm(capsule)
        self._cached_normalized = capsule_normalized  # Cache for backward

        # Optional DentateGyrus expansion
        if self.use_dg and self.dg is not None:
            capsule_normalized = self.dg(capsule_normalized)

        return capsule_normalized

    def backward(self, grad_output: np.ndarray, embeddings: np.ndarray = None) -> np.ndarray:
        """
        Backward pass for CapsuleEmbedding.

        Args:
            grad_output: Gradient w.r.t. output
            embeddings: Input embeddings (optional, uses cached if not provided)

        Returns:
            grad_input: Gradient w.r.t. input embeddings
        """
        # Backprop through DentateGyrus if used
        if self.use_dg and self.dg is not None:
            grad_capsule = self.dg.backward(grad_output)
        else:
            grad_capsule = grad_output

        # Backprop through normalization
        # LayerNorm backward needs the normalized input
        normalized_capsule = getattr(self, "_cached_normalized", None)
        if hasattr(self.norm, "backward") and normalized_capsule is not None:
            try:
                grad_capsule = self.norm.backward(grad_capsule, normalized_capsule)
            except Exception:
                # Fallback if backward signature doesn't match
                pass

        # Backprop through capsule projection
        grad_input = self.capsule_proj.backward(grad_capsule)

        return grad_input

    def __repr__(self):
        """Return a debug representation."""

        dg_str = f", dg_dim={self.dg.out_dim}" if self.use_dg and self.dg is not None else ""
        return f"CapsuleEmbedding(embedding_dim={self.embedding_dim}, capsule_dim={self.capsule_dim}{dg_str})"


class ContrastiveLoss(Module):
    """
    Contrastive Loss for training capsule embeddings.

    Uses triplet loss: max(0, margin + pos_distance - neg_distance)
    Or cosine similarity loss for positive/negative pairs.
    """

    def __init__(self, margin: float = 0.5, distance_metric: str = "cosine"):
        """
        Initialize ContrastiveLoss.

        Args:
            margin: Margin for triplet loss (default: 0.5)
            distance_metric: 'cosine' or 'euclidean' (default: 'cosine')
        """
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        self._cached_anchor = None
        self._cached_positive = None
        self._cached_negative = None

    def forward(
        self, anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Compute contrastive loss.

        Args:
            anchor: Anchor embeddings (batch, dim)
            positive: Positive embeddings (batch, dim) - should be similar to anchor
            negative: Optional negative embeddings (batch, dim) - should be dissimilar

        Returns:
            Loss value (scalar)
        """
        self._cached_anchor = anchor
        self._cached_positive = positive
        self._cached_negative = negative

        # Normalize embeddings (for cosine similarity)
        anchor_norm = anchor / (np.linalg.norm(anchor, axis=1, keepdims=True) + 1e-8)
        positive_norm = positive / (np.linalg.norm(positive, axis=1, keepdims=True) + 1e-8)

        if self.distance_metric == "cosine":
            # Cosine distance = 1 - cosine similarity
            pos_sim = np.sum(anchor_norm * positive_norm, axis=1)  # (batch,)
            pos_dist = 1.0 - pos_sim
        else:
            # Euclidean distance
            pos_dist = np.linalg.norm(anchor - positive, axis=1)  # (batch,)

        if negative is not None:
            negative_norm = negative / (np.linalg.norm(negative, axis=1, keepdims=True) + 1e-8)

            if self.distance_metric == "cosine":
                neg_sim = np.sum(anchor_norm * negative_norm, axis=1)  # (batch,)
                neg_dist = 1.0 - neg_sim
            else:
                neg_dist = np.linalg.norm(anchor - negative, axis=1)  # (batch,)

            # Triplet loss: max(0, margin + pos_dist - neg_dist)
            loss = np.maximum(0.0, self.margin + pos_dist - neg_dist)
        else:
            # Positive loss only (pull similar pairs together)
            loss = pos_dist**2

        return np.mean(loss)

    def backward(
        self,
        grad_output: np.ndarray = None,
        anchor: np.ndarray = None,
        positive: np.ndarray = None,
        negative: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Backward pass for ContrastiveLoss.

        Args:
            grad_output: Gradient w.r.t. loss (usually 1.0)
            anchor: Anchor embeddings (optional, uses cached)
            positive: Positive embeddings (optional, uses cached)
            negative: Negative embeddings (optional, uses cached)

        Returns:
            (grad_anchor, grad_positive, grad_negative)
        """
        if grad_output is None:
            grad_output = 1.0

        if anchor is None:
            anchor = self._cached_anchor
        if positive is None:
            positive = self._cached_positive
        if negative is None:
            negative = self._cached_negative

        if anchor is None or positive is None:
            raise ValueError("anchor and positive must be provided")

        batch_size = anchor.shape[0]

        # Normalize embeddings
        anchor_norm = anchor / (np.linalg.norm(anchor, axis=1, keepdims=True) + 1e-8)
        positive_norm = positive / (np.linalg.norm(positive, axis=1, keepdims=True) + 1e-8)

        if self.distance_metric == "cosine":
            pos_sim = np.sum(anchor_norm * positive_norm, axis=1, keepdims=True)  # (batch, 1)

            if negative is not None:
                negative_norm = negative / (np.linalg.norm(negative, axis=1, keepdims=True) + 1e-8)
                neg_sim = np.sum(anchor_norm * negative_norm, axis=1, keepdims=True)  # (batch, 1)

                # Triplet loss gradients
                pos_dist = 1.0 - pos_sim
                neg_dist = 1.0 - neg_sim
                loss_active = (self.margin + pos_dist - neg_dist > 0.0).astype(np.float32)

                # Gradient w.r.t. anchor
                grad_anchor = (positive_norm - negative_norm) * loss_active / batch_size
                # Gradient w.r.t. positive
                grad_positive = -anchor_norm * loss_active / batch_size
                # Gradient w.r.t. negative
                grad_negative = anchor_norm * loss_active / batch_size
            else:
                # Positive loss only
                grad_anchor = 2.0 * (anchor_norm - positive_norm) / batch_size
                grad_positive = 2.0 * (positive_norm - anchor_norm) / batch_size
                grad_negative = None
        else:
            # Euclidean distance gradients
            if negative is not None:
                pos_dist = np.linalg.norm(anchor - positive, axis=1, keepdims=True)
                neg_dist = np.linalg.norm(anchor - negative, axis=1, keepdims=True)
                loss_active = (self.margin + pos_dist - neg_dist > 0.0).astype(np.float32)

                # Gradient w.r.t. anchor
                grad_anchor = (
                    (
                        (anchor - positive) / (pos_dist + 1e-8)
                        - (anchor - negative) / (neg_dist + 1e-8)
                    )
                    * loss_active
                    / batch_size
                )
                # Gradient w.r.t. positive
                grad_positive = -(anchor - positive) / (pos_dist + 1e-8) * loss_active / batch_size
                # Gradient w.r.t. negative
                grad_negative = (anchor - negative) / (neg_dist + 1e-8) * loss_active / batch_size
            else:
                # Positive loss only
                diff = anchor - positive
                grad_anchor = 2.0 * diff / batch_size
                grad_positive = -2.0 * diff / batch_size
                grad_negative = None

        return (
            grad_anchor * grad_output,
            grad_positive * grad_output,
            grad_negative * grad_output if grad_negative is not None else None,
        )

    def __repr__(self):
        """Return a debug representation."""

        return f"ContrastiveLoss(margin={self.margin}, distance={self.distance_metric})"
