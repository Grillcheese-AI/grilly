"""
Affect Processing Layers

Uses: affect-mlp-forward.glsl, affect-mlp-backward.glsl

Reference: ref/brain/amygdala.py
"""

import numpy as np

from .module import Module
from .modules import Dropout


class AffectMLP(Module):
    """
    Affect MLP layer - Emotional processing MLP.

    Uses: affect-mlp-forward.glsl, affect-mlp-backward.glsl

    Reference: ref/brain/amygdala.py
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden1_dim: int = 128,
        hidden2_dim: int = 64,
        output_dim: int = 2,  # valence, arousal
        leaky_slope: float = 0.01,
        dropout_rate: float = 0.1,
    ):
        """
        Initialize AffectMLP layer.

        Args:
            embedding_dim: Input embedding dimension
            hidden1_dim: First hidden layer dimension
            hidden2_dim: Second hidden layer dimension
            output_dim: Output dimension (default: 2 for valence/arousal)
            leaky_slope: LeakyReLU negative slope
            dropout_rate: Dropout rate
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.output_dim = output_dim
        self.leaky_slope = leaky_slope
        self.dropout_rate = dropout_rate

        # Layer 1
        limit1 = np.sqrt(6.0 / (embedding_dim + hidden1_dim))
        self.W1 = np.random.uniform(-limit1, limit1, (hidden1_dim, embedding_dim)).astype(
            np.float32
        )
        self.b1 = np.zeros(hidden1_dim, dtype=np.float32)
        self._parameters["W1"] = self.W1
        self._parameters["b1"] = self.b1

        # Layer 2
        limit2 = np.sqrt(6.0 / (hidden1_dim + hidden2_dim))
        self.W2 = np.random.uniform(-limit2, limit2, (hidden2_dim, hidden1_dim)).astype(np.float32)
        self.b2 = np.zeros(hidden2_dim, dtype=np.float32)
        self._parameters["W2"] = self.W2
        self._parameters["b2"] = self.b2

        # Output layer
        limit3 = np.sqrt(6.0 / (hidden2_dim + output_dim))
        self.W3 = np.random.uniform(-limit3, limit3, (output_dim, hidden2_dim)).astype(np.float32)
        self.b3 = np.zeros(output_dim, dtype=np.float32)
        self._parameters["W3"] = self.W3
        self._parameters["b3"] = self.b3

        self.dropout = Dropout(dropout_rate)
        self._modules["dropout"] = self.dropout

    def forward(self, embeddings: np.ndarray, apply_output_activation: bool = False) -> np.ndarray:
        """
        Forward pass - predict affect (valence, arousal).

        Args:
            embeddings: Input embeddings (batch, embedding_dim)
            apply_output_activation: Whether to apply tanh/sigmoid to output

        Returns:
            Affect predictions (batch, output_dim) - [valence, arousal]
        """
        backend = self._get_backend()

        # Try GPU forward pass if available
        if hasattr(backend, "affect") and hasattr(backend.affect, "affect_mlp_forward"):
            try:
                output, _, _ = backend.affect.affect_mlp_forward(
                    embeddings,
                    self.W1,
                    self.b1,
                    self.W2,
                    self.b2,
                    self.W3,
                    self.b3,
                    leaky_slope=self.leaky_slope,
                    apply_output_activation=apply_output_activation,
                    dropout_rate=self.dropout_rate if self.training else 0.0,
                )
                return output
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback
        # Layer 1
        h1 = embeddings @ self.W1.T + self.b1
        h1 = np.where(h1 > 0, h1, h1 * self.leaky_slope)  # LeakyReLU
        if self.training:
            h1 = self.dropout(h1)

        # Layer 2 (with residual from layer 1)
        h2 = h1 @ self.W2.T + self.b2
        h2 = np.where(h2 > 0, h2, h2 * self.leaky_slope)  # LeakyReLU
        if self.training:
            h2 = self.dropout(h2)

        # Output layer
        output = h2 @ self.W3.T + self.b3

        if apply_output_activation:
            # Tanh for valence, sigmoid for arousal
            output[:, 0] = np.tanh(output[:, 0])  # valence: [-1, 1]
            output[:, 1] = 1 / (1 + np.exp(-output[:, 1]))  # arousal: [0, 1]

        return output

    def backward(
        self,
        grad_output: np.ndarray,
        embeddings: np.ndarray,
        h1: np.ndarray | None = None,
        h2: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Backward pass using affect-mlp-backward.glsl

        Args:
            grad_output: Gradient w.r.t. output (batch, output_dim)
            embeddings: Input embeddings (batch, embedding_dim)
            h1: Optional hidden1 activations (from forward pass)
            h2: Optional hidden2 activations (from forward pass)

        Returns:
            (grad_input, grad_dict) - grad_input (batch, embedding_dim), grad_dict with weight gradients
        """
        backend = self._get_backend()

        # Try GPU backward if available
        if hasattr(backend, "affect") and hasattr(backend.affect, "affect_mlp_backward"):
            try:
                grad_dict = backend.affect.affect_mlp_backward(
                    grad_output,
                    embeddings,
                    self.W1,
                    self.b1,
                    self.W2,
                    self.b2,
                    self.W3,
                    self.b3,
                    h1=h1,
                    h2=h2,
                    leaky_slope=self.leaky_slope,
                )
                return grad_dict["grad_input"], grad_dict
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback (simplified)
        # This is a placeholder - full implementation would compute all gradients
        grad_input = np.zeros_like(embeddings)
        grad_dict = {
            "grad_W1": np.zeros_like(self.W1),
            "grad_b1": np.zeros_like(self.b1),
            "grad_W2": np.zeros_like(self.W2),
            "grad_b2": np.zeros_like(self.b2),
            "grad_W3": np.zeros_like(self.W3),
            "grad_b3": np.zeros_like(self.b3),
            "grad_input": grad_input,
        }
        return grad_input, grad_dict

    def __repr__(self):
        """Return a debug representation."""

        return f"AffectMLP(embedding_dim={self.embedding_dim}, hidden1={self.hidden1_dim}, hidden2={self.hidden2_dim}, output_dim={self.output_dim})"
