"""
LoRA (Low-Rank Adaptation) Module for Grilly

This module provides efficient fine-tuning of large language models by adding
low-rank decomposition matrices to existing weights. Instead of updating the
full weight matrix W (d_out x d_in), we add ΔW = B @ A where:
- A is (rank x d_in)
- B is (d_out x rank)
- rank << d_in, d_out

This reduces trainable parameters dramatically while maintaining model quality.

Usage:
    from grilly.nn.lora import LoRALinear, LoRAConfig, LoRAModel

    # Create LoRA layer
    lora_layer = LoRALinear(768, 768, rank=8, alpha=16)

    # Create LoRA model wrapper
    config = LoRAConfig(rank=8, alpha=16, target_modules=['q_proj', 'v_proj'])
    model = LoRAModel(config)

References:
    - LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
    - https://arxiv.org/abs/2106.09685
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .autograd import Variable, matmul


@dataclass
class LoRAConfig:
    """
    Configuration for LoRA adapters.

    Args:
        rank: The rank of the low-rank decomposition (default: 8)
        alpha: Scaling factor for LoRA updates (default: 16)
        dropout: Dropout probability for LoRA layers (default: 0.0)
        target_modules: List of module names to apply LoRA to
            Common targets: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        bias: Whether to train bias parameters ('none', 'all', 'lora_only')
        init_lora_weights: Initialization method ('gaussian', 'kaiming', 'zeros')
        modules_to_save: Additional modules to make trainable (not LoRA)
    """

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"  # 'none', 'all', 'lora_only'
    init_lora_weights: str = "gaussian"  # 'gaussian', 'kaiming', 'zeros'
    modules_to_save: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "init_lora_weights": self.init_lora_weights,
            "modules_to_save": self.modules_to_save,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LoRAConfig":
        """Create config from dictionary."""
        return cls(**d)

    def save(self, path: str | Path) -> None:
        """Save config to JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "LoRAConfig":
        """Load config from JSON file."""
        path = Path(path)
        with open(path) as f:
            return cls.from_dict(json.load(f))


class LoRALinear:
    """
    Low-Rank Adaptation (LoRA) Linear Layer.

    Instead of updating the full weight matrix W (d_out x d_in),
    we add a low-rank decomposition: ΔW = B @ A
    where A is (rank x d_in) and B is (d_out x rank), with rank << d_in, d_out

    Forward: output = x @ (W + scaling * B @ A)^T = x @ W^T + scaling * x @ A^T @ B^T

    This reduces trainable parameters from (d_out * d_in) to (d_out * rank + rank * d_in)

    Args:
        in_features: Size of input features
        out_features: Size of output features
        rank: Rank of the low-rank decomposition (default: 8)
        alpha: Scaling factor (default: 16.0)
        dropout: Dropout probability (default: 0.0)
        bias: Whether to include bias (default: False)
        init_weights: Initialization method ('gaussian', 'kaiming', 'zeros')
        base_weights: Optional pre-trained weights to freeze

    Example:
        >>> lora = LoRALinear(768, 768, rank=8)
        >>> x = Variable(np.random.randn(4, 768).astype(np.float32))
        >>> y = lora.forward(x)
        >>> print(f"Trainable params: {lora.num_trainable_params()}")
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = False,
        init_weights: str = "gaussian",
        base_weights: np.ndarray | None = None,
    ):
        """Initialize the instance."""

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = dropout
        self.has_bias = bias

        # Frozen pre-trained weights (not trainable)
        if base_weights is not None:
            assert base_weights.shape == (out_features, in_features), (
                f"Base weights shape mismatch: expected ({out_features}, {in_features}), got {base_weights.shape}"
            )
            self.W = Variable(base_weights.astype(np.float32), requires_grad=False)
        else:
            # Initialize with small random values if no base weights provided
            self.W = Variable(
                np.random.randn(out_features, in_features).astype(np.float32) * 0.01,
                requires_grad=False,
            )

        # LoRA adapters (trainable)
        # A: (rank, in_features) - projects input to low-rank space
        # B: (out_features, rank) - projects from low-rank space to output
        self.lora_A = self._init_lora_A(init_weights)
        self.lora_B = self._init_lora_B()  # Always zeros

        # Optional bias
        if bias:
            self.bias = Variable(np.zeros(out_features, dtype=np.float32), requires_grad=True)
        else:
            self.bias = None

        # Training state
        self._merged = False
        self._enabled = True

    def _init_lora_A(self, method: str) -> Variable:
        """Initialize LoRA A matrix."""
        if method == "gaussian":
            data = np.random.randn(self.rank, self.in_features).astype(np.float32) * 0.01
        elif method == "kaiming":
            # Kaiming/He initialization
            std = np.sqrt(2.0 / self.in_features)
            data = np.random.randn(self.rank, self.in_features).astype(np.float32) * std
        elif method == "zeros":
            data = np.zeros((self.rank, self.in_features), dtype=np.float32)
        else:
            raise ValueError(f"Unknown init method: {method}")
        return Variable(data, requires_grad=True)

    def _init_lora_B(self) -> Variable:
        """Initialize LoRA B matrix (always zeros for stable training start)."""
        return Variable(
            np.zeros((self.out_features, self.rank), dtype=np.float32), requires_grad=True
        )

    def forward(self, x: Variable) -> Variable:
        """
        Forward pass with LoRA adaptation.

        Args:
            x: Input tensor of shape (batch, ..., in_features)

        Returns:
            Output tensor of shape (batch, ..., out_features)
        """
        # Base model forward (frozen weights)
        if self._merged:
            # Weights already merged, just do linear
            output = matmul(x, self.W.T)
        else:
            # Compute base + LoRA
            base_output = matmul(x, self.W.T)

            if self._enabled:
                # LoRA forward: x @ A^T @ B^T * scaling
                lora_output = matmul(matmul(x, self.lora_A.T), self.lora_B.T)
                output = base_output + lora_output * self.scaling
            else:
                output = base_output

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias

        return output

    def __call__(self, x: Variable) -> Variable:
        """Allow calling layer directly."""
        return self.forward(x)

    def parameters(self) -> list[Variable]:
        """Get trainable parameters (only LoRA adapters and optional bias)."""
        params = [self.lora_A, self.lora_B]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def lora_parameters(self) -> list[Variable]:
        """Get only LoRA adapter parameters (A and B matrices)."""
        return [self.lora_A, self.lora_B]

    def num_trainable_params(self) -> int:
        """Count trainable parameters."""
        count = self.rank * (self.in_features + self.out_features)
        if self.bias is not None:
            count += self.out_features
        return count

    def num_total_params(self) -> int:
        """Count total parameters (including frozen base weights)."""
        base = self.out_features * self.in_features
        if self.bias is not None:
            base += self.out_features
        return base + self.num_trainable_params()

    def merge_weights(self) -> None:
        """
        Merge LoRA weights into base weights for inference.

        After merging, the layer acts as a regular linear layer with no overhead.
        Call unmerge_weights() to restore LoRA adapters for continued training.
        """
        if self._merged:
            return

        # W_merged = W + scaling * B @ A
        delta_W = self.scaling * np.matmul(self.lora_B.data, self.lora_A.data)
        self.W = Variable(self.W.data + delta_W, requires_grad=False)
        self._merged = True

    def unmerge_weights(self) -> None:
        """
        Unmerge LoRA weights from base weights.

        Restores the original base weights and separate LoRA adapters.
        """
        if not self._merged:
            return

        # W_original = W_merged - scaling * B @ A
        delta_W = self.scaling * np.matmul(self.lora_B.data, self.lora_A.data)
        self.W = Variable(self.W.data - delta_W, requires_grad=False)
        self._merged = False

    def enable_lora(self) -> None:
        """Enable LoRA adaptation."""
        self._enabled = True

    def disable_lora(self) -> None:
        """Disable LoRA adaptation (use only base weights)."""
        self._enabled = False

    def reset_lora_parameters(self) -> None:
        """Reset LoRA parameters to initial values."""
        self.lora_A = self._init_lora_A("gaussian")
        self.lora_B = self._init_lora_B()

    def get_state_dict(self) -> dict[str, np.ndarray]:
        """Get state dict for saving (only LoRA weights)."""
        state = {
            "lora_A": self.lora_A.data.copy(),
            "lora_B": self.lora_B.data.copy(),
        }
        if self.bias is not None:
            state["bias"] = self.bias.data.copy()
        return state

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        """Load LoRA weights from state dict."""
        self.lora_A = Variable(state["lora_A"].astype(np.float32), requires_grad=True)
        self.lora_B = Variable(state["lora_B"].astype(np.float32), requires_grad=True)
        if "bias" in state and self.bias is not None:
            self.bias = Variable(state["bias"].astype(np.float32), requires_grad=True)

    def __repr__(self) -> str:
        """Return a debug representation."""

        merged_str = " (merged)" if self._merged else ""
        enabled_str = "" if self._enabled else " (disabled)"
        return (
            f"LoRALinear(in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}{merged_str}{enabled_str})"
        )


class LoRAEmbedding:
    """
    LoRA-adapted Embedding layer.

    Adds low-rank adapters to embedding lookups for vocabulary adaptation.

    Args:
        num_embeddings: Size of vocabulary
        embedding_dim: Dimension of embeddings
        rank: Rank of low-rank decomposition
        alpha: Scaling factor
        base_weights: Pre-trained embedding weights
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        rank: int = 8,
        alpha: float = 16.0,
        base_weights: np.ndarray | None = None,
    ):
        """Initialize the instance."""

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Base embeddings (frozen)
        if base_weights is not None:
            self.weight = Variable(base_weights.astype(np.float32), requires_grad=False)
        else:
            self.weight = Variable(
                np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.01,
                requires_grad=False,
            )

        # LoRA adapters
        self.lora_A = Variable(
            np.random.randn(rank, num_embeddings).astype(np.float32) * 0.01, requires_grad=True
        )
        self.lora_B = Variable(
            np.zeros((embedding_dim, rank), dtype=np.float32), requires_grad=True
        )

        self._merged = False
        self._enabled = True

    def forward(self, input_ids: np.ndarray) -> Variable:
        """
        Embedding lookup with LoRA adaptation.

        Args:
            input_ids: Integer tensor of token IDs (batch, seq_len)

        Returns:
            Embeddings of shape (batch, seq_len, embedding_dim)
        """
        # Base embedding lookup
        embeddings = self.weight.data[input_ids]

        if self._enabled and not self._merged:
            # LoRA adaptation: one-hot @ A^T @ B^T
            one_hot = np.zeros((input_ids.size, self.num_embeddings), dtype=np.float32)
            one_hot[np.arange(input_ids.size), input_ids.flatten()] = 1.0
            one_hot = one_hot.reshape(input_ids.shape + (self.num_embeddings,))

            lora_out = np.matmul(np.matmul(one_hot, self.lora_A.data.T), self.lora_B.data.T)
            embeddings = embeddings + self.scaling * lora_out

        return Variable(embeddings, requires_grad=False)

    def parameters(self) -> list[Variable]:
        """Get trainable parameters."""
        return [self.lora_A, self.lora_B]

    def merge_weights(self) -> None:
        """Merge LoRA into base embeddings."""
        if self._merged:
            return
        delta_W = self.scaling * np.matmul(self.lora_B.data, self.lora_A.data).T
        self.weight = Variable(self.weight.data + delta_W, requires_grad=False)
        self._merged = True

    def unmerge_weights(self) -> None:
        """Unmerge LoRA from base embeddings."""
        if not self._merged:
            return
        delta_W = self.scaling * np.matmul(self.lora_B.data, self.lora_A.data).T
        self.weight = Variable(self.weight.data - delta_W, requires_grad=False)
        self._merged = False


class LoRAAttention:
    """
    LoRA-adapted Multi-Head Attention.

    Applies LoRA to the Q, K, V, and output projections.

    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of attention heads
        rank: LoRA rank for all projections
        alpha: LoRA scaling factor
        qkv_weights: Pre-trained QKV weights (3, embed_dim, embed_dim)
        out_weights: Pre-trained output projection weights (embed_dim, embed_dim)
        apply_lora_to: Which projections to apply LoRA ('q', 'k', 'v', 'o')
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rank: int = 8,
        alpha: float = 16.0,
        qkv_weights: np.ndarray | None = None,
        out_weights: np.ndarray | None = None,
        apply_lora_to: list[str] = ["q", "v"],
    ):
        """Initialize the instance."""

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rank = rank
        self.alpha = alpha

        # Create LoRA layers for selected projections
        self.q_proj = (
            LoRALinear(
                embed_dim,
                embed_dim,
                rank=rank,
                alpha=alpha,
                base_weights=qkv_weights[0] if qkv_weights is not None else None,
            )
            if "q" in apply_lora_to
            else None
        )

        self.k_proj = (
            LoRALinear(
                embed_dim,
                embed_dim,
                rank=rank,
                alpha=alpha,
                base_weights=qkv_weights[1] if qkv_weights is not None else None,
            )
            if "k" in apply_lora_to
            else None
        )

        self.v_proj = (
            LoRALinear(
                embed_dim,
                embed_dim,
                rank=rank,
                alpha=alpha,
                base_weights=qkv_weights[2] if qkv_weights is not None else None,
            )
            if "v" in apply_lora_to
            else None
        )

        self.o_proj = (
            LoRALinear(embed_dim, embed_dim, rank=rank, alpha=alpha, base_weights=out_weights)
            if "o" in apply_lora_to
            else None
        )

        # Store which projections have LoRA
        self.apply_lora_to = apply_lora_to

    def parameters(self) -> list[Variable]:
        """Get all trainable LoRA parameters."""
        params = []
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            if proj is not None:
                params.extend(proj.parameters())
        return params

    def num_trainable_params(self) -> int:
        """Count trainable parameters."""
        count = 0
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            if proj is not None:
                count += proj.num_trainable_params()
        return count

    def merge_weights(self) -> None:
        """Merge all LoRA weights."""
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            if proj is not None:
                proj.merge_weights()

    def unmerge_weights(self) -> None:
        """Unmerge all LoRA weights."""
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            if proj is not None:
                proj.unmerge_weights()

    def get_state_dict(self) -> dict[str, dict[str, np.ndarray]]:
        """Get state dict for all LoRA projections."""
        state = {}
        for name, proj in [
            ("q_proj", self.q_proj),
            ("k_proj", self.k_proj),
            ("v_proj", self.v_proj),
            ("o_proj", self.o_proj),
        ]:
            if proj is not None:
                state[name] = proj.get_state_dict()
        return state

    def load_state_dict(self, state: dict[str, dict[str, np.ndarray]]) -> None:
        """Load LoRA weights for all projections."""
        for name, proj in [
            ("q_proj", self.q_proj),
            ("k_proj", self.k_proj),
            ("v_proj", self.v_proj),
            ("o_proj", self.o_proj),
        ]:
            if proj is not None and name in state:
                proj.load_state_dict(state[name])


class LoRAModel:
    """
    Wrapper to add LoRA adapters to an existing model.

    This class manages multiple LoRA layers and provides utilities for
    training, saving, and loading LoRA adapters.

    Args:
        config: LoRA configuration

    Example:
        >>> config = LoRAConfig(rank=8, alpha=16, target_modules=['q_proj', 'v_proj'])
        >>> lora_model = LoRAModel(config)
        >>> lora_model.add_lora_layer('layer.0.attention.q_proj', 768, 768, q_weights)
    """

    def __init__(self, config: LoRAConfig):
        """Initialize the instance."""

        self.config = config
        self.lora_layers: dict[str, LoRALinear] = {}
        self._merged = False

    def add_lora_layer(
        self,
        name: str,
        in_features: int,
        out_features: int,
        base_weights: np.ndarray | None = None,
    ) -> LoRALinear:
        """
        Add a LoRA layer.

        Args:
            name: Unique name for this layer
            in_features: Input dimension
            out_features: Output dimension
            base_weights: Pre-trained weights to freeze

        Returns:
            The created LoRA layer
        """
        layer = LoRALinear(
            in_features=in_features,
            out_features=out_features,
            rank=self.config.rank,
            alpha=self.config.alpha,
            dropout=self.config.dropout,
            init_weights=self.config.init_lora_weights,
            base_weights=base_weights,
        )
        self.lora_layers[name] = layer
        return layer

    def get_layer(self, name: str) -> LoRALinear | None:
        """Get a LoRA layer by name."""
        return self.lora_layers.get(name)

    def parameters(self) -> Iterator[Variable]:
        """Iterate over all trainable parameters."""
        for layer in self.lora_layers.values():
            yield from layer.parameters()

    def lora_parameters(self) -> Iterator[Variable]:
        """Iterate over only LoRA adapter parameters."""
        for layer in self.lora_layers.values():
            yield from layer.lora_parameters()

    def num_trainable_params(self) -> int:
        """Count total trainable parameters."""
        return sum(layer.num_trainable_params() for layer in self.lora_layers.values())

    def num_total_params(self) -> int:
        """Count total parameters including frozen."""
        return sum(layer.num_total_params() for layer in self.lora_layers.values())

    def merge_weights(self) -> None:
        """Merge all LoRA weights into base weights."""
        for layer in self.lora_layers.values():
            layer.merge_weights()
        self._merged = True

    def unmerge_weights(self) -> None:
        """Unmerge all LoRA weights."""
        for layer in self.lora_layers.values():
            layer.unmerge_weights()
        self._merged = False

    def save_checkpoint(self, path: str | Path) -> None:
        """
        Save LoRA checkpoint.

        Saves both config and adapter weights.

        Args:
            path: Directory to save checkpoint
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save(path / "config.json")

        # Save adapter weights
        state = {}
        for name, layer in self.lora_layers.items():
            state[name] = layer.get_state_dict()

        np.savez(
            path / "adapters.npz",
            **{
                f"{name}_{key}": value
                for name, layer_state in state.items()
                for key, value in layer_state.items()
            },
        )

        # Save layer metadata
        metadata = {
            name: {
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "rank": layer.rank,
                "alpha": layer.alpha,
            }
            for name, layer in self.lora_layers.items()
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "LoRAModel":
        """
        Load LoRA checkpoint.

        Args:
            path: Directory containing checkpoint

        Returns:
            Loaded LoRAModel
        """
        path = Path(path)

        # Load config
        config = LoRAConfig.load(path / "config.json")
        model = cls(config)

        # Load metadata
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        # Load adapter weights
        adapters = np.load(path / "adapters.npz")

        # Reconstruct layers
        for name, meta in metadata.items():
            layer = model.add_lora_layer(
                name=name,
                in_features=meta["in_features"],
                out_features=meta["out_features"],
            )

            # Load weights
            state = {}
            for key in ["lora_A", "lora_B", "bias"]:
                full_key = f"{name}_{key}"
                if full_key in adapters:
                    state[key] = adapters[full_key]
            layer.load_state_dict(state)

        return model

    def print_trainable_parameters(self) -> None:
        """Print summary of trainable parameters."""
        trainable = self.num_trainable_params()
        total = self.num_total_params()
        print(
            f"trainable params: {trainable:,} || all params: {total:,} || "
            f"trainable%: {100 * trainable / total:.4f}"
        )


# Convenience functions


def apply_lora_to_linear(
    weight: np.ndarray,
    config: LoRAConfig,
) -> LoRALinear:
    """
    Create a LoRA layer from existing linear weight.

    Args:
        weight: Weight matrix of shape (out_features, in_features)
        config: LoRA configuration

    Returns:
        LoRALinear layer with the weight frozen
    """
    out_features, in_features = weight.shape
    return LoRALinear(
        in_features=in_features,
        out_features=out_features,
        rank=config.rank,
        alpha=config.alpha,
        dropout=config.dropout,
        init_weights=config.init_lora_weights,
        base_weights=weight,
    )


def calculate_lora_params(
    model_params: int,
    num_lora_layers: int,
    in_features: int,
    out_features: int,
    rank: int,
) -> dict[str, Any]:
    """
    Calculate LoRA parameter count and memory usage.

    Args:
        model_params: Total base model parameters
        num_lora_layers: Number of layers with LoRA
        in_features: Input dimension of LoRA layers
        out_features: Output dimension of LoRA layers
        rank: LoRA rank

    Returns:
        Dictionary with parameter counts and memory estimates
    """
    lora_params_per_layer = rank * (in_features + out_features)
    total_lora_params = num_lora_layers * lora_params_per_layer

    # Memory in bytes (float32)
    base_memory = model_params * 4
    lora_memory = total_lora_params * 4
    # Adam optimizer: 2x for m and v
    optimizer_memory = total_lora_params * 4 * 2

    return {
        "base_params": model_params,
        "lora_params": total_lora_params,
        "trainable_ratio": total_lora_params / model_params,
        "base_memory_gb": base_memory / (1024**3),
        "lora_memory_gb": lora_memory / (1024**3),
        "optimizer_memory_gb": optimizer_memory / (1024**3),
        "total_training_memory_gb": (lora_memory + optimizer_memory) / (1024**3),
    }


# Export all classes and functions
__all__ = [
    "LoRAConfig",
    "LoRALinear",
    "LoRAEmbedding",
    "LoRAAttention",
    "LoRAModel",
    "apply_lora_to_linear",
    "calculate_lora_params",
]
