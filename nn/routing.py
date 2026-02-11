"""
Domain Routing Layers

Uses: domain-router.glsl, domain-predict.glsl, domain-classifier.glsl,
      domain-combine-experts.glsl

Reference: ref/brain/gpu_brain.py domain_routing
"""

import numpy as np

from .module import Module
from .modules import Linear, Softmax


class DomainRouter(Module):
    """
    Domain Router layer - Route inputs to domain experts.

    Uses: domain-router.glsl

    Reference: ref/brain/gpu_brain.py domain_routing
    """

    def __init__(self, embed_dim: int, num_domains: int, num_experts: int):
        """
        Initialize DomainRouter layer.

        Args:
            embed_dim: Embedding dimension
            num_domains: Number of domains
            num_experts: Number of expert networks
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_domains = num_domains
        self.num_experts = num_experts

        # Domain predictor
        self.domain_predictor = Linear(embed_dim, num_domains)
        self._modules["domain_predictor"] = self.domain_predictor

        # Expert weights per domain
        limit = np.sqrt(6.0 / (num_domains + num_experts))
        self.expert_weights = np.random.uniform(-limit, limit, (num_domains, num_experts)).astype(
            np.float32
        )
        self._parameters["expert_weights"] = self.expert_weights

        self.softmax = Softmax(dim=-1)
        self._modules["softmax"] = self.softmax

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass - route to experts.

        Args:
            x: Input embeddings (batch, embed_dim)

        Returns:
            Routing weights (batch, num_experts)
        """
        backend = self._get_backend()

        # Predict domain probabilities
        domain_probs = self.domain_predictor(x)  # (batch, num_domains)
        domain_probs = self.softmax(domain_probs)

        # Route to experts
        if hasattr(backend, "domain_route"):
            routing_weights = backend.domain_route(domain_probs, self.expert_weights)
        else:
            # CPU fallback: domain_probs @ expert_weights
            routing_weights = domain_probs @ self.expert_weights  # (batch, num_experts)

        return routing_weights

    def __repr__(self):
        """Return a debug representation."""

        return f"DomainRouter(embed_dim={self.embed_dim}, num_domains={self.num_domains}, num_experts={self.num_experts})"


class DomainPredictor(Module):
    """
    Domain Predictor layer - Predict domain from input.

    Uses: domain-predict.glsl
    """

    def __init__(self, embed_dim: int, num_domains: int):
        """
        Initialize DomainPredictor layer.

        Args:
            embed_dim: Embedding dimension
            num_domains: Number of domains
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_domains = num_domains

        self.predictor = Linear(embed_dim, num_domains)
        self._modules["predictor"] = self.predictor
        self.softmax = Softmax(dim=-1)
        self._modules["softmax"] = self.softmax

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass - predict domain.

        Args:
            x: Input embeddings (batch, embed_dim)

        Returns:
            Domain probabilities (batch, num_domains)
        """
        logits = self.predictor(x)
        return self.softmax(logits)

    def __repr__(self):
        """Return a debug representation."""

        return f"DomainPredictor(embed_dim={self.embed_dim}, num_domains={self.num_domains})"


class DomainClassifier(Module):
    """
    Domain Classifier layer - Classify domain.

    Uses: domain-classifier.glsl
    """

    def __init__(self, embed_dim: int, num_domains: int):
        """
        Initialize DomainClassifier layer.

        Args:
            embed_dim: Embedding dimension
            num_domains: Number of domains
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_domains = num_domains

        self.classifier = Linear(embed_dim, num_domains)
        self._modules["classifier"] = self.classifier
        self.softmax = Softmax(dim=-1)
        self._modules["softmax"] = self.softmax

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass - classify domain.

        Args:
            x: Input embeddings (batch, embed_dim)

        Returns:
            Domain class probabilities (batch, num_domains)
        """
        logits = self.classifier(x)
        return self.softmax(logits)

    def __repr__(self):
        """Return a debug representation."""

        return f"DomainClassifier(embed_dim={self.embed_dim}, num_domains={self.num_domains})"


class ExpertCombiner(Module):
    """
    Expert Combiner layer - Combine expert outputs.

    Uses: domain-combine-experts.glsl
    """

    def __init__(self, expert_dim: int, num_experts: int, output_dim: int):
        """
        Initialize ExpertCombiner layer.

        Args:
            expert_dim: Dimension of each expert output
            num_experts: Number of experts
            output_dim: Output dimension
        """
        super().__init__()
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.output_dim = output_dim

        # Combination weights
        limit = np.sqrt(6.0 / (expert_dim * num_experts + output_dim))
        self.combine_weight = np.random.uniform(
            -limit, limit, (output_dim, expert_dim * num_experts)
        ).astype(np.float32)
        self.combine_bias = np.zeros(output_dim, dtype=np.float32)

        self._parameters["combine_weight"] = self.combine_weight
        self._parameters["combine_bias"] = self.combine_bias

    def forward(self, expert_outputs: np.ndarray, routing_weights: np.ndarray) -> np.ndarray:
        """
        Forward pass - combine expert outputs.

        Args:
            expert_outputs: Expert outputs (batch, num_experts, expert_dim)
            routing_weights: Routing weights (batch, num_experts)

        Returns:
            Combined output (batch, output_dim)
        """
        backend = self._get_backend()

        # Try GPU shader if available
        if hasattr(backend, "shaders") and "domain-combine-experts" in backend.shaders:
            try:
                # GPU implementation would go here
                pass
            except Exception:
                pass  # Fall back to CPU

        # CPU fallback
        batch_size = expert_outputs.shape[0]

        # Weight expert outputs by routing weights
        weighted = expert_outputs * routing_weights[:, :, None]  # (batch, num_experts, expert_dim)

        # Flatten and combine
        weighted_flat = weighted.reshape(batch_size, -1)  # (batch, num_experts * expert_dim)
        output = weighted_flat @ self.combine_weight.T + self.combine_bias

        return output

    def __repr__(self):
        """Return a debug representation."""

        return f"ExpertCombiner(expert_dim={self.expert_dim}, num_experts={self.num_experts}, output_dim={self.output_dim})"
