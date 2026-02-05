"""
ResonatorMoE - Resonator-based Mixture of Experts routing.

Uses VSA operations to route queries to relevant experts.
"""

import numpy as np
from typing import Dict, List, Callable, Optional, Tuple
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
        experts: Dict[str, Callable],
        expert_vectors: Optional[Dict[str, np.ndarray]] = None
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
    
    def route(
        self,
        query: np.ndarray,
        top_k: int = 1,
        threshold: Optional[float] = None
    ) -> List[str]:
        """
        Route query to top-k most similar experts.
        
        Args:
            query: Query vector of shape (dim,)
            top_k: Number of experts to select
            threshold: Optional minimum similarity threshold
            
        Returns:
            List of expert names, ordered by similarity (descending)
        """
        # Compute similarities
        similarities = []
        for name, expert_vec in self.expert_vectors.items():
            sim = BinaryOps.similarity(query, expert_vec)
            similarities.append((name, sim))
        
        # Filter by threshold if provided
        if threshold is not None:
            similarities = [(n, s) for n, s in similarities if s >= threshold]
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k expert names
        return [name for name, _ in similarities[:top_k]]
    
    def get_weights(
        self,
        query: np.ndarray,
        normalize: bool = True
    ) -> Dict[str, float]:
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
            sim = BinaryOps.similarity(query, expert_vec)
            # Convert similarity [-1, 1] to non-negative weight [0, 2]
            weights[name] = sim + 1.0
        
        # Apply softmax normalization if requested
        if normalize:
            exp_weights = {k: np.exp(v) for k, v in weights.items()}
            total = sum(exp_weights.values())
            weights = {k: v / total for k, v in exp_weights.items()}
        
        return weights
    
    def forward(
        self,
        x: np.ndarray,
        query: np.ndarray,
        top_k: int = 1
    ) -> np.ndarray:
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


class RelationalMoE(ResonatorMoE):
    """
    RelationalMoE - MoE with relational expert codebook.
    
    Extends ResonatorMoE to use RelationalEncoder for creating
    expert vectors from relational concepts.
    """
    
    def __init__(
        self,
        dim: int,
        experts: Dict[str, Callable],
        expert_relations: Dict[str, Tuple[str, str]],
        relational_encoder: Optional['RelationalEncoder'] = None
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
