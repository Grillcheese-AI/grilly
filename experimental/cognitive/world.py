"""
WorldModel - Knowledge base for coherence checking.

Stores facts, constraints, and expectations for verifying statement coherence.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from grilly.experimental.vsa.ops import HolographicOps
from grilly.experimental.cognitive.capsule import CapsuleEncoder, cosine_similarity


@dataclass
class Fact:
    """A fact in the world model."""
    subject: str
    relation: str
    object: str
    vector: np.ndarray  # Holographic encoding
    capsule_vector: Optional[np.ndarray] = None
    confidence: float = 1.0
    source: str = "observed"


class WorldModel:
    """
    World model for coherence checking.
    
    Stores:
    - Facts (subject-relation-object triples)
    - Constraints (what can't be true together)
    - Expectations (what typically follows what)
    
    Used to verify that candidate outputs make sense.
    """
    
    DEFAULT_DIM = 4096
    
    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        capsule_dim: int = 32,
        semantic_dims: int = 28
    ):
        self.dim = dim
        self.capsule_dim = capsule_dim
        self.semantic_dims = semantic_dims
        self.capsule_encoder: Optional[CapsuleEncoder] = None
        if capsule_dim > 0:
            self.capsule_encoder = CapsuleEncoder(
                input_dim=dim,
                capsule_dim=capsule_dim,
                semantic_dims=semantic_dims
            )
        
        # Fact storage
        self.facts: List[Fact] = []
        self.fact_vectors: List[np.ndarray] = []
        self.fact_capsules: List[Optional[np.ndarray]] = []
        
        # Relation encodings
        self.relations: Dict[str, np.ndarray] = {}
        self._init_relations()
        
        # Constraint patterns (things that can't both be true)
        self.constraints: List[Tuple[np.ndarray, np.ndarray]] = []
        
        # Causal/temporal expectations
        self.expectations: Dict[str, List[Tuple[str, float]]] = {}
    
    def _init_relations(self):
        """Initialize relation vectors."""
        relations = [
            "is", "is_not", "has", "can", "cannot",
            "causes", "prevents", "before", "after",
            "part_of", "contains", "similar_to", "opposite_of",
            "wants", "believes", "knows", "thinks"
        ]
        for i, rel in enumerate(relations):
            self.relations[rel] = HolographicOps.random_vector(self.dim, seed=6000+i)
    
    def encode_fact(self, subject: str, relation: str, object_: str) -> np.ndarray:
        """Encode a fact as a holographic vector."""
        subj_vec = HolographicOps.random_vector(self.dim, seed=hash(subject) % (2**31))
        rel_vec = self.relations.get(relation, HolographicOps.random_vector(self.dim, seed=hash(relation) % (2**31)))
        obj_vec = HolographicOps.random_vector(self.dim, seed=hash(object_) % (2**31))
        
        # Fact = subject ⊗ relation ⊗ object
        return HolographicOps.convolve(
            HolographicOps.convolve(subj_vec, rel_vec),
            obj_vec
        )
    
    def add_fact(
        self,
        subject: str,
        relation: str,
        object_: str,
        confidence: float = 1.0,
        source: str = "observed",
        cognitive_features: Optional[np.ndarray] = None
    ):
        """Add a fact to the world model."""
        vector = self.encode_fact(subject, relation, object_)
        
        capsule_vec = None
        if self.capsule_encoder is not None:
            capsule_vec = self.capsule_encoder.encode_vector(vector, cognitive_features)

        fact = Fact(
            subject=subject,
            relation=relation,
            object=object_,
            vector=vector,
            capsule_vector=capsule_vec,
            confidence=confidence,
            source=source
        )
        
        self.facts.append(fact)
        self.fact_vectors.append(vector)
        self.fact_capsules.append(capsule_vec)
        
        # Also add the negation as a constraint
        neg_vector = self.encode_fact(subject, "is_not", object_)
        self.constraints.append((vector, neg_vector))
    
    def query_fact(self, subject: str, relation: str, object_: str) -> Tuple[bool, float]:
        """
        Query if a fact is in the world model.
        
        Returns (is_known, confidence)
        """
        query_vec = self.encode_fact(subject, relation, object_)
        
        best_sim = 0.0
        query_capsule = None
        if self.capsule_encoder is not None:
            query_capsule = self.capsule_encoder.encode_vector(query_vec)

        for fact, fact_vec, fact_capsule in zip(self.facts, self.fact_vectors, self.fact_capsules):
            sim = HolographicOps.similarity(query_vec, fact_vec)
            if sim > best_sim:
                best_sim = sim
            if query_capsule is not None and fact_capsule is not None:
                cap_sim = cosine_similarity(query_capsule, fact_capsule)
                if cap_sim > best_sim:
                    best_sim = cap_sim
        
        return best_sim > 0.7, best_sim
    
    def check_coherence(self, statement_vec: np.ndarray) -> Tuple[bool, float, str]:
        """
        Check if a statement is coherent with known facts.
        
        Returns (is_coherent, confidence, reason)
        """
        # Check against known facts (should be consistent)
        max_support = 0.0
        supporting_fact = None
        
        for fact, fact_vec in zip(self.facts, self.fact_vectors):
            sim = HolographicOps.similarity(statement_vec, fact_vec)
            if sim > max_support:
                max_support = sim
                supporting_fact = fact
        
        # Check against constraints (should not violate)
        max_violation = 0.0
        violating_constraint = None
        
        for fact_vec, neg_vec in self.constraints:
            # If statement is similar to the negation of a known fact, that's bad
            sim_to_neg = HolographicOps.similarity(statement_vec, neg_vec)
            if sim_to_neg > max_violation:
                max_violation = sim_to_neg

        if self.capsule_encoder is not None:
            statement_capsule = self.capsule_encoder.encode_vector(statement_vec)
            capsule_support = 0.0
            capsule_violation = 0.0

            for fact_capsule in self.fact_capsules:
                if fact_capsule is None:
                    continue
                cap_sim = cosine_similarity(statement_capsule, fact_capsule)
                if cap_sim > capsule_support:
                    capsule_support = cap_sim

            for _, neg_vec in self.constraints:
                neg_capsule = self.capsule_encoder.encode_vector(neg_vec)
                cap_neg_sim = cosine_similarity(statement_capsule, neg_capsule)
                if cap_neg_sim > capsule_violation:
                    capsule_violation = cap_neg_sim

            max_support = max(max_support, capsule_support)
            max_violation = max(max_violation, capsule_violation)
        
        # Compute coherence score
        coherence = max_support - max_violation
        
        if coherence > 0.3:
            reason = "Supported by known fact"
            return True, coherence, reason
        elif max_violation > 0.5:
            reason = "Contradicts known fact"
            return False, coherence, reason
        else:
            reason = "No strong evidence either way"
            return True, 0.5, reason  # Uncertain but not incoherent
    
    def predict_consequence(self, action: str) -> List[Tuple[str, float]]:
        """
        Predict consequences of an action based on causal knowledge.
        """
        if action in self.expectations:
            return self.expectations[action]
        return []
    
    def add_causal_link(self, cause: str, effect: str, strength: float = 0.8):
        """Add a causal expectation."""
        if cause not in self.expectations:
            self.expectations[cause] = []
        self.expectations[cause].append((effect, strength))
