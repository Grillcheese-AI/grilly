"""
WorldModel - Knowledge base for coherence checking.

Stores facts, constraints, and expectations for verifying statement coherence.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from grilly.experimental.vsa.ops import HolographicOps


@dataclass
class Fact:
    """A fact in the world model."""
    subject: str
    relation: str
    object: str
    vector: np.ndarray  # Holographic encoding
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
    
    def __init__(self, dim: int = DEFAULT_DIM):
        self.dim = dim
        
        # Fact storage
        self.facts: List[Fact] = []
        self.fact_vectors: List[np.ndarray] = []
        
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
        source: str = "observed"
    ):
        """Add a fact to the world model."""
        vector = self.encode_fact(subject, relation, object_)
        
        fact = Fact(
            subject=subject,
            relation=relation,
            object=object_,
            vector=vector,
            confidence=confidence,
            source=source
        )
        
        self.facts.append(fact)
        self.fact_vectors.append(vector)
        
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
        for fact, fact_vec in zip(self.facts, self.fact_vectors):
            sim = HolographicOps.similarity(query_vec, fact_vec)
            if sim > best_sim:
                best_sim = sim
        
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
