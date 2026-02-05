"""
ResonatorParser for parsing sentences via resonator factorization.

Recovers word-role pairs from sentence vectors.
"""

import numpy as np
from typing import Dict, List, Tuple
from grilly.experimental.language.encoder import SentenceEncoder
from grilly.experimental.vsa.ops import HolographicOps


class ResonatorParser:
    """
    Parse sentences using resonator factorization.
    
    Given a sentence vector, recover:
    - What words are present
    - What roles they fill
    - The phrase structure
    
    This is PARALLEL - all factors resolved simultaneously!
    """
    
    def __init__(
        self,
        sentence_encoder: SentenceEncoder,
        max_iterations: int = 15
    ):
        self.encoder = sentence_encoder
        self.word_encoder = sentence_encoder.word_encoder
        self.dim = sentence_encoder.dim
        self.max_iterations = max_iterations
        
        # Build codebooks for resonator
        self._build_codebooks()
    
    def _build_codebooks(self):
        """Build codebooks from known words and roles."""
        # Role codebook
        self.role_codebook = np.array([
            vec for vec in self.encoder.roles.values()
        ])
        self.role_names = list(self.encoder.roles.keys())
        
        # Position codebook
        self.position_codebook = np.array(self.encoder.position_vectors[:20])
    
    def parse(
        self,
        sentence_vec: np.ndarray,
        num_slots: int = 5
    ) -> List[Tuple[str, str, float]]:
        """
        Parse sentence vector into word-role pairs.
        
        Uses resonator dynamics to factorize:
        sentence = Σ (word_i ⊗ role_i ⊗ position_i)
        
        Returns:
            List of (word, role, confidence) tuples
        """
        results = []
        residual = sentence_vec.copy()
        
        for slot in range(num_slots):
            # Find best role for this slot
            role_sims = []
            for role_vec in self.role_codebook:
                sim = HolographicOps.similarity(residual, role_vec)
                role_sims.append(abs(sim))
            
            best_role_idx = np.argmax(role_sims)
            best_role = self.role_names[best_role_idx]
            best_role_vec = self.role_codebook[best_role_idx]
            
            # Unbind role to get word
            word_estimate = HolographicOps.correlate(residual, best_role_vec)
            
            # Also unbind position
            pos_vec = self.encoder.position_vectors[slot % len(self.encoder.position_vectors)]
            word_estimate = HolographicOps.correlate(word_estimate, pos_vec)
            
            # Find closest word
            closest_words = self.word_encoder.find_closest(word_estimate, top_k=1)
            
            if closest_words:
                word, confidence = closest_words[0]
                results.append((word, best_role, confidence))
                
                # Remove this component from residual
                word_vec = self.word_encoder.encode_word(word)
                component = HolographicOps.convolve(word_vec, best_role_vec)
                component = HolographicOps.convolve(component, pos_vec)
                residual = residual - component * 0.5  # Soft removal
            
            # Stop if residual is too small
            if np.linalg.norm(residual) < 0.1:
                break
        
        return results
    
    def parallel_parse(
        self,
        sentence_vec: np.ndarray,
        known_words: List[str],
        num_iterations: int = 10
    ) -> Dict[str, Tuple[str, float]]:
        """
        Parallel resonator parsing.
        
        Given known vocabulary, find which words are present
        and what roles they fill - ALL SIMULTANEOUSLY.
        """
        # Build word codebook from known words
        word_codebook = np.array([
            self.word_encoder.encode_word(w) for w in known_words
        ])
        
        # Initialize estimates
        word_estimates = np.random.randn(len(known_words), self.dim).astype(np.float32)
        role_estimates = np.random.randn(len(self.role_names), self.dim).astype(np.float32)
        
        # Resonator iterations
        for _ in range(num_iterations):
            # Update word estimates
            for i in range(len(known_words)):
                # Unbind all role estimates
                unbound = sentence_vec.copy()
                for j, role_est in enumerate(role_estimates):
                    if HolographicOps.similarity(word_estimates[i], word_codebook[i]) > 0.5:
                        unbound = HolographicOps.correlate(unbound, role_est)
                
                # Project onto word codebook
                sims = []
                for word_vec in word_codebook:
                    sim = HolographicOps.similarity(unbound, word_vec)
                    sims.append(sim)
                word_estimates[i] = word_codebook[np.argmax(sims)]
            
            # Update role estimates
            for j in range(len(self.role_names)):
                # Unbind word estimates
                unbound = sentence_vec.copy()
                for i, word_est in enumerate(word_estimates):
                    unbound = HolographicOps.correlate(unbound, word_est)
                
                # Project onto role codebook
                sims = []
                for role_vec in self.role_codebook:
                    sim = HolographicOps.similarity(unbound, role_vec)
                    sims.append(sim)
                role_estimates[j] = self.role_codebook[np.argmax(sims)]
        
        # Extract results
        results = {}
        for i, word in enumerate(known_words):
            # Find which role this word has
            word_vec = self.word_encoder.encode_word(word)
            
            best_role = None
            best_conf = 0.0
            
            for j, role_name in enumerate(self.role_names):
                role_vec = self.encoder.roles[role_name]
                # Check if word ⊗ role is in sentence
                bound = HolographicOps.convolve(word_vec, role_vec)
                sim = HolographicOps.similarity(sentence_vec, bound)
                
                if sim > best_conf:
                    best_conf = sim
                    best_role = role_name
            
            if best_role and best_conf > 0.1:
                results[word] = (best_role, best_conf)
        
        return results
