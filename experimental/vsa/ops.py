"""
Vector Symbolic Architecture (VSA) Operations.

Provides O(d) operations for hyperdimensional computing:
- Binding: Combine two vectors into a composite
- Unbinding: Recover a vector from a composite
- Bundling: Superposition of multiple vectors
- Similarity: Measure relatedness between vectors

Two operation classes are provided:
- BinaryOps: For bipolar (+1/-1) vectors - exact binding/unbinding
- HolographicOps: For continuous vectors using FFT - approximate binding/unbinding

Author: Grilly Team
Date: February 2026
"""

import numpy as np
from typing import List, Optional


class BinaryOps:
    """
    Operations for bipolar (+1/-1) vectors.
    
    Bipolar vectors are efficient for hardware implementation and have
    exact inverse properties: bind(a, a) = identity, unbind = bind.
    
    All operations are O(d) and embarrassingly parallel.
    """
    
    @staticmethod
    def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Bind two bipolar vectors via element-wise multiplication.
        
        Properties:
            - Commutative: bind(a, b) == bind(b, a)
            - Associative: bind(bind(a, b), c) == bind(a, bind(b, c))
            - Self-inverse: bind(a, a) == identity (all ones)
            - Preserves bipolarity: output is +1/-1
        
        Args:
            a: First bipolar vector
            b: Second bipolar vector
            
        Returns:
            Bound bipolar vector
        """
        return (a * b).astype(np.float32)
    
    @staticmethod
    def unbind(composite: np.ndarray, known: np.ndarray) -> np.ndarray:
        """
        Unbind a known vector from a composite.
        
        For bipolar vectors, unbind is identical to bind since
        each element is its own inverse: x * x = 1.
        
        Args:
            composite: The composite vector
            known: The known factor to remove
            
        Returns:
            The recovered vector (approximately the original bound vector)
        """
        return (composite * known).astype(np.float32)
    
    @staticmethod
    def bundle(vectors: List[np.ndarray], normalize: bool = True) -> np.ndarray:
        """
        Bundle multiple vectors via majority voting.
        
        The result preserves similarity to each component, allowing
        multiple items to be stored in superposition.
        
        Args:
            vectors: List of vectors to bundle
            normalize: If True, apply sign function for bipolar output
            
        Returns:
            Bundled vector (bipolar if normalize=True)
        """
        if not vectors:
            raise ValueError("Cannot bundle empty list of vectors")
        
        result = np.sum(vectors, axis=0)
        
        if normalize:
            # Majority vote: sign of sum (with small epsilon to break ties)
            result = np.sign(result + 1e-8).astype(np.float32)
        
        return result
    
    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        For bipolar vectors, this is equivalent to normalized Hamming distance.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Similarity in range [-1, 1]
        """
        return float(np.dot(a, b) / len(a))
    
    @staticmethod
    def random_bipolar(dim: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a random bipolar vector.
        
        Args:
            dim: Dimension of the vector
            seed: Optional random seed for reproducibility
            
        Returns:
            Random bipolar vector with values +1 or -1
        """
        if seed is not None:
            np.random.seed(seed)
        return np.sign(np.random.randn(dim)).astype(np.float32)
    
    @staticmethod
    def hash_to_bipolar(s: str, dim: int) -> np.ndarray:
        """
        Deterministically hash a string to a bipolar vector.
        
        Same string always produces same vector.
        
        Args:
            s: String to hash
            dim: Dimension of output vector
            
        Returns:
            Deterministic bipolar vector for the string
        """
        seed = hash(s) % (2**31)
        return BinaryOps.random_bipolar(dim, seed)


class HolographicOps:
    """
    Operations for continuous vectors using Holographic Reduced Representations (HRR).
    
    Uses circular convolution for binding and correlation for unbinding.
    Implemented via FFT for O(d log d) complexity.
    
    HRR preserves more information than binary binding but unbinding
    is approximate rather than exact.
    """
    
    @staticmethod
    def convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Circular convolution (binding) via frequency domain multiplication.
        
        Properties:
            - Commutative: convolve(a, b) == convolve(b, a)
            - Associative: convolve(convolve(a, b), c) == convolve(a, convolve(b, c))
            - Approximate inverse via correlate
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Convolved vector
        """
        return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b))).astype(np.float32)
    
    @staticmethod
    def correlate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Circular correlation (unbinding) - approximate inverse of convolve.
        
        Given composite = convolve(x, key), correlate(composite, key) ≈ x.
        
        Args:
            a: The composite vector
            b: The known factor (key) to remove
            
        Returns:
            Approximate recovered vector
        """
        return np.real(np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b)))).astype(np.float32)
    
    @staticmethod
    def bundle(vectors: List[np.ndarray], normalize: bool = True) -> np.ndarray:
        """
        Bundle multiple vectors via element-wise sum.
        
        Args:
            vectors: List of vectors to bundle
            normalize: If True, normalize result to unit length
            
        Returns:
            Bundled vector
        """
        if not vectors:
            raise ValueError("Cannot bundle empty list of vectors")
        
        result = np.sum(vectors, axis=0).astype(np.float32)
        
        if normalize:
            norm = np.linalg.norm(result)
            if norm > 0:
                result = result / norm
        
        return result
    
    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Similarity in range [-1, 1] for unit vectors
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    @staticmethod
    def random_vector(dim: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a random unit vector.
        
        Args:
            dim: Dimension of the vector
            seed: Optional random seed for reproducibility
            
        Returns:
            Random unit vector
        """
        if seed is not None:
            np.random.seed(seed)
        v = np.random.randn(dim).astype(np.float32)
        return v / np.linalg.norm(v)
