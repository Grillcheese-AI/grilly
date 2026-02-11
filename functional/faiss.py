"""
Functional FAISS Operations

Uses: faiss-distance.glsl, faiss-topk.glsl, faiss-ivf-filter.glsl,
      faiss-kmeans-update.glsl, faiss-quantize.glsl
"""

import numpy as np


def _get_backend():
    """Get backend instance"""
    try:
        from ..backend.compute import Compute

        return Compute()
    except Exception:
        return None


def faiss_distance(query: np.ndarray, vectors: np.ndarray, distance_type: str = "l2") -> np.ndarray:
    """
    Compute distances between query and vectors.

    Uses: faiss-distance.glsl

    Args:
        query: Query vectors (batch, dim) or (dim,)
        vectors: Database vectors (num_vectors, dim)
        distance_type: Distance type ('l2', 'cosine', 'dot')

    Returns:
        Distances (batch, num_vectors) or (num_vectors,)
    """
    from grilly import Compute

    backend = Compute()
    if hasattr(backend, "faiss") and hasattr(backend.faiss, "compute_distances"):
        return backend.faiss.compute_distances(query, vectors, distance_type=distance_type)
    else:
        # CPU fallback
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if distance_type == "l2":
            diff = query[:, None, :] - vectors[None, :, :]
            return np.sqrt(np.sum(diff**2, axis=2))
        elif distance_type == "cosine":
            q_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
            v_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
            return 1 - np.dot(q_norm, v_norm.T)
        else:  # dot
            return -np.dot(query, vectors.T)


def faiss_topk(distances: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Get top-k nearest neighbors.

    Uses: faiss-topk.glsl

    Args:
        distances: Distance matrix (batch, num_vectors)
        k: Number of neighbors to retrieve

    Returns:
        (indices, topk_distances) - both (batch, k)
    """
    from grilly import Compute

    backend = Compute()
    if hasattr(backend, "faiss") and hasattr(backend.faiss, "topk"):
        return backend.faiss.topk(distances, k)
    else:
        # CPU fallback
        indices = np.argsort(distances, axis=1)[:, :k]
        topk_distances = np.take_along_axis(distances, indices, axis=1)
        return indices, topk_distances


def faiss_ivf_filter(vectors: np.ndarray, centroids: np.ndarray, nlist: int = 100) -> np.ndarray:
    """
    Filter vectors using IVF (Inverted File Index) structure.

    Uses: faiss-ivf-filter.glsl

    Args:
        vectors: Input vectors (num_vectors, dim)
        centroids: Cluster centroids (nlist, dim)
        nlist: Number of clusters

    Returns:
        Cluster assignments (num_vectors,)
    """
    backend = _get_backend()

    # Try GPU shader if available
    if backend and hasattr(backend, "shaders") and "faiss-kmeans-update" in backend.shaders:
        try:
            # GPU FAISS kmeans update would go here
            # For now, use CPU fallback
            pass
        except Exception:
            pass  # Fall back to CPU

    # CPU fallback - Assign each vector to nearest centroid
    distances = np.linalg.norm(vectors[:, None, :] - centroids[None, :, :], axis=2)
    assignments = np.argmin(distances, axis=1)
    return assignments


def faiss_kmeans_update(
    vectors: np.ndarray, centroids: np.ndarray, assignments: np.ndarray, nlist: int
) -> np.ndarray:
    """
    Update K-means centroids.

    Uses: faiss-kmeans-update.glsl

    Args:
        vectors: Input vectors (num_vectors, dim)
        centroids: Current centroids (nlist, dim)
        assignments: Cluster assignments (num_vectors,)
        nlist: Number of clusters

    Returns:
        Updated centroids (nlist, dim)
    """
    backend = _get_backend()

    # Try GPU shader if available
    if backend and hasattr(backend, "shaders") and "faiss-kmeans-update" in backend.shaders:
        try:
            # GPU FAISS kmeans update would go here
            # For now, use CPU fallback
            pass
        except Exception:
            pass  # Fall back to CPU

    # CPU fallback
    new_centroids = np.zeros_like(centroids)
    counts = np.zeros(nlist, dtype=np.int32)

    for i, vec in enumerate(vectors):
        cluster = assignments[i]
        new_centroids[cluster] += vec
        counts[cluster] += 1

    # Normalize by counts
    for c in range(nlist):
        if counts[c] > 0:
            new_centroids[c] /= counts[c]
        else:
            new_centroids[c] = centroids[c]  # Keep old centroid if no vectors assigned

    return new_centroids


def faiss_quantize(
    vectors: np.ndarray, codebook: np.ndarray, nbits: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize vectors using codebook.

    Uses: faiss-quantize.glsl

    Args:
        vectors: Input vectors (num_vectors, dim)
        codebook: Codebook (num_codes, dim)
        nbits: Number of bits for quantization

    Returns:
        (quantized_vectors, codes) - codes (num_vectors,)
    """
    backend = _get_backend()

    # Try GPU shader if available
    if backend and hasattr(backend, "shaders") and "faiss-quantize" in backend.shaders:
        try:
            # GPU FAISS quantization would go here
            # For now, use CPU fallback
            pass
        except Exception:
            pass  # Fall back to CPU

    # CPU fallback - Find nearest codebook entry for each vector
    distances = np.linalg.norm(vectors[:, None, :] - codebook[None, :, :], axis=2)
    codes = np.argmin(distances, axis=1)
    quantized = codebook[codes]
    return quantized, codes
