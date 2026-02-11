"""
Example: Vector Symbolic Architecture Operations

Demonstrates binary and holographic VSA operations including binding,
unbinding, bundling, and similarity computation.
"""

from grilly.experimental.vsa import BinaryOps, HolographicOps

print("=" * 60)
print("VSA Operations Examples")
print("=" * 60)

# Binary Operations
print("\n1. Binary Operations (Bipolar Vectors)")
print("-" * 60)

dim = 1024
a = BinaryOps.random_bipolar(dim)
b = BinaryOps.random_bipolar(dim)

print(f"Input vectors: shape={a.shape}, dtype={a.dtype}")
print(f"Values: {a[:5]} ... (bipolar: +1 or -1)")

bound = BinaryOps.bind(a, b)
print(f"\nBound vector: shape={bound.shape}")
print(f"Values: {bound[:5]} ...")

unbound = BinaryOps.unbind(bound, b)
similarity = BinaryOps.similarity(a, unbound)
print(f"\nUnbind recovery similarity: {similarity:.4f} (should be ~1.0)")

bundled = BinaryOps.bundle([a, b, BinaryOps.random_bipolar(dim)])
print(f"\nBundled vector: shape={bundled.shape}")
print(f"Values: {bundled[:5]} ...")

# Holographic Operations
print("\n2. Holographic Operations (Continuous Vectors)")
print("-" * 60)

vec1 = HolographicOps.random_vector(dim)
vec2 = HolographicOps.random_vector(dim)

print(f"Input vectors: shape={vec1.shape}, dtype={vec1.dtype}")
print(f"Sample values: {vec1[:5]}")

bound_h = HolographicOps.bind(vec1, vec2)
print(f"\nBound vector: shape={bound_h.shape}")

unbound_h = HolographicOps.unbind(bound_h, vec2)
similarity_h = HolographicOps.similarity(vec1, unbound_h)
print(f"\nUnbind recovery similarity: {similarity_h:.4f} (approximate)")

bundled_h = HolographicOps.bundle([vec1, vec2, HolographicOps.random_vector(dim)])
print(f"\nBundled vector: shape={bundled_h.shape}")

# Similarity comparison
print("\n3. Similarity Comparison")
print("-" * 60)

similar = HolographicOps.random_vector(dim)
different = HolographicOps.random_vector(dim)

sim_same = HolographicOps.similarity(similar, similar)
sim_diff = HolographicOps.similarity(similar, different)

print(f"Self-similarity: {sim_same:.4f}")
print(f"Different vectors similarity: {sim_diff:.4f}")
