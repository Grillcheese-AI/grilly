"""
Example: Resonator Network

Demonstrates factorization of composite VSA vectors into constituent components.
"""

from grilly.experimental.vsa import HolographicOps, ResonatorNetwork

print("=" * 60)
print("Resonator Network Examples")
print("=" * 60)

dim = 2048
codebook_size = 20

# Create codebook
codebook = [HolographicOps.random_vector(dim) for _ in range(codebook_size)]
print(f"\nCodebook: {len(codebook)} vectors of dimension {dim}")

# Create composite vector
idx_a, idx_b = 3, 7
composite = HolographicOps.bind(codebook[idx_a], codebook[idx_b])
print(f"\nComposite vector created from indices {idx_a} and {idx_b}")

# Initialize resonator
resonator = ResonatorNetwork(codebook=codebook, max_iterations=50)
print(f"Resonator initialized with max_iterations={resonator.max_iterations}")

# Factorize
factors, iterations = resonator.factorize(composite, num_factors=2)
print(f"\nFactorization complete in {iterations} iterations")
print(f"Recovered factors: {factors}")
print(f"Expected factors: [{idx_a}, {idx_b}]")

# Verify recovery
if set(factors) == {idx_a, idx_b}:
    print("Factorization successful: correct factors recovered")
else:
    print("Factorization partial: some factors may differ")

# Single factor example
print("\n" + "-" * 60)
print("Single Factor Factorization")
print("-" * 60)

single_composite = codebook[5]
single_factors, single_iters = resonator.factorize(single_composite, num_factors=1)
print(f"Single factor recovered: {single_factors[0]} (expected: 5)")
print(f"Iterations: {single_iters}")
