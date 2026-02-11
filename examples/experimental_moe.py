"""
Example: Resonator-based Mixture of Experts

Demonstrates relational encoding and expert routing using VSA operations.
"""

from grilly.experimental.moe import RelationalEncoder, RelationalMoE, ResonatorMoE
from grilly.experimental.vsa import HolographicOps

print("=" * 60)
print("Mixture of Experts Examples")
print("=" * 60)

dim = 2048

# Relational Encoder
print("\n1. Relational Encoder")
print("-" * 60)

encoder = RelationalEncoder(dim=dim)

cat_vec = encoder.encode("cat", modality="text")
dog_vec = encoder.encode("dog", modality="text")
print(f"Encoded 'cat': shape={cat_vec.shape}")
print(f"Encoded 'dog': shape={dog_vec.shape}")

relation = encoder.extract_relation("cat", "chases", "mouse")
print(f"\nExtracted relation 'chases': shape={relation.shape}")

opposite = encoder.get_opposite(relation)
print(f"Opposite relation: shape={opposite.shape}")

# Resonator MoE
print("\n2. Resonator MoE Routing")
print("-" * 60)


def expert_a(x):
    return x * 1.1


def expert_b(x):
    return x * 0.9


def expert_c(x):
    return x * 1.0


experts = {"expert_a": expert_a, "expert_b": expert_b, "expert_c": expert_c}

expert_vectors = {
    "expert_a": HolographicOps.random_vector(dim),
    "expert_b": HolographicOps.random_vector(dim),
    "expert_c": HolographicOps.random_vector(dim),
}

moe = ResonatorMoE(dim=dim, experts=experts, expert_vectors=expert_vectors)
print(f"MoE initialized with {len(experts)} experts")

query = HolographicOps.random_vector(dim)
result = moe.route(query, top_k=2)

print("\nQuery routed to top-2 experts:")
print(f"Expert names: {result}")
weights = moe.get_weights(query, normalize=True)
print(f"Weights: {weights}")

# Relational MoE
print("\n3. Relational MoE")
print("-" * 60)

relational_experts = {
    "chase_expert": lambda x: x,
    "eat_expert": lambda x: x,
    "sleep_expert": lambda x: x,
}

expert_relations = {
    "chase_expert": ("cat", "mouse"),
    "eat_expert": ("cat", "fish"),
    "sleep_expert": ("cat", "bed"),
}

rel_moe = RelationalMoE(
    dim=dim,
    experts=relational_experts,
    expert_relations=expert_relations,
    relational_encoder=encoder,
)

query_entity = encoder.encode("cat", modality="text")
rel_result = rel_moe.route(query_entity, top_k=2)

print("Relational routing for 'cat':")
print(f"Selected experts: {rel_result}")
rel_weights = rel_moe.get_weights(query_entity, normalize=True)
print(f"Weights: {rel_weights}")
