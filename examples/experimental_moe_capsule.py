"""
Example: Capsule-Aware MoE Routing

Demonstrates routing with capsule similarity blending.
"""

import numpy as np
from grilly.experimental.cognitive.capsule import CapsuleEncoder
from grilly.experimental.moe.routing import ResonatorMoE


def main() -> None:
    np.random.seed(7)
    dim = 1024

    def expert_a(x: np.ndarray) -> np.ndarray:
        return x * 1.0

    def expert_b(x: np.ndarray) -> np.ndarray:
        return x * 1.0

    experts = {"expert_a": expert_a, "expert_b": expert_b}

    query = np.random.randn(dim).astype(np.float32)

    encoder = CapsuleEncoder(input_dim=dim)
    query_capsule = encoder.encode_vector(query)

    expert_capsules = {"expert_a": query_capsule, "expert_b": -query_capsule}

    moe_capsule = ResonatorMoE(
        dim=dim,
        experts=experts,
        expert_capsules=expert_capsules,
        capsule_encoder=encoder,
        capsule_weight=1.0,
    )

    selected_capsule = moe_capsule.route(query, top_k=1)
    print("Capsule-weighted selection:", selected_capsule)

    moe_vsa = ResonatorMoE(
        dim=dim, experts=experts, expert_vectors=moe_capsule.expert_vectors, capsule_weight=0.0
    )

    selected_vsa = moe_vsa.route(query, top_k=1)
    print("VSA-only selection:", selected_vsa)


if __name__ == "__main__":
    main()
