"""
Example: Capsule-Enhanced Cognitive Components

Shows capsule vectors for working memory and world facts.
"""

from grilly.experimental.cognitive import WorkingMemory, WorkingMemorySlot, WorldModel
from grilly.experimental.vsa.ops import HolographicOps


def main() -> None:
    dim = 1024

    print("=" * 60)
    print("Capsule-Enhanced Cognitive Components")
    print("=" * 60)

    wm = WorkingMemory(dim=dim, capacity=5)

    vec_a = HolographicOps.random_vector(dim)
    vec_b = HolographicOps.random_vector(dim)

    wm.add(vec_a, "dog runs", WorkingMemorySlot.CONTEXT, confidence=0.9)
    wm.add(vec_b, "cat sleeps", WorkingMemorySlot.CONTEXT, confidence=0.8)

    context_vec = wm.get_context_vector()
    context_capsule = wm.get_context_capsule()

    print(f"Context vector shape: {context_vec.shape}")
    print(f"Context capsule shape: {context_capsule.shape}")

    world = WorldModel(dim=dim)
    world.add_fact("dog", "is", "animal")

    fact = world.facts[0]
    print(f"Fact capsule available: {fact.capsule_vector is not None}")

    is_known, confidence = world.query_fact("dog", "is", "animal")
    print(f"Query fact: {is_known}, confidence: {confidence:.3f}")


if __name__ == "__main__":
    main()
