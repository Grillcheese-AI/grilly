"""
Example: VSA Batch Operations

Demonstrates CPU and GPU batch bind, bundle, and similarity.
"""

import numpy as np
from grilly.experimental.vsa.ops import BinaryOps


def main() -> None:
    dim = 1024
    batch_size = 4
    num_vectors = 5

    print("=" * 60)
    print("VSA Batch Operations")
    print("=" * 60)

    a_batch = np.array([BinaryOps.random_bipolar(dim) for _ in range(batch_size)])
    b_batch = np.array([BinaryOps.random_bipolar(dim) for _ in range(batch_size)])

    cpu_bind = BinaryOps.bind_batch(a_batch, b_batch)
    print(f"CPU bind batch shape: {cpu_bind.shape}")

    vectors = np.array(
        [[BinaryOps.random_bipolar(dim) for _ in range(num_vectors)] for _ in range(batch_size)],
        dtype=np.float32,
    )

    cpu_bundle = BinaryOps.bundle_batch(vectors)
    print(f"CPU bundle batch shape: {cpu_bundle.shape}")

    query = BinaryOps.random_bipolar(dim)
    codebook = np.array([BinaryOps.random_bipolar(dim) for _ in range(8)])
    cpu_sims = BinaryOps.similarity_batch(query, codebook)
    print(f"CPU similarity batch shape: {cpu_sims.shape}")

    try:
        from grilly.backend.core import VulkanCore
        from grilly.backend.experimental.vsa import VulkanVSA

        core = VulkanCore()
        vsa = VulkanVSA(core)

        gpu_bind = vsa.bind_bipolar_batch(a_batch, b_batch)
        print(f"GPU bind batch matches CPU: {np.allclose(cpu_bind, gpu_bind, atol=1e-4)}")

        gpu_bundle = vsa.bundle_batch(vectors)
        print(f"GPU bundle batch matches CPU: {np.allclose(cpu_bundle, gpu_bundle, atol=1e-4)}")

        gpu_sims = vsa.similarity_batch(query, codebook)
        print(f"GPU similarity batch matches CPU: {np.allclose(cpu_sims, gpu_sims, atol=1e-4)}")
    except RuntimeError:
        print("Vulkan not available, GPU batch skipped")
    except ImportError:
        print("Vulkan backend not available, GPU batch skipped")


if __name__ == "__main__":
    main()
