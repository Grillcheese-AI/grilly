"""
Benchmark: VSA (Vector Symbolic Architecture) operations.
"""

import sys

import numpy as np

sys.path.insert(0, ".")

from benchmarks.utils import (
    print_header,
    print_row,
    print_summary_table,
    time_cpu,
    time_fn,
)


def cpu_bind(a, b):
    return a * b


def cpu_bundle(vectors):
    s = np.sum(vectors, axis=0)
    return np.sign(s)


def cpu_similarity(query, codebook):
    norms_q = np.linalg.norm(query)
    norms_cb = np.linalg.norm(codebook, axis=1)
    dots = codebook @ query
    return dots / (norms_q * norms_cb + 1e-8)


def main():
    print_header("VSA Operations Benchmark")

    # Try to get GPU VSA backend
    gpu_vsa = None
    try:
        from grilly.backend.core import VulkanCore

        core = VulkanCore()
        from grilly.backend.experimental.vsa import VulkanVSA

        gpu_vsa = VulkanVSA(core)
    except Exception as e:
        print(f"GPU VSA unavailable: {e}")

    dims = [1024, 4096, 10000]
    results = []

    # Bind benchmark
    print_header("Bind (element-wise multiply)")
    for dim in dims:
        a = np.random.choice([-1.0, 1.0], size=dim).astype(np.float32)
        b = np.random.choice([-1.0, 1.0], size=dim).astype(np.float32)

        cpu_res = time_cpu(cpu_bind, a, b, warmup=3, repeats=10)

        if gpu_vsa is not None:
            try:
                gpu_res = time_fn(gpu_vsa.bind_bipolar, a, b, warmup=2, repeats=5)
                print_row(f"bind d={dim}", gpu_res["mean"], cpu_res["mean"])
                results.append(
                    {
                        "label": f"bind d={dim}",
                        "gpu_ms": gpu_res["mean"],
                        "cpu_ms": cpu_res["mean"],
                        "shape": f"d={dim}",
                    }
                )
            except Exception as e:
                print(f"  bind d={dim}: GPU failed: {e}")
        else:
            print(f"  bind d={dim}: CPU only: {cpu_res['mean']:.3f} ms")

    # Bundle benchmark
    print_header("Bundle (majority vote)")
    for dim in dims:
        n_vectors = 5
        vectors = [
            np.random.choice([-1.0, 1.0], size=dim).astype(np.float32) for _ in range(n_vectors)
        ]
        vectors_stack = np.stack(vectors)

        cpu_res = time_cpu(cpu_bundle, vectors_stack, warmup=3, repeats=10)

        if gpu_vsa is not None:
            try:
                gpu_res = time_fn(gpu_vsa.bundle, vectors, warmup=2, repeats=5)
                print_row(f"bundle d={dim} n={n_vectors}", gpu_res["mean"], cpu_res["mean"])
                results.append(
                    {
                        "label": f"bundle d={dim}",
                        "gpu_ms": gpu_res["mean"],
                        "cpu_ms": cpu_res["mean"],
                        "shape": f"d={dim}",
                    }
                )
            except Exception as e:
                print(f"  bundle d={dim}: GPU failed: {e}")
        else:
            print(f"  bundle d={dim}: CPU only: {cpu_res['mean']:.3f} ms")

    # Similarity benchmark
    print_header("Similarity (cosine)")
    for dim in dims:
        vocab_size = 1000
        query = np.random.randn(dim).astype(np.float32)
        codebook = np.random.randn(vocab_size, dim).astype(np.float32)

        cpu_res = time_cpu(cpu_similarity, query, codebook, warmup=2, repeats=5)

        if gpu_vsa is not None:
            try:
                gpu_res = time_fn(
                    gpu_vsa.similarity_batch, query, codebook, vocab_size, dim, warmup=2, repeats=5
                )
                print_row(f"sim V={vocab_size} d={dim}", gpu_res["mean"], cpu_res["mean"])
                results.append(
                    {
                        "label": f"sim V={vocab_size} d={dim}",
                        "gpu_ms": gpu_res["mean"],
                        "cpu_ms": cpu_res["mean"],
                        "shape": f"V={vocab_size} d={dim}",
                    }
                )
            except Exception as e:
                print(f"  sim V={vocab_size} d={dim}: GPU failed: {e}")
        else:
            print(f"  sim V={vocab_size} d={dim}: CPU only: {cpu_res['mean']:.3f} ms")

    if results:
        print_header("VSA Summary")
        print_summary_table(results)


if __name__ == "__main__":
    main()
