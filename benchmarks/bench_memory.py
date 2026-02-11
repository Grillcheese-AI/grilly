"""
Benchmark: Memory read/write operations at various bank sizes.
"""

import sys

import numpy as np

sys.path.insert(0, ".")

from benchmarks.utils import (
    get_gpu_backend,
    print_header,
    print_row,
    print_summary_table,
    time_cpu,
    time_fn,
)


def cpu_memory_read(memory_bank, keys, num_heads=1):
    """CPU reference for memory read (attention-based)."""
    # Simplified: softmax(keys @ memory.T) @ memory
    scores = keys @ memory_bank.T
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / (np.sum(weights, axis=-1, keepdims=True) + 1e-8)
    return weights @ memory_bank


def cpu_memory_write(memory_bank, keys, values, lr=0.1):
    """CPU reference for memory write."""
    update = np.outer(
        keys.flatten()[: memory_bank.shape[0]], values.flatten()[: memory_bank.shape[1]]
    )
    return memory_bank + lr * update[: memory_bank.shape[0], : memory_bank.shape[1]]


def main():
    print_header("Memory Operations Benchmark")

    backend = get_gpu_backend()
    has_memory = backend is not None and hasattr(backend, "memory")

    configs = [
        # (num_slots, dim)
        (64, 128),
        (128, 256),
        (256, 512),
        (512, 512),
        (1024, 256),
    ]

    results = []

    print_header("Memory Read")
    for num_slots, dim in configs:
        label = f"read slots={num_slots} dim={dim}"
        memory_bank = np.random.randn(num_slots, dim).astype(np.float32)
        keys = np.random.randn(1, dim).astype(np.float32)

        cpu_res = time_cpu(cpu_memory_read, memory_bank, keys, warmup=2, repeats=5)

        if has_memory:
            try:
                gpu_res = time_fn(
                    backend.memory.memory_read, memory_bank, keys, 1, warmup=2, repeats=3
                )
                print_row(label, gpu_res["mean"], cpu_res["mean"])
                results.append(
                    {
                        "label": label,
                        "gpu_ms": gpu_res["mean"],
                        "cpu_ms": cpu_res["mean"],
                        "shape": f"({num_slots},{dim})",
                    }
                )
            except Exception as e:
                print(f"  {label}: GPU failed: {e}")
        else:
            print(f"  {label}: CPU only: {cpu_res['mean']:.3f} ms")

    print_header("Memory Write")
    for num_slots, dim in configs:
        label = f"write slots={num_slots} dim={dim}"
        memory_bank = np.random.randn(num_slots, dim).astype(np.float32)
        keys = np.random.randn(1, dim).astype(np.float32)
        values = np.random.randn(1, dim).astype(np.float32)

        cpu_res = time_cpu(cpu_memory_write, memory_bank, keys, values, warmup=2, repeats=5)

        if has_memory:
            try:
                gpu_res = time_fn(
                    backend.memory.memory_write, memory_bank, keys, values, 1, warmup=2, repeats=3
                )
                print_row(label, gpu_res["mean"], cpu_res["mean"])
                results.append(
                    {
                        "label": label,
                        "gpu_ms": gpu_res["mean"],
                        "cpu_ms": cpu_res["mean"],
                        "shape": f"({num_slots},{dim})",
                    }
                )
            except Exception as e:
                print(f"  {label}: GPU failed: {e}")
        else:
            print(f"  {label}: CPU only: {cpu_res['mean']:.3f} ms")

    if results:
        print_header("Memory Summary")
        print_summary_table(results)


if __name__ == "__main__":
    main()
