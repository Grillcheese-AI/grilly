"""
Benchmark: End-to-end inference for typical model architectures.

Compares three modes:
  1. CPU-only (pure numpy)
  2. GPU standard (numpy arrays, full CPU<->GPU transfers each op)
  3. GPU-resident (VulkanTensor, data stays on GPU between ops)
"""

import sys
import time

import numpy as np

sys.path.insert(0, ".")

from benchmarks.utils import (
    format_time,
    get_gpu_backend,
    print_header,
    print_summary_table,
)


def bench_stacked_linear(backend, label, batch, dims, repeats=5):
    """Benchmark a stack of linear layers (standard numpy mode)."""
    from grilly.nn.modules import Linear, ReLU

    layers = []
    for i in range(len(dims) - 1):
        layers.append(Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(ReLU())

    x = np.random.randn(batch, dims[0]).astype(np.float32)

    def forward(x):
        h = x
        for layer in layers:
            h = layer(h)
        return h

    # Warmup
    for _ in range(2):
        forward(x)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        forward(x)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return {
        "label": label,
        "gpu_ms": np.mean(times),
        "shape": f"B={batch} {' -> '.join(str(d) for d in dims)}",
    }


def bench_stacked_linear_gpu_resident(backend, label, batch, dims, repeats=5):
    """Benchmark a stack of linear layers (GPU-resident VulkanTensor mode)."""
    from grilly.nn.modules import Linear, ReLU
    from grilly.utils.tensor_conversion import VulkanTensor

    layers = []
    for i in range(len(dims) - 1):
        layers.append(Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(ReLU())

    # Enable GPU-resident mode on all layers
    for layer in layers:
        layer.gpu_mode(True)

    x_np = np.random.randn(batch, dims[0]).astype(np.float32)
    x_gpu = VulkanTensor(x_np)

    def forward(x):
        h = x
        for layer in layers:
            h = layer(h)
        # Download final result
        if isinstance(h, VulkanTensor):
            return h.numpy()
        return h

    # Warmup
    for _ in range(2):
        forward(x_gpu)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        forward(x_gpu)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return {
        "label": label + " (GPU-res)",
        "gpu_ms": np.mean(times),
        "shape": f"B={batch} {' -> '.join(str(d) for d in dims)}",
    }


def bench_cpu_stacked_linear(label, batch, dims, repeats=5):
    """CPU-only benchmark for comparison."""
    weights = []
    biases = []
    for i in range(len(dims) - 1):
        limit = np.sqrt(6.0 / (dims[i] + dims[i + 1]))
        W = np.random.uniform(-limit, limit, (dims[i + 1], dims[i])).astype(np.float32)
        b = np.zeros(dims[i + 1], dtype=np.float32)
        weights.append(W)
        biases.append(b)

    x = np.random.randn(batch, dims[0]).astype(np.float32)

    def forward(x):
        h = x
        for W, b in zip(weights, biases):
            h = h @ W.T + b
            h = np.maximum(0, h)
        return h

    # Warmup
    for _ in range(2):
        forward(x)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        forward(x)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return np.mean(times)


def main():
    print_header("Model Shape Inference Benchmark")

    backend = get_gpu_backend()

    # Model configs: (label, batch, [dim0, dim1, ..., dimN])
    configs = [
        ("Small MLP 3-layer", 32, [512, 256, 128, 64]),
        ("Medium MLP 4-layer", 64, [1024, 512, 256, 128, 64]),
        ("BERT-like FFN", 32, [768, 3072, 768]),
        ("GPT-like FFN", 16, [512, 2048, 512]),
        ("Large MLP 6-layer", 16, [2048, 1024, 512, 256, 128, 64, 32]),
        ("Wide MLP", 8, [4096, 4096, 4096]),
    ]

    std_results = []
    gpu_results = []

    for label, batch, dims in configs:
        print(f"\n  {label}: batch={batch}, dims={dims}")

        # CPU
        cpu_ms = bench_cpu_stacked_linear(label, batch, dims, repeats=5)

        if backend is not None:
            # GPU standard mode
            try:
                r = bench_stacked_linear(backend, label, batch, dims, repeats=5)
                r["cpu_ms"] = cpu_ms
                std_results.append(r)
                speedup = cpu_ms / r["gpu_ms"] if r["gpu_ms"] > 0 else 0
                print(
                    f"    Standard  GPU: {format_time(r['gpu_ms'])}  "
                    f"CPU: {format_time(cpu_ms)}  Speedup: {speedup:.2f}x"
                )
            except Exception as e:
                print(f"    Standard GPU failed: {e}")

            # GPU-resident mode
            try:
                r2 = bench_stacked_linear_gpu_resident(backend, label, batch, dims, repeats=5)
                r2["cpu_ms"] = cpu_ms
                gpu_results.append(r2)
                speedup2 = cpu_ms / r2["gpu_ms"] if r2["gpu_ms"] > 0 else 0
                print(
                    f"    GPU-res   GPU: {format_time(r2['gpu_ms'])}  "
                    f"CPU: {format_time(cpu_ms)}  Speedup: {speedup2:.2f}x"
                )
            except Exception as e:
                print(f"    GPU-resident failed: {e}")
        else:
            print(f"    CPU only: {format_time(cpu_ms)}")

    # Summary tables
    if std_results:
        print_header("Standard Mode Summary")
        print_summary_table(std_results)

    if gpu_results:
        print_header("GPU-Resident Mode Summary")
        print_summary_table(gpu_results)

    # Side-by-side comparison
    if std_results and gpu_results and len(std_results) == len(gpu_results):
        print_header("Standard vs GPU-Resident Comparison")
        print(f"  {'Model':<28} {'Standard':>10} {'GPU-res':>10} {'Transfer':>10} {'vs CPU':>8}")
        print("  " + "-" * 70)
        for std, gpur in zip(std_results, gpu_results):
            std_ms = std["gpu_ms"]
            gpur_ms = gpur["gpu_ms"]
            cpu_ms = std["cpu_ms"]
            transfer_elim = ""
            vs_cpu = ""
            if std_ms > 0 and gpur_ms > 0:
                transfer_elim = f"{std_ms / gpur_ms:.2f}x"
            if cpu_ms > 0 and gpur_ms > 0:
                vs_cpu = f"{cpu_ms / gpur_ms:.2f}x"
            print(
                f"  {std['label']:<28} "
                f"{format_time(std_ms):>10} "
                f"{format_time(gpur_ms):>10} "
                f"{transfer_elim:>10} "
                f"{vs_cpu:>8}"
            )
    print()


if __name__ == "__main__":
    main()
