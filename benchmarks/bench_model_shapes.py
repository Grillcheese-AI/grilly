"""
Benchmark: End-to-end inference for typical model architectures.
"""

import numpy as np
import sys
import time
sys.path.insert(0, '.')

from benchmarks.utils import (
    print_header, print_summary_table, format_time, get_gpu_backend,
)


def bench_stacked_linear(backend, label, batch, dims, repeats=5):
    """Benchmark a stack of linear layers."""
    from grilly.nn.modules import Linear

    layers = []
    for i in range(len(dims) - 1):
        layers.append(Linear(dims[i], dims[i + 1]))

    x = np.random.randn(batch, dims[0]).astype(np.float32)

    def forward(x):
        h = x
        for layer in layers:
            h = layer(h)
            h = np.maximum(0, h)  # ReLU
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
        'label': label,
        'gpu_ms': np.mean(times),
        'shape': f"B={batch} {' -> '.join(str(d) for d in dims)}",
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

    results = []

    for label, batch, dims in configs:
        print(f"\n  {label}: batch={batch}, dims={dims}")

        # CPU
        cpu_ms = bench_cpu_stacked_linear(label, batch, dims, repeats=5)

        # GPU
        if backend is not None:
            try:
                r = bench_stacked_linear(backend, label, batch, dims, repeats=5)
                r['cpu_ms'] = cpu_ms
                results.append(r)
                speedup = cpu_ms / r['gpu_ms'] if r['gpu_ms'] > 0 else 0
                print(f"    GPU: {format_time(r['gpu_ms'])}  CPU: {format_time(cpu_ms)}  "
                      f"Speedup: {speedup:.1f}x")
            except Exception as e:
                print(f"    GPU failed: {e}")
                results.append({
                    'label': label, 'gpu_ms': 0, 'cpu_ms': cpu_ms,
                    'shape': f"B={batch}"
                })
        else:
            print(f"    CPU only: {format_time(cpu_ms)}")

    if results:
        print_header("Model Shape Summary")
        print_summary_table(results)


if __name__ == '__main__':
    main()
