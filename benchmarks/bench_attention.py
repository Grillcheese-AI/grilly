"""
Benchmark: Multi-head attention at various shapes.
"""

import sys

import numpy as np

sys.path.insert(0, ".")

from benchmarks.utils import (
    format_size,
    get_gpu_backend,
    print_header,
    print_row,
    print_summary_table,
    time_cpu,
    time_fn,
)


def cpu_attention_scores(q, k, scale):
    """CPU reference for attention scores."""
    # q, k: (batch, seq, heads, head_dim)
    # transpose to (batch, heads, seq, head_dim)
    q_t = q.transpose(0, 2, 1, 3)
    k_t = k.transpose(0, 2, 1, 3)
    scores = np.matmul(q_t, k_t.transpose(0, 1, 3, 2)) * scale
    return scores


def main():
    print_header("Multi-Head Attention Benchmark")

    backend = get_gpu_backend()
    if backend is None:
        print("GPU backend unavailable, skipping.")
        return

    has_attention = hasattr(backend, "attention")
    if not has_attention:
        print("Attention module not available on backend.")
        return

    configs = [
        # (batch, seq_len, num_heads, head_dim)
        (1, 32, 4, 32),
        (2, 64, 8, 64),
        (4, 128, 8, 64),
        (2, 256, 12, 64),
        (1, 512, 12, 64),
    ]

    results = []

    for batch, seq, heads, hdim in configs:
        label = f"B={batch} S={seq} H={heads} D={hdim}"
        embed_dim = heads * hdim
        q = np.random.randn(batch, seq, heads, hdim).astype(np.float32)
        k = np.random.randn(batch, seq, heads, hdim).astype(np.float32)
        scale = 1.0 / np.sqrt(hdim)

        data_bytes = q.nbytes + k.nbytes
        print(f"\n  Config: {label}  Data: {format_size(data_bytes)}")

        # CPU timing
        cpu_result = time_cpu(cpu_attention_scores, q, k, scale, warmup=1, repeats=3)
        cpu_ms = cpu_result["mean"]

        # GPU timing
        try:
            gpu_result = time_fn(
                backend.attention.attention_scores,
                q.reshape(batch, seq, embed_dim),
                k.reshape(batch, seq, embed_dim),
                heads,
                hdim,
                scale,
                warmup=1,
                repeats=3,
            )
            gpu_ms = gpu_result["mean"]
            print_row(label, gpu_ms, cpu_ms)
            results.append(
                {
                    "label": label,
                    "gpu_ms": gpu_ms,
                    "cpu_ms": cpu_ms,
                    "shape": f"({batch},{seq},{heads},{hdim})",
                }
            )
        except Exception as e:
            print(f"    GPU failed: {e}")
            results.append(
                {
                    "label": label,
                    "gpu_ms": 0,
                    "cpu_ms": cpu_ms,
                    "shape": f"({batch},{seq},{heads},{hdim})",
                }
            )

    if results:
        print_header("Attention Summary")
        print_summary_table(results)


if __name__ == "__main__":
    main()
