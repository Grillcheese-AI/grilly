"""
Profile GPU-vs-CPU bottlenecks for parity-critical ops.

Usage:
    python benchmarks/profile_gpu_bottlenecks.py
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time

import numpy as np

from grilly.nn.attention import MultiheadAttention
from grilly.nn.linear import Linear


def _print_stats(pr: cProfile.Profile, title: str, top_n: int = 25):
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumtime").print_stats(top_n)
    print(f"\n=== {title} ===")
    print(s.getvalue())


def profile_linear():
    x = np.random.randn(64, 512).astype(np.float32)
    linear = Linear(512, 512)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(20):
        _ = linear(x)
    pr.disable()
    _print_stats(pr, "Linear forward profile")

    w = np.asarray(linear.weight, dtype=np.float32)
    b = np.asarray(linear.bias, dtype=np.float32)
    t0 = time.perf_counter()
    for _ in range(20):
        _ = x @ w.T + b
    t1 = time.perf_counter()
    print(f"CPU linear baseline (20 iters): {(t1 - t0) * 1000.0:.3f} ms")


def profile_attention():
    attn = MultiheadAttention(embed_dim=512, num_heads=8)
    q = np.random.randn(2, 64, 512).astype(np.float32)
    k = np.random.randn(2, 64, 512).astype(np.float32)
    v = np.random.randn(2, 64, 512).astype(np.float32)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(10):
        _ = attn(q, k, v)
    pr.disable()
    _print_stats(pr, "MultiheadAttention forward profile")


def main():
    print("Profiling parity-critical GPU paths...")
    profile_linear()
    profile_attention()


if __name__ == "__main__":
    main()
