#!/usr/bin/env python3
"""
INT8 weight GEMM baseline (Workstream C2) — `VulkanFNN.gemm_int8` if shader loaded.

Usage:
  uv run python benchmarks/benchmark_int8_gemm.py
"""

from __future__ import annotations

import time
import warnings

import numpy as np

from grilly.backend.compute import VulkanCompute


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        backend = VulkanCompute()
    try:
        if "int8-gemm" not in backend.fnn.shaders:
            print("int8-gemm shader not loaded; skipping benchmark.")
            return

        rng = np.random.default_rng(1)
        M, K, N = 256, 512, 256
        group_size = 64
        num_groups = (K + group_size - 1) // group_size

        act = rng.standard_normal((M, K), dtype=np.float32)
        w_i8 = rng.integers(-128, 127, size=(N, K), dtype=np.int8)
        scales = np.abs(rng.standard_normal((N, num_groups), dtype=np.float32)) + 0.01

        backend.fnn.gemm_int8(act, w_i8, scales, group_size=group_size)

        n = 10
        t0 = time.perf_counter()
        for _ in range(n):
            backend.fnn.gemm_int8(act, w_i8, scales, group_size=group_size)
        elapsed = (time.perf_counter() - t0) / n
        print(f"int8_gemm M={M} K={K} N={N}: {elapsed * 1000:.3f} ms/iter (mean of {n})")
    finally:
        backend.cleanup()


if __name__ == "__main__":
    main()
