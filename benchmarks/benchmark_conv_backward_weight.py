#!/usr/bin/env python3
"""
Baseline timings for conv2d backward weight (Workstream C1).

The Vulkan training path uses `conv2d-backward-weight` (non-atomic, one thread
per weight slot) or the GEMM path (im2col + CPU matmul when `convd_im2col` is
available). This script prints wall time for a representative backward-weight call.

Usage:
  uv run python benchmarks/benchmark_conv_backward_weight.py
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
        rng = np.random.default_rng(0)
        batch_size = 4
        in_ch, out_ch = 32, 64
        h, w = 16, 16
        kh, kw = 3, 3
        grad_out = rng.standard_normal((batch_size, out_ch, h, w), dtype=np.float32)
        inp = rng.standard_normal((batch_size, in_ch, h, w), dtype=np.float32)

        # Warmup
        backend.conv.conv2d_backward_weight(
            grad_out,
            inp,
            (kh, kw),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            has_bias=True,
        )

        n = 5
        t0 = time.perf_counter()
        for _ in range(n):
            backend.conv.conv2d_backward_weight(
                grad_out,
                inp,
                (kh, kw),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                groups=1,
                has_bias=True,
            )
        elapsed = (time.perf_counter() - t0) / n
        print(f"conv2d_backward_weight: {elapsed * 1000:.3f} ms/iter (mean of {n})")
    finally:
        backend.cleanup()


if __name__ == "__main__":
    main()
