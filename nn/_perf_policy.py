"""Adaptive runtime policy to avoid slower-than-CPU execution."""

from __future__ import annotations

import os
import time

_AUTO_FASTEST = os.getenv("GRILLY_AUTO_FASTEST", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_DECISIONS: dict[str, str] = {}


def _time_ms(fn, warmup: int = 1, repeats: int = 2):
    for _ in range(max(0, warmup)):
        fn()
    t = []
    out = None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        out = fn()
        t1 = time.perf_counter()
        t.append((t1 - t0) * 1000.0)
    return (sum(t) / len(t)), out


def choose_fastest(op_key: str, gpu_fn, cpu_fn):
    """Run fastest backend for this op/shape key and cache choice."""
    if not _AUTO_FASTEST:
        return gpu_fn()

    decision = _DECISIONS.get(op_key)
    if decision == "gpu":
        try:
            out = gpu_fn()
            if out is not None:
                return out
        except Exception:
            pass
        _DECISIONS[op_key] = "cpu"
        return cpu_fn()
    if decision == "cpu":
        return cpu_fn()

    # First encounter for this shape/op: benchmark both paths.
    gpu_ms = float("inf")
    gpu_out = None
    try:
        gpu_ms, gpu_out = _time_ms(gpu_fn, warmup=1, repeats=2)
    except Exception:
        gpu_ms = float("inf")
        gpu_out = None

    try:
        cpu_ms, cpu_out = _time_ms(cpu_fn, warmup=1, repeats=2)
    except Exception:
        # If CPU path fails, force GPU.
        _DECISIONS[op_key] = "gpu"
        return gpu_out

    if gpu_out is not None and gpu_ms <= cpu_ms * 0.98:
        _DECISIONS[op_key] = "gpu"
        return gpu_out

    _DECISIONS[op_key] = "cpu"
    return cpu_out


def get_perf_decisions() -> dict[str, str]:
    return dict(_DECISIONS)


def reset_perf_decisions():
    _DECISIONS.clear()
