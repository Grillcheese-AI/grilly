"""
Linear layer benchmark for Grilly.

Compares GPU-accelerated Linear layer (Vulkan compute shaders) against
CPU-based numpy matmul across model-relevant shapes. Tests both standard
numpy-array mode and GPU-resident VulkanTensor mode.

Usage:
    python benchmarks/bench_linear.py
"""

import os
import sys

import numpy as np

# Ensure the repo root is on the path so grilly and benchmarks are importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.utils import (
    format_size,
    format_time,
    get_gpu_backend,
    print_header,
    print_row,
    print_summary_table,
    time_cpu,
    time_fn,
)

# Model-relevant shapes: (batch_size, in_features) -> out_features
SHAPES = [
    {"name": "Small", "batch": 32, "in_features": 128, "out_features": 64},
    {"name": "Medium", "batch": 64, "in_features": 512, "out_features": 512},
    {"name": "Large", "batch": 256, "in_features": 1024, "out_features": 2048},
    {"name": "XL", "batch": 512, "in_features": 2048, "out_features": 4096},
]


def generate_input(batch, in_features, dtype=np.float32):
    """Generate a random float32 input tensor of shape (batch, in_features)."""
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal((batch, in_features)).astype(dtype)


def cpu_linear(x, weight, bias):
    """CPU reference: y = x @ W^T + b (mirrors nn.Linear semantics)."""
    out = x @ weight.T
    if bias is not None:
        out = out + bias
    return out


def verify_correctness(warmup=1):
    """Verify that GPU Linear layer output matches CPU numpy reference.

    Returns:
        list of (name, passed, max_diff) tuples.
    """
    from grilly.nn.modules import Linear

    print_header("Linear Correctness Verification")
    results = []

    for shape in SHAPES:
        name = shape["name"]
        batch = shape["batch"]
        in_f = shape["in_features"]
        out_f = shape["out_features"]

        model = Linear(in_f, out_f)
        x = generate_input(batch, in_f)

        # Extract weight and bias arrays for CPU reference
        weight = np.asarray(model.weight, dtype=np.float32)
        bias_param = model.bias
        bias = np.asarray(bias_param, dtype=np.float32) if bias_param is not None else None

        # GPU forward pass
        out_gpu = model(x)

        # CPU reference
        out_cpu = cpu_linear(x, weight, bias)

        # Tolerance scales with accumulation depth (in_features)
        atol = 1e-2 * max(1, in_f / 128)
        rtol = 1e-3

        passed = np.allclose(out_gpu, out_cpu, atol=atol, rtol=rtol)
        max_diff = float(np.max(np.abs(out_gpu - out_cpu)))

        status = "PASS" if passed else "FAIL"
        print(f"  {name:<8} ({batch},{in_f})->{out_f}  {status}  max_diff={max_diff:.6e}")
        results.append((name, passed, max_diff))

    return results


def benchmark_standard_mode(warmup=2, repeats=5):
    """Benchmark Linear layer in standard mode (numpy arrays).

    Returns:
        list of result dicts suitable for print_summary_table.
    """
    from grilly.nn.modules import Linear

    print_header("Linear Forward Pass: GPU vs CPU (numpy array mode)")
    summary = []

    for shape in SHAPES:
        name = shape["name"]
        batch = shape["batch"]
        in_f = shape["in_features"]
        out_f = shape["out_features"]

        model = Linear(in_f, out_f)
        x = generate_input(batch, in_f)

        # Extract weight and bias for CPU reference
        weight = np.asarray(model.weight, dtype=np.float32)
        bias_param = model.bias
        bias = np.asarray(bias_param, dtype=np.float32) if bias_param is not None else None

        label = f"{name} ({batch},{in_f})->{out_f}"
        data_bytes = x.nbytes + weight.nbytes

        # GPU timing (grilly Linear forward)
        gpu_stats = time_fn(model, x, warmup=warmup, repeats=repeats)
        gpu_ms = gpu_stats["mean"]

        # CPU timing (numpy matmul reference)
        cpu_stats = time_cpu(
            lambda w=weight, b=bias, inp=x: cpu_linear(inp, w, b),
            warmup=warmup,
            repeats=repeats,
        )
        cpu_ms = cpu_stats["mean"]

        # FLOP count: 2 * batch * in_features * out_features (multiply + add)
        flops = 2.0 * batch * in_f * out_f
        gpu_gflops = (flops / (gpu_ms / 1000.0)) / 1e9 if gpu_ms > 0 else 0
        cpu_gflops = (flops / (cpu_ms / 1000.0)) / 1e9 if cpu_ms > 0 else 0

        extra = f"GPU: {gpu_gflops:.1f} GFLOP/s  CPU: {cpu_gflops:.1f} GFLOP/s"
        print_row(label, gpu_ms, cpu_ms, extra=extra)

        summary.append(
            {
                "label": label,
                "gpu_ms": gpu_ms,
                "cpu_ms": cpu_ms,
                "shape": f"({batch},{in_f})->({batch},{out_f}) [{format_size(data_bytes)}]",
                "gpu_gflops": gpu_gflops,
                "cpu_gflops": cpu_gflops,
                "mode": "standard",
            }
        )

    return summary


def benchmark_gpu_mode(warmup=2, repeats=5):
    """Benchmark Linear layer in GPU-resident VulkanTensor mode.

    Returns:
        list of result dicts suitable for print_summary_table.
    """
    from grilly.nn.modules import Linear
    from grilly.utils.tensor_conversion import VulkanTensor

    print_header("Linear Forward Pass: GPU-resident VulkanTensor mode")
    summary = []

    for shape in SHAPES:
        name = shape["name"]
        batch = shape["batch"]
        in_f = shape["in_features"]
        out_f = shape["out_features"]

        model = Linear(in_f, out_f)
        model.gpu_mode(True)

        x_np = generate_input(batch, in_f)
        x_gpu = VulkanTensor(x_np)

        # Extract weight and bias for CPU reference
        weight = np.asarray(model.weight, dtype=np.float32)
        bias_param = model.bias
        bias = np.asarray(bias_param, dtype=np.float32) if bias_param is not None else None

        label = f"{name} GPU-res ({batch},{in_f})->{out_f}"
        data_bytes = x_np.nbytes + weight.nbytes

        # GPU-resident timing (VulkanTensor in, VulkanTensor out -- no CPU roundtrip)
        gpu_stats = time_fn(model, x_gpu, warmup=warmup, repeats=repeats)
        gpu_ms = gpu_stats["mean"]

        # CPU timing (numpy matmul reference, same as standard for comparison)
        cpu_stats = time_cpu(
            lambda w=weight, b=bias, inp=x_np: cpu_linear(inp, w, b),
            warmup=warmup,
            repeats=repeats,
        )
        cpu_ms = cpu_stats["mean"]

        flops = 2.0 * batch * in_f * out_f
        gpu_gflops = (flops / (gpu_ms / 1000.0)) / 1e9 if gpu_ms > 0 else 0
        cpu_gflops = (flops / (cpu_ms / 1000.0)) / 1e9 if cpu_ms > 0 else 0

        extra = f"GPU: {gpu_gflops:.1f} GFLOP/s  CPU: {cpu_gflops:.1f} GFLOP/s"
        print_row(label, gpu_ms, cpu_ms, extra=extra)

        summary.append(
            {
                "label": label,
                "gpu_ms": gpu_ms,
                "cpu_ms": cpu_ms,
                "shape": f"({batch},{in_f})->({batch},{out_f}) [{format_size(data_bytes)}]",
                "gpu_gflops": gpu_gflops,
                "cpu_gflops": cpu_gflops,
                "mode": "gpu_resident",
            }
        )

    return summary


def main():
    backend = get_gpu_backend()
    if backend is None:
        print("ERROR: GPU backend is required for this benchmark. Exiting.")
        sys.exit(1)

    # -- Correctness verification --
    correctness = verify_correctness()
    all_passed = all(passed for _, passed, _ in correctness)
    if not all_passed:
        print("\nWARNING: Some correctness checks failed! Results may be unreliable.")
    else:
        print("\nAll correctness checks passed.")

    # -- Standard mode benchmark (numpy arrays) --
    standard_results = benchmark_standard_mode()

    # -- GPU-resident mode benchmark (VulkanTensor) --
    gpu_results = benchmark_gpu_mode()

    # -- Combined summary table --
    all_results = standard_results + gpu_results
    print_header("Summary")
    print_summary_table(all_results)

    # -- Mode comparison: standard vs GPU-resident --
    print_header("Standard vs GPU-resident Mode Comparison")
    print(f"  {'Shape':<28} {'Standard':>10} {'GPU-res':>10} {'Speedup':>8}")
    print("  " + "-" * 60)
    for std, gpur in zip(standard_results, gpu_results):
        std_ms = std["gpu_ms"]
        gpur_ms = gpur["gpu_ms"]
        speedup = ""
        if std_ms > 0 and gpur_ms > 0:
            speedup = f"{std_ms / gpur_ms:.2f}x"
        shape_label = std["label"]
        print(
            f"  {shape_label:<28} {format_time(std_ms):>10} {format_time(gpur_ms):>10} {speedup:>8}"
        )

    # -- Peak throughput --
    if all_results:
        best = max(all_results, key=lambda r: r.get("gpu_gflops", 0))
        print(f"\n  Peak GPU throughput: {best['gpu_gflops']:.1f} GFLOP/s at {best['label']}")
        best_cpu = max(all_results, key=lambda r: r.get("cpu_gflops", 0))
        print(f"  Peak CPU throughput: {best_cpu['cpu_gflops']:.1f} GFLOP/s at {best_cpu['label']}")

    print()


if __name__ == "__main__":
    main()
