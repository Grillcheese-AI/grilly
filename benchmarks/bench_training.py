"""
Benchmark: Full training loop (forward -> loss -> backward -> optimizer step).
"""

import sys
import time

import numpy as np

sys.path.insert(0, ".")

from benchmarks.utils import (
    format_time,
    get_gpu_backend,
    print_header,
    print_row,
    print_summary_table,
)


def train_step_gpu(model, optimizer, x, y_true):
    """Single GPU training step."""
    # Forward
    y_pred = model(x)
    # Loss (MSE)
    diff = y_pred - y_true
    loss = np.mean(diff**2)
    # Backward (manual gradient for linear: dL/dW = x.T @ dL/dy)
    2.0 * diff / diff.size
    model.backward(loss)
    # Optimizer step
    optimizer.step()
    return loss


def train_step_cpu(W, b, x, y_true, lr=0.001):
    """Single CPU training step (pure numpy)."""
    # Forward
    y_pred = x @ W.T + b
    # Loss
    diff = y_pred - y_true
    loss = np.mean(diff**2)
    # Backward
    grad_out = 2.0 * diff / diff.size
    grad_W = grad_out.T @ x
    grad_b = np.sum(grad_out, axis=0)
    # Update
    W -= lr * grad_W
    b -= lr * grad_b
    return loss


def main():
    print_header("Training Loop Benchmark")

    backend = get_gpu_backend()

    configs = [
        # (batch, in_features, out_features, steps)
        (32, 128, 64, 20),
        (64, 512, 256, 20),
        (128, 1024, 512, 10),
        (256, 2048, 1024, 5),
    ]

    results = []

    for batch, in_f, out_f, steps in configs:
        label = f"B={batch} {in_f}->{out_f}"
        print(f"\n  Config: {label}, steps={steps}")

        x = np.random.randn(batch, in_f).astype(np.float32)
        y_true = np.random.randn(batch, out_f).astype(np.float32)

        # CPU benchmark
        W = np.random.randn(out_f, in_f).astype(np.float32) * 0.01
        b = np.zeros(out_f, dtype=np.float32)
        t0 = time.perf_counter()
        for _ in range(steps):
            train_step_cpu(W, b, x, y_true)
        cpu_total = (time.perf_counter() - t0) * 1000
        cpu_per_step = cpu_total / steps

        # GPU benchmark
        gpu_per_step = 0
        if backend is not None:
            try:
                from grilly.nn.modules import Linear
                from grilly.optim import Adam

                model = Linear(in_f, out_f)
                optimizer = Adam(model.parameters(), lr=0.001)

                # Warmup
                for _ in range(2):
                    train_step_gpu(model, optimizer, x, y_true)

                t0 = time.perf_counter()
                for _ in range(steps):
                    train_step_gpu(model, optimizer, x, y_true)
                gpu_total = (time.perf_counter() - t0) * 1000
                gpu_per_step = gpu_total / steps
            except Exception as e:
                print(f"    GPU failed: {e}")
                gpu_per_step = 0

        if gpu_per_step > 0:
            print_row(f"{label} /step", gpu_per_step, cpu_per_step)
            results.append(
                {
                    "label": f"{label} /step",
                    "gpu_ms": gpu_per_step,
                    "cpu_ms": cpu_per_step,
                    "shape": f"({batch},{in_f})->({out_f})",
                }
            )
        else:
            print(f"  {label}: CPU only: {format_time(cpu_per_step)} /step")

    if results:
        print_header("Training Summary")
        print_summary_table(results)


if __name__ == "__main__":
    main()
