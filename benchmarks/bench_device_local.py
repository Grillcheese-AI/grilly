"""
DEVICE_LOCAL Pipeline Benchmark

Compares three modes for a Linear -> ReLU -> Linear chain:
  1. CPU numpy (baseline)
  2. GPU HOST_VISIBLE (gpu_mode without device_local)
  3. GPU DEVICE_LOCAL (gpu_mode with device_local=True)

The key bottleneck being tested is PCIe transfer overhead for intermediate
activations.  DEVICE_LOCAL keeps them in VRAM (384 GB/s GDDR6) instead of
crossing PCIe (14 GB/s on PCIe 3.0).

Usage:
    cd grilly && uv run python benchmarks/bench_device_local.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.utils import (
    format_size,
    format_time,
    get_gpu_backend,
    print_header,
    time_cpu,
    time_fn,
)

# ── Shapes to test ──────────────────────────────────────────────────────────
SHAPES = [
    {"name": "Small",  "batch": 32,  "in": 128,  "hidden": 256,  "out": 64},
    {"name": "Medium", "batch": 64,  "in": 512,  "hidden": 1024, "out": 512},
    {"name": "Large",  "batch": 256, "in": 1024, "hidden": 2048, "out": 1024},
    {"name": "XL",     "batch": 512, "in": 2048, "hidden": 4096, "out": 2048},
]

WARMUP = 3
REPEATS = 10


def cpu_chain(x, w1, b1, w2, b2):
    """CPU reference: Linear -> ReLU -> Linear."""
    h = x @ w1.T + b1
    h = np.maximum(0, h)
    return h @ w2.T + b2


def build_chain(in_f, hidden, out_f):
    """Build a grilly Linear->ReLU->Linear chain."""
    from grilly.nn.modules import Linear

    class Chain:
        def __init__(self):
            self.linear1 = Linear(in_f, hidden)
            self.linear2 = Linear(hidden, out_f)

        def forward(self, x):
            backend = self.linear1._get_backend()
            h = self.linear1(x)
            h = backend.fnn.activation_relu(
                h, return_gpu_tensor=self.linear1._return_gpu_tensor
            )
            return self.linear2(h)

        def gpu_mode(self, enable, device_local=True):
            self.linear1.gpu_mode(enable, device_local)
            self.linear2.gpu_mode(enable, device_local)
            return self

        def weights(self):
            w1 = np.asarray(self.linear1.weight, dtype=np.float32)
            b1 = np.asarray(self.linear1.bias, dtype=np.float32)
            w2 = np.asarray(self.linear2.weight, dtype=np.float32)
            b2 = np.asarray(self.linear2.bias, dtype=np.float32)
            return w1, b1, w2, b2

    return Chain()


def verify_correctness():
    """Check that all three modes produce matching results."""
    from grilly.utils.tensor_conversion import VulkanTensor

    print_header("Correctness Verification")
    all_ok = True

    for s in SHAPES:
        rng = np.random.default_rng(42)
        x = rng.standard_normal((s["batch"], s["in"])).astype(np.float32)
        chain = build_chain(s["in"], s["hidden"], s["out"])
        w1, b1, w2, b2 = chain.weights()

        ref = cpu_chain(x, w1, b1, w2, b2)

        # GPU HOST_VISIBLE
        chain.gpu_mode(False)
        out_hv = chain.forward(x)
        if isinstance(out_hv, VulkanTensor):
            out_hv = out_hv.numpy()
        out_hv = np.asarray(out_hv, dtype=np.float32)

        # GPU DEVICE_LOCAL
        chain.gpu_mode(True, device_local=True)
        x_vt = VulkanTensor(x)
        out_dl = chain.forward(x_vt)
        if isinstance(out_dl, VulkanTensor):
            out_dl = out_dl.numpy()
        out_dl = np.asarray(out_dl, dtype=np.float32)

        # Tolerance scales with hidden dim accumulation
        atol = 1e-2 * max(1, s["hidden"] / 128)
        ok_hv = np.allclose(out_hv, ref, atol=atol, rtol=1e-3)
        ok_dl = np.allclose(out_dl, ref, atol=atol, rtol=1e-3)

        diff_hv = float(np.max(np.abs(out_hv - ref)))
        diff_dl = float(np.max(np.abs(out_dl - ref)))

        tag_hv = "PASS" if ok_hv else "FAIL"
        tag_dl = "PASS" if ok_dl else "FAIL"
        print(f"  {s['name']:<8}  HOST_VISIBLE: {tag_hv} (diff={diff_hv:.2e})"
              f"  DEVICE_LOCAL: {tag_dl} (diff={diff_dl:.2e})")
        if not (ok_hv and ok_dl):
            all_ok = False

    return all_ok


def benchmark_all():
    """Run the three-mode benchmark."""
    from grilly.utils.tensor_conversion import VulkanTensor

    print_header("Linear -> ReLU -> Linear  Pipeline Benchmark")
    print(f"  {'Shape':<28} {'CPU':>10} {'GPU(HV)':>10} {'GPU(DL)':>10}"
          f" {'HV/CPU':>8} {'DL/CPU':>8} {'DL/HV':>8}")
    print("  " + "-" * 88)

    summary = []

    for s in SHAPES:
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((s["batch"], s["in"])).astype(np.float32)

        chain = build_chain(s["in"], s["hidden"], s["out"])
        w1, b1, w2, b2 = chain.weights()

        label = (f"{s['name']} ({s['batch']},{s['in']})"
                 f"->{s['hidden']}->{s['out']}")
        data_bytes = x_np.nbytes + w1.nbytes + w2.nbytes

        # ── CPU ─────────────────────────────────────────────────────
        cpu_stats = time_cpu(
            lambda w1=w1, b1=b1, w2=w2, b2=b2, inp=x_np: cpu_chain(inp, w1, b1, w2, b2),
            warmup=WARMUP, repeats=REPEATS,
        )
        cpu_ms = cpu_stats["mean"]

        # ── GPU HOST_VISIBLE (gpu_mode off — numpy round-trips) ────
        chain.gpu_mode(False)
        hv_stats = time_fn(chain.forward, x_np, warmup=WARMUP, repeats=REPEATS)
        hv_ms = hv_stats["mean"]

        # ── GPU DEVICE_LOCAL (gpu_mode on, device_local=True) ──────
        chain.gpu_mode(True, device_local=True)
        x_vt = VulkanTensor(x_np)
        dl_stats = time_fn(chain.forward, x_vt, warmup=WARMUP, repeats=REPEATS)
        dl_ms = dl_stats["mean"]

        # Speedup ratios
        hv_vs_cpu = f"{cpu_ms / hv_ms:.2f}x" if hv_ms > 0 else "N/A"
        dl_vs_cpu = f"{cpu_ms / dl_ms:.2f}x" if dl_ms > 0 else "N/A"
        dl_vs_hv = f"{hv_ms / dl_ms:.2f}x" if dl_ms > 0 else "N/A"

        print(f"  {label:<28} {format_time(cpu_ms):>10} {format_time(hv_ms):>10}"
              f" {format_time(dl_ms):>10} {hv_vs_cpu:>8} {dl_vs_cpu:>8} {dl_vs_hv:>8}")

        summary.append({
            "label": label,
            "cpu_ms": cpu_ms,
            "hv_ms": hv_ms,
            "dl_ms": dl_ms,
            "data": format_size(data_bytes),
        })

    # ── Summary ─────────────────────────────────────────────────────
    print_header("Summary — DEVICE_LOCAL vs HOST_VISIBLE Speedup")
    for r in summary:
        speedup = r["hv_ms"] / r["dl_ms"] if r["dl_ms"] > 0 else 0
        saved = r["hv_ms"] - r["dl_ms"]
        print(f"  {r['label']:<28}  {speedup:.2f}x faster  "
              f"(saved {format_time(saved)} per call)  [{r['data']}]")


def benchmark_layernorm():
    """Benchmark layernorm 3-pass: single-shot vs batched CommandRecorder."""
    print_header("LayerNorm 3-Pass Benchmark (single-shot vs batched)")
    print(f"  {'Shape':<28} {'CPU':>10} {'GPU(HV)':>10} {'GPU(DL)':>10}"
          f" {'DL/HV':>8}")
    print("  " + "-" * 75)

    backend = get_gpu_backend()
    if backend is None:
        print("  GPU backend unavailable")
        return

    for batch, features in [(32, 128), (64, 512), (256, 1024), (512, 2048)]:
        rng = np.random.default_rng(42)
        x = rng.standard_normal((batch, features)).astype(np.float32)
        gamma = np.ones(features, dtype=np.float32)
        beta = np.zeros(features, dtype=np.float32)

        label = f"LN ({batch},{features})"

        # CPU
        def cpu_ln(x=x, g=gamma, b=beta):
            m = x.mean(axis=-1, keepdims=True)
            v = x.var(axis=-1, keepdims=True)
            return (x - m) / np.sqrt(v + 1e-5) * g + b

        cpu_stats = time_cpu(cpu_ln, warmup=WARMUP, repeats=REPEATS)
        cpu_ms = cpu_stats["mean"]

        # GPU HOST_VISIBLE (3 separate dispatches)
        hv_stats = time_fn(
            backend.fnn.layernorm, x, gamma, beta, 1e-5, False,
            warmup=WARMUP, repeats=REPEATS,
        )
        hv_ms = hv_stats["mean"]

        # GPU batched (CommandRecorder, return_gpu_tensor=True)
        from grilly.utils.tensor_conversion import VulkanTensor
        x_vt = VulkanTensor(x)
        dl_stats = time_fn(
            backend.fnn.layernorm, x_vt, gamma, beta, 1e-5, True,
            warmup=WARMUP, repeats=REPEATS,
        )
        dl_ms = dl_stats["mean"]

        dl_vs_hv = f"{hv_ms / dl_ms:.2f}x" if dl_ms > 0 else "N/A"
        print(f"  {label:<28} {format_time(cpu_ms):>10} {format_time(hv_ms):>10}"
              f" {format_time(dl_ms):>10} {dl_vs_hv:>8}")


def main():
    backend = get_gpu_backend()
    if backend is None:
        print("ERROR: GPU backend required. Exiting.")
        sys.exit(1)

    ok = verify_correctness()
    if not ok:
        print("\nWARNING: Correctness checks failed!")
    else:
        print("\n  All correctness checks passed.")

    benchmark_all()
    benchmark_layernorm()
    print()


if __name__ == "__main__":
    main()
