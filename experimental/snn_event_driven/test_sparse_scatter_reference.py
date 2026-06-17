"""
Tier-0 event-driven synaptic scatter: reference correctness + cost crossover.

Validates that the scatter formulation in synapse_scatter.glsl
    I_acc[post] += sum over fired pre of W[pre, post]
is exactly equal to the dense propagation
    I = spikes @ W
and measures where the sparse path wins as a function of spike activity.

Pure NumPy: tests the ALGORITHM, independent of Vulkan. The on-GPU dispatch
test is a separate step (needs the shaders registered in the build).
"""
import numpy as np

RNG = np.random.default_rng(0xC0DEB00C)


def dense_propagate(spikes, W):
    # spikes: (N,) in {0,1}; W: (N, N) row-major [pre, post]
    return spikes @ W


def sparse_scatter(spikes, W):
    fired = np.flatnonzero(spikes).astype(np.uint32)   # compaction (Kernel A)
    I_acc = np.zeros(W.shape[1], dtype=W.dtype)
    for pre in fired:                                  # scatter (Kernel B)
        I_acc += W[pre]
    return I_acc, fired.size


def test_correctness(N=1024, trials=20):
    max_abs = 0.0
    for _ in range(trials):
        p = float(RNG.uniform(0.0, 0.3))
        spikes = (RNG.random(N) < p).astype(np.float32)
        W = RNG.standard_normal((N, N)).astype(np.float32)
        ref = dense_propagate(spikes, W)
        got, _ = sparse_scatter(spikes, W)
        max_abs = max(max_abs, float(np.max(np.abs(ref - got))))
    # float32 add-order differs between matmul and sequential scatter -> expect
    # tiny rounding, not exact bit-equality. Assert it's at the fp32 noise floor.
    print(f"[correctness] N={N} trials={trials} max|dense-sparse|={max_abs:.3e}")
    assert max_abs < 1e-2, "scatter diverged from dense beyond fp32 rounding"
    return max_abs


def cost_crossover(N=1024):
    """Cost model in units of weight-element reads (the dominant memory traffic).

    dense           : N*N      (reads every W[pre,post] every step)
    sparse work     : fired*N  (reads only fired rows)  -- Tier-1/2 indirect
    tier0 scheduled : N*N invocations launched, but only fired*N touch W;
                      empty invocations cost ~1 cached uint read + return.
    """
    print(f"\n[crossover] N={N}  (dense baseline = {N*N} weight reads/step)")
    print(f"{'activity':>9} {'fired':>6} {'sparse_reads':>13} "
          f"{'work_ratio':>11} {'speedup(x)':>11}")
    for p in (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00):
        fired = max(1, int(round(p * N)))
        sparse_reads = fired * N
        ratio = sparse_reads / (N * N)
        print(f"{p:>9.3f} {fired:>6d} {sparse_reads:>13d} "
              f"{ratio:>11.4f} {1.0/ratio:>11.2f}")
    print("\nReads scale linearly with activity. The Tier-0 early-out captures the\n"
          "weight-read saving (empty invocations skip W); only workgroup launch\n"
          "stays worst-case. True work-proportional LAUNCH needs indirect dispatch.")


if __name__ == "__main__":
    print("=" * 64)
    print("Event-driven synaptic scatter -- reference test")
    print("=" * 64)
    test_correctness(N=1024, trials=20)
    test_correctness(N=4096, trials=8)
    cost_crossover(N=1024)
    cost_crossover(N=4096)
    print("\nAll reference assertions passed.")
