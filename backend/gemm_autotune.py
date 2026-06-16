"""GEMM autotuner: per-shape fp32-tiled vs fp16-coopmat selection, cached to disk.

The resident training GEMM (cpp/src/autograd.cpp: forward_linear / backward_linear)
is currently HARDCODED to `gemm_tiled` (fp32). Whether that or the fp16
`gemm-coopmat-shared` kernel is faster is shape-dependent (coopmat pays a
padding tax — M,K->mult of 16, N->mult of 64 — that only amortizes on large,
well-aligned GEMMs; tiled wins on small/odd shapes). This module measures the
crossover per (M,K,N) on the actual hardware via TapeContext.bench_gemm and
persists a decision table, so a future C++ dispatch hook can pick per shape
instead of always using tiled.

This is the Python-level half: a benchmark + cache framework that runs against
the real C++ kernels. It changes nothing about training on its own — it produces
the table. Wiring the table into the C++ GEMM dispatch is a separate, later step.

Usage:
    from grilly.backend.gemm_autotune import GemmAutotuner
    tuner = GemmAutotuner()                 # loads ~/.grilly/gemm_autotune.json
    variant = tuner.decide(M, K, N)         # "tiled" | "coopmat" (benched+cached)
    # ... later, cheap lookups ...
    variant = tuner.lookup(M, K, N)         # cached value or None

The cache is keyed by device name so a table tuned on one GPU is never applied
to another (coopmat support and the crossover both vary by device/driver).
"""

from __future__ import annotations

import json
import logging
import os
import time

logger = logging.getLogger("grilly.gemm_autotune")

# Decision constants — also the strings a C++ dispatch hook would consult.
TILED = "tiled"      # gemm_tiled, fp32 in/out
COOPMAT = "coopmat"  # gemm-coopmat-shared, fp16 in -> fp32 out (WMMA)

_DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".grilly")
_DEFAULT_CACHE_PATH = os.path.join(_DEFAULT_CACHE_DIR, "gemm_autotune.json")

# Below this fp16 speedup the coopmat path isn't worth the precision/padding
# cost — require a real margin, not noise, before preferring it.
_COOPMAT_MARGIN = 1.05  # fp16 must be >=5% faster than fp32 to win

# SAFETY CAP for the coopmat benchmark dispatch. bench_gemm's coopmat path
# dispatches gx=N/64 workgroups wide and gy=M/16 deep; a vocab-sized M (e.g. the
# 65536x512x1024 grad_weight transpose) means gy=4096 deep over fp16 buffers,
# which wedged the RADV queue hard enough to drop the display (TDR). Above this
# bound on M or N we DO NOT issue the coopmat dispatch — we return TILED by
# policy and record it as capped. Tiled is the current resident default anyway,
# so the worst case is "no speedup measured on that shape", never a GPU hang.
# 65536-on-N (the forward head) benched cleanly pre-cap, but re-issuing huge
# fp16 dispatches is exactly what we want to avoid; its result is already
# cached. The cap sits below any vocab-sized (65536) dimension so neither the
# M=65536 transpose (the crasher) nor the N=65536 head re-benches. Raise via
# the coopmat_max_dim ctor arg only with eyes open.
_DEFAULT_COOPMAT_MAX_DIM = 32768  # refuse coopmat bench when M > this or N > this


def shape_key(M: int, K: int, N: int) -> str:
    """Canonical cache key for a single GEMM shape."""
    return f"{int(M)}x{int(K)}x{int(N)}"


class GemmAutotuner:
    """Benchmark-and-cache GEMM variant selector.

    Holds an in-memory table {shape_key: variant} loaded from / saved to a
    device-scoped section of a JSON file. `decide` benchmarks a shape once
    (warmup is handled inside bench_gemm) and memoizes the winner; `lookup`
    is a pure cache read.
    """

    def __init__(self, ctx=None, *, cache_path: str | None = None,
                 device_name: str | None = None, iters: int = 50,
                 coopmat_max_dim: int = _DEFAULT_COOPMAT_MAX_DIM):
        """Create an autotuner.

        Args:
            ctx: a TapeContext (anything with bench_gemm(M,K,N,iters) ->
                (fp32_ms, fp16_ms)). Optional — only needed to `decide` new
                shapes; `lookup` works without it. If None, `decide` lazily
                builds one via _default_context().
            cache_path: JSON cache file. Defaults to ~/.grilly/gemm_autotune.json.
            device_name: identifier for the GPU section in the cache. Auto-probed
                if None (falls back to env GRILLY_DEVICE_NAME, then "unknown").
            iters: dispatches per timed submit inside bench_gemm.
            coopmat_max_dim: safety cap — if M or N exceeds this, the coopmat
                bench dispatch is skipped entirely and the shape is recorded as
                TILED (capped). Guards against the oversized fp16 dispatch that
                can wedge the GPU queue. See _DEFAULT_COOPMAT_MAX_DIM.
        """
        self._ctx = ctx
        self._iters = int(iters)
        self._coopmat_max_dim = int(coopmat_max_dim)
        self._cache_path = cache_path or _DEFAULT_CACHE_PATH
        self._device_name = device_name or _probe_device_name()
        self._table: dict[str, str] = {}
        self._timings: dict[str, tuple[float, float]] = {}
        self._capped: set[str] = set()
        self._load()

    # ── public API ───────────────────────────────────────────────────────

    def lookup(self, M: int, K: int, N: int) -> str | None:
        """Return the cached variant for a shape, or None if untuned."""
        return self._table.get(shape_key(M, K, N))

    def decide(self, M: int, K: int, N: int, *, force: bool = False) -> str:
        """Return the best variant for a shape, benchmarking + caching if needed.

        Args:
            force: re-benchmark even if the shape is already cached.
        Returns TILED or COOPMAT.
        """
        key = shape_key(M, K, N)
        if not force and key in self._table:
            return self._table[key]

        # SAFETY: never issue the coopmat bench dispatch above the cap — that is
        # the oversized fp16 dispatch that can wedge the GPU. Record TILED
        # (the resident default) by policy and move on; no GPU work issued.
        if int(M) > self._coopmat_max_dim or int(N) > self._coopmat_max_dim:
            self._table[key] = TILED
            self._capped.add(key)
            logger.info("autotune %s -> tiled (capped: M/N > %d, coopmat bench skipped)",
                        key, self._coopmat_max_dim)
            self._save()
            return TILED

        fp32_ms, fp16_ms = self._bench(M, K, N)
        # fp16<=0 (or NaN) means the coopmat path didn't run (unsupported kernel
        # / device) — fall back to tiled. Otherwise require a real margin.
        if fp16_ms is None or fp16_ms <= 0.0 or fp32_ms <= 0.0:
            variant = TILED
        else:
            variant = COOPMAT if (fp32_ms / fp16_ms) >= _COOPMAT_MARGIN else TILED

        self._table[key] = variant
        self._timings[key] = (fp32_ms, fp16_ms)
        logger.info("autotune %s -> %s (fp32=%.3fms fp16=%.3fms)",
                    key, variant, fp32_ms, fp16_ms if fp16_ms else float("nan"))
        self._save()
        return variant

    def tune_shapes(self, shapes, *, force: bool = False) -> dict[str, str]:
        """Decide a batch of (M,K,N) shapes; returns {shape_key: variant}."""
        out = {}
        for (M, K, N) in shapes:
            out[shape_key(M, K, N)] = self.decide(M, K, N, force=force)
        return out

    def timings(self) -> dict[str, tuple[float, float]]:
        """Return measured (fp32_ms, fp16_ms) for shapes benched this session."""
        return dict(self._timings)

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def table(self) -> dict[str, str]:
        return dict(self._table)

    # ── internals ────────────────────────────────────────────────────────

    def _bench(self, M: int, K: int, N: int) -> tuple[float, float]:
        ctx = self._ensure_ctx()
        res = ctx.bench_gemm(int(M), int(K), int(N), self._iters)
        # bench_gemm returns {fp32_ms, fp16_ms} per iter as a length-2 sequence.
        fp32_ms = float(res[0])
        fp16_ms = float(res[1]) if len(res) > 1 else -1.0
        return fp32_ms, fp16_ms

    def _ensure_ctx(self):
        if self._ctx is None:
            self._ctx = _default_context()
        return self._ctx

    def _load(self) -> None:
        try:
            with open(self._cache_path, encoding="utf-8") as fh:
                blob = json.load(fh)
            section = blob.get(self._device_name, {})
            if isinstance(section, dict):
                # Stored as {key: {"variant":..., "fp32":..., "fp16":...,
                # "capped":...}} or the compact {key: variant}. Accept both, and
                # round-trip timings + capped flag so re-saving keeps a complete
                # record instead of dropping prior measurements.
                for k, v in section.items():
                    if isinstance(v, dict):
                        self._table[k] = v.get("variant", TILED)
                        if v.get("capped"):
                            self._capped.add(k)
                        if "fp32" in v and "fp16" in v:
                            self._timings[k] = (float(v["fp32"]), float(v["fp16"]))
                    else:
                        self._table[k] = v
            # Re-assert the safety cap over anything loaded: a result cached
            # before the cap existed (e.g. a measured coopmat win on the
            # N=65536 head) must be demoted so a downstream dispatch hook never
            # picks coopmat on a shape we refuse to even benchmark.
            self._enforce_cap_on_loaded()
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("could not read autotune cache %s: %s", self._cache_path, e)

    def _enforce_cap_on_loaded(self) -> None:
        changed = False
        for k in list(self._table.keys()):
            try:
                M, _, N = (int(x) for x in k.split("x"))
            except ValueError:
                continue
            if (M > self._coopmat_max_dim or N > self._coopmat_max_dim) \
                    and self._table[k] != TILED:
                logger.info("autotune %s: demoting cached %s -> tiled (over cap %d)",
                            k, self._table[k], self._coopmat_max_dim)
                self._table[k] = TILED
                self._capped.add(k)
                self._timings.pop(k, None)
                changed = True
        if changed:
            self._save()

    def _save(self) -> None:
        # Read-modify-write so other devices' sections survive. Atomic replace.
        blob = {}
        try:
            with open(self._cache_path, encoding="utf-8") as fh:
                blob = json.load(fh)
            if not isinstance(blob, dict):
                blob = {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            blob = {}

        section = {}
        for k, variant in self._table.items():
            entry = {"variant": variant}
            if k in self._capped:
                entry["capped"] = True
            if k in self._timings:
                fp32_ms, fp16_ms = self._timings[k]
                entry["fp32"] = round(fp32_ms, 5)
                entry["fp16"] = round(fp16_ms, 5)
            section[k] = entry
        blob[self._device_name] = section

        try:
            os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
            tmp = self._cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(blob, fh, indent=2, sort_keys=True)
            os.replace(tmp, self._cache_path)
        except OSError as e:
            logger.warning("could not write autotune cache %s: %s", self._cache_path, e)


# ── device / context helpers ─────────────────────────────────────────────


def _probe_device_name() -> str:
    """Best-effort GPU identifier for cache scoping."""
    env = os.getenv("GRILLY_DEVICE_NAME")
    if env:
        return env.strip()
    try:
        import grilly_core as _core
        dev = _core.Device()
        for attr in ("device_name", "name", "gpu_name"):
            val = getattr(dev, attr, None)
            if callable(val):
                val = val()
            if isinstance(val, str) and val:
                return val
    except Exception:
        pass
    return "unknown"


def _default_context():
    """Build a TapeContext on a fresh device + loaded shaders.

    Mirrors grilly.backend._bridge._get_device(): Device() then
    load_shaders(<repo>/shaders/spv). bench_gemm needs the gemm_tiled and
    gemm-coopmat-shared SPIR-V loaded.
    """
    import grilly_core as _core

    dev = _core.Device()
    shader_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shaders", "spv"
    )
    if os.path.isdir(shader_dir):
        dev.load_shaders(shader_dir)
    # TapeContext(device, arena_capacity). The binding takes the core context
    # (the Device exposes the pool/batch/cache the TapeContext needs).
    return _core.TapeContext(dev)


# ── model-shape derivation ───────────────────────────────────────────────


def cubby_gemm_shapes(cfg, *, batch: int = 1) -> list[tuple[int, int, int]]:
    """Enumerate the distinct (M,K,N) GEMMs a Cubby config issues per step.

    M = batch*seq (rows), and for each Linear of logical shape in_features=K ->
    out_features=N the forward GEMM is (M,K)x(K,N). The dominant ones:
      - SwiGLU gate+up:  d_model -> 2*d_ffn
      - SwiGLU down:     d_ffn   -> d_model
      - MinGRU proj:     d_model -> 3*d_model   (G,V,D fused)
      - output head:     d_model -> vocab       (tied linear; the big one)
    Backward adds the transposed companions (M<->N swap and the grad_weight
    GEMM), so we include those too since they hit the same dispatch.
    """
    M = int(batch) * int(cfg.seq_len)
    d = int(cfg.d_model)
    dffn = int(cfg.d_ffn)
    vocab = int(cfg.total_vocab)

    fwd = [
        (M, d, 3 * d),        # MinGRU G/V/D projection
        (M, d, 2 * dffn),     # SwiGLU gate+up
        (M, dffn, d),         # SwiGLU down
        (M, d, vocab),        # tied output head (v3.3 bottleneck)
    ]
    # Backward grad_input GEMM reuses (M, out, in); grad_weight GEMM is
    # (out, M, in) after transpose. Include the head's, which dominates.
    bwd = [
        (M, vocab, d),        # head grad_input
        (vocab, M, d),        # head grad_weight (transposed)
        (M, 2 * dffn, d),     # swiglu up grad_input
    ]
    # De-dup while preserving order.
    seen, shapes = set(), []
    for s in fwd + bwd:
        if s not in seen:
            seen.add(s)
            shapes.append(s)
    return shapes
