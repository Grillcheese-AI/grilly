# Numerical parity tests (PyTorch reference)

This directory holds tests that compare Grilly outputs to **numpy references** and, when
`torch` is installed, to **`torch.nn.functional`** equivalents.

## Running

```bash
# Core parity (numpy reference only; no optional deps)
uv run pytest tests/parity/ -v

# Include PyTorch cross-checks (requires: pip install "grilly[torch]" or torch)
uv run pytest tests/parity/ -v
```

## Conventions

- **Tolerances**: `rtol=1e-4`, `atol=1e-5` for float32 unless an op documents a looser policy.
- **Weight layout**: `grilly.functional.linear` uses `weight` shaped `(out_features, in_features)`, matching `F.linear` / `nn.Linear.weight`.
- **Markers**: tests are tagged `parity` (see root `tests/conftest.py`).

## Optimizers

`test_optimizers_parity.py` compares **SGD** and **Adam** (CPU path, `use_gpu=False`) against
`torch.optim` on a single tensor when `torch` is installed.

## Dispatch / batching notes

See `docs/PERF_DISPATCH.md` for `VulkanCompute.record_commands`, async dispatch, and Sequential fusion.

## Roadmap

See `docs/PYTORCH_PARITY_TASKLIST.md` (workstream A1). Planned additions: small CNN/MLP modules,
transformer encoder blocks, and a summarized pass/fail table in CI.
