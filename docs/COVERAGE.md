# Test Coverage

## Priority: GPU Coverage

**GPU coverage is the primary metric.** Grilly is a GPU-accelerated framework; the Vulkan/GPU code paths are what matter most. CPU-only runs exercise fallbacks and skip many backend operations.

## Current Status

- **Target:** 80%
- **Full run (with GPU):** ~45%
- **CPU-only:** ~32% (secondary; many GPU paths not exercised)
- **CI:** Installs Vulkan (lavapipe) and runs full test suite; fails if coverage < 40%

## Configuration

Coverage is configured in `pyproject.toml` under `[tool.coverage.run]` and `[tool.coverage.report]`.

**Excluded from measurement:**
- `tests/`, `benchmarks/`, `tutorials/`, `datasets/`, `scripts/`, `examples/`
- `experimental_datasets/`, `mcp-servers/`, `docs/`
- Optional integration modules: `vulkan_sentence_transformer`, `hippocampal_checkpoint`, `visualization`, `vma_wrapper`, `snn_visualizer`, `hilbert`, `_glu_append`

## Reaching 80%

Priority modules to add tests for (high stmt count, low coverage):

| Module | Coverage | Statements |
|--------|----------|------------|
| `nn/autograd.py` | ~26% | 1112 |
| `backend/fnn.py` | ~51% | 1068 |
| `utils/onnx_loader.py` | ~43% | 629 |
| `nn/modules.py` | ~38% | 761 |
| `backend/capsule_transformer.py` | ~16% | 857 |
| `backend/attention.py` | ~29% | 292 |
| `backend/learning.py` | ~30% | 344 |
| `utils/huggingface_bridge.py` | ~34% | 304 |

## Commands

```bash
# Full run with GPU (primary coverage metric)
uv run pytest tests/ --cov=. --cov-report=term --cov-report=html --cov-fail-under=40

# CPU-only (fallback when Vulkan unavailable)
uv run pytest tests/ -m "not gpu" --cov=. --cov-report=term
```
